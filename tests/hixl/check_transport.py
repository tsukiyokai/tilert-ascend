"""Check one HIXL device WRITE with an independent receiver process."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.pd_vllm.hixl_transport import HixlTransport  # noqa: E402


def pattern(torch, size):
    return torch.arange(size, dtype=torch.int32).remainder(251).byte()


def receive(connection, args):
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(1)
    device = f"npu:{args.receiver_device}"
    torch.npu.set_device(device)
    storage = torch.full((args.bytes,), 255, dtype=torch.uint8, device=device)
    torch.npu.synchronize()
    transport = HixlTransport(args.library, port=args.port_base + 1)
    transport.init(args.host)
    transport.register(storage.data_ptr(), storage.numel(),
                       args.receiver_device)
    connection.send({"base_ptr": storage.data_ptr(), **transport.local_meta()})
    written = connection.recv()
    torch.npu.synchronize()
    actual = storage.cpu()
    connection.send({
        "receiver_pid": os.getpid(),
        "written": written,
        "exact_pattern": bool(torch.equal(actual, pattern(torch, args.bytes))),
        "unchanged": bool(torch.all(actual == 255)),
        "received_sha256": hashlib.sha256(bytes(actual.tolist())).hexdigest(),
    })
    connection.recv()
    transport.close()
    connection.close()


def run(args):
    context = multiprocessing.get_context("spawn")
    connection, child_connection = context.Pipe()
    child = context.Process(target=receive, args=(child_connection, args))
    child.start()
    child_connection.close()
    endpoint = connection.recv()

    import torch
    import torch_npu

    torch.set_num_threads(1)
    device = f"npu:{args.sender_device}"
    torch.npu.set_device(device)
    source = pattern(torch, args.bytes).to(device)
    transport = HixlTransport(args.library, port=args.port_base)
    transport.init(args.host)
    transport.register(source.data_ptr(), source.numel(), args.sender_device)
    error = None
    try:
        transport.write(endpoint, [source.data_ptr()], [endpoint["base_ptr"]],
                        [source.numel()])
    except RuntimeError as failure:
        error = str(failure)
    connection.send(error is None)
    received = connection.recv()
    try:
        transport.close()
    finally:
        connection.send(None)
    child.join()
    connection.close()
    passed = (error is None and received["exact_pattern"]
              and child.exitcode == 0)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail", "error": error,
        "bytes": args.bytes, "sender_device": args.sender_device,
        "receiver_device": args.receiver_device, "endpoint": endpoint,
        "receiver": received, "receiver_exit": child.exitcode,
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "command": [sys.executable, *sys.argv],
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "library_sha256": hashlib.sha256(
            args.library.read_bytes()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--sender-device", type=int, default=0)
    parser.add_argument("--receiver-device", type=int, default=1)
    parser.add_argument("--port-base", type=int, default=29000)
    parser.add_argument("--bytes", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
