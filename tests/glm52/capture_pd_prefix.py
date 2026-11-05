"""Receive and freeze TP16 vLLM/HIXL prefixes without decode weights."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.pd_vllm.decode_timing import atomic_json  # noqa: E402
from tilert.pd_vllm.prefix_capture import digest  # noqa: E402


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hixl-host", required=True)
    parser.add_argument("--hixl-port-base", type=int, default=27100)
    parser.add_argument("--hixl-library", type=Path)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--ctrl-port", type=int, default=5557)
    parser.add_argument("--http-port", type=int, default=18081)
    parser.add_argument("--capacity", type=int, default=9344)
    parser.add_argument("--mtp-parent-hidden", action="store_true",
                        help="Receive the final normalized target row")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.hixl_port_base <= 65520 or args.timeout <= 0:
        parser.error("HIXL ports must fit sixteen endpoints; timeout > 0")
    return args


def run(args):
    import torch
    import torch.distributed as distributed
    import torch_npu
    import uvicorn

    from tilert.models.glm_5_2_ascend.tp_runtime import (
        CoordinatedBackend, RankExecutor, TorchControlChannel,
    )
    from tilert.pd_vllm import decode_server
    from tilert.pd_vllm.ascend_cache import (
        AscendReceiveBuffer, checkpoint_fingerprint, receive_device_id,
    )
    from tilert.pd_vllm.ascend_receiver import AscendReceiveServer
    from tilert.pd_vllm.hixl_transport import HixlTransport
    from tilert.pd_vllm.prefix_capture import (
        CaptureRankEngine, PrefixCaptureEngine,
    )

    torch.set_num_threads(1)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"npu:{receive_device_id(local_rank)}")
    torch.npu.set_device(device)
    distributed.init_process_group("hccl")
    rank = distributed.get_rank()
    if distributed.get_world_size() != 16 or rank != local_rank:
        raise ValueError("Prefix capture requires one local TP16 group")
    checkpoint = checkpoint_fingerprint(args.checkpoint)
    transport = HixlTransport(library_path=args.hixl_library,
                              port=args.hixl_port_base + rank)
    buffer = AscendReceiveBuffer(rank, device, args.capacity, transport,
                                args.hixl_host,
                                parent_hidden=args.mtp_parent_hidden)
    endpoints = [None] * 16
    distributed.all_gather_object(endpoints, buffer.endpoint())
    channel = TorchControlChannel()

    def gather(receipt):
        values = [None] * 16
        distributed.all_gather_object(values, receipt)
        return values

    rank_engine = CaptureRankEngine(buffer, checkpoint, args.output, gather)
    executor = RankExecutor(rank_engine, rank, lambda request, _rank: request)
    files = [Path(__file__), *(ROOT / "tilert/pd_vllm").glob("*.py"),
             ROOT / "tilert/models/glm_5_2_ascend/session.py",
             ROOT / "tilert/models/glm_5_2_ascend/tp_runtime.py"]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "command": [sys.executable, *sys.argv],
        "rank": rank, "world_size": 16, "device": str(device),
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "python": platform.python_version(), "layout": buffer.layout,
        "checkpoint_sha256": checkpoint,
        "source_sha256": {str(path.relative_to(ROOT)): digest(path)
                          for path in files},
        "scope": "Original HIXL prefix export; no decode inference",
        "decode_weights_loaded": False, "numerical_execution": False,
        "tpot_measured": False, "status": "initialized",
    }
    report = args.output / f"runtime-rank-{rank}.json"
    atomic_json(report, metadata)
    if rank == 0:
        backend = CoordinatedBackend(executor, channel)
        engine = PrefixCaptureEngine(
            backend, rank_engine, args.output, metadata,
            device_context=lambda: torch.npu.device(device),
        )
        server = AscendReceiveServer(
            buffer.layout, endpoints, checkpoint, host=args.bind_host,
            ctrl_port=args.ctrl_port, request_timeout=args.timeout,
        )
        try:
            uvicorn.run(decode_server.build_app(server, engine),
                        host=args.bind_host, port=args.http_port)
        finally:
            server.close()
            if not backend.failed:
                with torch.npu.device(device):
                    backend.close()
    else:
        executor.serve(channel)
    distributed.barrier()
    transport.close()
    metadata.update(status="stopped", command_serial=executor.serial,
                    final_position=rank_engine.position)
    atomic_json(report, metadata)
    distributed.destroy_process_group()


def main(argv=None):
    run(parse_arguments(argv))


if __name__ == "__main__":
    main()
