"""Run rank-zero PDEngine requests with fifteen TP16 CANN worker loops."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=9344)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def run(args, report):
    import torch
    import torch.distributed as distributed
    import torch_npu

    from tilert.models.glm_5_2_ascend.native_engine import NativeDecodeEngine
    from tilert.models.glm_5_2_ascend.session import (
        AscendPDEngine, PrefixRequest,
    )
    from tilert.models.glm_5_2_ascend.tp_runtime import (
        CoordinatedBackend, RankExecutor, TorchControlChannel,
    )

    requests = json.loads(args.requests.read_text())
    if not isinstance(requests, list) or not requests:
        raise ValueError("--requests must contain a nonempty JSON list")
    config = json.loads((args.checkpoint / "config.json").read_text())
    stop_ids = config["eos_token_id"]
    if isinstance(stop_ids, int):
        stop_ids = [stop_ids]
    torch.set_num_threads(1)
    device = f"npu:{int(os.environ['LOCAL_RANK'])}"
    torch.npu.set_device(device)
    distributed.init_process_group("hccl")
    rank = distributed.get_rank()
    report.update(rank=rank, world_size=distributed.get_world_size(),
                  device=device, torch=torch.__version__,
                  torch_npu=torch_npu.__version__)
    engine = NativeDecodeEngine(args.checkpoint, device, args.capacity)

    def prefix_loader(request, local_rank):
        directory = request.metadata.get("directory")
        if request.seq_len == 0 and directory is None:
            return None
        path = Path(directory) / f"rank-{local_rank}.pt"
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if bundle["metadata_sha256"] != report["metadata_sha256"]:
            raise ValueError("Prefix checkpoint differs from decode")
        return [{name: value.to(device) for name, value in item.items()}
                for item in bundle["layers"]]

    executor = RankExecutor(engine, rank, prefix_loader)
    channel = TorchControlChannel()
    if rank == 0:
        backend = CoordinatedBackend(executor, channel)
        session = AscendPDEngine(backend, stop_ids=stop_ids)
        report["requests"] = []
        for request in requests:
            prefix = PrefixRequest(
                request["request_id"], request["seq_len"],
                {"directory": request["prefix"]} if request.get("prefix")
                else {},
            )
            delivered = []
            session.inject(prefix)
            tokens = session.decode(
                request["first_token_id"], request["max_tokens"],
                sampling=request.get("sampling"), on_token=delivered.append,
            )
            report["requests"].append({
                "request_id": prefix.request_id, "seq_len": prefix.seq_len,
                "tokens": tokens, "callback_tokens": delivered,
                "stats": dict(session.last_stats),
                "delivery_matches_return": tokens == delivered,
            })
            session.reset()
        backend.close()
    else:
        executor.serve(channel)
    positions = torch.tensor([engine.position], dtype=torch.int64,
                             device=device)
    values = [torch.empty_like(positions) for _ in range(16)]
    distributed.all_gather(values, positions)
    report["final_positions"] = [int(value.item()) for value in values]
    if any(report["final_positions"]):
        raise ValueError("Session reset left nonzero cache positions")
    report["command_serial"] = executor.serial
    report["status"] = "capture_complete"
    distributed.barrier()
    distributed.destroy_process_group()


def main():
    args = parse_arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), *(root / "tilert/models/glm_5_2_ascend").rglob(
        "*.py"
    )]
    report = {
        "status": "error", "git_head": args.source_commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "python": platform.python_version(),
        "command": [sys.executable, *sys.argv],
        "scope": "TP16 request/control and frozen-prefix injection capture",
        "numerical_parity": "not compared", "hixl_transfer": "not executed",
        "timed": False, "profiled": False,
    }
    try:
        report["source_sha256"] = {
            str(path.relative_to(root)): digest(path) for path in sources
        }
        report["requests_sha256"] = digest(args.requests)
        report["metadata_sha256"] = {
            name: digest(args.checkpoint / name) for name in (
                "config.json", "quant_model_description.json",
                "quant_model_weights.safetensors.index.json",
            )
        }
        run(args, report)
    except Exception:
        report["error"] = traceback.format_exc()
    rank = int(os.environ.get("RANK", "0"))
    output = args.output / f"rank-{rank}.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rank": rank, "status": report["status"],
                      "report": str(output)}))
    return int(report["status"] != "capture_complete")


if __name__ == "__main__":
    raise SystemExit(main())
