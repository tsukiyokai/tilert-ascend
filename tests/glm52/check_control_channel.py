"""Check the production TP16 host channel after an idle request interval."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

import torch
import torch.distributed as distributed


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.models.glm_5_2_ascend.tp_runtime import (  # noqa: E402
    TorchControlChannel,
)


def run(args):
    torch.set_num_threads(1)
    distributed.init_process_group("gloo")
    channel = TorchControlChannel()
    commands = [
        {"serial": index + 1, "operation": operation,
         "request_id": "idle-check", "position": position}
        for index, (operation, position) in enumerate(
            (("inject", 0), ("step", 8192), ("reset", 8193), ("close", 0))
        )
    ]
    distributed.barrier()
    started = time.monotonic()
    if channel.rank == 0:
        time.sleep(args.idle_seconds)
        for command in commands:
            channel.send(command)
        received = commands
    else:
        received = [channel.receive() for _ in commands]
    receipt = {
        "rank": channel.rank, "commands": received,
        "elapsed_seconds": time.monotonic() - started,
        "torch_npu_imported": "torch_npu" in sys.modules,
    }
    receipts = [None] * 16
    distributed.all_gather_object(receipts, receipt, group=channel.group)
    passed = all(item["commands"] == commands
                 and not item["torch_npu_imported"] for item in receipts)
    if channel.rank == 0:
        sources = (Path(__file__), ROOT / (
            "tilert/models/glm_5_2_ascend/tp_runtime.py"
        ))
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pass" if passed else "fail",
            "scope": "CPU Gloo TP16 idle and ordered object broadcasts",
            "idle_seconds": args.idle_seconds, "receipts": receipts,
            "default_backend": distributed.get_backend(),
            "control_backend": distributed.get_backend(channel.group),
            "torch": str(torch.__version__), "hostname": socket.gethostname(),
            "command": [sys.executable, *sys.argv],
            "source_sha256": {str(path.relative_to(ROOT)):
                              hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "ranks": 16,
                          "idle_seconds": args.idle_seconds}), flush=True)
    distributed.destroy_process_group()
    return int(not passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idle-seconds", type=float, default=141)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
