"""Check real NPU token payloads and stream ordering on one selected die."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.models.glm_5_2_ascend.cann_model import (  # noqa: E402
    CannModel, VOCAB_SIZE, device_tokens,
)
from tilert.pd_vllm.decode_timing import atomic_json  # noqa: E402


def check(device, report):
    import torch
    import torch_npu  # noqa: F401

    device = torch.device(device)
    torch.npu.set_device(device)
    model = SimpleNamespace(device=device)
    stream = torch.npu.Stream(device=device)
    report["packages"] = {name: version(name)
                          for name in ("torch", "torch-npu")}
    report["device"] = str(device)
    report["device_name"] = torch.npu.get_device_name(device)
    for width in (1, 2, 4, 6, 8, 12, 16, 24, 32):
        for generation in range(4):
            expected = [(generation * 257 + row * 19) % VOCAB_SIZE
                        for row in range(width)]
            with torch.inference_mode(), torch.npu.stream(stream):
                host = device_tokens(expected, device)
                produced = host + 1
                prediction = tuple(token + 1 for token in expected)
                assert CannModel.host_tokens(model, produced) == (prediction,)
                mixed = [expected[0], *host[1:].unbind()]
                assert device_tokens(mixed, device).tolist() == expected
                assert CannModel.host_tokens(model, expected, produced) == (
                    tuple(expected), prediction,
                )
                assert device_tokens(host.int(), device).tolist() == expected
            report["checks"].append({
                "width": width, "generation": generation, "passed": True,
            })
    stream.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [Path(__file__), ROOT / "tilert/pd_vllm/decode_timing.py",
             ROOT / "tilert/models/glm_5_2_ascend/cann_model.py"]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Selected-die integer token staging; no model or timing",
        "hostname": platform.node(), "command": sys.argv, "checks": [],
        "precision_verified": False, "performance_measured": False,
        "source_sha256": {str(path.relative_to(ROOT)):
                          hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in paths},
    }
    try:
        check(args.device, report)
        report["status"] = "pass"
    except Exception as error:
        report.update(status="fail", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "device")}))


if __name__ == "__main__":
    main()
