"""Check the frozen diagnostic and reproduce fixture data without an NPU."""

import hashlib
import json
from pathlib import Path
import platform
import sys

import torch
import torch_npu

from attention_check import metric
from vendor_hook_forward import initial_cache_values, page_plan
import vendor_hook_diagnostic_test as checks


def main():
    source = Path("/workspace/outer-forward-int8-m12-r1.pt")
    saved = torch.load(source, map_location="cpu", weights_only=False)
    generator = torch.Generator().manual_seed(41023 + 12 * 100 + 1)
    hidden = torch.randn(12, 6144, generator=generator).bfloat16()
    plan = page_plan(torch, 12, 1, 8192, 66, 66)
    caches = initial_cache_values(torch, saved["initial_caches"])
    tests = []
    for name in sorted(vars(checks)):
        if name.startswith("test_"):
            getattr(checks, name)()
            tests.append({"name": name, "pass": True})
    comparisons = {"hidden": metric(torch, hidden, saved["hidden"])}
    comparisons.update({f"metadata.{name}": metric(torch, value, saved["metadata"][name])
                        for name, value in plan.items()})
    comparisons.update({f"initial_cache.{index}": metric(torch, actual, expected)
                        for index, (actual, expected) in enumerate(
                            zip(caches, saved["initial_caches"]))})
    npu_initialized = torch.npu.is_initialized()
    report = {
        "scope": "Same-image CPU data regeneration and diagnostic contract checks; no NPU devices exposed",
        "python": sys.version, "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "machine": platform.machine(),
        "image_id": "sha256:9afbed90a0f92e228cc1ba13adf3cee2e89e34f1dd4cad1ed8eb5ce8c091b4af",
        "input_dump_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scripts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in Path("/workspace").glob("*.py")},
        "comparisons": comparisons, "tests": tests,
        "npu_initialized": npu_initialized,
        "pass": not npu_initialized and all(
            value["pass"] and value["bitwise_equal"] for value in comparisons.values()),
    }
    Path("/evidence/vendor-hook-diagnostic-cpu-v10.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return int(not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())
