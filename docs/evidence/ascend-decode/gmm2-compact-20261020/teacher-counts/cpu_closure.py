"""Check the new declaration and reuse the audited per-core schedule model."""

import hashlib
import json
from pathlib import Path
import runpy

root = Path("/workspace/experiments/gmm2-compact-20261020")
out = root / "teacher-counts-screen"
source = out / "fixed-wide-buffer-build/source/ops_transformer/ascendc/grouped_matmul/op_kernel"
helper = root / "retile-cpu-agent/audit.py"
assert hashlib.sha256(helper.read_bytes()).hexdigest() == (
    "13b90f39649c404e337465e183d353c97d6fef6e55af646dd81d3d820b413d91")
check = runpy.run_path(str(helper), run_name="schedule_helper")["check"]
fixture = json.loads((out / "counts.json").read_text())
result = {"status": "pass", "npu_launched": False, "cases": []}
for case in fixture["cases"]:
    compact = check(case["counts"], 64, 512)
    wide = check(case["counts"], 32, 1024)
    assert compact["tiles"] == case["compact"]["tasks"]
    assert wide["tiles"] == case["wide"]["tasks"]
    result["cases"].append({"name": case["name"],
                            "compact": compact, "wide": wide})
header = source / "grouped_matmul_antiquant_a8w4_msd.h"
assert "4 * tiling->ubCalSize * sizeof(float) : tiling->ubRestBytes" in header.read_text()
result["tmp_declared_bytes"] = 4 * 8192 * 4
result["ub_platform_bytes"] = 196352
result["ub_end_by_arm"] = {}
for arm, n in (("compact", 512), ("wide", 1024)):
    end = n * 4 + 128 + 32768 + 16384 + result["tmp_declared_bytes"]
    assert end <= result["ub_platform_bytes"]
    result["ub_end_by_arm"][arm] = end
result["ub_scope"] = "Existing allocation model: withOffset=0, 32-byte sequential queues, Matmul shareUbSize=0; no asynchronous or arithmetic simulation."
result["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [helper, header, Path(__file__)]}
(out / "cpu-closure.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
