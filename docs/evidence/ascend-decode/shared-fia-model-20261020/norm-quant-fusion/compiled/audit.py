"""Recompute graph comparisons from saved device tensors on server CPU."""

import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch


ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


torch.set_num_threads(1)
results = {}
for arm in ("on", "off"):
    root = ROOT / f"device-01-{arm}"
    report = json.loads((root / "result.json").read_text())
    device = report["device_result"]
    assert report["status"] == "pass"
    assert digest(root / "outputs.pt") == device["outputs_sha256"]
    assert digest(root / "compiled-ir.py") == device["compiled_ir_sha256"]
    operators = Counter(
        ast.unparse(node.func).removeprefix("torch.ops.")
        for node in ast.walk(ast.parse((root / "compiled-ir.py").read_text()))
        if isinstance(node, ast.Call)
    )
    fused = arm == "on"
    assert operators["npu.npu_add_rms_norm_dynamic_quant.default"] == fused
    assert operators["_C_ascend.npu_add_rms_norm_bias.default"] == (not fused)
    assert operators["npu.npu_dynamic_quant.default"] == (not fused)
    saved = torch.load(root / "outputs.pt", map_location="cpu",
                       weights_only=True)
    assert [v["input"] for v in saved["graph_replays"]] == [
        "original", "changed", "restored",
    ]
    comparisons = []
    for row in saved["graph_replays"]:
        counts = {}
        for name, actual, expected, eager in zip(
            ("q", "scale", "residual"), row["graph"], row["expected"],
            row["eager"],
        ):
            assert actual.dtype == expected.dtype == eager.dtype
            assert actual.shape == expected.shape == eager.shape
            assert torch.equal(actual.view(torch.uint8),
                               expected.view(torch.uint8))
            counts[name] = int((actual != eager).sum())
        assert counts["residual"] == 0
        assert (counts["q"] > 0) == fused
        assert (counts["scale"] > 0) == fused
        comparisons.append({"input": row["input"],
                            "different_from_unfused": counts})
    first, changed, restored = [r["graph"] for r in saved["graph_replays"]]
    assert all(torch.equal(a, b) for a, b in zip(first, restored))
    assert any(not torch.equal(a, b) for a, b in zip(first, changed))
    assert all(torch.equal(a, b)
               for a, b in zip(first, saved["compiled_warm"]))
    results[arm] = {
        "comparisons": comparisons, "peak_bytes": device["peak_allocated"],
        "ir_sha256": device["compiled_ir_sha256"],
        "output_sha256": device["outputs_sha256"],
        "report_sha256": digest(root / "result.json"),
    }
result = {"status": "pass", "recomputed_tensor_comparisons": 18,
          "changed_input_observed": True, "restore_matches_original": True,
          "arms": results, "auditor_sha256": digest(Path(__file__))}
(ROOT / "raw-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
