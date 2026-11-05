"""CPU binding and byte audit before current-kernel teacher-route reconstruction."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same_bytes(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.view(torch.uint8), right.view(torch.uint8)))


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("frozen_collect", ROOT / "collect.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt = json.loads((ROOT / "norm-result.json").read_text())
    prepared = json.loads((ROOT / "prepare-result.json").read_text())
    report = json.loads((module.OLD / "rank-0.json").read_text())
    teacher = json.loads(module.TEACHER.read_text())
    binding = {
        "script_sha256": sha(ROOT / "collect.py"),
        "teacher_sha256": sha(module.TEACHER),
        "checkpoint_sha256": {name: sha(module.CHECKPOINT / name)
                              for name in report["checkpoint_sha256"]},
        "old_report_sha256": sha(module.OLD / "rank-0.json"),
        "weight_reader_sha256": sha(ROOT / "source/weights.py"),
        "byte_estimator_sha256": sha(ROOT / "source/estimate_traffic.py"),
    }
    assert receipt["status"] == "pass"
    assert binding == receipt["binding"] == prepared["binding"]
    assert report["rank"] == 0 and report["layers"] == 78 and report["steps"] == 32
    assert report.get("query_width", 1) == 1
    assert teacher["teacher_tokens"][1:] == teacher["expected_next_tokens"][:-1]
    assert teacher["first_token_id"] == teacher["teacher_tokens"][0]
    assert report["teacher_tokens"] == teacher["teacher_tokens"]
    assert report["expected_tokens"] == teacher["expected_next_tokens"]
    source = report["teacher_source"]
    assert source["sha256"] == binding["teacher_sha256"]
    assert source["prompt_file_sha256"] == teacher["prompt_sha256"]
    assert source["teacher_tokens"] == teacher["teacher_tokens"]
    assert source["expected_next_tokens"] == teacher["expected_next_tokens"]
    assert report["inputs"]["prompt_length"] == teacher["prompt_tokens"] == 8192
    assert report["inputs"]["first_token_id"] == teacher["first_token_id"]
    assert source["prompt_identity"]["prompt_sha256"] == report["inputs"]["prompt_sha256"]

    layers = []
    for layer in module.LAYERS:
        path = module.raw_path(layer)
        expected = [value for name, value in report["artifact_sha256"].items()
                    if name.endswith(f"/rank-0/layer-{layer:02d}.pt")]
        assert path is not None and len(expected) == 1 and sha(path) == expected[0]
        data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        assert data["layer"] == layer and data["rank"] == 0
        assert data["positions"] == list(range(8192, 8224))
        assert data["teacher_tokens"] == teacher["teacher_tokens"]
        assert data["inputs"] == report["inputs"]
        assert data["teacher_source"] == source
        assert data["checkpoint_sha256"] == binding["checkpoint_sha256"]
        value = data["residual"]
        assert value.shape == (32, 6144) and value.dtype == torch.bfloat16
        assert torch.isfinite(value).all()
        layers.append({"layer": layer, "path": str(path), "sha256": expected[0]})
        del data, value

    outputs_path = ROOT / "norm-outputs.pt"
    assert sha(outputs_path) == receipt["norm_output_sha256"]
    outputs = torch.load(outputs_path, map_location="cpu", weights_only=True)
    value = module.residual(3, report, teacher)
    cases = module.fixture(value)
    assert list(outputs) == [name for name, _, _ in cases] == receipt["norm_cases"]
    checks = []
    for name, x, residual in cases:
        saved = outputs[name]
        assert set(saved) == {"y", "recovered", "residual"}
        assert all(t.shape == (1, 6144) and t.dtype == torch.bfloat16
                   and torch.isfinite(t).all() for t in saved.values())
        assert same_bytes(saved["y"], saved["recovered"])
        assert same_bytes(saved["residual"], (x.float() + residual.float()).to(torch.bfloat16))
        checks.append({"case": name, "y_recovered_bytes_equal": True,
                       "residual_expected_bytes_equal": True, "finite": True})

    evidence, launch_keys, runtime_names = [], [], []
    logs = sorted((ROOT / "norm-logs/debug/plog").glob("*.log"))
    for path in logs:
        for line, text in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if "Launch dynamic kernel AddRmsNormBias task" in text:
                launch_keys.append(int(re.search(r"tilingKey is (\d+)", text)[1]))
            elif "StreamLaunchKernelV2: kernel info" in text and "kernel_name=AddRmsNormBias_" in text:
                runtime_names.append(re.search(r"kernel_name=([^,]+)", text)[1])
            elif not ("AddRmsNormBias" in text and "Tiling outputs of" in text):
                continue
            evidence.append({"path": str(path), "line": line, "text": text})
    assert launch_keys == [33] * 12
    assert len(runtime_names) == 12 and len(set(runtime_names)) == 1
    assert runtime_names[0].endswith("_33")
    assert receipt["peak_allocated_bytes"] <= module.QUOTA
    result = {"status": "pass", "cpu_only": True, "binding": binding,
              "layers": layers, "layer_count": len(layers), "norm_checks": checks,
              "norm_result_sha256": sha(ROOT / "norm-result.json"),
              "norm_output_sha256": sha(outputs_path), "actual_launch_keys": launch_keys,
              "actual_runtime_kernel": runtime_names[0], "launch_evidence": evidence,
              "launch_log_sha256": {str(path): sha(path) for path in logs},
              "scope": receipt["scope"], "historical_native_identity_verified": False,
              "saved_roundtrip_residual_again_available": False,
              "saved_roundtrip_note": "Device script asserted merged_again equality; raw file saves only first merged residual.",
              "gate_source_sha256": sha(Path(__file__))}
    (ROOT / "cpu-gate-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "cpu_only", "layer_count", "actual_runtime_kernel")}))


if __name__ == "__main__":
    main()
