"""Check HF trace/prefix metadata gates with stdlib tensor metadata doubles."""

import argparse
import ast
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from tilert.models.glm_5_2_ascend.ops.indexer import (
    cann_arch22_scoring_contract,
)


@dataclass
class TensorMetadata:
    shape: tuple
    dtype: str
    device: object = None

    def __post_init__(self):
        self.device = SimpleNamespace(type="cpu")


def load_runner(filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_prefix_validator(path):
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "prefix_length")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"LAYERS": 78, "torch": SimpleNamespace(
        bfloat16="BF16", float16="FP16", int8="INT8",
    )}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["prefix_length"]


def make_prefix(length):
    entries = []
    for layer in range(78):
        entry = {
            "latent": TensorMetadata((length, 512), "BF16"),
            "rope": TensorMetadata((length, 64), "BF16"),
        }
        if layer < 3 or (layer - 2) % 4 == 0:
            quantized = 6 <= layer <= 66
            entry["index_keys"] = TensorMetadata(
                (length, 128), "INT8" if quantized else "BF16"
            )
            if quantized:
                entry["index_scales"] = TensorMetadata((length,), "FP16")
        entries.append(entry)
    return entries


def make_trace():
    rows, routes, scores = [], [], []
    for layer in range(78):
        full = layer < 3 or (layer - 2) % 4 == 0
        rows.append({
            "latent": object(), "rope": object(),
            "index": object() if full else None,
            "index_scale": object() if full and 6 <= layer <= 66 else None,
        })
        routes.append(object() if layer >= 3 else None)
        scores.append(TensorMetadata((8193,), "FP32") if full else None)
    return {
        "token": 1, "position": 8192, "committed_position": 8193,
        "hidden_states": (None,) * 79, "layer_logits": (None,) * 78,
        "indices": [object() for _ in range(78)], "routes": routes,
        "cache_rows": rows,
        "index_scores": scores,
        "attention_weights": [TensorMetadata((64, 8193), "BF16")
                              for _ in range(78)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    reference = root / "tilert/models/glm_5_2_ascend/reference.py"
    hf = load_runner("run_hf_reference.py")
    native = load_runner("run_native_decode.py")
    validate_prefix = load_prefix_validator(reference)
    checks = []

    def verify(name, operation, accepted):
        try:
            operation()
            actual = True
        except ValueError:
            actual = False
        checks.append({"name": name, "expected_acceptance": accepted,
                       "passed": actual == accepted})

    for length in (0, 1, 8192):
        verify(f"prefix.valid.{length}",
               lambda: validate_prefix(make_prefix(length)), True)
    for name, layer, field, value in (
        ("short_index", 6, "index_keys", TensorMetadata((0, 128), "INT8")),
        ("short_rope", 77, "rope", TensorMetadata((8191, 64), "BF16")),
        ("wrong_latent", 4, "latent", TensorMetadata((8192, 64), "BF16")),
        ("wrong_c8_dtype", 6, "index_keys",
         TensorMetadata((8192, 128), "BF16")),
        ("wrong_scale_dtype", 6, "index_scales",
         TensorMetadata((8192,), "BF16")),
        ("short_scale", 6, "index_scales", TensorMetadata((1,), "FP16")),
    ):
        prefix = make_prefix(8192)
        prefix[layer][field] = value
        verify("prefix." + name, lambda: validate_prefix(prefix), False)
    prefix = make_prefix(8192)
    del prefix[6]["index_scales"]
    verify("prefix.missing_scale", lambda: validate_prefix(prefix), False)
    verify("prefix.missing_layer", lambda: validate_prefix(prefix[:-1]), False)

    trace = make_trace()
    verify("trace.valid", lambda: hf.validate_trace(trace), True)
    for field in ("hidden_states", "layer_logits", "indices", "routes",
                  "cache_rows", "index_scores", "attention_weights"):
        incomplete = copy.deepcopy(trace)
        incomplete[field] = []
        verify("trace.empty." + field,
               lambda: hf.validate_trace(incomplete), False)
    for name, layer, field in (("index", 6, "index"),
                               ("scale", 6, "index_scale"),
                               ("latent", 77, "latent")):
        incomplete = copy.deepcopy(trace)
        incomplete["cache_rows"][layer][field] = None
        verify("trace.missing_" + name,
               lambda: hf.validate_trace(incomplete), False)
    wrong = copy.deepcopy(trace)
    wrong["committed_position"] = 8192
    verify("trace.uncommitted", lambda: hf.validate_trace(wrong), False)

    scoring = cann_arch22_scoring_contract()
    scoring["source_verified"] = True
    scorer = "tilert/models/glm_5_2_ascend/ops/indexer.py"
    manifest = {
        "implementation": "transformers.GlmMoeDsaForCausalLM",
        "status": "pass", "numerical_execution": True,
        "tp_size": 16, "metadata_sha256": {"config.json": "checkpoint"},
        "source_sha256": {
            "generator.py": "generator",
            scorer: scoring["reference_source_sha256"],
        },
        "weight_arithmetic": {"c8_index_scoring": scoring},
        "installed_hf_source_sha256": {"modeling.py": "installed"},
        "trace_sha256": {"step-00000.pt": "trace"},
        "rms_norm_eps": {"ordinary": 1e-5, "q_a": 1e-5, "kv_a": 1e-5},
        "index_rope_basis": "interleaved",
    }

    def validate_manifest(value):
        native.validate_reference_manifest(value, manifest["metadata_sha256"])

    verify("manifest.valid", lambda: validate_manifest(manifest), True)
    for field, value in (
        ("status", "running"), ("status", "fail"), ("status", "error"),
        ("status", "metadata_only"), ("numerical_execution", False),
        ("tp_size", 8), ("metadata_sha256", {}), ("source_sha256", {}),
        ("source_sha256", {"generator.py": "generator"}),
        ("source_sha256", {scorer: "0" * 64}),
        ("weight_arithmetic", {}),
        ("installed_hf_source_sha256", {}), ("trace_sha256", {}),
        ("index_rope_basis", "half"),
        ("fp32_head", True), ("fp32_head", "false"),
        ("rms_norm_eps", {"ordinary": 1e-5, "q_a": 1e-6, "kv_a": 1e-6}),
    ):
        wrong = copy.deepcopy(manifest)
        wrong[field] = value
        verify(f"manifest.reject.{field}.{value}",
               lambda: validate_manifest(wrong), False)
    for field, value in (("source_verified", False), ("source_sha256", {}),
                         ("reference_source_sha256", "0" * 64)):
        wrong = copy.deepcopy(manifest)
        wrong["weight_arithmetic"]["c8_index_scoring"][field] = value
        verify("manifest.reject.scoring." + field,
               lambda: validate_manifest(wrong), False)

    head_sources = (
        "tests/glm52/run_hf_reference.py",
        "tilert/models/glm_5_2_ascend/reference.py",
        "tilert/models/glm_5_2_ascend/reference_weights.py",
    )
    precise = copy.deepcopy(manifest)
    precise["fp32_head"] = True
    precise["loaded_reference_sources"] = {}
    for name in head_sources:
        sha = hashlib.sha256((root / name).read_bytes()).hexdigest()
        precise["source_sha256"][name] = sha
        if name.startswith("tilert/"):
            precise["loaded_reference_sources"][name] = {
                "path": str(root / name), "sha256": sha,
            }
    precise["weight_arithmetic"]["lm_head"] = {
        "input_dtype": "bfloat16", "weight_dtype": "bfloat16",
        "accumulator_dtype": "float32", "projection_output_dtype": "float32",
        "tensor_parallel_size": 16, "vocabulary_range": [0, 154880],
    }

    def validate_precise(value):
        native.validate_reference_manifest(
            value, manifest["metadata_sha256"], fp32_head=True,
        )

    verify("manifest.fp32.valid", lambda: validate_precise(precise), True)
    verify("manifest.fp32.reject_legacy", lambda: validate_precise(manifest),
           False)
    for field, value in (("input_dtype", "float32"),
                         ("weight_dtype", "float32"),
                         ("accumulator_dtype", "bfloat16"),
                         ("projection_output_dtype", "bfloat16"),
                         ("tensor_parallel_size", 8),
                         ("vocabulary_range", [0, 154879])):
        wrong = copy.deepcopy(precise)
        wrong["weight_arithmetic"]["lm_head"][field] = value
        verify("manifest.fp32.reject." + field,
               lambda: validate_precise(wrong), False)
    for name in head_sources:
        for sha in (None, "0" * 64):
            wrong = copy.deepcopy(precise)
            if sha is None:
                wrong["source_sha256"].pop(name)
            else:
                wrong["source_sha256"][name] = sha
            verify("manifest.fp32.reject_source." + name + str(sha),
                   lambda: validate_precise(wrong), False)
        if name.startswith("tilert/"):
            wrong = copy.deepcopy(precise)
            wrong["loaded_reference_sources"].pop(name)
            verify("manifest.fp32.reject_loaded." + name,
                   lambda: validate_precise(wrong), False)

    sources = [Path(__file__), reference, *(Path(__file__).with_name(name)
               for name in ("run_native_decode.py", "run_hf_reference.py"))]
    sources.append(root / scorer)
    for name in ("reference.py", "reference_weights.py"):
        path = reference.with_name(name)
        tree = ast.parse(path.read_text())
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                forbidden.extend(item.name for item in node.names
                                 if item.name == "torch_npu")
            if isinstance(node, ast.ImportFrom) and node.module in (
                "weights", "native_engine", "ops", "ops.quantized",
            ):
                forbidden.append(node.module)
        checks.append({"name": "independent_imports." + name,
                       "passed": not forbidden, "forbidden": forbidden})
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "stdlib metadata and AST; no Torch, numerical decode or NPU",
        "checks": checks, "checks_total": len(checks),
        "checks_failed": sum(not item["passed"] for item in checks),
        "source_sha256": {str(path.relative_to(root)):
                          hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }
    report["status"] = "pass" if not report["checks_failed"] else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: report[name] for name in (
        "status", "checks_total", "checks_failed",
    )}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
