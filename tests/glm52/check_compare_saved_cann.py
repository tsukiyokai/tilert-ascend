"""Exercise the saved CANN validator using synthetic CPU-only fixtures."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch

from compare_saved_cann import CHOICE_SOURCE_SUFFIXES, compare_saved, digest
from prefix_io import prompt_identity
from run_cann_model import compare_choices, compare_step
from tilert.models.glm_5_2_ascend.ops.indexer import (
    cann_arch22_scoring_contract,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_json(path):
    return json.loads(path.read_text())


def fingerprint(text):
    return hashlib.sha256(text.encode()).hexdigest()


def trace_fixture(position, choices=False):
    hidden = torch.ones(1, 6144, dtype=torch.bfloat16)
    logits = torch.zeros(1, 154880, dtype=torch.bfloat16)
    logits[0, 5] = 1
    rows = []
    for layer in range(78):
        full = layer < 3 or (layer - 2) % 4 == 0
        quantized = full and 6 <= layer <= 66
        rows.append({
            "latent": torch.ones(512, dtype=torch.bfloat16),
            "rope": torch.ones(64, dtype=torch.bfloat16),
            "index": torch.ones(128, dtype=(
                torch.int8 if quantized else torch.bfloat16
            )) if full else None,
            "index_scale": torch.ones(1, dtype=torch.float16)
            if quantized else None,
        })
    trace = {
        "token": 42, "position": position,
        "committed_position": position + 1,
        "token_ids": torch.tensor([5]), "logits": logits,
        "last_hidden": hidden, "hidden_states": (hidden,) * 79,
        "layer_logits": (logits,) * 78, "cache_rows": rows,
    }
    if choices:
        trace.update(
            indices=(torch.arange(2048, dtype=torch.int32),) * 78,
            routes=(None,) * 3 + (torch.arange(8, dtype=torch.int32),) * 75,
            choice_source_sha256={name: fingerprint(name)
                                  for name in CHOICE_SOURCE_SUFFIXES},
        )
    return trace


def make_fixture(root, choices=False):
    native, reference = root / "native", root / "reference"
    native.mkdir(parents=True)
    reference.mkdir()
    trace = trace_fixture(8192, choices)
    filename = "step-08192.pt"
    metadata = {name: fingerprint(name) for name in (
        "config.json", "quant_model_description.json",
        "quant_model_weights.safetensors.index.json",
    )}
    identity = prompt_identity([1 + index % 31 for index in range(8192)])
    reset_checks = [{"name": "reset.position", "passed": True}]
    compare_step("reset.0", trace, trace, reset_checks, exact=True)
    for phase in ("decode", "reset"):
        for layer, row in enumerate(trace["cache_rows"]):
            for field, value in row.items():
                if value is None:
                    continue
                field = {"index": "index_keys",
                         "index_scale": "index_scales"}.get(field, field)
                reset_checks.append({
                    "name": f"{phase}.prefix.{layer}.{field}",
                    "passed": True, "exact": True,
                })
    prefix_hashes = {
        f"rank-{rank}.pt": fingerprint(f"host-validator prefix {rank}")
        for rank in range(16)
    }
    receipts = []
    for rank in range(16):
        directory = native / f"rank-{rank}"
        directory.mkdir()
        torch.save(trace, directory / filename)
        report = {
            "scope": "synthetic host validator; not NPU capture",
            "rank": rank, "world_size": 16, "status": "captured",
            "initial_position": 8192, "final_position": 8193, "steps": 1,
            "consumed_tokens": [42], "predicted_tokens": [5],
            "checkpoint_sha256": metadata,
            "source_sha256": {"synthetic_fixture": fingerprint("fixture"),
                              **trace.get("choice_source_sha256", {})},
            "repeatability_passed": True, "checks": reset_checks,
            "trace_sha256": {filename: digest(directory / filename)},
            "coverage": {name: True for name in (
                "intermediate_hidden_states", "intermediate_layer_logits",
                "all_layers_new_kv", "reset_exact",
            )},
            "inputs": {
                **identity, "initial_position": 8192, "first_token_id": 42,
                "prefix_sha256": prefix_hashes[f"rank-{rank}.pt"],
                "capture_manifest_sha256": fingerprint("host capture"),
                "request_id": "host-validator", "generation": 1,
            },
        }
        report["coverage"].update(routing_choices=choices,
                                  index_choices=choices)
        path = native / f"rank-{rank}.json"
        write_json(path, report)
        receipts.append({"rank": rank, "report_sha256": digest(path),
                         "consumed_tokens": [42], "predicted_tokens": [5],
                         "passed": True})
    write_json(native / "summary.json", {
        "status": "captured", "all_ranks_agree": True, "ranks": receipts,
    })
    expected = dict(trace)
    expected.pop("token_ids")
    expected.pop("last_hidden")
    torch.save(expected, reference / filename)
    contract = cann_arch22_scoring_contract()
    contract["source_verified"] = True
    write_json(reference / "manifest.json", {
        "scope": "synthetic host validator; not HF numerical execution",
        "implementation": "transformers.GlmMoeDsaForCausalLM",
        "status": "pass", "numerical_execution": True, "tp_size": 16,
        "layer_count": 78, "cpu_matmul": {"mkldnn_enabled": False},
        "metadata_sha256": metadata, "initial_position": 8192,
        "teacher_tokens": [42], "prefix_files_sha256": prefix_hashes,
        **identity,
        "source_sha256": {
            "tilert/models/glm_5_2_ascend/ops/indexer.py":
                contract["reference_source_sha256"],
        },
        "installed_hf_source_sha256": {"host_fixture": fingerprint("HF")},
        "weight_arithmetic": {"c8_index_scoring": contract},
        "trace_sha256": {filename: digest(reference / filename)},
        "rms_norm_eps": {"ordinary": 1e-5, "q_a": 1e-5, "kv_a": 1e-5},
        "index_rope_basis": "interleaved",
        "checks": [{"name": "reset.cache_lengths", "passed": True}],
    })


def change_rank(root, rank, change):
    path = root / "native" / f"rank-{rank}.json"
    report = read_json(path)
    change(report)
    write_json(path, report)
    path_summary = root / "native" / "summary.json"
    summary = read_json(path_summary)
    summary["ranks"][rank]["report_sha256"] = digest(path)
    write_json(path_summary, summary)


def change_reference(root, change):
    path = root / "reference" / "manifest.json"
    report = read_json(path)
    change(report)
    write_json(path, report)


def change_trace(root, change, rank=0):
    path = root / "native" / f"rank-{rank}" / "step-08192.pt"
    trace = torch.load(path, map_location="cpu", weights_only=True)
    change(trace)
    torch.save(trace, path)
    change_rank(root, rank, lambda report: report["trace_sha256"].update(
        {path.name: digest(path)},
    ))


def mutate_reset(report, *, remove=False, exact=False):
    name = "reset.0.hidden_states.77"
    if remove:
        report["checks"] = [item for item in report["checks"]
                            if item["name"] != name]
    else:
        item = next(item for item in report["checks"] if item["name"] == name)
        item["exact" if exact else "passed"] = False


def corrupt_hidden(trace):
    values = list(trace["hidden_states"])
    values[3] = values[3] * 100
    trace["hidden_states"] = tuple(values)


def tree_hashes(root):
    return {str(path.relative_to(root)): digest(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def check_choices(root, checks):
    base = root / "choices"
    make_fixture(base, choices=True)
    result = compare_saved(base / "native", base / "reference")
    checks.append({"name": "actual_choices_complete_tp16",
                   "passed": result["status"] == "diagnostic"
                   and result["coverage"]["routing_choices"]
                   and result["coverage"]["index_choices"]})

    def wrong_selection(trace):
        values = list(trace["routes"])
        values[3] = values[3].clone()
        values[3][0] = 255
        trace["routes"] = tuple(values)

    cases = (
        ("missing_indices", lambda t: t.pop("indices")),
        ("missing_dense_route", lambda t: t.update(routes=t["routes"][3:])),
        ("float_indices", lambda t: t.update(
            indices=tuple(x.float() for x in t["indices"])
        )),
        ("wrong_actual_route", wrong_selection),
        ("missing_choice_sources", lambda t: t.pop("choice_source_sha256")),
        ("partial_choice_sources", lambda t: t.update(
            choice_source_sha256={CHOICE_SOURCE_SUFFIXES[0]:
                                  fingerprint(CHOICE_SOURCE_SUFFIXES[0])}
        )),
        ("duplicate_choice_source_suffix", lambda t:
         t["choice_source_sha256"].update({
             "/another/" + CHOICE_SOURCE_SUFFIXES[0]:
                 fingerprint(CHOICE_SOURCE_SUFFIXES[0]),
         })),
    )
    for name, mutate in cases:
        current = root / name
        shutil.copytree(base, current)
        change_trace(current, mutate, rank=15)
        result = compare_saved(current / "native", current / "reference")
        if name == "wrong_actual_route":
            differences = [item for item in result["diagnostics"]
                           if item["name"] == "rank.15.step.8192.routes.3"]
            accepted = (result["status"] == "diagnostic"
                        and len(differences) == 1
                        and differences[0]["overlap"] == 7
                        and differences[0]["sets_equal"] is False)
            label = "choices.diagnostic."
        else:
            accepted = result["status"] == "fail"
            label = "choices.reject."
        checks.append({"name": label + name, "passed": accepted})
        shutil.rmtree(current)

    padded = torch.full((2048,), -1, dtype=torch.int32)
    padded[:3] = torch.tensor([2, 0, 1], dtype=torch.int32)
    actual = {"position": 2, "indices": (padded,) * 78,
              "routes": (None,) * 3
              + (torch.arange(8, dtype=torch.int32),) * 75}
    expected = {"indices": (torch.arange(3),) * 78,
                "routes": (None,) * 3 + (torch.arange(8),) * 75}
    numeric = []
    compare_choices("padded", actual, expected, numeric)
    checks.append({"name": "choices.padding_and_set_order",
                   "passed": all(item["passed"] for item in numeric)})
    padded[0] = 3
    numeric = []
    compare_choices("future", actual, expected, numeric)
    checks.append({"name": "choices.future_index_rejected",
                   "passed": any(not item["passed"] for item in numeric
                                 if item["name"].endswith("causal_range"))})
    reordered = {"indices": tuple(x.flip(0) for x in actual["indices"]),
                 "routes": actual["routes"]}
    numeric = []
    compare_choices("reset", actual, reordered, numeric, exact=True)
    checks.append({"name": "choices.reset_preserves_raw_order",
                   "passed": any(not item["passed"] for item in numeric)})
    for field, invalid in (
        ("indices", torch.tensor([0, 0, 1])),
        ("indices", torch.tensor([0, 1])),
        ("routes", torch.tensor([0, 0, 2, 3, 4, 5, 6, 7])),
        ("routes", torch.arange(256, 264)),
    ):
        malformed = {**expected, "position": 2,
                     field: (invalid,) * 78 if field == "indices" else
                     (None,) * 3 + (invalid,) * 75}
        numeric = []
        compare_choices("invalid", malformed, malformed, numeric)
        checks.append({"name": "choices.same_invalid_selection_rejected."
                       + field + "." + str(invalid.tolist()),
                       "passed": any(not item["passed"]
                                     for item in numeric)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch_npu

    torch.set_num_threads(4)
    before_npu = torch_npu.npu._initialized
    checks = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        root = Path(temporary)
        base = root / "base"
        make_fixture(base)
        original = tree_hashes(base)
        valid_output = root / "valid.json"
        command = [sys.executable, str(Path(__file__).with_name(
            "compare_saved_cann.py")), "--native", str(base / "native"),
            "--reference", str(base / "reference"),
            "--output", str(valid_output)]
        completed = subprocess.run(command, capture_output=True, text=True)
        valid = read_json(valid_output)
        checks.append({
            "name": "actual_cli_complete_tp16_8k",
            "passed": completed.returncode == 0
            and valid["status"] == "diagnostic"
            and valid["completed_rank_steps"] == 16
            and valid["coverage"]["hidden_states_per_step"] == 79
            and valid["coverage"]["layer_logits_per_step"] == 78
            and valid["coverage"]["prompt_tokens"] == 8192
            and valid["hf_diagnostics_complete"] is True
            and valid["continuation_parity"] is False
            and valid["layerwise_logits_parity"] is False
            and valid["precision_verified"] is False,
            "checks_total": valid["checks_total"],
            "stdout": completed.stdout, "stderr": completed.stderr,
        })
        cases = [
            ("missing_rank15", lambda p: (p / "native" / "rank-15.json")
             .unlink()),
            ("rank_report_hash", lambda p: (p / "native" / "rank-15.json")
             .write_text("{}\n")),
            ("rank15_trace_hash", lambda p: (
                p / "native" / "rank-15" / "step-08192.pt"
            ).write_bytes(b"changed")),
            ("hf_trace_hash", lambda p: (
                p / "reference" / "step-08192.pt"
            ).write_bytes(b"changed")),
            ("checkpoint", lambda p: change_rank(p, 15, lambda r:
             r["checkpoint_sha256"].update({"config.json": "changed"}))),
            ("teacher", lambda p: change_reference(p, lambda r:
             r.update(teacher_tokens=[43]))),
            ("initial_position", lambda p: change_reference(p, lambda r:
             r.update(initial_position=8191))),
            ("prefix_hash", lambda p: change_reference(p, lambda r:
             r["prefix_files_sha256"].update({"rank-15.pt": "changed"}))),
            ("prompt_hash", lambda p: change_reference(p, lambda r:
             r.update(prompt_sha256="changed"))),
            ("prompt_length", lambda p: change_reference(p, lambda r:
             r.update(prompt_length=8191))),
            ("last_prompt_token", lambda p: change_reference(p, lambda r:
             r.update(last_prompt_token=500))),
            ("capture_generation", lambda p: change_rank(p, 15, lambda r:
             r["inputs"].update(generation=2))),
            ("reset_not_completed", lambda p: change_rank(p, 15, lambda r:
             r.update(repeatability_passed=False))),
            ("missing_reset_hidden", lambda p: change_rank(p, 15, lambda r:
             mutate_reset(r, remove=True))),
            ("reset_not_exact", lambda p: change_rank(p, 15, lambda r:
             mutate_reset(r, exact=True))),
            ("reset_failed", lambda p: change_rank(p, 15, mutate_reset)),
            ("hf_reset_failed", lambda p: change_reference(p, lambda r:
             r["checks"][0].update(passed=False))),
            ("final_only", lambda p: change_trace(p, lambda t:
             t.pop("hidden_states"))),
            ("missing_layer_logits", lambda p: change_trace(p, lambda t:
             t.update(layer_logits=t["layer_logits"][:-1]))),
            ("rank15_layer3_numeric_difference", lambda p: change_trace(
                p, corrupt_hidden, rank=15)),
        ]
        for name, mutate in cases:
            current = root / name
            shutil.copytree(base, current)
            mutate(current)
            result = compare_saved(current / "native", current / "reference")
            failed = [item for item in result["checks"] if not item["passed"]]
            if name == "rank15_layer3_numeric_difference":
                differences = [item for item in result["diagnostics"]
                               if item["name"]
                               == "rank.15.step.8192.hidden_states.3"]
                accepted = (result["status"] == "diagnostic" and not failed
                            and len(differences) == 1
                            and differences[0]["max_abs"] == 99
                            and differences[0]["equal_elements"] == 0)
            else:
                accepted = result["status"] == "fail" and bool(failed)
            checks.append({"name": name,
                           "passed": accepted,
                           "failures": failed})
            shutil.rmtree(current)
        current = root / "old_hf_failure"
        shutil.copytree(base, current)
        change_rank(current, 0, lambda r: r["checks"].append({
            "name": "hf.0.logits", "passed": False,
        }))
        result = compare_saved(current / "native", current / "reference")
        checks.append({"name": "old_hf_failure_recompared",
                       "passed": result["status"] == "diagnostic"
                       and result["completed_rank_steps"] == 16})
        checks.append({"name": "original_files_unchanged",
                       "passed": tree_hashes(base) == original})
        check_choices(root, checks)
    after_npu = torch_npu.npu._initialized
    checks.append({"name": "no_npu_initialization",
                   "passed": not before_npu and not after_npu})
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Synthetic host-validator fixtures with real CPU tensors; "
                 "not HF/NPU precision or model execution",
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "npu_initialized_before": before_npu,
        "npu_initialized_after": after_npu,
        "source_sha256": {name: digest(Path(__file__).with_name(name))
                          for name in (Path(__file__).name,
                                       "compare_saved_cann.py",
                                       "run_cann_model.py",
                                       "run_native_decode.py")},
        "checks": checks, "passed": all(item["passed"] for item in checks),
    }
    write_json(args.output, output)
    print(json.dumps({"passed": output["passed"], "checks": len(checks),
                      "output": str(args.output)}))
    return int(not output["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
