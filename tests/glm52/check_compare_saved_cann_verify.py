"""Synthetic CPU fixtures for direct saved-wide/HF validation.

The fixture files have the real trace shapes and serialization contract;
they contain no captured NPU results or independently computed HF model.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch  # noqa: E402

from check_compare_saved_cann import (  # noqa: E402
    change_rank, change_reference, make_fixture, read_json, trace_fixture,
    tree_hashes, write_json,
)
from compare_saved_cann_verify import (  # noqa: E402
    compare_saved, digest, single_row,
)
from run_cann_model import compare_step, diagnose_tensor  # noqa: E402


def make_wide_fixture(root, width, *, choices=False):
    make_fixture(root, choices)
    native, reference = root / "native", root / "reference"
    teachers = list(range(42, 42 + width))
    trace = trace_fixture(8192, choices)
    hidden = trace["last_hidden"].expand(width, -1).clone()
    logits = trace["logits"].expand(width, -1).clone()
    wide = {
        "token": teachers, "position": 8192,
        "committed_position": 8192 + width,
        "token_ids": torch.full((width,), 5), "logits": logits,
        "last_hidden": hidden, "hidden_states": (hidden,) * 79,
        "layer_logits": (logits,) * 78,
        "model_inputs": {
            "input_ids": torch.tensor(teachers),
            "positions": torch.arange(8192, 8192 + width),
            "slots": torch.arange(8192, 8192 + width, dtype=torch.int32),
            "query_cpu": torch.tensor([0, width], dtype=torch.int32),
            "seq_lens": torch.tensor([8192 + width], dtype=torch.int32),
        },
        "cache_rows": tuple({
            key: value.unsqueeze(0).expand(width, *value.shape).clone()
            if value is not None and key != "index_scale"
            else value.expand(width).clone() if value is not None else None
            for key, value in layer.items()
        } for layer in trace["cache_rows"]),
    }
    if choices:
        wide.update({field: tuple(
            None if value is None else value.expand(width, -1).clone()
            for value in trace[field]
        ) for field in ("indices", "routes")})
        wide["choice_source_sha256"] = trace["choice_source_sha256"]
    reset_checks = []
    for row in range(width):
        value = single_row(wide, row, choices=choices)
        compare_step(f"reset.row-{row}", value, value, reset_checks,
                     exact=True)
    original = read_json(native / "rank-0.json")
    for item in original["checks"]:
        if item["name"].startswith(("decode.prefix.", "reset.prefix.")):
            phase, suffix = item["name"].split(".", 1)
            phase = "wide" if phase == "decode" else "reset"
            reset_checks.append({
                **item, "name": f"{phase}.prefix.after.{suffix}",
            })
    summaries = []
    for rank in range(16):
        path = native / f"rank-{rank}" / "wide.pt"
        torch.save(wide, path)
        report = read_json(native / f"rank-{rank}.json")
        report.update(
            status="pass", width=width, final_position=8192 + width,
            checks=reset_checks, trace_sha256={"wide.pt": digest(path)},
        )
        report["inputs"]["teacher_tokens"] = teachers
        report["coverage"].update(routing_choices=choices,
                                  index_choices=choices)
        report_path = native / f"rank-{rank}.json"
        write_json(report_path, report)
        summaries.append({"rank": rank, "passed": True,
                          "predicted_tokens": [5] * width,
                          "report_sha256": digest(report_path)})
    write_json(native / "summary.json", {
        "status": "pass", "width": width,
        "all_ranks_agree": True, "ranks": summaries,
    })
    manifest = read_json(reference / "manifest.json")
    manifest["teacher_tokens"] = teachers
    manifest["trace_sha256"] = {}
    for row, token in enumerate(teachers):
        expected = trace_fixture(8192 + row, choices)
        expected.update(token=token)
        expected.pop("last_hidden")
        expected.pop("token_ids")
        name = f"step-{8192 + row:05d}.pt"
        torch.save(expected, reference / name)
        manifest["trace_sha256"][name] = digest(reference / name)
    write_json(reference / "manifest.json", manifest)


def change_wide(root, mutate, rank=15):
    path = root / "native" / f"rank-{rank}" / "wide.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    mutate(saved)
    torch.save(saved, path)
    change_rank(root, rank, lambda report: report["trace_sha256"].update(
        {"wide.pt": digest(path)},
    ))


def nontransitive_hidden(saved):
    values = list(saved["hidden_states"])
    values[3] = torch.full_like(values[3], 1.015625)
    saved["hidden_states"] = tuple(values)


def check_nontransitivity():
    expected = torch.ones((1, 6144), dtype=torch.bfloat16)
    serial = torch.full_like(expected, 1.0078125)
    wide = torch.full_like(expected, 1.015625)
    checks, diagnostics = [], []
    diagnose_tensor("serial_to_hf", serial, expected, checks, diagnostics)
    diagnose_tensor("wide_to_serial", wide, serial, checks, diagnostics)
    diagnose_tensor("wide_to_hf", wide, expected, checks, diagnostics)
    return checks, diagnostics


def break_reset(report):
    item = next(value for value in report["checks"]
                if value["name"] == "reset.row-1.hidden_states.77")
    item["exact"] = False


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
        make_wide_fixture(base, 4, choices=True)
        original = tree_hashes(base)
        command = [sys.executable, str(Path(__file__).with_name(
            "compare_saved_cann_verify.py")), "--native", str(base / "native"),
            "--reference", str(base / "reference"),
            "--output", str(root / "valid.json")]
        completed = subprocess.run(command, capture_output=True, text=True)
        valid = read_json(root / "valid.json")
        checks.append({
            "name": "actual_cli_m4_all_16_ranks",
            "passed": completed.returncode == 0
            and valid["status"] == "diagnostic"
            and valid["completed_rank_steps"] == 64
            and valid["coverage"]["hidden_states_per_step"] == 79
            and valid["coverage"]["layer_logits_per_step"] == 78
            and valid["coverage"]["routing_choices"] is True
            and valid["coverage"]["index_choices"] is True
            and valid["hf_diagnostics_complete"] is True
            and valid["saved_wide_hf_parity"] is False
            and valid["layerwise_logits_parity"] is False
            and valid["precision_verified"] is False
            and valid["current_runtime_verified"] is False,
            "checks_total": valid["checks_total"],
            "stdout": completed.stdout, "stderr": completed.stderr,
        })
        two = root / "m2"
        make_wide_fixture(two, 2)
        result = compare_saved(two / "native", two / "reference")
        checks.append({"name": "m2_all_16_ranks",
                       "passed": result["status"] == "diagnostic"
                       and result["completed_rank_steps"] == 32
                       and result["coverage"]["routing_choices"] is False
                       and result["coverage"]["index_choices"] is False})
        pair_checks, pairs = check_nontransitivity()
        checks.append({"name": "nontransitive_differences_recorded",
                       "passed": all(item["passed"] for item in pair_checks)
                       and [item["max_abs"] for item in pairs]
                       == [0.0078125, 0.0078125, 0.015625],
                       "comparisons": pairs})
        cases = [
            ("direct_records_numeric_difference", lambda path: change_wide(
                path, nontransitive_hidden)),
            ("rank_report_hash", lambda path: (
                path / "native/rank-15.json").write_text("{}\n")),
            ("wide_hash", lambda path: (
                path / "native/rank-15/wide.pt").write_bytes(b"changed")),
            ("hf_trace_hash", lambda path: (
                path / "reference/step-08193.pt").write_bytes(b"changed")),
            ("teacher", lambda path: change_reference(path, lambda value:
             value.update(teacher_tokens=[42, 43, 44, 46]))),
            ("prefix", lambda path: change_reference(path, lambda value:
             value["prefix_files_sha256"].update({"rank-15.pt": "changed"}))),
            ("rank", lambda path: change_rank(path, 15, lambda value:
             value.update(rank=14))),
            ("checkpoint", lambda path: change_rank(path, 15, lambda value:
             value["checkpoint_sha256"].update({"config.json": "changed"}))),
            ("source", lambda path: change_rank(path, 15, lambda value:
             value.update(source_sha256={}))),
            ("capture_generation", lambda path: change_rank(
                path, 15, lambda value: value["inputs"].update(generation=2))),
            ("missing_hidden", lambda path: change_wide(path, lambda value:
             value.pop("hidden_states"))),
            ("missing_layer_logits", lambda path: change_wide(
                path, lambda value: value.update(
                    layer_logits=value["layer_logits"][:-1]))),
            ("missing_indices", lambda path: change_wide(
                path, lambda value: value.pop("indices"))),
            ("missing_routes", lambda path: change_wide(
                path, lambda value: value.pop("routes"))),
            ("choice_sources", lambda path: change_wide(
                path, lambda value: value.pop("choice_source_sha256"))),
            ("partial_choice_coverage", lambda path: change_rank(
                path, 15, lambda value: value["coverage"].update(
                    index_choices=False))),
            ("reset_not_exact", lambda path: change_rank(path, 15,
                                                         break_reset)),
        ]
        for name, mutate in cases:
            current = root / name
            shutil.copytree(base, current)
            mutate(current)
            result = compare_saved(current / "native", current / "reference")
            failures = [item for item in result["checks"]
                        if not item["passed"]]
            if name == "direct_records_numeric_difference":
                expected = [f"rank.15.row.{row}.hidden_states.3"
                            for row in range(4)]
                differences = [item for item in result["diagnostics"]
                               if item["name"] in expected]
                accepted = (result["status"] == "diagnostic" and not failures
                            and [item["name"] for item in differences]
                            == expected
                            and all(item["max_abs"] == 0.015625
                                    for item in differences)
                            and result["completed_rank_steps"] == 64)
            else:
                accepted = result["status"] == "fail" and bool(failures)
            checks.append({"name": name,
                           "passed": accepted,
                           "failures": failures})
            shutil.rmtree(current)
        checks.append({"name": "original_files_unchanged",
                       "passed": tree_hashes(base) == original})
    after_npu = torch_npu.npu._initialized
    checks.append({"name": "no_npu_initialization",
                   "passed": not before_npu and not after_npu})
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Synthetic host-validator fixtures with real CPU tensors; "
                 "not HF/NPU model precision or execution",
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "npu_initialized_before": before_npu,
        "npu_initialized_after": after_npu,
        "source_sha256": {name: digest(Path(__file__).with_name(name))
                          for name in (Path(__file__).name,
                                       "compare_saved_cann_verify.py",
                                       "compare_saved_cann.py",
                                       "check_compare_saved_cann.py",
                                       "run_cann_verify.py",
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
