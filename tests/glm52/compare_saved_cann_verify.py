"""Diagnose saved TP16 M2/M4 target rows against matching HF traces.

Every wide row must contain 79 hidden states, 78 layer logits and new KV.
HF value and selection differences are diagnostics. Input, layout, causal
range, cache length and the saved exact reset remain execution checks.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from compare_saved_cann import (
    CHECKPOINT_FILES, FIELDS, IDENTITY, read_report, require,
    reset_tensor_names, valid_tokens, validate_choice_sources, validate_inputs,
    validate_trace,
)
from precision_gate import is_digest
from run_cann_model import compare_step, digest, validate_reference_manifest
from run_cann_verify import check_record, snapshot_rows


def validate_saved_reset(report, checks, label):
    saved = report["checks"]
    names = {item["name"]: item for item in saved}
    require(len(names) == len(saved), label + ".unique_checks", checks)
    require(all(report["coverage"].get(field) is True for field in (
        "intermediate_hidden_states", "intermediate_layer_logits",
        "all_layers_new_kv", "reset_exact",
    )), label + ".complete_capture_coverage", checks)
    for row in range(report["width"]):
        tensors = {name.replace(f"reset.{row}.", f"reset.row-{row}.")
                   for name in reset_tensor_names(row)}
        counts = {f"reset.row-{row}.{field}" for field in (
            "position_and_input", "hidden_states.count",
            "layer_logits.count", "cache_layer_count",
        )}
        if report["coverage"].get("routing_choices") is True:
            tensors.update(f"reset.row-{row}.{field}.{layer}"
                           for field in ("indices", "routes")
                           for layer in range(78))
            counts.update(f"reset.row-{row}.{field}.count"
                          for field in ("indices", "routes"))
        require(tensors | counts <= names.keys()
                and all(names[name].get("passed") is True
                        for name in tensors | counts)
                and all(names[name].get("exact") is True for name in tensors),
                f"{label}.reset.row-{row}.complete_exact", checks)
    if report["initial_position"]:
        for phase in ("wide", "reset"):
            required = set()
            for layer in range(78):
                fields = ["latent", "rope"]
                if layer < 3 or (layer - 2) % 4 == 0:
                    fields.append("index_keys")
                    if 6 <= layer <= 66:
                        fields.append("index_scales")
                required.update(
                    f"{phase}.prefix.after.prefix.{layer}.{field}"
                    for field in fields
                )
            require(required <= names.keys() and all(
                names[name].get("passed") is True
                and names[name].get("exact") is True for name in required
            ), f"{label}.{phase}.prefix_exact", checks)


def input_view(report):
    inputs = dict(report["inputs"])
    teacher = inputs["teacher_tokens"]
    if report["initial_position"] == 0:
        inputs["cold_teacher_tokens"] = teacher
    return {**report, "steps": report["width"], "inputs": inputs,
            "consumed_tokens": teacher}


def validate_wide(trace, report, checks, label, *, choices=False):
    import torch

    width, initial = report["width"], report["initial_position"]
    teacher = report["inputs"]["teacher_tokens"]
    require(type(trace["position"]) is int
            and trace["position"] == initial
            and type(trace["committed_position"]) is int
            and trace["committed_position"] == initial + width
            and valid_tokens(trace["token"]) and trace["token"] == teacher,
            label + ".wide_input", checks)
    for name, shape in (("token_ids", (width,)),
                        ("logits", (width, 154880)),
                        ("last_hidden", (width, 6144))):
        require(isinstance(trace[name], torch.Tensor)
                and trace[name].shape == shape, label + "." + name, checks)
    for name, count, size in (("hidden_states", 79, 6144),
                              ("layer_logits", 78, 154880)):
        values = trace.get(name, ())
        require(len(values) == count and all(
            isinstance(value, torch.Tensor) and value.shape == (width, size)
            for value in values
        ), label + ".complete_" + name, checks)
    rows = trace["cache_rows"]
    require(len(rows) == 78 and all(
        set(layer) == set(FIELDS) and all(
            value is None or (isinstance(value, torch.Tensor)
                              and value.ndim > 0 and value.shape[0] == width)
            for value in layer.values()
        ) for layer in rows
    ), label + ".cache_row_counts", checks)
    if choices:
        for field, size in (("indices", 2048), ("routes", 8)):
            values = trace.get(field, ())
            require(len(values) == 78 and all(
                value is None if field == "routes" and layer < 3 else (
                    isinstance(value, torch.Tensor)
                    and value.dtype in (torch.int32, torch.int64)
                    and value.shape == (width, size)
                ) for layer, value in enumerate(values)
            ), label + ".complete_" + field, checks)
        validate_choice_sources(trace, report, checks, label)
    check_record(label, trace, teacher, checks)


def single_row(wide, row, *, choices=False):
    saved = snapshot_rows(wide, row, row + 1)
    saved["token"] = saved["token"][0]
    saved["cache_rows"] = tuple({
        name: value[0] if value is not None and name != "index_scale"
        else value for name, value in layer.items()
    } for layer in saved["cache_rows"])
    for field in ("indices", "routes"):
        if choices:
            saved[field] = tuple(None if value is None else value.reshape(-1)
                                 for value in saved[field])
        else:
            saved.pop(field, None)
    return saved


def compare_saved(native, reference_dir):
    import torch

    checks, diagnostics, artifacts = [], [], {}
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "HF cross-configuration diagnostics from saved wide rows",
        "comparison_kind": "cross_configuration_hf",
        "native": str(native), "reference": str(reference_dir),
        "precision_verified": False, "performance_measured": False,
        "current_runtime_verified": False,
        "coverage": {"routing_choices": False, "index_choices": False,
                     "mtp": False, "session": False},
        "checks": checks, "diagnostics": diagnostics,
        "artifact_sha256": artifacts,
        "completed_rank_steps": 0,
        "source_sha256": {
            name: digest(Path(__file__).with_name(name)) for name in (
                Path(__file__).name, "compare_saved_cann.py",
                "run_cann_verify.py", "run_cann_model.py",
                "run_native_decode.py", "precision_gate.py",
            )
        },
    }
    try:
        summary = read_report(native / "summary.json", artifacts)
        receipts = summary["ranks"]
        require(summary["width"] in (2, 4)
                and summary["all_ranks_agree"] is True
                and len(receipts) == 16
                and all(type(item["rank"]) is int for item in receipts)
                and {item["rank"] for item in receipts} == set(range(16)),
                "native.all_16_rank_receipts", checks)
        reference = read_report(reference_dir / "manifest.json", artifacts)
        reports = []
        for rank in range(16):
            label = f"rank.{rank}"
            path = native / f"rank-{rank}.json"
            report = read_report(path, artifacts)
            receipt = next(item for item in receipts if item["rank"] == rank)
            require(artifacts[str(path)] == receipt["report_sha256"],
                    label + ".report_hash", checks)
            sources = report["source_sha256"]
            require(type(report["rank"]) is int and report["rank"] == rank
                    and report["world_size"] == 16
                    and report["width"] == summary["width"]
                    and bool(sources) and all(is_digest(value)
                                             for value in sources.values())
                    and set(report["checkpoint_sha256"]) == CHECKPOINT_FILES,
                    label + ".execution_identity", checks)
            validate_reference_manifest(reference, report["checkpoint_sha256"])
            checks.append({"name": label + ".hf_manifest", "passed": True})
            validate_inputs(input_view(report), reference, checks, label)
            selection = [report["coverage"].get(field, False)
                         for field in ("routing_choices", "index_choices")]
            require(all(type(value) is bool for value in selection)
                    and selection[0] == selection[1],
                    label + ".choice_coverage", checks)
            validate_saved_reset(report, checks, label)
            require(valid_tokens(receipt["predicted_tokens"])
                    and len(receipt["predicted_tokens"]) == report["width"],
                    label + ".receipt_predictions", checks)
            report["receipt_predictions"] = receipt["predicted_tokens"]
            if report["initial_position"]:
                inputs = report["inputs"]
                require(is_digest(inputs["capture_manifest_sha256"])
                        and isinstance(inputs["request_id"], str)
                        and bool(inputs["request_id"])
                        and type(inputs["generation"]) is int
                        and inputs["generation"] > 0,
                        label + ".capture_identity", checks)
            if reports:
                first = reports[0]
                require(all(report[key] == first[key] for key in (
                    "initial_position", "width", "checkpoint_sha256",
                    "source_sha256", "receipt_predictions",
                )), label + ".rank_agreement", checks)
                if report["initial_position"]:
                    require(all(report["inputs"][key]
                                == first["inputs"][key] for key in (
                                    "capture_manifest_sha256", "request_id",
                                    "generation", *IDENTITY,
                                )), label + ".capture_agreement", checks)
            reports.append(report)
        first = reports[0]
        initial, width = first["initial_position"], first["width"]
        teacher = first["inputs"]["teacher_tokens"]
        require(valid_tokens(reference["teacher_tokens"])
                and len(reference["teacher_tokens"]) >= width
                and reference["checks"]
                and all(item["passed"] is True for item in reference["checks"])
                and any(item["name"] == "reset.cache_lengths"
                        for item in reference["checks"]),
                "hf.reset_passed", checks)
        expected = []
        for row in range(width):
            position = initial + row
            name = f"step-{position:05d}.pt"
            path = reference_dir / name
            artifacts[str(path)] = digest(path)
            require(artifacts[str(path)] == reference["trace_sha256"][name],
                    f"hf.{name}.hash", checks)
            trace = torch.load(path, map_location="cpu", weights_only=True)
            validate_trace(trace, position, teacher[row], checks,
                           f"hf.{position}", native=False, choices=any(
                               r["coverage"].get("routing_choices", False)
                               for r in reports
                           ))
            expected.append(trace)
        for rank, report in enumerate(reports):
            path = native / f"rank-{rank}" / "wide.pt"
            artifacts[str(path)] = digest(path)
            require(artifacts[str(path)] == report["trace_sha256"]["wide.pt"],
                    f"rank.{rank}.wide_hash", checks)
            wide = torch.load(path, map_location="cpu", weights_only=True)
            choices = report["coverage"].get("routing_choices", False)
            validate_wide(wide, report, checks, f"rank.{rank}",
                          choices=choices)
            require(wide["token_ids"].tolist()
                    == report["receipt_predictions"],
                    f"rank.{rank}.reported_predictions", checks)
            for row, reference_trace in enumerate(expected):
                label = f"rank.{rank}.row.{row}"
                actual = single_row(wide, row, choices=choices)
                validate_trace(actual, initial + row, teacher[row], checks,
                               label, native=True, choices=choices)
                compare_step(label, actual, reference_trace, checks,
                             diagnostics=diagnostics)
                output["completed_rank_steps"] += 1
        output.update(
            checkpoint_sha256=first["checkpoint_sha256"],
            initial_position=initial, width=width, teacher_tokens=teacher,
            input_identity=first["inputs"],
            native_source_sha256=first["source_sha256"],
            reference_source_sha256=reference["source_sha256"],
            saved_run_status=summary["status"],
        )
        output["coverage"].update(
            tp_ranks=16, steps=width, hidden_states_per_step=79,
            layer_logits_per_step=78, all_layers_new_kv=True,
            saved_reset_exact=True, prompt_tokens=initial,
            frozen_prefix_exact=initial > 0,
            routing_choices=all(r["coverage"].get("routing_choices") is True
                                for r in reports),
            index_choices=all(r["coverage"].get("index_choices") is True
                              for r in reports),
        )
    except Exception as error:
        checks.append({"name": "validation_error", "passed": False,
                       "error": f"{type(error).__name__}: {error}"})
    failed = sum(not item["passed"] for item in checks)
    output.update(status="fail" if failed else "diagnostic",
                  saved_wide_hf_parity=False, layerwise_logits_parity=False,
                  hf_diagnostics_complete=not failed,
                  checks_total=len(checks), checks_failed=failed)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(path.resolve())
           for path in (args.native, args.reference)):
        parser.error("Write the report outside both input directories")
    import torch

    torch.set_num_threads(4)
    output = compare_saved(args.native, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
    )
    print(json.dumps({key: output[key] for key in (
        "status", "checks_total", "checks_failed", "completed_rank_steps",
        "hf_diagnostics_complete", "precision_verified",
    )}))
    return int(output["checks_failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
