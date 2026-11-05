"""Diagnose saved CANN/HF differences; check capture identity and reset."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from precision_gate import is_digest
from run_cann_model import compare_step, digest, validate_reference_manifest


FIELDS = ("latent", "rope", "index", "index_scale")
IDENTITY = ("prompt_length", "prompt_sha256", "last_prompt_token")
CHECKPOINT_FILES = {
    "config.json", "quant_model_description.json",
    "quant_model_weights.safetensors.index.json",
}
CHOICE_SOURCE_SUFFIXES = (
    "tilert/models/glm_5_2_ascend/cann_choices.py",
    "vllm_ascend/attention/sfa_v1.py",
    "vllm_ascend/ops/fused_moe/routed_experts.py",
)


def require(condition, name, checks):
    checks.append({"name": name, "passed": bool(condition)})
    if not condition:
        raise ValueError(name)


def read_report(path, artifacts):
    artifacts[str(path)] = digest(path)
    return json.loads(path.read_text())


def validate_choice_sources(trace, report, checks, label):
    sources = trace.get("choice_source_sha256", {})
    for suffix in CHOICE_SOURCE_SUFFIXES:
        matches = [name for name in sources
                   if Path(name).as_posix() == suffix
                   or Path(name).as_posix().endswith("/" + suffix)]
        require(len(matches) == 1,
                label + ".choice_source." + suffix, checks)
    require(all(is_digest(sha) and report["source_sha256"].get(path) == sha
                for path, sha in sources.items()),
            label + ".choice_sources", checks)


def valid_tokens(tokens):
    return (isinstance(tokens, list) and bool(tokens)
            and all(type(value) is int and 0 <= value < 154880
                    for value in tokens))


def reset_tensor_names(step):
    prefix = f"reset.{step}"
    return {
        prefix + ".logits", prefix + ".last_hidden",
        *(f"{prefix}.hidden_states.{layer}" for layer in range(79)),
        *(f"{prefix}.layer_logits.{layer}" for layer in range(78)),
        *(f"{prefix}.cache.{layer}.{field}"
          for layer in range(78) for field in FIELDS),
    }


def validate_reset(report, checks, label):
    saved = report["checks"]
    names = {item["name"]: item for item in saved}
    require(len(names) == len(saved), label + ".unique_checks", checks)
    require(report.get("repeatability_passed") is True
            and names.get("reset.position", {}).get("passed") is True,
            label + ".reset_completed", checks)
    require(all(item["passed"] is True for item in saved
                if not item["name"].startswith("hf.")),
            label + ".device_checks_passed", checks)
    for step in range(report["steps"]):
        tensors = reset_tensor_names(step)
        counts = {f"reset.{step}.{field}" for field in (
            "position_and_input", "hidden_states.count",
            "layer_logits.count", "cache_layer_count",
        )}
        if any(report["coverage"].get(field) for field in (
            "routing_choices", "index_choices",
        )):
            tensors.update(f"reset.{step}.{field}.{layer}"
                           for field in ("indices", "routes")
                           for layer in range(78))
            counts.update(f"reset.{step}.{field}.count"
                          for field in ("indices", "routes"))
        require(tensors | counts <= names.keys(),
                f"{label}.reset.{step}.complete", checks)
        require(all(names[name].get("exact") is True for name in tensors),
                f"{label}.reset.{step}.exact", checks)
    if report["initial_position"]:
        for phase in ("decode", "reset"):
            expected = set()
            for layer in range(78):
                fields = ["latent", "rope"]
                if layer < 3 or (layer - 2) % 4 == 0:
                    fields.append("index_keys")
                    if 6 <= layer <= 66:
                        fields.append("index_scales")
                expected.update(f"{phase}.prefix.{layer}.{field}"
                                for field in fields)
            require(expected <= names.keys() and all(
                names[name].get("exact") is True for name in expected
            ), f"{label}.{phase}.prefix_exact", checks)


def validate_inputs(report, reference, checks, label):
    initial, steps = report["initial_position"], report["steps"]
    require(type(initial) is int and initial >= 0
            and type(steps) is int and steps > 0,
            label + ".position_range", checks)
    inputs, consumed = report["inputs"], report["consumed_tokens"]
    require(valid_tokens(consumed) and len(consumed) == steps
            and consumed == reference["teacher_tokens"][:steps]
            and initial == reference["initial_position"]
            and inputs["initial_position"] == initial
            and report["final_position"] == initial + steps,
            label + ".teacher_tokens_and_positions", checks)
    if initial:
        require(inputs["prompt_length"] == initial
                and all(inputs[key] == reference[key] for key in IDENTITY)
                and type(inputs["last_prompt_token"]) is int
                and 0 <= inputs["last_prompt_token"] < 154880
                and inputs["first_token_id"] == consumed[0],
                label + ".original_prompt_identity", checks)
        prefix_hashes = reference["prefix_files_sha256"]
        require(set(prefix_hashes) == {
            f"rank-{rank}.pt" for rank in range(16)
        } and prefix_hashes[f"rank-{report['rank']}.pt"]
            == inputs["prefix_sha256"], label + ".prefix_hash", checks)
    else:
        cold = inputs["cold_teacher_tokens"]
        require(valid_tokens(cold) and consumed[:len(cold)] == cold
                and not reference["prefix_files_sha256"],
                label + ".cold_inputs", checks)


def validate_trace(trace, position, token, checks, label, *, native,
                   choices=False):
    import torch

    require(all(type(trace[field]) is int for field in (
        "position", "committed_position", "token",
    )) and trace["position"] == position
        and trace["committed_position"] == position + 1
        and trace["token"] == token, label + ".input", checks)
    for field, count, width in (
        ("hidden_states", 79, 6144), ("layer_logits", 78, 154880),
    ):
        values = trace.get(field, ())
        require(isinstance(values, (list, tuple)) and len(values) == count
                and all(isinstance(value, torch.Tensor)
                        and value.shape == (1, width) for value in values),
                label + ".complete_" + field, checks)
    require(isinstance(trace["logits"], torch.Tensor)
            and trace["logits"].shape == (1, 154880)
            and torch.equal(trace["logits"], trace["layer_logits"][-1]),
            label + ".final_logits_identity", checks)
    if native:
        require(torch.equal(trace["last_hidden"], trace["hidden_states"][-1])
                and trace["token_ids"].shape == (1,)
                and torch.equal(trace["token_ids"],
                                trace["logits"].argmax(-1)),
                label + ".final_hidden_and_prediction", checks)
    if choices:
        for field in ("indices", "routes"):
            values = trace.get(field, ())
            require(isinstance(values, (tuple, list)) and len(values) == 78,
                    label + ".complete_" + field, checks)
            for layer, value in enumerate(values):
                if field == "routes" and layer < 3:
                    require(value is None,
                            f"{label}.{field}.{layer}.dense", checks)
                    continue
                width = 8 if field == "routes" else (
                    2048 if native else min(position + 1, 2048)
                )
                require(isinstance(value, torch.Tensor)
                        and value.dtype in (torch.int32, torch.int64)
                        and value.shape == (width,),
                        f"{label}.{field}.{layer}.layout", checks)
    rows = trace["cache_rows"]
    require(len(rows) == 78, label + ".cache_count", checks)
    for layer, row in enumerate(rows):
        full = layer < 3 or (layer - 2) % 4 == 0
        quantized = full and 6 <= layer <= 66
        required = {"latent": ((512,), torch.bfloat16),
                    "rope": ((64,), torch.bfloat16)}
        if full:
            required["index"] = (
                (128,), torch.int8 if quantized else torch.bfloat16,
            )
        if quantized:
            required["index_scale"] = ((1,), torch.float16)
        require(set(row) == set(FIELDS) and all(
            (isinstance(row[field], torch.Tensor)
             and row[field].shape == required[field][0]
             and row[field].dtype == required[field][1])
            if field in required else row[field] is None
            for field in FIELDS
        ), f"{label}.cache.{layer}.layout", checks)


def compare_saved(native, reference_dir):
    import torch

    checks, diagnostics, artifacts = [], [], {}
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "HF cross-configuration diagnostics from saved TP16 M1 "
                 "tensors; execution and exact reset remain checks",
        "comparison_kind": "cross_configuration_hf",
        "native": str(native), "reference": str(reference_dir),
        "precision_verified": False, "performance_measured": False,
        "coverage": {"routing_choices": False, "index_choices": False,
                     "mtp": False},
        "checks": checks, "diagnostics": diagnostics,
        "artifact_sha256": artifacts,
        "completed_rank_steps": 0,
        "source_sha256": {
            name: digest(Path(__file__).with_name(name)) for name in (
                Path(__file__).name, "run_cann_model.py",
                "run_native_decode.py", "precision_gate.py",
            )
        },
    }
    try:
        summary = read_report(native / "summary.json", artifacts)
        receipts = summary["ranks"]
        require(len(receipts) == 16
                and all(type(item["rank"]) is int for item in receipts)
                and {item["rank"] for item in receipts} == set(range(16))
                and summary["all_ranks_agree"] is True,
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
            require(report["rank"] == rank and report["world_size"] == 16
                    and report["source_sha256"]
                    and set(report["checkpoint_sha256"]) == CHECKPOINT_FILES,
                    label + ".execution_identity", checks)
            validate_reference_manifest(reference, report["checkpoint_sha256"])
            checks.append({"name": label + ".hf_manifest", "passed": True})
            validate_inputs(report, reference, checks, label)
            validate_reset(report, checks, label)
            require(all(report["coverage"].get(field) is True for field in (
                "intermediate_hidden_states", "intermediate_layer_logits",
                "all_layers_new_kv", "reset_exact",
            )), label + ".layer_trace_enabled", checks)
            selection = [report["coverage"].get(field, False)
                         for field in ("routing_choices", "index_choices")]
            require(all(type(value) is bool for value in selection)
                    and selection[0] == selection[1],
                    label + ".choice_coverage", checks)
            require(receipt["consumed_tokens"] == report["consumed_tokens"]
                    and receipt["predicted_tokens"]
                    == report["predicted_tokens"]
                    and valid_tokens(report["predicted_tokens"])
                    and len(report["predicted_tokens"]) == report["steps"],
                    label + ".receipt_tokens", checks)
            if reports:
                first = reports[0]
                keys = ("initial_position", "steps", "consumed_tokens",
                        "predicted_tokens", "checkpoint_sha256",
                        "source_sha256")
                require(all(report[key] == first[key] for key in keys),
                        label + ".rank_agreement", checks)
                if report["initial_position"]:
                    require(all(report["inputs"][key]
                                == first["inputs"][key] for key in (
                                    "capture_manifest_sha256", "request_id",
                                    "generation", *IDENTITY,
                                )), label + ".capture_agreement", checks)
            reports.append(report)
        first = reports[0]
        initial, steps = first["initial_position"], first["steps"]
        require(valid_tokens(reference["teacher_tokens"])
                and len(reference["teacher_tokens"]) >= steps
                and reference["checks"]
                and all(item["passed"] is True for item in reference["checks"])
                and any(item["name"] == "reset.cache_lengths"
                        for item in reference["checks"]),
                "hf.reset_passed", checks)
        names = [f"step-{initial + step:05d}.pt" for step in range(steps)]
        for rank, report in enumerate(reports):
            require(set(report["trace_sha256"]) == set(names),
                    f"rank.{rank}.complete_trace_files", checks)
            for name in names:
                path = native / f"rank-{rank}" / name
                artifacts[str(path)] = digest(path)
                require(artifacts[str(path)] == report["trace_sha256"][name],
                        f"rank.{rank}.{name}.hash", checks)
        for offset, name in enumerate(names):
            path = reference_dir / name
            artifacts[str(path)] = digest(path)
            require(artifacts[str(path)] == reference["trace_sha256"][name],
                    f"hf.{name}.hash", checks)
            expected = torch.load(path, map_location="cpu", weights_only=True)
            position = initial + offset
            token = first["consumed_tokens"][offset]
            validate_trace(expected, position, token, checks, f"hf.{position}",
                           native=False, choices=any(
                               r["coverage"].get("routing_choices", False)
                               for r in reports
                           ))
            for rank, report in enumerate(reports):
                label = f"rank.{rank}.step.{position}"
                actual = torch.load(native / f"rank-{rank}" / name,
                                    map_location="cpu", weights_only=True)
                validate_trace(actual, position, token, checks, label,
                               native=True, choices=report["coverage"].get(
                                   "routing_choices", False,
                               ))
                if report["coverage"].get("routing_choices"):
                    validate_choice_sources(actual, report, checks, label)
                require(actual["token_ids"].item()
                        == report["predicted_tokens"][offset],
                        label + ".reported_prediction", checks)
                compare_step(label, actual, expected, checks,
                             diagnostics=diagnostics)
                output["completed_rank_steps"] += 1
        output.update(
            checkpoint_sha256=first["checkpoint_sha256"],
            initial_position=initial, teacher_tokens=first["consumed_tokens"],
            input_identity=first["inputs"],
            native_source_sha256=first["source_sha256"],
            reference_source_sha256=reference["source_sha256"],
        )
        output["coverage"].update(
            tp_ranks=16, steps=steps, hidden_states_per_step=79,
            layer_logits_per_step=78, all_layers_new_kv=True, reset_exact=True,
            prompt_tokens=initial, frozen_prefix_exact=initial > 0,
            routing_choices=all(r["coverage"].get("routing_choices") is True
                                for r in reports),
            index_choices=all(r["coverage"].get("index_choices") is True
                              for r in reports),
        )
    except Exception as exc:
        checks.append({"name": "validation_error", "passed": False,
                       "error": f"{type(exc).__name__}: {exc}"})
    failed = sum(not item["passed"] for item in checks)
    output.update(status="fail" if failed else "diagnostic",
                  continuation_parity=False, layerwise_logits_parity=False,
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
        parser.error("Write the report outside the input directories")
    import torch

    torch.set_num_threads(4)
    output = compare_saved(args.native, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({key: output[key] for key in (
        "status", "checks_total", "checks_failed", "completed_rank_steps",
        "hf_diagnostics_complete", "precision_verified",
    )}))
    return int(output["checks_failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
