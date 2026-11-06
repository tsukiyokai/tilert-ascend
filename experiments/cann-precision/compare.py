"""Strictly compare a completed TP16 capture with its independent HF traces."""

import argparse
import hashlib
import json
import traceback
from datetime import UTC, datetime
from importlib import util
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
spec = util.spec_from_file_location("saved_cann_precision_entry", HERE / "run.py")
entry = util.module_from_spec(spec)
spec.loader.exec_module(entry)
require = entry.require
TRACE_NAMES = {f"serial-{row:02d}.pt" for row in range(entry.WIDTH)} | {
    "wide.pt",
    "reset.pt",
    "wide-stamps.pt",
    "reset-stamps.pt",
}
STAMP_COUNT = 2 + 79 + 78 + 75


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def capture_receipts(directory, exit_receipt):
    """Bind successful process exit, summary and all sixteen actual reports."""
    summary_path = directory / "summary.json"
    summary = read_json(summary_path)
    completed = read_json(exit_receipt)
    require(
        completed.get("schema") == "cann-capture-exit-v1"
        and type(completed.get("returncode")) is int
        and completed["returncode"] == 0,
        "Capture launcher did not record exit zero",
    )
    require(
        completed.get("summary_sha256") == digest(summary_path),
        "Launcher exit belongs to a different capture summary",
    )
    receipts = summary["ranks"]
    require(
        len(receipts) == 16
        and all(type(item["rank"]) is int for item in receipts)
        and {item["rank"] for item in receipts} == set(range(16)),
        "Capture requires sixteen unique integer ranks",
    )
    require(
        summary.get("mode") == "capture_only"
        and summary.get("status") == "captured_unverified"
        and all(
            summary.get(key) is True
            for key in (
                "capture_complete",
                "capture_checks_passed",
                "entry_passed",
                "all_rank_state_agreement",
            )
        ),
        "Capture execution did not pass",
    )
    reports = []
    for receipt in sorted(receipts, key=lambda item: item["rank"]):
        rank = receipt["rank"]
        path = directory / f"rank-{rank}.json"
        require(
            digest(path) == receipt["report_sha256"],
            "Rank report bytes changed",
        )
        report = read_json(path)
        require(
            report["source_commit"] == completed["source_commit"]
            and report["command"] == completed["command"],
            "Launcher exit belongs to a different capture command",
        )
        require(
            type(report["rank"]) is int
            and report["rank"] == rank
            and type(report["world_size"]) is int
            and report["world_size"] == 16
            and report["schema"] == "cann-single-case-eager-v1"
            and report["mode"] == "capture_only"
            and report["status"] == "captured_unverified",
            "Rank report identity or mode differs",
        )
        for item in (summary, report):
            require(
                all(
                    item.get(key) is False
                    for key in (
                        "case_precision_passed",
                        "precision_verified",
                        "formal_precision_gate_passed",
                        "performance_measured",
                        "compiled_graph_verified",
                        "super_kernel_verified",
                    )
                ),
                "Capture must retain its original unverified scope",
            )
        require(
            all(
                report.get(key) is True
                for key in (
                    "capture_complete",
                    "capture_checks_passed",
                    "entry_passed",
                    "npu_executed",
                )
            )
            and report["timed"] is False
            and report["profiled"] is False
            and report["checks"]
            and all(check.get("passed") is True for check in report["checks"]),
            "Rank execution checks did not pass",
        )
        require(
            report["states"] == receipt["states"] == receipts[0]["states"]
            and type(report["final_position"]) is int
            and report["final_position"] == receipt["final_position"] == 0,
            "Rank state agreement differs",
        )
        require(
            set(report["trace_sha256"]) == TRACE_NAMES,
            "Capture raw inventory is incomplete",
        )
        execution = report["execution"]
        require(
            execution
            == {
                "widths": [1, 32],
                "layers": 78,
                "fp32_head": True,
                "attention_kernel": "sfa",
                "enforce_eager": True,
                "compile_target": False,
                "super_kernel": False,
                "graph": None,
                "mtp_model_loaded": False,
                "capacity": execution["capacity"],
            }
            and type(execution["capacity"]) is int
            and execution["capacity"] >= entry.INITIAL + entry.WIDTH
            and report["model_tp_reduction"] == "float32"
            and report["device"] == f"npu:{rank}"
            and report["loaded_libraries_sha256"]
            and all(
                report["packages"].get(name)
                for name in ("torch", "torch-npu", "vllm", "vllm-ascend")
            ),
            "Captured runtime differs from the eager SFA FP32 contract",
        )
        if reports:
            require(
                report["packages"] == reports[0]["packages"]
                and execution == reports[0]["execution"],
                "Rank runtime differs",
            )
        for key in ("checkpoint_sha256", "source_sha256"):
            require(report[key] == summary[key], "Capture ranks disagree: " + key)
        require(
            report["teacher"]["sha256"] == summary["teacher_sha256"],
            "Capture teacher identity differs",
        )
        for name, expected in entry.source_identity().items():
            if name.endswith(".py") and name != "experiments/cann-precision/check.py":
                require(
                    report["source_sha256"].get(name) == expected,
                    "Captured execution source differs: " + name,
                )
        reports.append(report)
    return (
        summary,
        reports,
        {
            "summary_sha256": digest(summary_path),
            "exit_receipt_sha256": digest(exit_receipt),
            "rank_reports_sha256": {str(item["rank"]): item["report_sha256"] for item in receipts},
        },
    )


def validate_frame(trace, tokens, position):
    """Reject missing coverage before the legacy comparator can skip a field."""
    width = len(tokens)
    require(
        trace["token"] == list(tokens)
        and all(type(token) is int for token in trace["token"])
        and type(trace["position"]) is int
        and trace["position"] == position
        and type(trace["committed_position"]) is int
        and trace["committed_position"] == position + width,
        "Raw frame input or position differs",
    )

    def tensor(value, shape, dtypes):
        require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and tuple(value.shape) == shape
            and value.dtype in dtypes,
            "Raw tensor layout or dtype differs",
        )

    tensor(trace["token_ids"], (width,), (torch.int64,))
    tensor(trace["logits"], (width, 154880), (torch.float32,))
    tensor(trace["last_hidden"], (width, 6144), (torch.bfloat16,))
    for field, count, size, dtypes in (
        ("hidden_states", 79, 6144, (torch.bfloat16,)),
        ("layer_logits", 78, 154880, (torch.float32,)),
        ("indices", 78, 2048, (torch.int32, torch.int64)),
        ("routes", 78, 8, (torch.int32, torch.int64)),
    ):
        require(len(trace[field]) == count, "Incomplete raw " + field)
        for layer, value in enumerate(trace[field]):
            if field == "routes" and layer < 3:
                require(value is None, "Dense layers cannot have routed choices")
            else:
                tensor(value, (width, size), dtypes)
    require(len(trace["cache_rows"]) == 78, "Incomplete raw KV")
    for layer, values in enumerate(trace["cache_rows"]):
        require(
            set(values) == {"latent", "rope", "index", "index_scale"},
            "Raw cache fields differ",
        )
        tensor(values["latent"], (width, 512), (torch.bfloat16,))
        tensor(values["rope"], (width, 64), (torch.bfloat16,))
        full = layer < 3 or (layer - 2) % 4 == 0
        quantized = full and 6 <= layer <= 66
        if full:
            tensor(
                values["index"],
                (width, 128),
                (torch.int8,) if quantized else (torch.bfloat16,),
            )
        else:
            require(
                values["index"] is None,
                "Shared index layer stores no new index KV",
            )
        if quantized:
            scale = values["index_scale"]
            require(
                isinstance(scale, torch.Tensor) and tuple(scale.shape) in ((width,), (width, 1)),
                "Raw index scale must retain one scalar per row",
            )
        else:
            require(values["index_scale"] is None, "Unexpected index scale")


def validate_stamps(stamps, epoch):
    require(
        isinstance(stamps, torch.Tensor)
        and stamps.dtype == torch.int64
        and tuple(stamps.shape) == (STAMP_COUNT,)
        and bool(torch.all(stamps == epoch)),
        "Missing or stale observation stamps",
    )


def validate_prefix_checks(saved, prefixes):
    checks = {item["name"]: item for item in saved["checks"]}
    require(len(checks) == len(saved["checks"]), "Duplicate saved checks")
    for phase in ("serial", "wide", "reset"):
        for layer, values in enumerate(prefixes):
            for field, value in values.items():
                if value is not None:
                    check = checks.get(f"{phase}.prefix.{layer}.{field}", {})
                    require(
                        check.get("passed") is True and check.get("exact") is True,
                        "Missing exact frozen-prefix check",
                    )


def compare_rank(args, saved, reference, teacher, inputs, checkpoint, prefixes):
    rank, checks = saved["rank"], []
    require(
        saved["checkpoint_sha256"] == checkpoint
        and {key: value for key, value in saved["teacher"].items() if key != "file"}
        == {key: value for key, value in teacher.items() if key != "file"},
        "Checkpoint or teacher differs from admitted HF inputs",
    )
    validate_prefix_checks(saved, prefixes)
    for key in (
        "prompt_length",
        "prompt_sha256",
        "last_prompt_token",
        "initial_position",
        "first_token_id",
        "request_id",
        "generation",
        "capture_manifest_sha256",
    ):
        require(
            saved["inputs"][key] == inputs[key],
            "Capture input differs: " + key,
        )
    prefix = args.capture_case / "main-prefix" / f"rank-{rank}.pt"
    require(
        digest(prefix)
        == saved["inputs"]["prefix_sha256"]
        == reference["manifest"]["prefix_files_sha256"][f"rank-{rank}.pt"],
        "HF and raw capture prefix bytes differ",
    )
    directory = args.capture / f"rank-{rank}"
    for name, expected in saved["trace_sha256"].items():
        require(
            digest(directory / name) == expected,
            "Raw capture bytes changed: " + name,
        )

    def load(name):
        return torch.load(directory / name, map_location="cpu", weights_only=True, mmap=True)

    tokens = teacher["teacher_tokens"]
    wide, reset = load("wide.pt"), load("reset.pt")
    for epoch, (phase, trace) in enumerate((("wide", wide), ("reset", reset)), start=1):
        validate_frame(trace, tokens, entry.INITIAL)
        validate_stamps(load(phase + "-stamps.pt"), epoch)
    predictions = {phase: [] for phase in ("serial", "wide", "reset")}
    choice_sources = None
    for row, token in enumerate(tokens):
        serial = load(f"serial-{row:02d}.pt")
        validate_frame(serial, (token,), entry.INITIAL + row)
        actual_sources = serial["choice_source_sha256"]
        require(
            actual_sources and (choice_sources is None or actual_sources == choice_sources),
            "Actual choice-source identity is missing or changed",
        )
        for path, source_digest in actual_sources.items():
            expected_digest = saved["source_sha256"].get(path)
            if expected_digest is not None:
                require(source_digest == expected_digest, "Choice and model source hashes differ")
        choice_sources = actual_sources
        name = f"step-{entry.INITIAL + row:05d}.pt"
        expected = torch.load(
            args.reference / name,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        rows = {
            "serial": entry.single_row(serial, 0),
            "wide": entry.single_row(wide, row),
            "reset": entry.single_row(reset, row),
        }
        for phase, actual in rows.items():
            entry.strict_row(f"hf.{phase}.{row}", actual, expected, checks)
            predictions[phase].append(int(actual["token_ids"][0]))
        entry.strict_row(f"serial_vs_wide.{row}", rows["wide"], rows["serial"], checks)
        entry.strict_row(f"serial_vs_reset.{row}", rows["reset"], rows["serial"], checks)
        entry.strict_row(
            f"reset_exact.{row}",
            rows["reset"],
            rows["wide"],
            checks,
            exact=True,
        )
    states = saved["states"]
    require(
        [state["phase"] for state in states] == ["serial", "wide", "reset"],
        "Capture phases differ",
    )
    for state in states:
        require(
            type(state["position"]) is int
            and state["position"] == entry.INITIAL + entry.WIDTH
            and state["predictions"] == predictions[state["phase"]],
            "Raw and saved states differ",
        )
    for epoch, state in enumerate(states[1:], start=1):
        require(
            type(state["epoch"]) is int
            and state["epoch"] == epoch
            and type(state["stamp_count"]) is int
            and state["stamp_count"] == STAMP_COUNT,
            "Capture epoch or stamp coverage differs",
        )
    passed = bool(checks) and all(check["passed"] for check in checks)
    report = {
        "rank": rank,
        "case_precision_passed": passed,
        "checks": checks,
        "trace_sha256": saved["trace_sha256"],
        "choice_source_sha256": choice_sources,
        "runtime": {
            key: saved[key]
            for key in (
                "execution",
                "packages",
                "device",
                "device_properties",
                "model_tp_reduction",
                "loaded_libraries_sha256",
            )
        },
    }
    path = args.output / f"rank-{rank}.json"
    entry.atomic_json(path, report)
    return {
        "rank": rank,
        "case_precision_passed": passed,
        "report_sha256": digest(path),
        "checks_total": len(checks),
        "checks_failed": sum(not check["passed"] for check in checks),
        "choice_source_sha256": choice_sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "capture",
        "capture-exit",
        "checkpoint",
        "capture-case",
        "teacher",
        "reference",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory; preserve existing comparison receipts")
    args.output.mkdir(parents=True)
    args.capture_only, args.capacity = False, 9344
    report = {
        "schema": "cann-single-case-saved-hf-v1",
        "status": "error",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "one frozen TP16 eager 8K/32-row saved capture vs independent HF",
        "case_precision_passed": False,
        "precision_verified": False,
        "formal_precision_gate_passed": False,
        "performance_measured": False,
        "compiled_graph_verified": False,
        "super_kernel_verified": False,
        "comparison_source_sha256": digest(Path(__file__)),
        "ranks": [],
    }
    try:
        torch.set_num_threads(1)
        summary, ranks, binding = capture_receipts(args.capture, args.capture_exit)
        prefixes, teacher, inputs, reference, checkpoint = entry.admit(args, 0)
        report.update(
            capture_binding=binding,
            checkpoint_sha256=checkpoint,
            reference_manifest_sha256=reference["manifest_sha256"],
            teacher_sha256=teacher["sha256"],
            source_sha256=summary["source_sha256"],
        )
        for saved in ranks:
            result = compare_rank(args, saved, reference, teacher, inputs, checkpoint, prefixes)
            report["ranks"].append(result)
            entry.atomic_json(args.output / "summary.json", report)
            print(json.dumps(result), flush=True)
        require(
            all(
                item["choice_source_sha256"] == report["ranks"][0]["choice_source_sha256"]
                for item in report["ranks"]
            ),
            "Ranks used different choice sources",
        )
        passed = all(item["case_precision_passed"] for item in report["ranks"])
        report.update(
            status="numerical_pass" if passed else "fail",
            case_precision_passed=passed,
        )
    except Exception:  # noqa: BLE001 - Preserve a failed CLI receipt.
        report["error"] = traceback.format_exc()
    entry.atomic_json(args.output / "summary.json", report)
    print(json.dumps({key: report[key] for key in ("status", "case_precision_passed")}))
    return int(not report["case_precision_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
