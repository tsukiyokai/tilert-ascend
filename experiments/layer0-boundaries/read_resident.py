"""Strictly read a resident TP16 attention pair without changing the oracle."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
FIELDS = (
    "token_ids",
    "logits",
    "last_hidden",
    "hidden_states",
    "layer_logits",
    "indices",
    "routes",
    "cache_rows",
    "routing_receipts",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def state_digest(value):
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {name: state_digest(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [state_digest(item) for item in value]
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(
            value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }


def route_basis(actual):
    receipts = actual["routing_receipts"]
    assert sorted(receipts) == list(range(3, 78))
    verified = []
    for number, row in receipts.items():
        assert torch.equal(
            row["ids"].flatten(), actual["routes"][number].flatten()
        )
        assert row["ids"].shape == row["weights"].shape == (1, 8)
        metadata = row["metadata"]
        verified.append(
            metadata["eplb_state_known"]
            and metadata["eplb_state_present"] is False
            and metadata["log2phy_present"] is False
            and not metadata["mix_placement"]
            and not metadata["enable_force_eplb"]
            and not metadata["enable_force_load_balance"]
        )
    return all(verified)


def read_phase(args, phase, expected, manifest, contract, entry):
    folder = args.pair / phase
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert sorted(row["rank"] for row in summary["ranks"]) == list(range(16))
    ranks, states = [], []
    for rank in range(16):
        report_path = folder / f"rank-{rank}.json"
        report = json.loads(report_path.read_text())
        receipt = next(row for row in summary["ranks"] if row["rank"] == rank)
        assert digest(report_path) == receipt["report_sha256"]
        assert report["rank"] == rank and report["world_size"] == 16
        assert report["steps"] == 1 and report["initial_position"] == 8192
        assert report["consumed_tokens"] == [expected["token"]]
        assert report["checkpoint_sha256"] == manifest["metadata_sha256"]
        assert (
            report["inputs"]["prefix_sha256"]
            == manifest["prefix_files_sha256"][f"rank-{rank}.pt"]
        )
        assert report["inputs"]["prompt_sha256"] == manifest["prompt_sha256"]
        for path, sha in contract["source_sha256"].items():
            assert report["source_sha256"]["/work/" + path] == sha
        assert report["coverage"]["intermediate_layer_logits"]
        assert report["coverage"]["routing_choices"]
        assert report["coverage"]["index_choices"]
        if phase == "expanded":
            assert report["parent_control_report_sha256"] == digest(
                args.pair / "control" / f"rank-{rank}.json"
            )
        traces = {}
        paths = {}
        for name, field in (
            ("decode", "trace_sha256"),
            ("reset", "reset_trace_sha256"),
        ):
            assert len(report[field]) == 1
            filename, sha = next(iter(report[field].items()))
            path = folder / f"rank-{rank}" / filename
            assert digest(path) == sha
            traces[name] = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
            paths[name] = {"path": str(path), "sha256": sha}
            assert traces[name]["attention_recipe"]["recipe"] == phase
            assert (
                traces[name]["attention_recipe"]["matmul_allow_hf32"] is False
            )
        actual, repeated = traces["decode"], traces["reset"]
        checks, reset = [], []
        entry.strict_row("hf", actual, expected, checks)
        entry.strict_row("reset", repeated, actual, reset, exact=True)
        for number in range(3, 78):
            left = repeated["routing_receipts"][number]
            right = actual["routing_receipts"][number]
            for field in ("ids", "weights", "router_logits"):
                entry.compare_tensor(
                    f"reset.routing_receipts.{number}.{field}",
                    left[field],
                    right[field],
                    reset,
                    exact=True,
                )
            reset.append(
                {
                    "name": f"reset.routing_receipts.{number}.metadata",
                    "passed": left["metadata"] == right["metadata"],
                }
            )
        hidden = [
            row
            for row in checks
            if row["name"].startswith("hf.hidden_states.")
            and row["name"].rsplit(".", 1)[-1].isdigit()
        ]
        assert len(hidden) == 79
        first_hidden = next(
            (
                int(row["name"].rsplit(".", 1)[-1])
                for row in hidden
                if not row["passed"]
            ),
            79,
        )
        first_route = next(
            (
                number
                for number in range(3, 78)
                if any(
                    row["name"] == f"hf.routes.{number}" and not row["passed"]
                    for row in checks
                )
            ),
            None,
        )
        ranks.append(
            {
                "rank": rank,
                "report_sha256": digest(report_path),
                "artifacts": paths,
                "checks": checks,
                "reset_checks": reset,
                "raw_capture_checks_passed": all(
                    row["passed"] for row in report["checks"]
                ),
                "strict_hf_passed": all(row["passed"] for row in checks),
                "exact_reset_passed": all(row["passed"] for row in reset),
                "route_id_basis_verified": route_basis(actual)
                and route_basis(repeated),
                "first_hidden_breach": first_hidden,
                "first_route_disagreement": first_route,
                "final_logits": next(
                    row for row in checks if row["name"] == "hf.logits"
                ),
            }
        )
        states.append(
            {
                name: {field: state_digest(trace[field]) for field in FIELDS}
                for name, trace in traces.items()
            }
        )
    agreement = {
        name: {
            field: all(
                state[name][field] == states[0][name][field]
                for state in states
            )
            for field in FIELDS
        }
        for name in ("decode", "reset")
    }
    return {
        "summary_sha256": digest(summary_path),
        "ranks": ranks,
        "rank_state_agreement": agreement,
        "all_rank_states_agree": all(
            all(values.values()) for values in agreement.values()
        ),
        "all_raw_capture_checks_passed": all(
            row["raw_capture_checks_passed"] for row in ranks
        ),
        "all_resets_exact": all(row["exact_reset_passed"] for row in ranks),
        "all_first_frames_pass_hf": all(
            row["strict_hf_passed"] for row in ranks
        ),
        "route_id_basis_verified": all(
            row["route_id_basis_verified"] for row in ranks
        ),
        "first_hidden_breach": min(
            row["first_hidden_breach"] for row in ranks
        ),
        "first_route_disagreement": min(
            (
                row["first_route_disagreement"]
                for row in ranks
                if row["first_route_disagreement"] is not None
            ),
            default=None,
        ),
        "final_logits_nrmse_max": max(
            row["final_logits"]["nrmse"] for row in ranks
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "pair", "reference", "contract", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    global torch
    import torch

    torch.set_num_threads(8)
    path = args.source / "experiments/cann-precision/run.py"
    spec = importlib.util.spec_from_file_location("frozen_precision", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    assert "torch_npu" not in sys.modules
    manifest = json.loads((args.reference / "manifest.json").read_text())
    entry.validate_reference_manifest(
        manifest, manifest["metadata_sha256"], fp32_head=True
    )
    frame_path = args.reference / "step-08192.pt"
    assert digest(frame_path) == manifest["trace_sha256"][frame_path.name]
    contract = json.loads(args.contract.read_text())
    assert digest(frame_path) == contract["hf_frame_sha256"]
    expected = torch.load(
        frame_path, map_location="cpu", weights_only=True, mmap=True
    )
    entry.validate_trace(expected, 78, fp32_head=True)
    phases = {
        phase: read_phase(args, phase, expected, manifest, contract, entry)
        for phase in ("control", "expanded")
    }
    control, expanded = phases["control"], phases["expanded"]
    prediction = {
        "hidden_delayed": expanded["first_hidden_breach"]
        > control["first_hidden_breach"],
        "final_logits_improved": expanded["final_logits_nrmse_max"]
        < control["final_logits_nrmse_max"],
    }
    output = {
        "status": "diagnostic_complete",
        "scope": "Resident A3 TP16, original/expanded first M1 and exact reset, one8Kinput",
        "phases": phases,
        "prediction": prediction,
        "prediction_met": all(prediction.values()),
        "hf_frame_sha256": digest(frame_path),
        "contract_sha256": digest(args.contract),
        "comparison_source_sha256": {
            str(Path(inspect.getfile(value)).resolve()): digest(
                inspect.getfile(value)
            )
            for value in (
                entry.strict_row,
                entry.compare_step,
                entry.compare_tensor,
                entry.validate_trace,
            )
        },
        "script_sha256": digest(__file__),
        "formal_precision_gate_passed": False,
        "performance_measured": False,
        "missing_full_gate_evidence": [
            "M1x32/M32",
            "eight inputs",
            "graph/SK",
            "online P/D",
        ],
        "thresholds_modified": False,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "prediction": prediction,
                "phases": {
                    name: {
                        key: value
                        for key, value in result.items()
                        if key != "ranks"
                    }
                    for name, result in phases.items()
                },
                "formal_precision_gate_passed": False,
            }
        )
    )
    assert "torch_npu" not in sys.modules


if __name__ == "__main__":
    main()
