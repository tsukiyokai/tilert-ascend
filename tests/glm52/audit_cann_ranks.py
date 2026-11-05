"""Locate rank and reset divergence in frozen TP16 CANN M1 captures.

This CPU audit reads first-pass tensors and recorded reset comparisons. It
does not reconstruct reset tensors or identify the responsible operator.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


CACHE_FIELDS = ("latent", "rope", "index", "index_scale")
TRACE_COUNTS = {"hidden_states": 79, "layer_logits": 78,
                "cache_rows": 78, "indices": 78, "routes": 78}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def tensor_difference(actual, expected):
    import torch

    if actual is None or expected is None:
        return {"equal": actual is expected}
    result = {"shape": list(actual.shape), "dtype": str(actual.dtype)}
    same_layout = (actual.shape == expected.shape
                   and actual.dtype == expected.dtype)
    result["equal"] = same_layout and torch.equal(actual, expected)
    if result["equal"]:
        return result
    if not same_layout:
        result.update(expected_shape=list(expected.shape),
                      expected_dtype=str(expected.dtype))
        return result
    different = actual != expected
    coordinate = different.nonzero()[0].tolist()
    left, right = actual.double(), expected.double()
    delta = left - right
    denominator = float(right.norm())
    result.update(
        unequal=int(different.sum()), first_coordinate=coordinate,
        actual=float(actual[tuple(coordinate)]),
        expected=float(expected[tuple(coordinate)]),
        max_abs=float(delta.abs().max()),
        nrmse=float(delta.norm()) / denominator if denominator else None,
    )
    return result


def event_groups(trace):
    """Order observed boundaries by decoder execution, not report order."""
    yield None, "embedding", {"hidden_states.0": trace["hidden_states"][0]}
    for layer in range(78):
        yield layer, "cache_write", {
            f"cache.{layer}.{field}": trace["cache_rows"][layer][field]
            for field in CACHE_FIELDS
        }
        for field in ("indices", "routes"):
            yield layer, field, {f"{field}.{layer}": trace[field][layer]}
        yield layer, "decoder_output", {
            f"hidden_states.{layer + 1}": trace["hidden_states"][layer + 1],
        }


def first_boundary(actual, expected):
    for (layer, stage, left), (_, _, right) in zip(
        event_groups(actual), event_groups(expected),
    ):
        differences = {name: tensor_difference(value, right[name])
                       for name, value in left.items()}
        failed = {name: value for name, value in differences.items()
                  if not value["equal"]}
        if failed:
            return {"layer": layer, "stage": stage, "differences": failed}
    return None


def rank_comparison(actual, expected, rank):
    return {
        "rank": rank, "first_boundary": first_boundary(actual, expected),
        "embedding": tensor_difference(actual["hidden_states"][0],
                                       expected["hidden_states"][0]),
        "layer0_output": tensor_difference(actual["hidden_states"][1],
                                           expected["hidden_states"][1]),
        "layer0_cache_equal": {
            field: tensor_difference(actual["cache_rows"][0][field],
                                     expected["cache_rows"][0][field])["equal"]
            for field in CACHE_FIELDS
        },
        "layer1_latent": tensor_difference(
            actual["cache_rows"][1]["latent"],
            expected["cache_rows"][1]["latent"],
        ),
        "token_ids": tensor_difference(actual["token_ids"],
                                       expected["token_ids"]),
        "logits": tensor_difference(actual["logits"], expected["logits"]),
    }


def reset_comparison(report, trace, step):
    prefix = f"reset.{step}."
    recorded = {item["name"][len(prefix):]: item for item in report["checks"]
                if item["name"].startswith(prefix)}
    first = None
    for layer, stage, values in event_groups(trace):
        failures = {name: recorded[name] for name in values
                    if not recorded[name]["passed"]}
        if failures:
            first = {"layer": layer, "stage": stage, "checks": failures}
            break
    return {
        "rank": report["rank"], "first_boundary": first,
        "layer0_output": recorded["hidden_states.1"],
        "layer1_latent": recorded["cache.1.latent"],
        "layer0_cache_equal": {
            field: recorded[f"cache.0.{field}"]["passed"]
            for field in CACHE_FIELDS
        },
        "logits": recorded["logits"],
        "indices_exact": all(recorded[f"indices.{layer}"]["passed"]
                             for layer in range(78)),
        "routes_order_different_layers": [
            layer for layer in range(78)
            if not recorded[f"routes.{layer}"]["passed"]
        ],
        "failed_check_count": sum(not item["passed"]
                                  for item in recorded.values()),
    }


def load_trace(root, report, step):
    import torch

    position = report["initial_position"] + step
    name = f"step-{position:05d}.pt"
    path = root / f"rank-{report['rank']}" / name
    actual_hash = digest(path)
    if actual_hash != report["trace_sha256"][name]:
        raise ValueError(f"Trace SHA differs: {path}")
    trace = torch.load(path, map_location="cpu", weights_only=True)
    expected = (report["consumed_tokens"][step], position, position + 1)
    if tuple(trace[key] for key in (
        "token", "position", "committed_position",
    )) != expected:
        raise ValueError(f"Trace token/position differs: {path}")
    if trace["token_ids"].tolist() != [report["predicted_tokens"][step]]:
        raise ValueError(f"Trace predicted token differs: {path}")
    for field, count in TRACE_COUNTS.items():
        if len(trace[field]) != count:
            raise ValueError(f"Incomplete {field}: {path}")
    return trace, {"path": str(path), "sha256": actual_hash}


def audit(root):
    summary_path = root / "summary.json"
    summary = read_json(summary_path)
    entries = summary["ranks"]
    if (len(entries) != 16
            or {item["rank"] for item in entries} != set(range(16))):
        raise ValueError("Summary must contain exactly 16 ranks")
    reports, artifacts = [], []
    for rank in range(16):
        path = root / f"rank-{rank}.json"
        actual_hash = digest(path)
        entry = next(item for item in entries if item["rank"] == rank)
        if actual_hash != entry["report_sha256"]:
            raise ValueError(f"Rank report SHA differs: {path}")
        report = read_json(path)
        if report["rank"] != rank or report["world_size"] != 16:
            raise ValueError(f"Report rank/world size differs: {path}")
        if reports:
            for key in ("checkpoint_sha256", "source_sha256", "steps",
                        "initial_position", "consumed_tokens"):
                if report[key] != reports[0][key]:
                    raise ValueError(f"Rank {rank} differs in {key}")
        reports.append(report)
        artifacts.append({"path": str(path), "sha256": actual_hash})
    steps = []
    for step in range(reports[0]["steps"]):
        reference, identity = load_trace(root, reports[0], step)
        artifacts.append(identity)
        cross_rank = []
        reset = [reset_comparison(reports[0], reference, step)]
        for rank in range(1, 16):
            trace, identity = load_trace(root, reports[rank], step)
            artifacts.append(identity)
            cross_rank.append(rank_comparison(trace, reference, rank))
            reset.append(reset_comparison(reports[rank], trace, step))
        steps.append({
            "step": step, "position": reference["position"],
            "input_token": reference["token"],
            "predicted_tokens_by_rank": [r["predicted_tokens"][step]
                                         for r in reports],
            "all_rank_logits_equal": all(row["logits"]["equal"]
                                         for row in cross_rank),
            "cross_rank_against_rank0": cross_rank,
            "reset_recorded_comparisons": reset,
        })
    return {
        "artifact_validation": "pass", "precision_verified": False,
        "scope": "Frozen first-pass TP16 tensors and recorded reset checks",
        "reset_evidence": "Recorded comparisons, without reset tensor reads",
        "cache_evidence": "Current row only",
        "capture_root": str(root),
        "summary": {"path": str(summary_path),
                    "sha256": digest(summary_path)},
        "checkpoint_sha256": reports[0]["checkpoint_sha256"],
        "captured_source_sha256": reports[0]["source_sha256"],
        "artifacts": artifacts, "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new evidence path")
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch

    torch.set_num_threads(4)
    result = audit(args.capture.resolve())
    result.update(timestamp=datetime.now(timezone.utc).isoformat(),
                  audit_source_sha256=digest(Path(__file__)),
                  torch_version=torch.__version__,
                  device_execution=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output),
                      "sha256": digest(args.output),
                      "artifact_validation": result["artifact_validation"],
                      "steps": len(result["steps"])}))


if __name__ == "__main__":
    main()
