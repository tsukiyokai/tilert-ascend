"""Plot three layerwise CANN/HF differences from frozen CPU traces."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys


LAYERS = 78
HIDDEN_SIZE = 6144
VOCAB_SIZE = 154880


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def bf16_spacing(value):
    """Return positive-binade spacing at |value|, not an error ULP count."""
    if value == 0:
        return math.ldexp(1.0, -133)
    _, exponent = math.frexp(abs(value))
    return math.ldexp(1.0, max(exponent - 8, -133))


def validate_pair(left, right, width):
    import torch

    if (left.shape != (1, width) or right.shape != left.shape
            or not left.is_floating_point() or not right.is_floating_point()
            or not torch.isfinite(left).all()
            or not torch.isfinite(right).all()):
        raise ValueError(f"Expected finite floating tensors [1,{width}]")


def hidden_metrics(left, right):
    import torch

    validate_pair(left, right, HIDDEN_SIZE)
    if left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 hidden states")
    left, right = left.double().flatten(), right.double().flatten()
    delta = left - right
    left_norm, right_norm = left.norm(), right.norm()
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Hidden NRMSE/cosine requires nonzero states")
    cosine = torch.dot(left, right) / (left_norm * right_norm)
    return {
        "hidden_nrmse": float(delta.norm() / right_norm),
        "hidden_cosine": float(cosine.clamp(-1, 1)),
        "hidden_delta_rms": float(delta.square().mean().sqrt()),
        "hf_hidden_rms": float(right.square().mean().sqrt()),
        "hidden_mean_signed_error": float(delta.mean()),
    }


def logits_metrics(left, right):
    import torch

    validate_pair(left, right, VOCAB_SIZE)
    delta = (left.double() - right.double()).abs().flatten()
    index = int(delta.argmax())
    reference_range = float(right.max()) - float(right.min())
    result = {"logits_maxabs": float(delta[index]),
              "logits_max_error_token_id": index,
              "hf_logits_range": reference_range,
              "logits_maxabs_over_hf_range": (
                  float(delta[index]) / reference_range
                  if reference_range else None
              )}
    for name, value in (("cann", left), ("hf", right)):
        maximum = float(value.abs().max())
        at_error = float(value.flatten()[index])
        top = value.double().flatten().topk(2)
        result.update({
            f"{name}_logits_dtype": str(value.dtype),
            f"{name}_logits_bf16_representable": torch.equal(
                value, value.to(torch.bfloat16).to(value.dtype),
            ),
            f"{name}_logits_max_magnitude": maximum,
            f"{name}_bf16_spacing_at_max_magnitude": bf16_spacing(maximum),
            f"{name}_logit_at_max_error": at_error,
            f"{name}_bf16_spacing_at_max_error": bf16_spacing(at_error),
            f"{name}_top1_token": int(top.indices[0]),
            f"{name}_top1_margin": float(top.values[0] - top.values[1]),
        })
    result["logits_top1_equal"] = (
        result["cann_top1_token"] == result["hf_top1_token"]
    )
    return result


def selection(value, length):
    import torch

    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError("Attention indices must be INT32 or INT64")
    values = value.flatten().tolist()
    if any(index < -1 or index >= length for index in values):
        raise ValueError("Attention indices exceed the causal prefix")
    valid = [index for index in values if index >= 0]
    return set(valid), len(valid) - len(set(valid))


def choice_metrics(actual, expected, layer_count):
    import torch

    length = expected["position"] + 1
    if (len(expected["index_scores"]) != layer_count
            or len(expected["attention_weights"]) != layer_count):
        raise ValueError("Incomplete HF index scores or attention weights")
    origin, cutoff, next_score, margin = None, None, None, None
    rows = []
    for layer in range(layer_count):
        scores = expected["index_scores"][layer]
        if scores is not None:
            if scores.shape != (length,) or torch.isnan(scores).any():
                raise ValueError("HF scores must cover the causal prefix")
            origin, cutoff, next_score, margin = layer, None, None, None
            if length > 2048:
                ordered = scores.double().sort(descending=True).values
                if torch.isfinite(ordered[2047:2049]).all():
                    cutoff, next_score = map(float, ordered[2047:2049])
                    margin = cutoff - next_score
        if origin is None:
            raise ValueError("Shared indices have no preceding indexer scores")
        cann_ids, cann_duplicates = selection(actual["indices"][layer], length)
        hf_ids, hf_duplicates = selection(expected["indices"][layer], length)
        weights = expected["attention_weights"][layer]
        if (weights.shape != (64, length) or weights.dtype != torch.bfloat16
                or not torch.isfinite(weights).all()):
            raise ValueError("Expected HF BF16 attention weights [64,P]")
        hf_only = sorted(hf_ids - cann_ids)
        mass = weights[:, hf_only].double().sum(dim=-1)
        rows.append({
            "top2048_overlap": len(cann_ids & hf_ids),
            "cann_selected_count": len(cann_ids),
            "hf_selected_count": len(hf_ids),
            "cann_duplicate_indices": cann_duplicates,
            "hf_duplicate_indices": hf_duplicates,
            "hf_index_score_origin_layer": origin,
            "hf_index_cutoff_score": cutoff,
            "hf_index_next_score": next_score,
            "hf_index_cutoff_margin": margin,
            "hf_only_selected_count": len(hf_only),
            "hf_only_attention_mass_mean": float(mass.mean()),
            "hf_only_attention_mass_max": float(mass.max()),
            "cann_only_attention_mass_mean": None,
            "cann_only_attention_mass_max": None,
        })
    return rows


def collect(native, reference):
    import torch

    report_path = native / "rank-0.json"
    manifest_path = reference / "manifest.json"
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    checks = manifest.get("checks", [])
    if (manifest.get("status") != "pass"
            or manifest.get("numerical_execution") is not True
            or not checks or any(item.get("passed") is not True
                                 for item in checks)):
        raise ValueError("HF execution and reset checks must complete")
    if (report["checkpoint_sha256"] != manifest["metadata_sha256"]
            or report["inputs"]["prefix_sha256"]
            != manifest["prefix_files_sha256"]["rank-0.pt"]
            or report["initial_position"] != manifest["initial_position"]):
        raise ValueError("CANN and HF must use the same checkpoint and prefix")
    layer_count = manifest.get("layer_count", LAYERS)
    if type(layer_count) is not int or not 1 <= layer_count <= LAYERS:
        raise ValueError("HF layer_count must be within 1..78")
    names = sorted(manifest["trace_sha256"])
    steps = len(names)
    teacher = manifest["teacher_tokens"]
    if (steps == 0 or steps > report["steps"] or len(teacher) != steps
            or teacher != report["consumed_tokens"][:steps]):
        raise ValueError("CANN and HF teacher inputs differ")
    initial = report["initial_position"]
    if names != [f"step-{initial + step:05d}.pt" for step in range(steps)]:
        raise ValueError("HF trace hashes must cover consecutive positions")
    if manifest.get("cpu_matmul", {}).get("mkldnn_enabled") is not False:
        raise ValueError("HF manifest must record mkldnn_enabled=false")
    artifacts = {str(path): digest(path)
                 for path in (report_path, manifest_path)}
    rows = []
    for step in range(steps):
        position = report["initial_position"] + step
        name = f"step-{position:05d}.pt"
        traces = []
        for directory, hashes in (
            (native / "rank-0", report["trace_sha256"]),
            (reference, manifest["trace_sha256"]),
        ):
            path = directory / name
            artifacts[str(path)] = digest(path)
            if artifacts[str(path)] != hashes[name]:
                raise ValueError(f"Frozen trace hash differs: {path}")
            traces.append(torch.load(path, map_location="cpu",
                                     weights_only=True))
        actual, expected = traces
        for trace, count in ((actual, LAYERS), (expected, layer_count)):
            if (trace["token"] != teacher[step]
                    or trace["position"] != position
                    or trace["committed_position"] != position + 1
                    or len(trace["hidden_states"]) != count + 1
                    or len(trace["layer_logits"]) != count
                    or len(trace["indices"]) != count):
                raise ValueError("Trace input, position or count differs")
            if count == LAYERS and not torch.equal(
                trace["logits"], trace["layer_logits"][-1],
            ):
                raise ValueError("Final layer logits differ from saved logits")
        choices = choice_metrics(actual, expected, layer_count)
        for layer in range(layer_count):
            partial_last = layer_count < LAYERS and layer == layer_count - 1
            hf_hidden = (expected["last_merged_hidden"] if partial_last
                         else expected["hidden_states"][layer + 1])
            rows.append({
                "decode_position": position, "consumed_token": actual["token"],
                "layer": layer, "hidden_state_index": layer + 1,
                "boundary": ("merged_residual" if layer < 77
                             else "layer77_and_final_rmsnorm"),
                "hf_hidden_field": ("last_merged_hidden" if partial_last
                                    else f"hidden_states.{layer + 1}"),
                **hidden_metrics(actual["hidden_states"][layer + 1],
                                 hf_hidden),
                **logits_metrics(actual["layer_logits"][layer],
                                 expected["layer_logits"][layer]),
                **choices[layer],
            })
    return rows, artifacts, report, manifest


def draw(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False})
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                               layout="constrained")
    positions = sorted({row["decode_position"] for row in rows})
    last_layer = max(row["layer"] for row in rows)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    panels = (("hidden_nrmse", 100, "Hidden NRMSE (%)"),
              ("hidden_cosine", 1, "Hidden cosine similarity"),
              ("logits_maxabs", 1, "Logits max absolute difference"))
    for number, position in enumerate(positions):
        color = colors[number % len(colors)]
        series = [r for r in rows if r["decode_position"] == position]
        raw = [row for row in series if row["layer"] < 77]
        label = f"Position {position}, token {series[0]['consumed_token']}"
        for axis, (field, factor, ylabel) in zip(axes, panels):
            axis.plot([r["layer"] for r in raw],
                      [factor * r[field] for r in raw], color=color,
                      linewidth=1.6, label=label)
            if last_layer == 77:
                axis.scatter([77], [factor * series[-1][field]], s=40,
                             marker="D", facecolors="white",
                             edgecolors=color, zorder=4)
            axis.set(ylabel=ylabel, xlim=(-0.5, last_layer + 0.5))
            axis.grid(alpha=0.2)
    axes[0].set_ylim(bottom=0)
    axes[2].set_ylim(bottom=0)
    axes[0].legend(loc="upper left", frameon=False)
    ticks = (list(range(last_layer + 1)) if last_layer < 8
             else sorted({0, *range(8, last_layer, 8), last_layer}))
    axes[-1].set(xlabel="Decoder layer (zero based)", xticks=ticks)
    figure.suptitle("GLM-5.2 W4A8C8 | A3 TP16 | CANN / independent HF",
                    fontsize=13)
    figure.supxlabel(
        "Hidden: layers 0-76 are merged residuals; layer 77, when present, "
        "includes final RMSNorm (diamonds).\nAll logits include norm and "
        "head; these differences are descriptive.",
        fontsize=9,
    )
    figure.savefig(output / "layer-curves.png", dpi=180)
    figure.savefig(output / "layer-curves.svg", metadata={"Date": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch

    torch.set_num_threads(2)
    rows, artifacts, native, reference = collect(
        args.native_run, args.reference,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "layer-curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draw(rows, args.output)
    result = {
        "scope": "Layerwise differences with separate CANN/HF state histories",
        "layer_count": reference.get("layer_count", LAYERS),
        "decode_positions": sorted({row["decode_position"] for row in rows}),
        "comparison_kind": "cross_configuration", "representative_rank": 0,
        "formulas": {
            "hidden_nrmse": "norm(CANN - HF, 2) / norm(HF, 2)",
            "hidden_cosine": "dot(CANN, HF) / (norm(CANN, 2) * norm(HF, 2))",
            "logits_maxabs": "max(abs(CANN_logits - HF_logits))",
            "evaluation_dtype": "float64",
            "bf16_spacing": (
                "2^max(floor(log2(abs(x))) - 7, -133); zero: 2^-133"
            ),
        },
        "choice_diagnostics": {
            "top2048_overlap": "Intersection of saved causal index sets",
            "cutoff_margin": (
                "HF scores sorted descending: score[2047] - score[2048]; "
                "shared layers use the preceding indexer's scores"
            ),
            "hf_only_attention_mass": (
                "Sum of actual HF expanded-KV BF16 attention probabilities "
                "on HF-selected slots absent from the CANN selection; "
                "mean and maximum over 64 heads"
            ),
            "cann_attention_mass": "Not captured; reported as null",
        },
        "elementwise_ulp_count": False, "acceptance_evaluated": False,
        "performance_measured": False,
        "hf_reference": {
            "directory": str(args.reference.resolve()),
            "implementation": reference["implementation"],
            "mla_preprocess": reference.get("mla_preprocess"),
            "cpu_matmul": reference["cpu_matmul"],
            "partial_layer_logits": reference.get("partial_layer_logits"),
            "source_sha256": reference["source_sha256"],
        },
        "native_source_sha256": native["source_sha256"],
        "checkpoint_sha256": native["checkpoint_sha256"],
        "inputs": native["inputs"], "artifact_sha256": artifacts,
        "source": {"path": str(Path(__file__).resolve()),
                   "sha256": digest(__file__)},
        "command": [sys.executable, *sys.argv], "rows": rows,
    }
    result["npu_initialized"] = (
        hasattr(torch, "npu") and torch.npu.is_initialized()
    )
    if result["npu_initialized"]:
        raise RuntimeError("Plotting must leave NPU uninitialized")
    for name in ("layer-curves.csv", "layer-curves.png", "layer-curves.svg"):
        result["artifact_sha256"][str(args.output / name)] = digest(
            args.output / name,
        )
    (args.output / "layer-curves.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
