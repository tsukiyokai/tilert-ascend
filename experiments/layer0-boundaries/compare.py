"""Diagnose frozen layer0 CANN boundaries with the unchanged CPU HF tail."""

import argparse
import hashlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
TP = 16
HIDDEN = 6144


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def tensor(value, shape, dtype, name):
    require(
        isinstance(value, torch.Tensor)
        and value.device.type == "cpu"
        and tuple(value.shape) == tuple(shape)
        and value.dtype == dtype,
        f"Invalid CPU tensor: {name}; expected {shape}, {dtype}",
    )
    return value


def comparison(actual, expected):
    require(
        actual.shape == expected.shape and actual.dtype == expected.dtype,
        "Comparison tensor shape/dtype mismatch",
    )
    return {
        "exact": torch.equal(actual, expected),
        **metrics(actual, expected),
    }


def tensor_digest(value):
    raw = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def source_file(value):
    return Path(inspect.getsourcefile(inspect.unwrap(value))).resolve()


def load_inputs(args, weights, sources):
    manifest_path = args.hf_step.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest.get("status") == "pass"
        and manifest.get("numerical_execution") is True
        and manifest.get("tp_size") == TP,
        "HF generation/reset must be complete",
    )
    require(
        manifest["metadata_sha256"] == weights.contract["metadata_sha256"],
        "HF checkpoint metadata differs",
    )
    require(
        manifest.get("cpu_matmul", {}).get("mkldnn_enabled") is False,
        "Expected the existing mkldnn=False HF contract",
    )
    require(
        manifest["versions"]["torch"] == torch.__version__,
        "CPU Torch version differs from the frozen HF",
    )
    for name, loaded in (
        ("reference.py", ReferenceMlp),
        ("reference_weights.py", ReferenceWeights),
    ):
        relative = "tilert/models/glm_5_2_ascend/" + name
        require(
            manifest["source_sha256"].get(relative)
            == digest(args.source_root / relative),
            "HF arithmetic source differs: " + name,
        )
        require(
            digest(source_file(loaded)) == manifest["source_sha256"][relative],
            "Loaded arithmetic source differs: " + name,
        )
    hf_source = source_file(HfRmsNorm)
    require(
        digest(hf_source) in manifest["installed_hf_source_sha256"].values(),
        "Installed HF RMSNorm source differs from the frozen HF",
    )
    hf_sha = digest(args.hf_step)
    require(
        manifest["trace_sha256"].get(args.hf_step.name) == hf_sha,
        "HF frame hash differs",
    )
    hf = torch.load(args.hf_step, map_location="cpu", weights_only=True)
    require(
        hf["position"] == 8192 and hf["committed_position"] == 8193,
        "Expected the first full-prefix M1 frame",
    )
    for index in (0, 1):
        tensor(
            hf["hidden_states"][index],
            (1, HIDDEN),
            torch.bfloat16,
            f"HF h{index}",
        )
    width = weights.config["intermediate_size"] // TP
    require(
        width * TP == weights.config["intermediate_size"],
        "Dense TP split differs",
    )
    captures, provenance = [], []
    required = {
        **{
            name: ((1, HIDDEN), torch.bfloat16)
            for name in (
                "attention.output",
                "post_norm.input",
                "post_norm.residual_input",
                "post_norm.normalized",
                "post_norm.retained",
                "mlp.down_output",
                "mlp.output",
                "o_proj.output",
                "input_norm",
            )
        },
        "mlp.gate_up": ((1, 2 * width), torch.bfloat16),
        "mlp.down_input": ((1, width), torch.bfloat16),
        "o_proj.input": ((1, 1024), torch.bfloat16),
        **{
            name: ((1, HIDDEN), torch.float32)
            for name in (
                "o_proj.tp_input",
                "o_proj.tp_output",
                "down.tp_input",
                "down.tp_output",
            )
        },
    }
    for rank in range(TP):
        path = args.captures / f"rank-{rank}" / "layer-00.pt"
        receipt_path = args.captures / f"rank-{rank}.json"
        receipt = json.loads(receipt_path.read_text())
        sha = digest(path)
        matching = [
            value
            for key, value in receipt["artifact_sha256"].items()
            if Path(key).name == path.name
            and Path(key).parent.name == path.parent.name
        ]
        require(matching == [sha], f"Rank {rank} artifact hash differs")
        require(
            receipt["rank"] == rank
            and receipt["layers"] == 1
            and receipt["steps"] == 1
            and receipt["query_width"] == 1,
            f"Rank {rank} capture scope differs",
        )
        data = torch.load(path, map_location="cpu", weights_only=True)
        require(
            data["rank"] == rank
            and data["layer"] == 0
            and data["positions"] == [8192]
            and data["teacher_tokens"] == [hf["token"]],
            f"Rank {rank} row identity differs",
        )
        require(
            data["checkpoint_sha256"]
            == receipt["checkpoint_sha256"]
            == weights.contract["metadata_sha256"],
            f"Rank {rank} checkpoint differs",
        )
        require(
            data["inputs"]["prefix_sha256"]
            == manifest["prefix_files_sha256"][f"rank-{rank}.pt"],
            f"Rank {rank} prefix differs from HF",
        )
        for name in ("input_hidden", "hidden", "residual", "merged"):
            tensor(
                data[name], (1, HIDDEN), torch.bfloat16, f"rank{rank}.{name}"
            )
        require(
            data["input_residual"] is None,
            "Layer0 must start without residual",
        )
        for name, (shape, dtype) in required.items():
            tensor(
                data["boundaries"][name], shape, dtype, f"rank{rank}.{name}"
            )
        captures.append(data)
        provenance.append(
            {
                "rank": rank,
                "path": str(path),
                "sha256": sha,
                "receipt_sha256": digest(receipt_path),
                "capture_status": receipt["status"],
                "runtime_boundary_metadata": data["boundary_metadata"],
                "baseline_checks": [
                    item
                    for item in receipt["checks"]
                    if item["name"].startswith("layer0.token0.")
                ],
                "capture_sources": receipt["source_sha256"],
            }
        )
    sources[str(hf_source)] = digest(hf_source)
    return (
        captures,
        hf,
        {
            "hf_step": str(args.hf_step),
            "hf_step_sha256": hf_sha,
            "hf_manifest_sha256": digest(manifest_path),
            "checkpoint_metadata_sha256": weights.contract["metadata_sha256"],
            "hf_mla_preprocess": manifest["mla_preprocess"],
            "hf_rms_norm_eps": manifest["rms_norm_eps"],
            "captures": provenance,
        },
    )


class DenseObservations:
    """Observe the original ReferenceMlp calls without changing their arithmetic."""

    def __init__(self, weights):
        self.weights = weights
        self.projections = {}
        self.down_input = None
        self.partials = None

    def linear(self, prefix, value, partition):
        result = self.weights.linear(prefix, value, partition)
        self.projections[prefix.rsplit(".", 1)[-1]] = result
        return result

    def input_partials(self, prefix, value):
        self.down_input = value
        self.partials = self.weights.input_partials(prefix, value)
        return self.partials


def dense(weights, value):
    observed = DenseObservations(weights)
    result = ReferenceMlp(observed, "model.layers.0.mlp")(value)
    return result, observed


def dense_details(captures, normalized, observed):
    gate = observed.projections["gate_proj"]
    up = observed.projections["up_proj"]
    width = gate.shape[-1] // TP
    rows = []
    for rank, data in enumerate(captures):
        b = data["boundaries"]
        columns = slice(rank * width, (rank + 1) * width)
        rows.append(
            {
                "rank": rank,
                "same_normalized_input": torch.equal(
                    b["post_norm.normalized"], normalized
                ),
                "gate_up": comparison(
                    b["mlp.gate_up"],
                    torch.cat((gate[:, columns], up[:, columns]), dim=-1),
                ),
                "down_input": comparison(
                    b["mlp.down_input"], observed.down_input[:, columns]
                ),
                "down_partial": comparison(
                    b["down.tp_input"], observed.partials[rank].float()
                ),
            }
        )
    return rows


def reduction_details(captures, label):
    partials = tuple(
        data["boundaries"][label + ".tp_input"] for data in captures
    )
    rank_order = tp_sum(partials)
    output_name = "o_proj.output" if label == "o_proj" else "mlp.down_output"
    return {
        "reference": "FP32 rank-order sum of actual partials; not HCCL-tree emulation",
        "input_is_exact_bf16_lift": [
            torch.equal(value, value.bfloat16().float()) for value in partials
        ],
        "ranks": [
            {
                "rank": rank,
                "fp32_vs_rank_order": comparison(
                    data["boundaries"][label + ".tp_output"], rank_order
                ),
                "bf16_vs_rank_order": comparison(
                    data["boundaries"][output_name], rank_order.bfloat16()
                ),
                "output_cast_link": comparison(
                    data["boundaries"][output_name],
                    data["boundaries"][label + ".tp_output"].bfloat16(),
                ),
            }
            for rank, data in enumerate(captures)
        ],
    }


def run(args, report):
    torch.set_num_threads(args.threads)
    weights = ReferenceWeights(args.checkpoint, dense_cache_bytes=1024**3)
    sources = {
        str(path): digest(path)
        for path in (
            Path(__file__).resolve(),
            source_file(ReferenceMlp),
            source_file(ReferenceWeights),
            source_file(metrics),
        )
    }
    captures, hf, identity = load_inputs(args, weights, sources)
    data = captures[args.rank]
    b = data["boundaries"]
    norm_eps = identity["hf_rms_norm_eps"]["ordinary"]
    norm = HfRmsNorm(HIDDEN, eps=norm_eps)
    norm.weight = torch.nn.Parameter(
        weights.tensor(
            "model.layers.0.post_attention_layernorm.weight",
            dtype=torch.bfloat16,
        ),
        requires_grad=False,
    )
    input_norm = HfRmsNorm(HIDDEN, eps=norm_eps)
    input_norm.weight = torch.nn.Parameter(
        weights.tensor(
            "model.layers.0.input_layernorm.weight", dtype=torch.bfloat16
        ),
        requires_grad=False,
    )
    normalized_input = (
        prolog_input_norm(input_norm, hf["hidden_states"][0])
        if identity["hf_mla_preprocess"] == "prolog_v3"
        else input_norm(hf["hidden_states"][0])
    )
    retained_a = hf["hidden_states"][0] + b["attention.output"]
    normalized_a = norm(retained_a)
    dense_a, observed_a = dense(weights, normalized_a)
    merged_a = retained_a + dense_a
    normalized_b, retained_b = (
        b["post_norm.normalized"],
        b["post_norm.retained"],
    )
    reused = torch.equal(normalized_a, normalized_b)
    dense_b, observed_b = (
        (dense_a, observed_a) if reused else dense(weights, normalized_b)
    )
    merged_b = retained_b + dense_b

    def outcomes(normalized, retained, output, merged):
        return {
            "normalized_vs_cann": comparison(
                normalized, b["post_norm.normalized"]
            ),
            "retained_vs_cann": comparison(retained, b["post_norm.retained"]),
            "dense_vs_cann": comparison(output, b["mlp.output"]),
            "merged_vs_cann": comparison(merged, data["merged"]),
            "merged_vs_hf": comparison(merged, hf["hidden_states"][1]),
            "merged_sha256": tensor_digest(merged),
        }

    report.update(
        identity=identity,
        source_sha256=sources,
        selected_rank=args.rank,
        cpu_backend={
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        original_h0_vs_cann=comparison(
            data["input_hidden"], hf["hidden_states"][0]
        ),
        original_h1_vs_cann=comparison(data["merged"], hf["hidden_states"][1]),
        input_norm_vs_hf=comparison(b["input_norm"], normalized_input),
        rank_agreement={
            name: all(
                torch.equal(
                    captures[0]["boundaries"][name], item["boundaries"][name]
                )
                for item in captures
            )
            for name in (
                "input_norm",
                "attention.output",
                "post_norm.normalized",
                "post_norm.retained",
                "mlp.output",
            )
        },
        boundary_links={
            "o_proj_to_attention": comparison(
                b["o_proj.output"], b["attention.output"]
            ),
            "attention_to_norm": comparison(
                b["attention.output"], b["post_norm.input"]
            ),
            "residual_to_h0": comparison(
                b["post_norm.residual_input"], data["input_hidden"]
            ),
            "retained_to_decoder": comparison(
                b["post_norm.retained"], data["residual"]
            ),
            "mlp_to_decoder": comparison(b["mlp.output"], data["hidden"]),
            "down_to_mlp": comparison(b["mlp.down_output"], b["mlp.output"]),
            "merged_from_decoder_parts": comparison(
                data["merged"], data["hidden"] + data["residual"]
            ),
        },
        interventions={
            "actual_attention_to_hf_tail": outcomes(
                normalized_a, retained_a, dense_a, merged_a
            ),
            "actual_post_norm_to_hf_dense": outcomes(
                normalized_b, retained_b, dense_b, merged_b
            ),
            "dense_result_reused_for_identical_input": reused,
        },
        dense_local={
            "actual_attention_to_hf_tail": dense_details(
                captures, normalized_a, observed_a
            ),
            "actual_post_norm_to_hf_dense": dense_details(
                captures, normalized_b, observed_b
            ),
        },
        reductions={
            label: reduction_details(captures, label)
            for label in ("o_proj", "down")
        },
    )
    report["attribution_prerequisites"] = {
        "original_h0_exact": report["original_h0_vs_cann"]["exact"],
        "boundary_links_exact": all(
            item["exact"] for item in report["boundary_links"].values()
        ),
        "rank_boundaries_agree": all(report["rank_agreement"].values()),
        "runtime_normalization_eps_match": all(
            item["runtime_boundary_metadata"][name]["variance_epsilon"]
            == norm_eps
            for item in identity["captures"]
            for name in ("input_norm", "post_norm")
        ),
        "full_capture_reproduced": all(
            item["capture_status"] == "pass"
            and item["baseline_checks"]
            and all(check["passed"] for check in item["baseline_checks"])
            for item in identity["captures"]
        ),
    }
    report["attribution_prerequisites_passed"] = all(
        report["attribution_prerequisites"].values()
    )
    joined_down = torch.cat(
        [item["boundaries"]["mlp.down_input"] for item in captures], dim=-1
    )
    down_partials = weights.input_partials(
        "model.layers.0.mlp.down_proj", joined_down
    )
    report["actual_down_input_to_reference"] = {
        "local_partials": [
            {
                "rank": rank,
                "partial": comparison(
                    item["boundaries"]["down.tp_input"],
                    down_partials[rank].float(),
                ),
            }
            for rank, item in enumerate(captures)
        ],
        "rank_order_vs_actual": comparison(
            tp_sum(down_partials), b["mlp.output"]
        ),
    }
    if args.check_o_proj:
        joined = torch.cat(
            [item["boundaries"]["o_proj.input"] for item in captures], dim=-1
        )
        partials = weights.input_partials(
            "model.layers.0.self_attn.o_proj", joined
        )
        report["o_proj_local"] = [
            {
                "rank": rank,
                "partial": comparison(
                    item["boundaries"]["o_proj.tp_input"],
                    partials[rank].float(),
                ),
            }
            for rank, item in enumerate(captures)
        ]
        report["o_proj_rank_order_vs_actual"] = comparison(
            tp_sum(partials), b["attention.output"]
        )
    if args.sfa_boundaries:
        from check_mla import attention_reference, value_reference
        from sfa import compare_sfa

        report["sfa"] = compare_sfa(
            weights,
            hf,
            captures,
            identity=identity,
            rms_norm_class=HfRmsNorm,
            dense=dense,
            comparison=comparison,
            tp_sum=tp_sum,
            attention_reference=attention_reference,
            value_reference=value_reference,
        )
        require(
            report["sfa"]["status"] == "diagnostic_complete",
            "The actual SFA output did not reconstruct the CANN tail",
        )
    if args.value_fixture:
        from value_order import compare_value_order

        report["value_order"] = compare_value_order(
            weights,
            hf,
            captures,
            args.value_fixture,
            identity=identity,
            rms_norm_class=HfRmsNorm,
            dense=dense,
            comparison=comparison,
            tp_sum=tp_sum,
        )
        require(
            report["value_order"]["status"] == "diagnostic_complete",
            "The fixed-probability Value reference did not close",
        )
    backend = sys.modules.get("torch_npu")
    initialized = bool(backend is not None and backend.npu._initialized)
    require(not initialized, "CPU diagnostic initialized NPU")
    report.update(
        status="diagnostic_complete",
        npu_initialized=False,
        source_stable=all(
            digest(path) == sha for path, sha in sources.items()
        ),
    )
    require(report["source_stable"], "Executed arithmetic source changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "checkpoint", "captures", "hf-step", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(TP), default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--check-o-proj", action="store_true")
    parser.add_argument("--value-fixture", type=Path)
    parser.add_argument("--sfa-boundaries", action="store_true")
    args = parser.parse_args()
    require(args.threads > 0, "Positive CPU thread count required")
    require(not args.output.exists(), "Preserve existing diagnostic output")
    sys.path[:0] = [
        str(args.source_root),
        str(args.source_root / "tests/glm52"),
    ]
    global torch, metrics, ReferenceWeights, ReferenceMlp, HfRmsNorm, tp_sum
    global prolog_input_norm
    import torch
    from check_mla import metrics

    with (
        patch("transformers.utils.is_torch_npu_available", return_value=False),
        patch(
            "transformers.utils.import_utils.is_torch_npu_available",
            return_value=False,
        ),
    ):
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaRMSNorm as HfRmsNorm,
        )

    from tilert.models.glm_5_2_ascend.reference import (
        ReferenceMlp,
        prolog_input_norm,
    )
    from tilert.models.glm_5_2_ascend.reference_weights import (
        ReferenceWeights,
        tp_sum,
    )

    report = {
        "status": "error",
        "scope": "Layer0 CPU tail interventions on frozen M1 inputs",
        "precision_verified": False,
        "formal_precision_gate_passed": False,
        "performance_measured": False,
        "thresholds_modified": False,
        "transformers_npu_backend_disabled": True,
        "command": [sys.executable, *sys.argv],
        "interpretation": "Local comparisons only; unequal rank inputs limit per-rank localization",
    }
    try:
        with (
            torch.inference_mode(),
            torch.backends.mkldnn.flags(
                enabled=False,
                deterministic=None,
                allow_tf32=None,
                fp32_precision=None,
            ),
        ):
            run(args, report)
    except Exception:  # noqa: BLE001 - preserve diagnostic failure in the receipt.
        report.update(status="error", error=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return int(report["status"] != "diagnostic_complete")


if __name__ == "__main__":
    raise SystemExit(main())
