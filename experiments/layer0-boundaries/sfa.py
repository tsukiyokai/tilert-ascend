"""CPU diagnostics for the admitted layer0 M1 SFA boundary capture."""

import hashlib
import inspect
from pathlib import Path
from statistics import median


def compare_sfa(
    weights,
    hf,
    captures,
    *,
    identity,
    rms_norm_class,
    dense,
    comparison,
    tp_sum,
    attention_reference,
    value_reference,
):
    """Return local closure/residual metrics, never a precision receipt.

    Use the existing compare.load_inputs outputs and original check_mla
    helpers in the frozen HF CPU environment, with inference_mode and
    mkldnn=False. Candidate probabilities are global CPU softmax results;
    neither candidate emulates CANN FlashAttention internals.
    """
    import torch

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def file_sha(path):
        result = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(chunk)
        return result.hexdigest()

    def tensor_sha(value):
        raw = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    def tensor(value, shape, dtype, name, *, finite=True):
        require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and tuple(value.shape) == tuple(shape)
            and value.dtype == dtype,
            "Invalid CPU tensor: " + name,
        )
        if finite:
            require(
                bool(torch.isfinite(value).all()), "Nonfinite tensor: " + name
            )

    require(
        not torch.backends.mkldnn.enabled, "Frozen HF requires mkldnn=False"
    )
    require(
        len(captures) == len(identity["captures"]) == 16,
        "TP16 admission required",
    )
    require(
        hf["position"] == 8192 and hf["committed_position"] == 8193,
        "Expected the first full-prefix M1 HF frame",
    )
    require(
        identity["checkpoint_metadata_sha256"]
        == weights.contract["metadata_sha256"],
        "Checkpoint metadata differs from admission",
    )
    require(
        weights.quantization.get(
            "model.layers.0.self_attn.kv_b_proj.weight", "FLOAT"
        )
        == "FLOAT",
        "The frozen KV-b projection must be unquantized",
    )
    for index in (0, 1):
        tensor(
            hf["hidden_states"][index],
            (1, 6144),
            torch.bfloat16,
            f"HF h{index}",
        )
    hf_probability = hf["attention_weights"][0]
    tensor(hf_probability, (64, 8193), torch.bfloat16, "HF probability")
    sources = {
        str(path): file_sha(path)
        for path in {
            Path(__file__).resolve(),
            Path(inspect.getsourcefile(type(weights))).resolve(),
            Path(inspect.getsourcefile(rms_norm_class)).resolve(),
            *(
                Path(inspect.getsourcefile(value)).resolve()
                for value in (
                    dense,
                    comparison,
                    tp_sum,
                    attention_reference,
                    value_reference,
                )
            ),
        }
    }
    expected = {
        "query_nope": ((1, 4, 512), torch.bfloat16),
        "query_rope": ((1, 4, 64), torch.bfloat16),
        "key_cache": ((73, 128, 1, 512), torch.bfloat16),
        "rope_cache": ((73, 128, 1, 64), torch.bfloat16),
        "indices": ((1, 1, 2048), torch.int32),
        "block_table": ((1, 73), torch.int32),
        "query_lengths": ((1,), torch.int32),
        "key_lengths": ((1,), torch.int32),
        "output": ((1, 4, 512), torch.bfloat16),
    }
    rank_inputs, actual_uv, rank_reports = [], [], []
    logical_position = torch.arange(8193, dtype=torch.int64, device="cpu")
    for rank, capture in enumerate(captures):
        require(
            capture["rank"] == identity["captures"][rank]["rank"] == rank,
            "Capture TP order differs",
        )
        b = capture["boundaries"]
        metadata = capture["boundary_metadata"]["sfa"]
        require(
            metadata["scale"] == 0.0625
            and metadata["source_path"]
            == "/assets/vllm-ascend-unified/vllm_ascend/attention/sfa_v1.py"
            and metadata["source_sha256"]
            == "b5ad39a05b1323ac5048962f137969158298e4f637b5bd3e401cb4c4248ecfa0",
            f"Rank {rank} frozen SFA source/scale differs",
        )
        for name, (shape, dtype) in expected.items():
            tensor(
                b["sfa." + name],
                shape,
                dtype,
                f"rank{rank}.sfa.{name}",
                finite=name not in ("key_cache", "rope_cache"),
            )
        require(
            b["sfa.query_lengths"].tolist() == [1]
            and b["sfa.key_lengths"].tolist() == [8193],
            f"Rank {rank} sequence lengths differ",
        )
        indices = b["sfa.indices"].reshape(-1).to(torch.int64)
        require(
            indices.unique().numel() == 2048
            and bool(torch.all(indices >= 0))
            and bool(torch.all(indices < 8193)),
            f"Rank {rank} sparse indices are duplicated/invalid",
        )
        table = b["sfa.block_table"][0]
        active_pages = table[:65].to(torch.int64)
        require(
            bool(torch.all((active_pages >= 0) & (active_pages < 73)))
            and active_pages.unique().numel() == 65,
            f"Rank {rank} active physical pages are invalid/aliased",
        )
        physical_page = active_pages[logical_position // 128]
        offset = logical_position % 128
        # Read exactly 8193 logical rows; unused physical cache tails are ignored.
        latent = b["sfa.key_cache"][physical_page, offset, 0]
        rope = b["sfa.rope_cache"][physical_page, offset, 0]
        tensor(latent, (8193, 512), torch.bfloat16, "logical latent")
        tensor(rope, (8193, 64), torch.bfloat16, "logical RoPE")
        require(
            torch.equal(capture["input_hidden"], hf["hidden_states"][0]),
            f"Rank {rank} h0 differs from HF",
        )
        weight = weights.tensor(
            "model.layers.0.self_attn.kv_b_proj.weight",
            rows=(rank * 1792, (rank + 1) * 1792),
            dtype=torch.bfloat16,
        )
        tensor(weight, (1792, 512), torch.bfloat16, "KV-b weight")
        uv = value_reference(b["sfa.output"], weight).to(torch.bfloat16)
        actual_uv.append(uv)
        rank_inputs.append((latent, rope, indices, weight))
        rank_reports.append(
            {
                "rank": rank,
                "metadata": metadata,
                "index_count": indices.numel(),
                "index_min": int(indices.min()),
                "index_max": int(indices.max()),
                "index_order_preserved": True,
                "block_table_is_identity": torch.equal(
                    table, torch.arange(73, dtype=torch.int32)
                ),
                "sha256": {
                    name: tensor_sha(b["sfa." + name])
                    for name in (
                        "query_nope",
                        "query_rope",
                        "indices",
                        "block_table",
                        "output",
                    )
                },
                "logical_latent_sha256": tensor_sha(latent),
                "logical_rope_sha256": tensor_sha(rope),
                "weight_sha256": tensor_sha(weight),
                "current_latent_vs_hf": comparison(
                    latent[-1:], hf["cache_rows"][0]["latent"].reshape(1, 512)
                ),
                "current_rope_vs_hf": comparison(
                    rope[-1:], hf["cache_rows"][0]["rope"].reshape(1, 64)
                ),
                "actual_sfa_uv_vs_cann": comparison(uv, b["o_proj.input"]),
            }
        )

    with torch.device("cpu"):
        norm = rms_norm_class(
            6144, eps=identity["hf_rms_norm_eps"]["ordinary"]
        )
    norm.weight = torch.nn.Parameter(
        weights.tensor(
            "model.layers.0.post_attention_layernorm.weight",
            dtype=torch.bfloat16,
        ),
        requires_grad=False,
    )

    def tail(local_outputs):
        joined = torch.cat(local_outputs, dim=-1)
        partials = weights.input_partials(
            "model.layers.0.self_attn.o_proj", joined
        )
        attention = tp_sum(partials)
        retained = hf["hidden_states"][0] + attention
        normalized = norm(retained)
        mlp, _ = dense(weights, normalized)
        h1 = retained + mlp
        return {
            "o_proj_input_sha256": tensor_sha(joined),
            "attention_sha256": tensor_sha(attention),
            "h1_sha256": tensor_sha(h1),
            "h1_vs_hf": comparison(h1, hf["hidden_states"][1]),
            "ranks": [
                {
                    "rank": rank,
                    "o_proj_partial_vs_cann": comparison(
                        partials[rank].float(),
                        capture["boundaries"]["o_proj.tp_input"],
                    ),
                    "attention_vs_cann": comparison(
                        attention, capture["boundaries"]["attention.output"]
                    ),
                    "normalized_vs_cann": comparison(
                        normalized,
                        capture["boundaries"]["post_norm.normalized"],
                    ),
                    "h1_vs_cann": comparison(h1, capture["merged"]),
                }
                for rank, capture in enumerate(captures)
            ],
        }

    actual_tail = tail(actual_uv)
    report = {
        "status": "actual_sfa_tail_not_reconstructed",
        "identity": identity,
        "scope": "CPU layer0 actual SFA boundary closure and two explicit arithmetic candidates",
        "precision_verified": False,
        "formal_precision_gate_passed": False,
        "performance_measured": False,
        "cann_internal_emulation_claimed": False,
        "source_sha256": sources,
        "ranks": rank_reports,
        "cpu": {
            "torch": torch.__version__,
            "mkldnn_enabled": False,
            "threads": torch.get_num_threads(),
        },
        "actual_sfa_to_tail": actual_tail,
        "actual_sfa_tail_reconstructed": all(
            row["h1_vs_cann"]["exact"] and row["attention_vs_cann"]["exact"]
            for row in actual_tail["ranks"]
        ),
        "formulas": {
            "scores": "FP32(QL @ C_selected.T + QPE @ KPE_selected.T) * scale",
            "fp32_probability": "BF16(softmax(scores)_FP32 @ C_selected_FP32)",
            "bf16_probability": "BF16(BF16(softmax(scores)).float() @ C_selected_FP32)",
            "bf16_score_probability": "BF16(BF16(softmax(BF16(scores).float())).float() @ C_selected_FP32)",
        },
    }
    outputs = {
        "fp32_probability": [],
        "bf16_probability": [],
        "bf16_score_probability": [],
    }
    expanded_outputs = {"hf_probability": [], "bf16_score_probability": []}
    for rank, (capture, inputs) in enumerate(zip(captures, rank_inputs)):
        latent, rope, indices, weight = inputs
        b = capture["boundaries"]
        keys, key_rope = latent[indices].float(), rope[indices].float()
        scores = b["sfa.query_nope"][0].float() @ keys.T
        scores = scores + b["sfa.query_rope"][0].float() @ key_rope.T
        scaled_scores = scores * 0.0625
        probability = torch.softmax(scaled_scores, dim=-1)
        context = attention_reference(
            b["sfa.query_nope"],
            b["sfa.query_rope"],
            latent,
            rope,
            indices,
            0.0625,
        )
        require(
            torch.equal(context, (probability @ keys).unsqueeze(0)),
            "Probability diagnostic differs from original attention_reference",
        )
        hf_local = hf_probability[rank * 4 : (rank + 1) * 4]
        outside = torch.ones(8193, dtype=torch.bool, device="cpu")
        outside[indices] = False
        rank_reports[rank]["hf_probability_mass_outside_selection"] = (
            hf_local[:, outside].float().sum(-1).tolist()
        )
        for label, local_outputs in outputs.items():
            if label == "fp32_probability":
                candidate_probability = probability
            elif label == "bf16_probability":
                candidate_probability = probability.to(torch.bfloat16)
            else:
                candidate_probability = torch.softmax(
                    scaled_scores.to(torch.bfloat16).float(), dim=-1
                ).to(torch.bfloat16)
            candidate_context = (
                context.to(torch.bfloat16)
                if label == "fp32_probability"
                else (candidate_probability.float() @ keys)
                .unsqueeze(0)
                .to(torch.bfloat16)
            )
            aligned = torch.zeros(
                (4, 8193), dtype=candidate_probability.dtype, device="cpu"
            )
            aligned[:, indices] = candidate_probability
            uv = value_reference(candidate_context, weight).to(torch.bfloat16)
            local_outputs.append(uv)
            rank_reports[rank][label] = {
                "context_sha256": tensor_sha(candidate_context),
                "uv_sha256": tensor_sha(uv),
                "probability_sha256": tensor_sha(candidate_probability),
                "context_vs_cann": comparison(
                    candidate_context, b["sfa.output"]
                ),
                "uv_vs_cann": comparison(uv, b["o_proj.input"]),
                "uv_vs_actual_sfa_uv": comparison(uv, actual_uv[rank]),
                "probability_vs_hf_logical_positions": comparison(
                    aligned, hf_local.to(aligned.dtype)
                ),
                "probability_sum": candidate_probability.float()
                .sum(-1)
                .tolist(),
            }
            if label == "bf16_score_probability":
                rows = (rank * 1792, (rank + 1) * 1792)
                prefix = weights._floating_product(
                    "model.layers.0.self_attn.kv_b_proj",
                    latent[:8192],
                    rows=rows,
                )
                current = weights._floating_product(
                    "model.layers.0.self_attn.kv_b_proj",
                    latent[8192:],
                    rows=rows,
                )
                expanded = torch.cat((prefix, current)).reshape(8193, 4, 448)
                values = expanded[:, :, 192:].permute(1, 0, 2).contiguous()
                for arm, probabilities in (
                    ("hf_probability", hf_local),
                    ("bf16_score_probability", aligned),
                ):
                    expanded_outputs[arm].append(
                        torch.matmul(
                            probabilities.unsqueeze(1), values
                        ).reshape(1, 1024)
                    )
    report["candidates"] = {
        label: tail(local) for label, local in outputs.items()
    }
    errors = {
        label: [
            row[label]["probability_vs_hf_logical_positions"]["nrmse"]
            for row in rank_reports
        ]
        for label in ("bf16_probability", "bf16_score_probability")
    }
    baseline, candidate = errors.values()
    report["score_rounding_hypothesis"] = {
        "prediction": "BF16 scaled scores reduce max probability NRMSE below 0.01 and median by at least half",
        "controls": "Actual QL/QPE/KV, index order, scale, BF16 probability output and Value computation fixed",
        "baseline_max_nrmse": max(baseline),
        "candidate_max_nrmse": max(candidate),
        "baseline_median_nrmse": median(baseline),
        "candidate_median_nrmse": median(candidate),
        "prediction_met": (
            max(candidate) < 0.01 and median(candidate) <= median(baseline) / 2
        ),
    }
    report["expanded_value"] = {
        label: tail(local) for label, local in expanded_outputs.items()
    }
    require(
        report["expanded_value"]["hf_probability"]["h1_vs_hf"]["exact"],
        "Frozen HF probabilities and rebuilt Value did not reconstruct HF h1",
    )
    error = report["expanded_value"]["bf16_score_probability"]["h1_vs_hf"]
    report["score_and_value_hypothesis"] = {
        "prediction": "BF16 score/probability with expanded Value gives h1 NRMSE below 0.001",
        "h1_nrmse": error["nrmse"],
        "prediction_met": error["nrmse"] < 0.001,
    }
    report["logical_cache_agreement"] = {
        "latent": all(
            torch.equal(rank_inputs[0][0], item[0]) for item in rank_inputs
        ),
        "rope": all(
            torch.equal(rank_inputs[0][1], item[1]) for item in rank_inputs
        ),
    }
    report["source_stable"] = all(
        file_sha(path) == sha for path, sha in sources.items()
    )
    require(
        report["source_stable"], "Executed source changed during comparison"
    )
    if report["actual_sfa_tail_reconstructed"]:
        report["status"] = "diagnostic_complete"
    return report
