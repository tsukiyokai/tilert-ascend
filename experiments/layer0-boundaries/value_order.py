"""CPU Value-order intervention; call after compare.load_inputs admission."""

import hashlib
import inspect
from pathlib import Path


def compare_value_order(
    weights,
    hf,
    captures,
    fixture_path,
    *,
    identity,
    rms_norm_class,
    dense,
    comparison,
    tp_sum,
):
    """Return JSON metrics using unchanged admitted HF arithmetic helpers.

    ``weights, hf, captures, identity`` must come from the existing
    ``experiments/layer0-boundaries/compare.py::load_inputs``. Run in the
    same frozen HF CPU environment. No tensor is moved to an accelerator.
    Both graphs keep all 64 frozen HF probability rows. A full 64-head
    batch preserves the HF eager PV call shape; rank0 also repeats the
    original four-head fixture calls and checks their equality.
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

    def cpu_bf16(value, shape, name):
        require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.bfloat16
            and tuple(value.shape) == tuple(shape)
            and bool(torch.isfinite(value).all()),
            "Invalid finite CPU BF16 tensor: " + name,
        )

    require(
        not torch.backends.mkldnn.enabled, "Existing HF requires mkldnn=False"
    )
    require(len(captures) == 16, "Sixteen admitted rank captures required")
    require(hf["position"] == 8192, "Expected layer0, first 8K decode row")
    require(
        identity["checkpoint_metadata_sha256"]
        == weights.contract["metadata_sha256"],
        "Checkpoint metadata differs from admission",
    )
    fixture_path = Path(fixture_path)
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    frozen, tensors = fixture["identity"], fixture["tensors"]
    require(
        frozen["rank"] == frozen["layer"] == 0
        and frozen["position"] == 8192
        and frozen["hf_sha256"] == identity["hf_step_sha256"]
        and frozen["prefix_sha256"] == captures[0]["inputs"]["prefix_sha256"]
        and frozen["cpu_mkldnn_enabled"] is False
        and frozen["prefix_and_current_projected_separately"] is True
        and frozen["probabilities_recomputed"] is False,
        "Value fixture identity or arithmetic contract differs",
    )
    latent = tensors["latent"]
    probability = hf["attention_weights"][0].unsqueeze(1).contiguous()
    cpu_bf16(latent, (8193, 512), "fixture latent")
    cpu_bf16(probability, (64, 1, 8193), "frozen HF probability")
    cpu_bf16(hf["hidden_states"][0], (1, 6144), "HF h0")
    cpu_bf16(hf["hidden_states"][1], (1, 6144), "HF h1")
    current = hf["cache_rows"][0]["latent"].reshape(1, 512)
    require(
        torch.equal(latent[-1:], current),
        "Fixture current latent differs from HF",
    )
    require(
        torch.equal(probability[:4], tensors["probability"]),
        "Rank0 fixture probabilities differ from the full HF frame",
    )
    for rank, capture in enumerate(captures):
        require(capture["rank"] == rank, "Capture TP order differs")
        cpu_bf16(
            capture["boundaries"]["o_proj.input"],
            (1, 1024),
            "CANN Oproj input",
        )
        require(
            torch.equal(capture["cache_rows"]["latent"], current),
            f"Rank {rank} actual new latent differs from HF",
        )
        require(
            torch.equal(capture["input_hidden"], hf["hidden_states"][0]),
            f"Rank {rank} h0 differs from HF",
        )

    sources = {
        str(path): file_sha(path)
        for path in {
            Path(__file__).resolve(),
            Path(inspect.getsourcefile(type(weights))).resolve(),
            Path(inspect.getsourcefile(rms_norm_class)).resolve(),
            *(
                Path(inspect.getsourcefile(value)).resolve()
                for value in (dense, comparison, tp_sum)
            ),
        }
    }
    name = "model.layers.0.self_attn.kv_b_proj"
    require(
        weights.quantization.get(name + ".weight", "FLOAT") == "FLOAT",
        "The frozen KV-b projection must be unquantized",
    )
    values = torch.empty((64, 8193, 256), dtype=torch.bfloat16, device="cpu")
    value_weights = torch.empty(
        (64, 512, 256), dtype=torch.bfloat16, device="cpu"
    )
    rank_metadata = []
    for rank in range(16):
        rows = (rank * 1792, (rank + 1) * 1792)
        weight = weights.tensor(
            name + ".weight", rows=rows, dtype=torch.bfloat16
        )
        cpu_bf16(weight, (1792, 512), f"rank{rank} KV-b weight")
        prefix = weights._floating_product(name, latent[:8192], rows=rows)
        new = weights._floating_product(name, latent[8192:], rows=rows)
        expanded = torch.cat((prefix, new)).reshape(8193, 4, 448)
        head = slice(rank * 4, (rank + 1) * 4)
        values[head] = expanded[:, :, 192:].permute(1, 0, 2)
        value_weights[head] = weight.reshape(4, 448, 512)[:, 192:].transpose(
            1, 2
        )
        rank_metadata.append(
            {
                "rank": rank,
                "kv_b_rows": list(rows),
                "weight_sha256": tensor_sha(weight),
                "value_sha256": tensor_sha(values[head]),
                "value_weight_sha256": tensor_sha(value_weights[head]),
                "probability_sha256": tensor_sha(probability[head]),
            }
        )
    cpu_bf16(values, (64, 8193, 256), "rebuilt expanded Value")
    cpu_bf16(value_weights, (64, 512, 256), "Value weight")
    # Same 1x64-head batch as HF eager_attention_forward, with BF16 outputs.
    expanded_output = torch.matmul(
        probability.unsqueeze(0), values.unsqueeze(0)
    )[0]
    latent_state = torch.matmul(probability.unsqueeze(0), latent)
    latent_output = torch.matmul(latent_state, value_weights.unsqueeze(0))[0]
    rank0_expanded = torch.matmul(probability[:4], values[:4])
    rank0_latent = torch.matmul(
        torch.matmul(probability[:4], latent), value_weights[:4]
    )
    rank0 = {
        "probability": comparison(probability[:4], tensors["probability"]),
        "value": comparison(values[:4], tensors["value"]),
        "value_weight": comparison(value_weights[:4], tensors["value_weight"]),
        "expanded_four_heads": comparison(
            rank0_expanded, tensors["cpu_expanded"]
        ),
        "latent_four_heads": comparison(rank0_latent, tensors["cpu_latent"]),
        "expanded_full_batch": comparison(
            expanded_output[:4], tensors["cpu_expanded"]
        ),
        "latent_full_batch": comparison(
            latent_output[:4], tensors["cpu_latent"]
        ),
    }
    require(
        rank_metadata[0]["weight_sha256"] == frozen["weight_tensor_sha256"],
        "Rank0 KV-b weight differs from the original fixture",
    )
    report = {
        "status": "rank0_fixture_mismatch",
        "scope": "CPU layer0 fixed-HF-probability Value reassociation through Oproj and HF tail",
        "precision_verified": False,
        "formal_precision_gate_passed": False,
        "performance_measured": False,
        "probabilities_recomputed": False,
        "qk_or_probability_equivalence_proven": False,
        "identity": identity,
        "source_sha256": sources,
        "fixture": {
            "path": str(fixture_path),
            "sha256": file_sha(fixture_path),
            "identity": frozen,
            "latent_sha256": tensor_sha(latent),
        },
        "cpu": {
            "torch": torch.__version__,
            "mkldnn_enabled": False,
            "threads": torch.get_num_threads(),
        },
        "rank0_fixture": rank0,
        "ranks": rank_metadata,
        "formulas": {
            "expanded": "R(P @ R(C @ U_V))",
            "latent": "R(R(P @ C) @ U_V)",
            "R": "BF16 output boundary; unchanged Torch CPU matmul contract",
        },
    }
    if not all(item["exact"] for item in rank0.values()):
        return report

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
    branches = {}
    for label, output in (
        ("expanded", expanded_output),
        ("latent", latent_output),
    ):
        cpu_bf16(output, (64, 1, 256), label + " Value output")
        joined = output.transpose(0, 1).reshape(1, 16384).contiguous()
        partials = weights.input_partials(
            "model.layers.0.self_attn.o_proj", joined
        )
        attention = tp_sum(partials)
        retained = hf["hidden_states"][0] + attention
        normalized = norm(retained)
        mlp, _ = dense(weights, normalized)
        h1 = retained + mlp
        branches[label] = {"attention": attention, "h1": h1}
        report[label] = {
            "o_proj_input_sha256": tensor_sha(joined),
            "attention_sha256": tensor_sha(attention),
            "h1_sha256": tensor_sha(h1),
            "h1_vs_hf": comparison(h1, hf["hidden_states"][1]),
            "ranks": [
                {
                    "rank": rank,
                    "o_proj_input_vs_cann": comparison(
                        joined[:, rank * 1024 : (rank + 1) * 1024],
                        capture["boundaries"]["o_proj.input"],
                    ),
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
    report["latent_vs_expanded"] = {
        field: comparison(
            branches["latent"][field], branches["expanded"][field]
        )
        for field in ("attention", "h1")
    }
    report["expanded_reconstructs_hf_h1_exactly"] = report["expanded"][
        "h1_vs_hf"
    ]["exact"]
    report["source_stable"] = all(
        file_sha(path) == sha for path, sha in sources.items()
    )
    require(
        report["source_stable"], "Executed source changed during comparison"
    )
    report["status"] = (
        "diagnostic_complete"
        if report["expanded_reconstructs_hf_h1_exactly"]
        else "expanded_h1_not_reconstructed"
    )
    return report
