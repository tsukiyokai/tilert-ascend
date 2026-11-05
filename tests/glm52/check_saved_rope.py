"""Compare rotary rounding on frozen native indexer inputs without NPU work."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
import torch.nn.functional as functional
from transformers import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaRotaryEmbedding,
)

from run_native_decode import digest
from tilert.models.glm_5_2_ascend.reference import hadamard_matrix
from tilert.models.glm_5_2_ascend.reference_weights import (
    ReferenceWeights,
    quantize_a8,
)


ROPE_SOURCE = Path(
    "/opt/ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/"
    "ops_transformer/ascendc/interleave_rope/op_kernel/"
    "interleave_rope_b11d.h"
)


def bf16_rope(x, cosine, sine):
    cosine = cosine[..., :32].unsqueeze(2)
    sine = sine[..., :32].unsqueeze(2)
    even, odd = x[..., 0::2], x[..., 1::2]
    real = even * cosine - odd * sine
    imaginary = odd * cosine + even * sine
    return torch.stack((real, imaginary), -1).flatten(-2)


def fused_rope(x, cosine, sine):
    """Keep BF16 trigonometric inputs and round the FP32 result once."""
    cosine = cosine[..., :32].float().unsqueeze(2)
    sine = sine[..., :32].float().unsqueeze(2)
    even, odd = x[..., 0::2].float(), x[..., 1::2].float()
    real = even * cosine + (-odd) * sine
    imaginary = odd * cosine + even * sine
    return torch.stack((real, imaginary), -1).flatten(-2).to(x.dtype)


def encode(key, quantized, rotation):
    if quantized:
        codes, scale = quantize_a8(key @ rotation)
        return codes.reshape(-1), scale.to(torch.float16).reshape(-1)
    return key.reshape(-1), None


def mismatch(actual, expected):
    return int((actual.reshape(-1) != expected.reshape(-1)).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--local-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    weights = ReferenceWeights(args.checkpoint)
    config = GlmMoeDsaConfig.from_dict(weights.config)
    rotary = GlmMoeDsaRotaryEmbedding(config, device="cpu")
    rotation = hadamard_matrix()
    native_report = args.native_run / "rank-0.json"
    metadata = json.loads(native_report.read_text())
    if metadata["metadata_sha256"] != weights.contract["metadata_sha256"]:
        raise ValueError("Native checkpoint metadata differs")
    artifacts = {str(native_report): digest(native_report)}
    results = []
    with torch.inference_mode():
        for offset, token in enumerate(metadata["teacher_tokens"]):
            position = metadata["initial_position"] + offset
            name = f"step-{position:05d}.pt"
            native_path = args.native_run / "trace" / name
            local_path = args.local_run / name
            actual = torch.load(native_path, weights_only=True)
            local = torch.load(local_path, weights_only=True)
            if actual["token"] != token or local["token"] != token:
                raise ValueError("Frozen trace inputs differ")
            artifacts.update({str(path): digest(path)
                              for path in (native_path, local_path)})
            for layer in range(78):
                saved = actual["cache_rows"][layer]
                if saved["index"] is None:
                    continue
                prefix = f"model.layers.{layer}"
                hidden = actual["hidden_states"][layer].unsqueeze(1)
                values = hidden.float()
                variance = values.square().mean(-1, keepdim=True)
                hidden = (values * (variance + 1e-5).rsqrt()).to(hidden.dtype)
                hidden = hidden * weights.tensor(
                    prefix + ".input_layernorm.weight", dtype=torch.bfloat16,
                )
                key = weights.linear(
                    prefix + ".self_attn.indexer.wk", hidden,
                    partition="replicated",
                )
                indexer = prefix + ".self_attn.indexer"
                key = functional.layer_norm(
                    key.float(), (128,),
                    weights.tensor(indexer + ".k_norm.weight").float(),
                    weights.tensor(indexer + ".k_norm.bias").float(), eps=1e-6,
                ).to(torch.bfloat16).unsqueeze(2)
                cosine, sine = rotary(hidden, torch.tensor([[position]]))
                quantized = saved["index"].dtype == torch.int8
                entry = {"position": position, "layer": layer,
                         "quantized": quantized, "variants": {}}
                for label, operation in (("bf16_steps", bf16_rope),
                                         ("fp32_fused", fused_rope)):
                    rotated = torch.cat((
                        operation(key[..., :64], cosine, sine), key[..., 64:],
                    ), -1).reshape(1, 128)
                    codes, scale = encode(rotated, quantized, rotation)
                    baseline = local["cache_rows"][layer]
                    entry["variants"][label] = {
                        "native_key_unequal": mismatch(codes, saved["index"]),
                        "local_key_unequal": mismatch(
                            codes, baseline["index"],
                        ),
                        "native_scale_unequal": (
                            mismatch(scale, saved["index_scale"])
                            if quantized else 0
                        ),
                        "local_scale_unequal": (
                            mismatch(scale, baseline["index_scale"])
                            if quantized else 0
                        ),
                    }
                results.append(entry)

    summary = {}
    for label in ("bf16_steps", "fp32_fused"):
        summary[label] = {
            metric: sum(row["variants"][label][metric] for row in results)
            for metric in results[0]["variants"][label]
        }
    baseline = summary["bf16_steps"]
    fused = summary["fp32_fused"]
    passed = not (baseline["local_key_unequal"]
                  or baseline["local_scale_unequal"]
                  or fused["native_key_unequal"]
                  or fused["native_scale_unequal"])
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen native hidden; CPU key projection, RoPE and C8",
        "changed_variable": "BF16 stepwise versus FP32 rotary intermediates",
        "source_sha256": {str(Path(__file__)): digest(Path(__file__)),
                          str(ROPE_SOURCE): digest(ROPE_SOURCE)},
        "artifact_sha256": artifacts,
        "weight_arithmetic": weights.contract,
        "rows": results, "summary": summary,
        "timed": False, "profiled": False,
        "status": "pass" if passed else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
