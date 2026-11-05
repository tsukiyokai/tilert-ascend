"""CPU arithmetic for the GLM-5.2 W8 MLA Prolog V3 boundary.

The raw checkpoint reader is shared with the independent HF reference.
No production loader or CANN operator is called. The returned NoPE query
uses HF's unabsorbed 192-channel space; CANN absorbs it into 512 channels.
"""

import torch

from .reference_weights import quantize_dynamic_a8


CONTRACT = {
    "implementation": "CPU W8 MLA Prolog V3; unabsorbed HF query",
    "input_quant": "BF16 DynamicQuant with calibrated A3 FP32 vdiv",
    "projection": "INT32 dot; FP32 token scale then channel scale",
    "rms": "FP32 64-lane fold; Div by sqrt(mean+1e-5); FP32 gamma",
    "query_quant": "FP32 Div(amax,127) and Div(127,amax); RINT to INT8",
    "q_nope": "BF16 after Q-B dequantization",
    "rope": "FP32 dequantization/products/addition; final BF16 half basis",
    "latent": "BF16 after weighted FP32 RMS",
    "index_query": "Reuse Prolog query codes/scale; channel then token scale",
    "index_query_boundary": "Scale order retains the existing W8 reference",
}
HIDDEN = 6144
Q_LORA = 2048
KV_LORA = 512
HEADS = 64
NOPE = 192
ROPE = 64
INDEX_HEADS = 32
INDEX_DIM = 128
EPSILON = 1e-5


def project_codes(weights, prefix, codes, token_scale, *, scale_order):
    """Return FP32 W8 projection with an explicit dequantization order."""
    if (codes.device.type != "cpu" or codes.dtype != torch.int8
            or codes.ndim != 2 or token_scale.dtype != torch.float32
            or token_scale.device.type != "cpu"
            or token_scale.numel() != codes.shape[0]):
        raise ValueError("W8 projection needs CPU INT8 rows and FP32 scales")
    if weights.quantization[prefix + ".weight"] != "W8A8_DYNAMIC":
        raise ValueError("MLA Prolog reference requires W8A8_DYNAMIC")
    integer = weights.tensor(prefix + ".weight")
    channel_scale = weights.tensor(prefix + ".weight_scale").reshape(-1)
    offset = weights.tensor(prefix + ".weight_offset")
    if (integer.dtype != torch.int8 or integer.ndim != 2
            or integer.shape[1] != codes.shape[1]
            or channel_scale.dtype != torch.float32
            or channel_scale.numel() != integer.shape[0]
            or bool(torch.count_nonzero(offset))):
        raise ValueError("Expected symmetric W8 with FP32 channel scales")
    product = (codes.to(torch.int32) @ integer.to(torch.int32).T).float()
    scale = token_scale.reshape(-1, 1)
    if scale_order == "token_first":
        return (product * scale) * channel_scale
    if scale_order == "channel_first":
        return (product * channel_scale) * scale
    raise ValueError("Scale order must be token_first or channel_first")


def rms_fp32(value, gamma):
    """Retain row-local FP32 folds and the Prolog Div/gamma boundary."""
    if (value.device.type != "cpu" or value.dtype != torch.float32
            or value.ndim != 2 or value.shape[1] % 64
            or gamma.device.type != "cpu" or gamma.dtype != torch.bfloat16
            or gamma.shape != (value.shape[1],)):
        raise ValueError("RMS needs FP32 rows and their BF16 gamma")
    squares = value.square().reshape(value.shape[0], -1, 64)
    folded = squares[:, 0].clone()
    for block in range(1, squares.shape[1]):
        folded = folded + squares[:, block]
    reciprocal = torch.tensor(1 / value.shape[1], dtype=torch.float32)
    mean = folded.sum(dim=-1, keepdim=True) * reciprocal
    return torch.div(value, (mean + EPSILON).sqrt()) * gamma.float()


def quantize_prolog(value):
    """Quantize FP32 normalized queries without an intervening BF16 cast."""
    if (value.device.type != "cpu" or value.dtype != torch.float32
            or value.ndim != 2):
        raise ValueError("Prolog query quantization needs CPU FP32 rows")
    maximum = value.abs().amax(dim=-1)
    denominator = torch.where(maximum > 0, maximum, 1.0)
    full = torch.full_like(maximum, 127)
    factor = torch.div(full, denominator)
    scale = torch.div(maximum, full)
    codes = torch.round(value * factor[:, None]).clamp(-127, 127)
    return codes.to(torch.int8), scale


def rotary_half(value, cosine, sine):
    """Rotate alternating FP32 lanes and return a BF16 half-basis tensor."""
    if (value.device.type != "cpu" or value.dtype != torch.float32
            or value.ndim != 3 or value.shape[-1] != ROPE):
        raise ValueError("RoPE needs CPU FP32 [tokens,heads,64]")
    expected = value.shape[0], ROPE
    for angles in (cosine, sine):
        if (angles.shape != expected or angles.dtype != torch.bfloat16
                or angles.device.type != "cpu"):
            raise ValueError("RoPE angles must be CPU BF16 [tokens,64]")
    cosine = cosine[:, None, :ROPE // 2].float()
    sine = sine[:, None, :ROPE // 2].float()
    even, odd = value[..., 0::2], value[..., 1::2]
    real = even * cosine + (-odd) * sine
    imaginary = odd * cosine + even * sine
    return torch.cat((real, imaginary), dim=-1).to(torch.bfloat16)


def preprocess(weights, prefix, hidden, cos, sin):
    """Return full-64-head HF inputs from already normalized BF16 rows.

    ``prefix`` names ``model.layers.N.self_attn``. The input is [M,6144];
    angles are [M,64]. Outputs retain the token axis without batch nesting.
    """
    if (hidden.device.type != "cpu" or hidden.dtype != torch.bfloat16
            or hidden.ndim != 2 or hidden.shape[1] != HIDDEN):
        raise ValueError("Prolog input must be CPU BF16 [tokens,6144]")
    codes, scale = quantize_dynamic_a8(hidden)
    query = project_codes(weights, prefix + ".q_a_proj", codes, scale,
                          scale_order="token_first")
    kv = project_codes(weights, prefix + ".kv_a_proj_with_mqa", codes,
                       scale, scale_order="token_first")
    q_norm = rms_fp32(query, weights.tensor(
        prefix + ".q_a_layernorm.weight", dtype=torch.bfloat16,
    ))
    latent = rms_fp32(kv[:, :KV_LORA], weights.tensor(
        prefix + ".kv_a_layernorm.weight", dtype=torch.bfloat16,
    )).to(torch.bfloat16)
    q_codes, q_scale = quantize_prolog(q_norm)
    expanded = project_codes(weights, prefix + ".q_b_proj", q_codes,
                             q_scale, scale_order="token_first")
    expanded = expanded.reshape(hidden.shape[0], HEADS, NOPE + ROPE)
    return {
        "q_nope": expanded[..., :NOPE].to(torch.bfloat16),
        "q_rope": rotary_half(expanded[..., NOPE:], cos, sin),
        "latent": latent,
        "k_rope": rotary_half(kv[:, None, KV_LORA:], cos, sin),
        "q_norm": q_norm, "q_codes": q_codes, "q_scale": q_scale,
    }


def index_query(weights, prefix, q_norm):
    """Return pre-RoPE BF16 [M,32,128] from the same Prolog INT8 pair."""
    if q_norm.ndim != 2 or q_norm.shape[1] != Q_LORA:
        raise ValueError("Indexer query input must have 2048 channels")
    codes, scale = quantize_prolog(q_norm)
    projected = project_codes(weights, prefix + ".wq_b", codes, scale,
                              scale_order="channel_first")
    return projected.to(torch.bfloat16).reshape(-1, INDEX_HEADS, INDEX_DIM)
