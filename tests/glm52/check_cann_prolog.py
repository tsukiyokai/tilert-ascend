"""Check one real TP16 shard of CANN MLA Prolog V3 without collectives.

The frozen layer-zero input is a BF16 embedding row at position zero or one.
The CPU reference retains FP32 dequantization, RMS and gamma multiplication.
Device reduction trees and Div/Sqrt implementations remain measured
boundaries; CPU arithmetic is not asserted to emulate Ascend instructions.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import sys


HIDDEN = 6144
Q_LORA = 2048
KV_LORA = 512
HEADS = 4
NOPE = 192
ROPE = 64
VALUE = 256
PAGE = 128
EPSILON = 1e-5
ALLOCATOR_LIMIT = 1536 * 1024 ** 2
CANN_ROOT = Path("/opt/ascend/cann-9.2.0")
ASCEND_ROOT = Path("/workspace/src/vllm-ascend-unified")
ARCH22 = Path("opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/"
              "ascendc/mla_prolog/op_kernel/arch22")
METADATA = ("config.json", "quant_model_description.json",
            "quant_model_weights.safetensors.index.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(value):
    import torch

    raw = value.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def describe(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": tensor_digest(value)}


def load_case(checkpoint, trace_path):
    import torch
    from safetensors import safe_open

    hashes = {name: digest(checkpoint / name) for name in METADATA}
    manifest_path = trace_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if (manifest["status"] != "pass"
            or manifest.get("numerical_execution") is not True
            or manifest["metadata_sha256"] != hashes
            or manifest["trace_sha256"][trace_path.name]
            != digest(trace_path)):
        raise ValueError("Frozen HF input identity does not match checkpoint")
    trace = torch.load(trace_path, map_location="cpu", weights_only=True)
    position = trace["position"]
    if (manifest.get("initial_position") != 0
            or type(position) is not int or position not in (0, 1)
            or trace["committed_position"] != position + 1
            or trace["token"] != manifest["teacher_tokens"][position]):
        raise ValueError("This probe requires cold teacher position 0 or 1")
    hidden = trace["hidden_states"][0].clone()
    if hidden.shape != (1, HIDDEN) or hidden.dtype != torch.bfloat16:
        raise ValueError("Frozen embedding must be BF16 [1,6144]")
    index = json.loads((checkpoint / METADATA[2]).read_text())["weight_map"]
    tensors = {}

    def read(name, rows=None):
        with safe_open(checkpoint / index[name], framework="pt",
                       device="cpu") as saved:
            value = (saved.get_tensor(name) if rows is None else
                     saved.get_slice(name)[slice(*rows)])
        value = value.clone().contiguous()
        tensors[name] = {**describe(value), "shard": index[name],
                         "output_rows": rows}
        return value

    def projection(name, rows=None):
        prefix = "model.layers.0.self_attn." + name
        weight = read(prefix + ".weight", rows)
        scale = read(prefix + ".weight_scale", rows).flatten()
        offset = read(prefix + ".weight_offset", rows)
        if (weight.dtype != torch.int8 or scale.dtype != torch.float32
                or torch.count_nonzero(offset)):
            raise ValueError("Expected symmetric W8 with FP32 channel scale")
        return {"weight": weight, "scale": scale}

    weights = {
        "q_a": projection("q_a_proj"),
        "q_b": projection("q_b_proj", (0, HEADS * (NOPE + ROPE))),
        "kv_a": projection("kv_a_proj_with_mqa"),
        "kv_b": read("model.layers.0.self_attn.kv_b_proj.weight",
                     (0, HEADS * (NOPE + VALUE))),
    }
    for key, name in (
        ("input_gamma", "input_layernorm"),
        ("q_gamma", "self_attn.q_a_layernorm"),
        ("kv_gamma", "self_attn.kv_a_layernorm"),
    ):
        weights[key] = read("model.layers.0." + name + ".weight").to(
            torch.bfloat16
        )
    embedding = read("model.embed_tokens.weight",
                     (trace["token"], trace["token"] + 1)).to(torch.bfloat16)
    if not torch.equal(hidden, embedding):
        raise ValueError("Frozen hidden is not the checkpoint embedding row")
    identity = {
        "checkpoint_sha256": hashes, "tensor_sources": tensors,
        "trace": str(trace_path), "trace_sha256": digest(trace_path),
        "manifest_sha256": digest(manifest_path), "token": trace["token"],
        "position": position, "layer": 0, "logical_rank": 0, "tp_size": 16,
        "hidden": describe(hidden),
    }
    return hidden, weights, trace["cache_rows"][0], identity


def load_rotary(checkpoint, trace_path, hidden, position):
    import torch
    from transformers import GlmMoeDsaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaRotaryEmbedding,
    )

    raw = json.loads((checkpoint / "config.json").read_text())
    rotary = GlmMoeDsaRotaryEmbedding(
        GlmMoeDsaConfig.from_dict(raw), device="cpu",
    )
    source = Path(inspect.getsourcefile(GlmMoeDsaRotaryEmbedding))
    manifest = json.loads(trace_path.with_name("manifest.json").read_text())
    if digest(source) not in manifest["installed_hf_source_sha256"].values():
        raise ValueError("Installed rotary source differs from frozen HF")
    positions = torch.tensor([[position]], dtype=torch.int64)
    cos, sin = rotary(hidden.unsqueeze(0), positions)
    cos, sin = cos.reshape(1, ROPE), sin.reshape(1, ROPE)
    sources = {str(source): digest(source)}
    if rotary.rope_type != "default":
        function = ROPE_INIT_FUNCTIONS[rotary.rope_type]
        implementation = Path(inspect.getsourcefile(function))
        sources[str(implementation)] = digest(implementation)
    metadata = {
        "position_ids": describe(positions), "cos": describe(cos),
        "sin": describe(sin), "rope_type": rotary.rope_type,
        "attention_scaling": float(rotary.attention_scaling),
        "source_sha256": sources,
    }
    return cos, sin, metadata


def rms_fp32(value, gamma):
    """Preserve Prolog's 64-lane folding and FP32 arithmetic boundaries."""
    import torch

    squares = value.float().square().reshape(-1, 64)
    folded = squares[0].clone()
    for block in squares[1:]:
        folded = folded + block
    reciprocal = torch.tensor(1 / value.shape[-1], dtype=torch.float32)
    mean = folded.sum() * reciprocal
    denominator = (mean + EPSILON).sqrt()
    return torch.div(value.float(), denominator) * gamma.float()


def quantize_input(value):
    """IEEE FP32 input-quantization formula for the counterfactual."""
    import torch

    maximum = value.float().abs().amax(dim=-1, keepdim=True)
    if torch.any(maximum == 0):
        raise ValueError("The frozen real input must have nonzero magnitude")
    factor = torch.div(torch.full_like(maximum, 127), maximum)
    scale = maximum * torch.tensor(1 / 127, dtype=torch.float32)
    return (value.float() * factor).round().to(torch.int8), scale


def quantize_query(value):
    """Prolog DynamicQuant uses separate Divs for scale and multiplier."""
    import torch

    maximum = value.abs().amax(dim=-1, keepdim=True)
    if torch.any(maximum == 0):
        raise ValueError("The frozen real query must have nonzero magnitude")
    full = torch.full_like(maximum, 127)
    scale = torch.div(maximum, full)
    factor = torch.div(full, maximum)
    return (value * factor).round().to(torch.int8), scale


def project(codes, scale, weight):
    import torch

    integer = codes.to(torch.int32) @ weight["weight"].to(torch.int32).T
    output = integer.float() * scale.reshape(-1, 1)
    return integer, output * weight["scale"]


def rotate(value, cos=None, sin=None):
    """Rotate FP32 alternating lanes, then round the half basis to BF16."""
    import torch

    if cos is None and sin is None:
        cos = torch.ones(1, ROPE, dtype=torch.bfloat16)
        sin = torch.zeros_like(cos)
    shape = (1,) * (value.ndim - 1) + (ROPE // 2,)
    cosine = cos[..., :ROPE // 2].float().reshape(shape)
    sine = sin[..., :ROPE // 2].float().reshape(shape)
    even, odd = value[..., 0::2].float(), value[..., 1::2].float()
    real = even * cosine + (-odd) * sine
    imaginary = odd * cosine + even * sine
    return torch.cat((real, imaginary), dim=-1).to(torch.bfloat16)


def query_reference(codes, scale, weights, cos=None, sin=None):
    import torch

    integer, query = project(codes, scale, weights["q_b"])
    query = query.reshape(1, HEADS, NOPE + ROPE)
    nope = query[..., :NOPE].to(torch.bfloat16)
    uk = weights["kv_b"].reshape(HEADS, NOPE + VALUE, KV_LORA)[:, :NOPE]
    absorbed = torch.bmm(nope.transpose(0, 1).float(), uk.float())
    return {
        "q_b_integer": integer, "q_b_fp32": query,
        "q_nope_bf16": nope,
        "query_nope": absorbed.transpose(0, 1).to(torch.bfloat16),
        "query_rope": rotate(query[..., NOPE:], cos, sin),
    }


def prolog_reference(codes, scale, weights, cos=None, sin=None):
    import torch

    q_integer, q_dequant = project(codes, scale, weights["q_a"])
    kv_integer, kv_dequant = project(codes, scale, weights["kv_a"])
    q_normalized = rms_fp32(q_dequant, weights["q_gamma"])
    kv_normalized = rms_fp32(kv_dequant[:, :KV_LORA], weights["kv_gamma"])
    q_codes, q_scale = quantize_query(q_normalized)
    return {
        "q_a_integer": q_integer, "q_a_dequant_fp32": q_dequant,
        "kv_a_integer": kv_integer, "kv_a_dequant_fp32": kv_dequant,
        "q_a_normalized_fp32": q_normalized,
        "kv_a_normalized_fp32": kv_normalized,
        "query_norm": q_codes, "query_norm_scale": q_scale,
        "latent": kv_normalized.to(torch.bfloat16),
        "rope": rotate(kv_dequant[:, KV_LORA:], cos, sin),
        **query_reference(q_codes, q_scale, weights, cos, sin),
    }


def bf16_counterfactual(reference, weights):
    """Expose the extra BF16 projection/normalization boundaries separately."""
    import torch

    outputs = {}
    for name, width, gamma in (("q_a", Q_LORA, "q_gamma"),
                               ("kv_a", KV_LORA, "kv_gamma")):
        value = reference[name + "_dequant_fp32"][:, :width].to(torch.bfloat16)
        unit = rms_fp32(value, torch.ones_like(weights[gamma])).to(
            torch.bfloat16
        )
        outputs[name] = unit * weights[gamma]
    codes, scale = quantize_input(outputs["q_a"])
    return {**outputs, "query_norm": codes, "query_norm_scale": scale}


def common_arguments(weights, cos=None, sin=None, position=0):
    import torch

    if cos is None and sin is None and position == 0:
        cos = torch.ones(1, ROPE, dtype=torch.bfloat16)
        sin = torch.zeros_like(cos)
    if cos is None or sin is None:
        raise ValueError("Nonzero positions require their actual HF angles")
    return {
        "weight_dq": weights["q_a"]["weight"].T.contiguous(),
        "weight_dkv_kr": weights["kv_a"]["weight"].T.contiguous(),
        "weight_uq_qr": weights["q_b"]["weight"].T.contiguous(),
        "weight_uk": weights["kv_b"].reshape(
            HEADS, NOPE + VALUE, KV_LORA
        )[:, :NOPE].contiguous(),
        "rmsnorm_gamma_cq": weights["q_gamma"],
        "rmsnorm_gamma_ckv": weights["kv_gamma"],
        "dequant_scale_w_dq": weights["q_a"]["scale"].reshape(1, -1),
        "dequant_scale_w_dkv_kr": weights["kv_a"]["scale"].reshape(1, -1),
        "dequant_scale_w_uq_qr": weights["q_b"]["scale"].reshape(1, -1),
        "rope_cos": cos,
        "rope_sin": sin,
        "cache_index": torch.tensor([position], dtype=torch.int64),
        "rmsnorm_epsilon_cq": EPSILON, "rmsnorm_epsilon_ckv": EPSILON,
        "cache_mode": "PA_BSND", "query_norm_flag": True,
        "weight_quant_mode": 2, "kv_cache_quant_mode": 0,
        "query_quant_mode": 0, "qc_qr_scale": 1.0, "kc_scale": 1.0,
    }


def check_cpu_layout(arguments, codes, reference, weights):
    import torch

    expected = {
        "weight_dq": ((HIDDEN, Q_LORA), torch.int8),
        "weight_dkv_kr": ((HIDDEN, KV_LORA + ROPE), torch.int8),
        "weight_uq_qr": ((Q_LORA, HEADS * (NOPE + ROPE)), torch.int8),
        "weight_uk": ((HEADS, NOPE, KV_LORA), torch.bfloat16),
        "rmsnorm_gamma_cq": ((Q_LORA,), torch.bfloat16),
        "rmsnorm_gamma_ckv": ((KV_LORA,), torch.bfloat16),
        "cache_index": ((1,), torch.int64),
        "rope_cos": ((1, ROPE), torch.bfloat16),
        "rope_sin": ((1, ROPE), torch.bfloat16),
    }
    checks = []
    for name, (shape, dtype) in expected.items():
        value = arguments[name]
        checks.append({"name": "layout." + name,
                       "passed": value.shape == shape and value.dtype == dtype
                       and value.is_contiguous()})
    row = codes[0].tolist()
    for name in ("q_a", "kv_a"):
        matrix = weights[name]["weight"]
        for column in (0, matrix.shape[0] // 2, matrix.shape[0] - 1):
            scalar = sum(x * y for x, y in zip(row, matrix[column].tolist()))
            checks.append({"name": f"integer_dot.{name}.{column}",
                           "passed": reference[name + "_integer"][0, column]
                           .item() == scalar})
    return checks


def run_device(hidden, weights, device, cos=None, sin=None, position=0):
    import torch
    import torch_npu

    torch.npu.set_device(device)
    properties = torch.npu.get_device_properties(device)
    torch.npu.set_per_process_memory_fraction(
        ALLOCATOR_LIMIT / properties.total_memory, device
    )
    torch.npu.config.allow_internal_format = True
    torch.npu.reset_peak_memory_stats(device)
    arguments = common_arguments(weights, cos, sin, position)
    for name, value in arguments.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if name in ("weight_dq", "weight_dkv_kr", "weight_uq_qr"):
                value = torch_npu.npu_format_cast(value, 29)
            arguments[name] = value
    x = hidden.to(device)
    gamma = weights["input_gamma"].to(device)
    normalized, _ = torch_npu.npu_rms_norm(x, gamma, EPSILON)
    codes, scale = torch_npu.npu_dynamic_quant(normalized.contiguous())
    arguments.update(
        token_x=codes, dequant_scale_x=scale.reshape(1, 1),
        kv_cache=torch.zeros(1, PAGE, 1, KV_LORA,
                             dtype=torch.bfloat16, device=device),
        kr_cache=torch.zeros(1, PAGE, 1, ROPE,
                             dtype=torch.bfloat16, device=device),
    )
    output = torch_npu.npu_mla_prolog_v3(**arguments)
    torch.npu.synchronize(device)
    names = ("query_nope", "query_rope", "query_scale",
             "query_norm", "query_norm_scale")
    actual = {name: value.detach().cpu().clone()
              for name, value in zip(names, output)}
    actual.update(
        input_norm=normalized.cpu().clone(), input_codes=codes.cpu().clone(),
        input_scale=scale.cpu().clone(),
        latent=arguments["kv_cache"][0, position, 0].cpu().reshape(1, KV_LORA),
        rope=arguments["kr_cache"][0, position, 0].cpu().reshape(1, ROPE),
        latent_prefix=arguments["kv_cache"][0, :position].cpu().clone(),
        rope_prefix=arguments["kr_cache"][0, :position].cpu().clone(),
        latent_tail=arguments["kv_cache"][0, position + 1:].cpu().clone(),
        rope_tail=arguments["kr_cache"][0, position + 1:].cpu().clone(),
    )
    layouts = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                      "format": torch_npu.get_npu_format(value)}
               for name, value in arguments.items()
               if isinstance(value, torch.Tensor)}
    memory = {"allocator_limit_bytes": ALLOCATOR_LIMIT,
              "peak_allocated_bytes": torch.npu.max_memory_allocated(device),
              "peak_reserved_bytes": torch.npu.max_memory_reserved(device),
              "scope": "Torch allocator; runtime allocations excluded"}
    return actual, layouts, memory


def comparison(name, actual, expected, exact=False):
    import torch
    import torch.nn.functional as functional

    raw_shape = list(actual.shape)
    actual = actual.reshape(expected.shape)
    left, right = actual.float(), expected.float()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    norm = right.norm().item()
    error = (left - right).norm().item() / max(norm, 1e-12)
    cosine = (functional.cosine_similarity(left.flatten(), right.flatten(),
                                          dim=0).item() if norm else
              float(torch.count_nonzero(left) == 0))
    equal = actual.dtype == expected.dtype and torch.equal(actual, expected)
    compatible = actual.dtype == expected.dtype
    passed = compatible and finite and (
        equal if exact else error <= 0.01 and cosine >= .999
    )
    return {
        "name": name, "passed": passed,
        "exact_required": exact, "bitwise_equal": equal,
        "dtype": str(actual.dtype), "expected_dtype": str(expected.dtype),
        "raw_shape": raw_shape, "canonical_shape": list(expected.shape),
        "nrmse": error, "cosine": cosine,
        "max_abs": (left - right).abs().max().item(),
        "unequal": int(torch.count_nonzero(actual != expected)),
    }


def source_hashes():
    paths = [Path(__file__).resolve(),
             ASCEND_ROOT / "vllm_ascend/attention/sfa_v1.py",
             ASCEND_ROOT / "vllm_ascend/ops/layernorm.py"]
    paths.extend(CANN_ROOT / ARCH22 / name for name in (
        "service_rms_norm_arch22.h", "rms_norm.h",
        "mla_prolog_vector_comm_arch22.h", "service_rope_arch22.h",
        "kernel_mla_prolog_split_n_arch22.h",
    ))
    return {str(path): digest(path) for path in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu-only", action="store_true")
    mode.add_argument("--device", help="One explicitly assigned npu:N")
    args = parser.parse_args(argv)
    import torch
    import torch_npu

    torch.set_num_threads(4)
    initialized_before = torch_npu.npu._initialized
    if initialized_before:
        raise RuntimeError("Start this probe in a fresh process")
    hidden, weights, hf_cache, identity = load_case(
        args.checkpoint, args.trace
    )
    position = identity["position"]
    cos, sin, rotary = load_rotary(args.checkpoint, args.trace, hidden,
                                   position)
    identity["rotary"] = rotary
    normalized = rms_fp32(hidden, weights["input_gamma"]).to(torch.bfloat16)
    codes, scale = quantize_input(normalized)
    arguments = common_arguments(weights, cos, sin, position)
    schema = torch.ops.npu.npu_mla_prolog_v3.default._schema
    argument_names = {argument.name for argument in schema.arguments}
    required = set(arguments) | {
        "token_x", "dequant_scale_x", "kv_cache", "kr_cache",
    }
    if not required <= argument_names:
        raise ValueError("Installed Prolog V3 schema does not accept inputs")
    actual, layouts, memory = None, None, None
    if args.device:
        if torch.device(args.device).type != "npu":
            raise ValueError("Device mode requires an explicitly assigned NPU")
        with torch.inference_mode():
            actual, layouts, memory = run_device(
                hidden, weights, args.device, cos, sin, position,
            )
        codes, scale = actual["input_codes"], actual["input_scale"]
    reference = prolog_reference(codes, scale, weights, cos, sin)
    if not all(bool(torch.isfinite(value.float()).all())
               for value in reference.values()):
        raise ValueError("CPU reference produced a nonfinite intermediate")
    checks = check_cpu_layout(arguments, codes, reference, weights)
    checks.append({"name": "cache_index_matches_position",
                   "passed": arguments["cache_index"].item() == position})
    checks.append({"name": "rotary_position_sensitive",
                   "passed": (bool(torch.count_nonzero(sin)) if position
                              else torch.equal(cos, torch.ones_like(cos))
                              and not bool(torch.count_nonzero(sin)))})
    if actual is not None:
        for name in ("query_norm", "query_norm_scale", "latent", "rope",
                     "query_nope", "query_rope"):
            exact = name.startswith("query_norm") or (
                position == 1 and name in ("query_rope", "rope")
            )
            checks.append(comparison(name, actual[name], reference[name],
                                     exact=exact))
        for name in ("latent_prefix", "rope_prefix", "latent_tail",
                     "rope_tail"):
            checks.append({"name": name, "passed": not bool(
                torch.count_nonzero(actual[name])), "exact_required": True})
    early_rope = {
        "query_rope": rotate(reference["q_b_fp32"][..., NOPE:].to(
            torch.bfloat16).float(), cos, sin),
        "rope": rotate(reference["kv_a_dequant_fp32"][:, KV_LORA:].to(
            torch.bfloat16).float(), cos, sin),
    }
    artifact = args.output.with_suffix(".pt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        "hidden": hidden, "weights": weights, "hf_cache": hf_cache,
        "cpu_input_norm": normalized, "input_codes": codes,
        "input_scale": scale, "cpu_reference": reference,
        "bf16_counterfactual": bf16_counterfactual(reference, weights),
        "early_bf16_rope": early_rope, "rotary_cos": cos, "rotary_sin": sin,
        "actual": actual, "identity": identity,
    }
    if actual is not None:
        saved["cpu_query_from_device_qnorm"] = query_reference(
            actual["query_norm"].reshape(1, Q_LORA),
            actual["query_norm_scale"], weights, cos, sin,
        )
    torch.save(saved, artifact)
    tensor_bytes = sum(value.numel() * value.element_size()
                       for value in arguments.values()
                       if isinstance(value, torch.Tensor))
    passed = all(item["passed"] for item in checks)
    status = "cpu_ready" if actual is None else "pass"
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status if passed else "fail",
        "scope": "Layer0 rank0 TP16 W8 Prolog V3; no collective or model load",
        "device_execution": actual is not None, "device": args.device,
        "precision_verified": False, "performance_measured": False,
        "npu_initialized_before": initialized_before,
        "npu_initialized_after": torch_npu.npu._initialized,
        "identity": identity,
        "source_sha256": {**source_hashes(), **rotary["source_sha256"]},
        "packages": {name: version(name) for name in ("torch", "torch-npu")},
        "schema": str(schema), "command": [sys.executable, *sys.argv],
        "input_layouts": layouts, "memory": memory,
        "cpu_argument_layouts": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arguments.items()
            if isinstance(value, torch.Tensor)
        },
        "static_argument_tensor_bytes": tensor_bytes,
        "cache_write": {"position": position, "initial_value": 0,
                        "unchanged_prefix": [0, position],
                        "unchanged_tail": [position + 1, PAGE]},
        "frozen_hf_cache_diagnostic": {
            "used_for_acceptance": False,
            "comparisons": [comparison(name, hf_cache[name], reference[name])
                            for name in ("latent", "rope")],
        },
        "early_bf16_rope_diagnostic": {
            "used_for_acceptance": False,
            "comparisons": [comparison(name, early_rope[name], reference[name])
                            for name in ("query_rope", "rope")],
        },
        "reference_contract": {
            "dequant": "FP32(INT32) * token_scale, then channel_scale",
            "rms": "FP32 square; sequential 64-lane folding; CPU sum; "
                   "multiply 1/H; add eps; sqrt; Div; BF16 gamma in FP32",
            "query_quant": "scale=Div(amax,127); factor=Div(127,amax); "
                           "round ties-even to INT8, without BF16 norm",
            "latent": "BF16 after FP32 weighted RMS",
            "rope": "Actual HF BF16 cos/sin; FP32 products/addition; "
                    "half basis; final BF16",
            "query_absorption": "Q-B dequant FP32 -> BF16 NoPE -> "
                                "FP32 dot BF16 UK -> BF16",
            "bf16_counterfactual": "Additional BF16 boundaries applied to "
                                   "the same token-then-channel dequant; "
                                   "not the current HF adapter's output",
            "unconfirmed": "CPU reduction/Div/Sqrt do not establish "
                           "Ascend instruction-level equivalence",
        },
        "artifact": str(artifact), "artifact_sha256": digest(artifact),
        "checks": checks,
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.write_text(text)
    print(json.dumps({key: report[key] for key in (
        "status", "device_execution", "npu_initialized_after",
        "static_argument_tensor_bytes", "artifact",
    )}))
    return int(report["status"] == "fail")


if __name__ == "__main__":
    raise SystemExit(main())
