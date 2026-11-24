"""CPU scalar and instance-isolation checks for the HF RoPE adapter."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys

import torch
from transformers import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as hf

from tilert.models.glm_5_2_ascend import reference


ROOT = Path("/workspace/src/TileRT-Ascend")
OUTPUT = Path(__file__).with_suffix(".json")
REFERENCE = ROOT / "tilert/models/glm_5_2_ascend/reference.py"
MTP_SOURCE = ROOT / "tilert/models/glm_5_2_ascend/mtp_reference.py"
CONFIG = Path("/workspace/models/GLM-5.2-w4a8c8/config.json")
checks = []


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(name, passed, **details):
    checks.append({"name": name, "passed": bool(passed), **details})


def bf16_bits(value):
    """Round a finite Python float directly to BF16, ties to even."""
    sign = 0x8000 if math.copysign(1.0, value) < 0 else 0
    magnitude = abs(value)
    if magnitude == 0:
        return sign
    if not math.isfinite(value):
        raise ValueError("The scalar fixture requires finite values")
    _, exponent = math.frexp(magnitude)
    if exponent < -125:
        raise ValueError("The scalar fixture excludes BF16 subnormals")
    significand = round(math.ldexp(magnitude, 8 - exponent))
    if significand == 256:
        significand, exponent = 128, exponent + 1
    return sign | ((exponent + 126) << 7) | (significand - 128)


def scalar_rows(values, cosines, sines, layout):
    """Use only Python float64 arithmetic for the independent oracle."""
    expected = []
    for position, heads in enumerate(values[0]):
        for row in heads:
            real, imaginary = [], []
            for pair in range(32):
                even, odd = row[2 * pair:2 * pair + 2]
                cosine = cosines[0][position][pair]
                sine = sines[0][position][pair]
                real.append(bf16_bits(even * cosine - odd * sine))
                imaginary.append(bf16_bits(odd * cosine + even * sine))
            if layout == "half":
                expected.extend(real + imaginary)
            else:
                expected.extend(value for pair in zip(real, imaginary)
                                for value in pair)
    return expected


def compare(name, actual, expected, shape, **details):
    bits = [value & 0xffff for value in
            actual.contiguous().view(torch.int16).reshape(-1).tolist()]
    mismatch = sum(left != right for left, right in zip(bits, expected))
    valid = (actual.device.type == "cpu"
             and actual.dtype == torch.bfloat16
             and list(actual.shape) == shape
             and len(bits) == len(expected) and mismatch == 0)
    record(name, valid, shape=list(actual.shape), elements=len(bits),
           mismatches=mismatch, **details)


def inputs(rows, heads, salt):
    # The input values are fixed independently of adapter/native outputs.
    values = [(((index * 73 + salt * 137) % 2049) - 1024) / 256
              for index in range(rows * heads * 64)]
    return torch.tensor(values, dtype=torch.bfloat16).view(
        1, rows, heads, 64
    )


def check_positions(config):
    rotary = hf.GlmMoeDsaRotaryEmbedding(config=config, device="cpu")
    for start in (0, 1, 8192):
        for rows in (1, 4):
            positions = torch.arange(start, start + rows).unsqueeze(0)
            hidden = torch.zeros((1, rows, 1), dtype=torch.bfloat16)
            cosine, sine = rotary(hidden, positions)
            record(
                f"coefficients.{start}.M{rows}",
                cosine.dtype == sine.dtype == torch.bfloat16
                and list(cosine.shape) == [1, rows, 64]
                and torch.equal(cosine[..., :32], cosine[..., 32:])
                and torch.equal(sine[..., :32], sine[..., 32:]),
                positions=positions[0].tolist(),
            )
            query, key = inputs(rows, 4, start + 1), inputs(rows, 1, start + 7)
            main = reference.attention_rope(
                query.transpose(1, 2), key.transpose(1, 2), cosine, sine
            )
            main_axis2 = reference.attention_rope(
                query, key, cosine, sine, unsqueeze_dim=2
            )
            for role, value, result, result_axis2 in zip(
                ("q4", "k1"), (query, key), main, main_axis2
            ):
                shape = list(value.shape)
                expected_half = scalar_rows(
                    value.tolist(), cosine.tolist(), sine.tolist(), "half"
                )
                expected_alternating = scalar_rows(
                    value.tolist(), cosine.tolist(), sine.tolist(),
                    "alternating",
                )
                name = f"position{start}.M{rows}.{role}"
                compare(name + ".main_BHMD", result.transpose(1, 2),
                        expected_half, shape, input_layout="BHMD")
                compare(name + ".main_BMHD", result_axis2,
                        expected_half, shape, input_layout="BMHD")
                compare(name + ".index_BMHD",
                        reference.interleaved_rope(value, cosine, sine),
                        expected_alternating, shape, input_layout="BMHD")


def check_rounding_witness():
    value = torch.full((1, 1, 1, 64), 1.0078125, dtype=torch.bfloat16)
    cosine = torch.full((1, 1, 64), 0.5390625, dtype=torch.bfloat16)
    sine = torch.full((1, 1, 64), 0.83984375, dtype=torch.bfloat16)
    old = value[..., 0::2] * cosine[..., :32].unsqueeze(2)
    old = old - value[..., 1::2] * sine[..., :32].unsqueeze(2)
    actual = reference.interleaved_rope(value, cosine, sine)
    expected = scalar_rows(value.tolist(), cosine.tolist(), sine.tolist(),
                           "alternating")
    compare("rounding_witness.scalar", actual, expected, [1, 1, 1, 64])
    record("rounding_witness.distinguishes_bf16_steps",
           old.flatten()[0].item() == -0.3046875
           and actual.flatten()[0].item() == -0.302734375,
           bf16_steps=old.flatten()[0].item(),
           adapter=actual.flatten()[0].item(),
           independent_float64_real=1.0078125 * (0.5390625 - 0.83984375))


def check_instance_isolation(config):
    with torch.device("meta"):
        first = hf.GlmMoeDsaAttention(config, layer_idx=0)
        second = hf.GlmMoeDsaAttention(config, layer_idx=0)
    original = first.forward.__func__
    second_original = second.forward.__func__
    module_globals = dict(vars(hf))
    reference.install_attention_rope(first)
    adapted = first.forward.__func__
    differing_names = [name for name in original.__globals__
                       if original.__globals__[name]
                       is not adapted.__globals__.get(name)]
    record("installation.same_code_object",
           adapted.__code__ is original.__code__
           and adapted.__code__.co_code == original.__code__.co_code)
    record("installation.same_call_defaults",
           adapted.__defaults__ == original.__defaults__
           and adapted.__kwdefaults__ == original.__kwdefaults__
           and adapted.__closure__ is original.__closure__)
    record("installation.instance_private_lookup",
           adapted.__globals__ is not original.__globals__
           and differing_names == ["apply_rotary_pos_emb_interleave"]
           and adapted.__globals__["apply_rotary_pos_emb_interleave"]
           is reference.attention_rope
           and first.forward.__self__ is first,
           changed_globals=differing_names)
    record("installation.hf_module_unchanged",
           module_globals.keys() == vars(hf).keys()
           and all(value is vars(hf)[name]
                   for name, value in module_globals.items()))
    record("installation.second_instance_and_class_unchanged",
           second.forward.__func__ is second_original is original
           and hf.GlmMoeDsaAttention.forward is original
           and second.forward.__globals__ is original.__globals__)



def check_mtp_constructor():
    from tilert.models.glm_5_2_ascend.mtp_reference import MtpReference

    original = hf.GlmMoeDsaAttention.forward
    original_decoder = hf.GlmMoeDsaDecoderLayer.forward
    module_globals = dict(vars(hf))
    oracle = MtpReference(CONFIG.parent, capacity=9216)
    attention = oracle.decoder.self_attn
    adapted = attention.forward.__func__
    record("mtp.constructs_real_checkpoint",
           oracle.contract["layer"] == 78 and oracle.position == 0
           and oracle.capacity == 9216 and len(oracle.cache.layers) == 1)
    record("mtp.instance_forward_binding",
           attention.forward.__self__ is attention
           and adapted.__code__ is original.__code__
           and adapted.__defaults__ == original.__defaults__
           and adapted.__kwdefaults__ == original.__kwdefaults__
           and adapted.__closure__ is original.__closure__)
    differing_names = [name for name in original.__globals__
                       if original.__globals__[name]
                       is not adapted.__globals__.get(name)]
    record("mtp.private_rope_lookup",
           adapted.__globals__ is not original.__globals__
           and differing_names == ["apply_rotary_pos_emb_interleave"]
           and adapted.__globals__["apply_rotary_pos_emb_interleave"]
           is reference.attention_rope,
           changed_globals=differing_names)
    contract = oracle.weights.contract["rotary"]
    record("mtp.rotary_contract_matches_independent_copy",
           contract == reference.ROTARY_CONTRACT
           and contract is not reference.ROTARY_CONTRACT, contract=contract)
    indexer = attention.indexer
    record("mtp.indexer_bf16_configuration",
           isinstance(indexer, reference.ReferenceIndexer)
           and indexer.quantized is False
           and oracle.contract["index_cache_dtype"] == "BF16"
           and oracle.weights.quantization.get(
               "model.layers.78.self_attn.indexer.quant_type"
           ) in (None, "FLOAT"))
    probe = torch.arange(256).reshape(2, 128).to(torch.bfloat16) / 128
    encoded, scale = indexer.encode(probe)
    record("mtp.indexer_bf16_encode",
           encoded is probe and encoded.dtype == torch.bfloat16
           and scale is None)
    indexer.inject(oracle.cache, probe, None)
    stored = oracle.cache.layers[0].indexer_keys
    record("mtp.indexer_bf16_cache",
           stored.dtype == torch.bfloat16
           and torch.equal(stored, probe.unsqueeze(0))
           and indexer.scales is None)
    oracle.reset()
    record("mtp.cpu_parameters_and_empty_reset",
           all(value.device.type == "cpu" for value in (
               *oracle.decoder.parameters(), *oracle.decoder.buffers()
           )) and oracle.position == 0
           and oracle.cache.layers[0].indexer_keys is None
           and indexer.external_indices is None)
    with torch.device("meta"):
        unaffected = hf.GlmMoeDsaAttention(oracle.config, layer_idx=0)
    record("mtp.hf_module_classes_and_other_instance_unchanged",
           module_globals.keys() == vars(hf).keys()
           and all(value is vars(hf)[name]
                   for name, value in module_globals.items())
           and hf.GlmMoeDsaAttention.forward is original
           and hf.GlmMoeDsaDecoderLayer.forward is original_decoder
           and unaffected.forward.__func__ is original
           and unaffected.forward.__globals__ is original.__globals__)


def main():
    torch.set_num_threads(1)
    source_before = digest(REFERENCE)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU RoPE arithmetic, instance isolation and real-checkpoint MTP construction",
        "device_execution": False, "full_model_execution": False,
        "oracle": "Python float64 scalar products/sums and independent BF16 RNE",
        "input_coefficients": "Installed HF rotary embedding, BF16 before rotation",
        "environment": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "autoload": os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"),
        },
        "source_sha256": {
            str(REFERENCE): source_before, str(Path(__file__)): digest(__file__),
            str(Path(hf.__file__)): digest(hf.__file__),
            str(CONFIG): digest(CONFIG),
            str(MTP_SOURCE): digest(MTP_SOURCE),
        },
        "checks": checks,
    }
    try:
        config = GlmMoeDsaConfig.from_dict(json.loads(CONFIG.read_text()))
        check_positions(config)
        check_rounding_witness()
        check_instance_isolation(config)
        check_mtp_constructor()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        record("execution_completed", False)
    record("reference_source_stable", digest(REFERENCE) == source_before)
    record("mtp_source_stable",
           digest(MTP_SOURCE) == report["source_sha256"][str(MTP_SOURCE)])
    npu_module = sys.modules.get("torch_npu.npu")
    initialized = (None if npu_module is None
                   else vars(npu_module).get("_initialized"))
    report["environment"]["torch_npu_imported"] = "torch_npu" in sys.modules
    report["environment"]["npu_runtime_initialized"] = initialized
    record("npu_runtime_not_initialized",
           npu_module is None or initialized is False)
    report["passed"] = sum(item["passed"] for item in checks)
    report["failed"] = sum(not item["passed"] for item in checks)
    report["status"] = "pass" if report["failed"] == 0 else "fail"
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key]
                      for key in ("status", "passed", "failed")}))
    print(OUTPUT)
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
