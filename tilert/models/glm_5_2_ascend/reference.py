from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from types import FunctionType, MethodType

import torch
from torch import nn
import torch.nn.functional as functional
from torch.overrides import TorchFunctionMode

from .ops.indexer import (
    CANN_ARCH22_SOURCE_DIRECTORY,
    cann_arch22_index_scores,
    cann_arch22_input_domain,
    cann_arch22_scoring_contract,
)

TP_SIZE = 16
LAYERS = 78
LAYER = 78
HIDDEN = 6144
VOCAB = 154880
TOP_K = 2048
Q_LORA = 2048
KV_LORA = 512
HEADS = 64
NOPE = 192
ROPE = 64
INDEX_HEADS = 32
INDEX_DIM = 128
EPSILON = 1e-5
PREFIX = f"model.layers.{LAYER}"
IMPLEMENTATION = "transformers.GlmMoeDsaDecoderLayer.MTP"

DYNAMIC_DIV_BIASED_MANTISSAS = (
    3, 7, 10, 12, 19, 22, 23, 28, 29, 35, 43, 48, 55, 73, 113,
)
STORAGE_DTYPES = {
    "BF16": (torch.bfloat16, 2),
    "F16": (torch.float16, 2),
    "F32": (torch.float32, 4),
    "I8": (torch.int8, 1),
    "I32": (torch.int32, 4),
    "I64": (torch.int64, 8),
}
ROTARY_CONTRACT = {
    "intermediates": "FP32 products and sum; one BF16 output cast",
    "angles": "HF FP32 generation followed by BF16 cast",
    "main_basis": "real half, imaginary half",
    "index_basis": "alternating real and imaginary",
    "cann_source_sha256": (
        "96028ea9c3f2b5426b60c00e03951fafbc"
        "d4903ceff5880b4ad31482d37e0a8a"
    ),
    "hf_adapter": "Instance forward globals; HF bytecode unchanged",
}
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


def quantize_a8(x):
    if x.device.type != "cpu":
        raise ValueError("The reference requires CPU tensors")
    values = x.to(torch.float32)
    maximum = torch.max(torch.abs(values), dim=-1).values
    scale = maximum * (1 / 127)
    denominator = torch.where(maximum > 0, maximum, 1.0)
    factor = torch.div(torch.full_like(denominator, 127), denominator)
    codes = torch.round(torch.mul(values, factor.unsqueeze(-1)))
    return codes.clamp(-127, 127).to(torch.int8), scale


def quantize_dynamic_a8(x):
    if x.device.type != "cpu" or x.dtype != torch.bfloat16:
        raise ValueError("Dynamic A8 reference requires CPU BF16 input")
    values = x.float()
    maximum = values.abs().amax(dim=-1)
    below_domain = (maximum > 0) & (maximum < 2.0**-120)
    if not bool(torch.isfinite(maximum).all()) or bool(below_domain.any()):
        raise ValueError("Dynamic A8 amax must be zero or finite and >=2^-120")
    scale = maximum * (1 / 127)
    denominator = torch.where(maximum > 0, maximum, 1.0)
    factor = torch.div(torch.full_like(maximum, 127), denominator)
    mantissa = (maximum.contiguous().view(torch.int32) >> 16) & 127
    corrected = torch.isin(
        mantissa, torch.tensor(DYNAMIC_DIV_BIASED_MANTISSAS, dtype=torch.int32)
    )
    factor_bits = factor.contiguous().view(torch.int32) - corrected.int()
    factor = factor_bits.view(torch.float32)
    codes = torch.round(values * factor.unsqueeze(-1))
    return codes.clamp(-127, 127).to(torch.int8), scale


def unpack_w4(packed):
    if packed.device.type != "cpu" or packed.dtype != torch.int8 or packed.ndim != 2:
        raise ValueError("Packed W4 must be INT8 [output/2, input]")
    unsigned = packed.to(torch.int16).remainder(256)
    low = unsigned.remainder(16)
    high = torch.div(unsigned, 16, rounding_mode="floor")
    output = torch.empty((packed.shape[0] * 2, packed.shape[1]), dtype=torch.int8)
    output[0::2] = torch.where(low < 8, low, low - 16)
    output[1::2] = torch.where(high < 8, high, high - 16)
    return output


def w4_msd_product(
    codes, integer, scales, correction, activation_scale, output_dtype=torch.bfloat16
):
    scale_bits = scales.contiguous().view(torch.int32) & -8192
    scales = scale_bits.view(torch.float32)
    high = torch.div(codes, 16, rounding_mode="floor")
    low = torch.bitwise_and(codes, 15) - 8
    high_product = high @ integer.T
    low_product = low @ integer.T
    high_scaled = (high_product.float() * scales).half().float()
    low_scaled = (low_product.float() * scales).half().float()
    result = high_scaled * 16 + low_scaled + correction
    return (result * activation_scale).to(output_dtype)


def tp_sum(partials):
    if len(partials) != TP_SIZE:
        raise ValueError("TP16 reduction requires sixteen partials")
    dtype = partials[0].dtype
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("TP partials must use BF16 or FP32")
    total = torch.zeros_like(partials[0], dtype=torch.float32)
    for partial in partials:
        if partial.device.type != "cpu" or partial.dtype != dtype:
            raise ValueError("TP partials must share one CPU dtype")
        if partial.shape != total.shape:
            raise ValueError("TP partial shapes differ")
        total.add_(partial.float())
    return total.to(dtype)


@dataclass(frozen=True)
class _IntegerWeight:
    values: torch.Tensor
    scales: torch.Tensor
    correction: torch.Tensor | None
    bits: int

    @property
    def shape(self):
        return self.values.shape[0] * (8 // self.bits), self.values.shape[1]


class ReferenceWeights:

    def __init__(
        self,
        directory,
        expert_cache_size=8,
        dense_cache_bytes=256 * 1024**2,
        *,
        fp32_head=False,
    ):
        if sys.byteorder != "little":
            raise ValueError("The raw reader requires a little-endian host")
        if expert_cache_size < 0 or dense_cache_bytes < 0:
            raise ValueError("Cache limits cannot be negative")
        self.directory = Path(directory)
        filenames = (
            "config.json",
            "quant_model_description.json",
            "quant_model_weights.safetensors.index.json",
        )
        documents = [self.directory.joinpath(name).read_bytes() for name in filenames]
        self.config, self.quantization, index = map(json.loads, documents)
        self.weight_map = index["weight_map"]
        if self.quantization.get("group_size") != 0:
            raise ValueError("The reference requires per-channel weights")
        head_recipe = self.quantization.get("lm_head.weight", "FLOAT")
        if fp32_head and head_recipe != "FLOAT":
            raise ValueError("FP32 head requires unquantized head weights")
        self.fp32_head = fp32_head
        self.contract = {
            "tp_size": TP_SIZE,
            "activation_quantization": (
                "BF16: calibrated A3 Div(127,amax), FP32 Mul, RINT; "
                "scale=amax*FP32(1/127)"
            ),
            "a8_source_sha256": (
                "fd0fb30d924b48ed4455cb505d2d530e"
                "822fa48d78b526db663d0bee69ffaefc"
            ),
            "integer_projection": (
                "W8: INT32 dot * FP32 weight scale * FP32 A8 scale; "
                "W4: CANN 9.2.0 API-dequant MSD with FP16 partials "
                "and Fixpipe scale truncated to 1+8+10 bits"
            ),
            "projection_output": "BF16 before TP reduction",
            "tp_sum": "FP32 rank-order sum, final BF16; not bitwise HCCL",
            "w4_scale_bias": "Checkpoint correction added once after 16H+L",
            "w4_msd_source_sha256": (
                "a8b7c446d5373f5f50d3ee22bdf6631a"
                "8468973da420791bb97ccf0e2f351fbb"
            ),
            "c8_quantization": "128-channel NPU operation order uncalibrated",
            "fused_rounding": (
                "W4 FP16 partials; activation boundaries set by HF adapters"
            ),
            "metadata_sha256": {
                name: hashlib.sha256(data).hexdigest()
                for name, data in zip(filenames, documents)
            },
        }
        self.expert_cache_size = expert_cache_size
        self.dense_cache_bytes = dense_cache_bytes
        self._active_layer = None
        self._dense = OrderedDict()
        self._dense_bytes = 0
        self._experts = OrderedDict()
        self._headers = OrderedDict()

    def clear(self):
        self._dense.clear()
        self._experts.clear()
        self._dense_bytes = 0

    def _cache_for(self, name, rows, columns):
        parts = name.split(".")
        if len(parts) < 3 or parts[:2] != ["model", "layers"]:
            return None
        layer = int(parts[2])
        if layer != self._active_layer:
            self.clear()
            self._active_layer = layer
        if "experts" not in parts:
            return self._dense
        if rows is not None or columns is not None or not self.expert_cache_size:
            return None
        expert = int(parts[parts.index("experts") + 1])
        cache = self._experts.pop(expert, {})
        self._experts[expert] = cache
        while len(self._experts) > self.expert_cache_size:
            self._experts.popitem(last=False)
        return cache

    def _metadata(self, name):
        filename = self.weight_map[name]
        cached = self._headers.pop(filename, None)
        if cached is None:
            with self.directory.joinpath(filename).open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                cached = 8 + size, json.loads(stream.read(size))
        self._headers[filename] = cached
        if len(self._headers) > 4:
            self._headers.popitem(last=False)
        base, header = cached
        return filename, base, header[name]

    def tensor(self, name, rows=None, columns=None, dtype=None):
        cache = self._cache_for(name, rows, columns)
        key = name, rows, columns
        value = None if cache is None else cache.get(key)
        if value is None:
            value = self._read_tensor(name, rows, columns)
            if cache is self._dense:
                size = value.numel() * value.element_size()
                if size <= self.dense_cache_bytes:
                    while self._dense_bytes + size > self.dense_cache_bytes:
                        _, old = self._dense.popitem(last=False)
                        self._dense_bytes -= old.numel() * old.element_size()
                    self._dense[key] = value
                    self._dense_bytes += size
            elif cache is not None:
                cache[key] = value
        elif cache is self._dense:
            self._dense.move_to_end(key)
        return value if dtype is None else value.to(dtype)

    def _read_tensor(self, name, rows, columns):
        filename, base, metadata = self._metadata(name)
        shape = metadata["shape"]
        if len(shape) not in (1, 2):
            raise ValueError("The GLM reference reads vectors and matrices")
        dtype, width = STORAGE_DTYPES[metadata["dtype"]]
        row_start, row_end = rows or (0, shape[0])
        row_width = shape[1] if len(shape) == 2 else 1
        col_start, col_end = columns or (0, row_width)
        if not (
            0 <= row_start < row_end <= shape[0]
            and 0 <= col_start < col_end <= row_width
        ):
            raise ValueError(f"Invalid tensor slice for {name}")
        offset, end = metadata["data_offsets"]
        if end - offset != shape[0] * row_width * width:
            raise ValueError(f"Unexpected storage size for {name}")
        payload = bytearray()
        with self.directory.joinpath(filename).open("rb") as stream:
            if col_start == 0 and col_end == row_width:
                stream.seek(base + offset + row_start * row_width * width)
                payload.extend(stream.read((row_end - row_start) * row_width * width))
            else:
                for row in range(row_start, row_end):
                    stream.seek(
                        base + offset + (row * row_width + col_start) * width
                    )
                    payload.extend(stream.read((col_end - col_start) * width))
        count = (row_end - row_start) * (col_end - col_start)
        if len(payload) != count * width:
            raise ValueError(f"Truncated tensor payload for {name}")
        value = torch.frombuffer(payload, dtype=dtype)
        return (
            value
            if len(shape) == 1
            else value.reshape(row_end - row_start, col_end - col_start)
        )

    def _integer_weight(self, prefix):
        recipe = self.quantization[prefix + ".weight"]
        if recipe not in ("W4A8_DYNAMIC", "W8A8_DYNAMIC"):
            raise ValueError(f"Unsupported integer recipe: {recipe}")
        values = self.tensor(prefix + ".weight")
        scales = self.tensor(prefix + ".weight_scale").flatten()
        offsets = self.tensor(prefix + ".weight_offset")
        if torch.count_nonzero(offsets):
            raise ValueError(f"Nonzero weight offsets: {prefix}")
        bits = 4 if recipe == "W4A8_DYNAMIC" else 8
        correction = self.tensor(prefix + ".scale_bias") if bits == 4 else None
        weight = _IntegerWeight(values, scales, correction, bits)
        if values.dtype != torch.int8 or scales.dtype != torch.float32:
            raise ValueError(f"Unexpected integer storage dtype: {prefix}")
        if scales.numel() != weight.shape[0]:
            raise ValueError(f"Per-channel scale shape mismatch: {prefix}")
        if weight.shape != self._expected_shape(prefix):
            raise ValueError(f"Weight dimensions differ from config: {prefix}")
        return weight

    def _expected_shape(self, prefix):
        config = self.config
        hidden = config["hidden_size"]
        if ".mlp." in prefix:
            key = (
                "moe_intermediate_size"
                if ".experts." in prefix or ".shared_experts." in prefix
                else "intermediate_size"
            )
            intermediate = config[key]
            return (
                (hidden, intermediate)
                if prefix.endswith("down_proj")
                else (intermediate, hidden)
            )
        if ".indexer." in prefix:
            return (
                config["index_n_heads"] * config["index_head_dim"],
                config["q_lora_rank"],
            )
        dimensions = {
            "q_a_proj": (config["q_lora_rank"], hidden),
            "kv_a_proj_with_mqa": (
                config["kv_lora_rank"] + config["qk_rope_head_dim"],
                hidden,
            ),
            "q_b_proj": (
                config["num_attention_heads"]
                * (config["qk_nope_head_dim"] + config["qk_rope_head_dim"]),
                config["q_lora_rank"],
            ),
            "o_proj": (
                hidden,
                config["num_attention_heads"] * config["v_head_dim"],
            ),
        }
        return dimensions[prefix.rsplit(".", 1)[1]]

    @staticmethod
    def _integer_product(
        weight, codes, activation_scale, rows, columns, output_dtype=torch.bfloat16
    ):
        start, stop = rows
        left, right = columns
        packing = 8 // weight.bits
        if start % packing or stop % packing:
            raise ValueError("W4 output slices must contain complete pairs")
        packed = weight.values[start // packing : stop // packing, left:right]
        integer = unpack_w4(packed) if weight.bits == 4 else packed
        scales = weight.scales[start:stop]
        if weight.correction is not None:
            partitions = weight.correction.shape[1]
            stride = weight.shape[1] // partitions
            if left % stride or right % stride:
                raise ValueError("Input slice splits an MSD correction group")
            actual = weight.correction[start:stop, left // stride : right // stride]
            expected = integer.to(torch.int32).sum(dim=1).float() * scales * 8
            if not torch.allclose(actual.sum(dim=1), expected, rtol=1e-5, atol=1e-8):
                raise ValueError("W4 scale_bias does not match signed weights")
        flat_codes = codes.reshape(-1, codes.shape[-1]).to(torch.int32)
        integer = integer.to(torch.int32)
        if weight.bits == 4:
            result = w4_msd_product(
                flat_codes,
                integer,
                scales,
                actual.sum(dim=1),
                activation_scale.reshape(-1, 1),
                output_dtype,
            )
            return result.reshape(*codes.shape[:-1], stop - start)
        accumulator = flat_codes @ integer.T
        accumulator = accumulator.reshape(*codes.shape[:-1], stop - start)
        result = accumulator.float() * scales
        return (result * activation_scale[..., None]).to(output_dtype)

    def fused_expert_partials(self, prefix, x, ranks=range(TP_SIZE)):
        self._check_input(x)
        codes, input_scale = quantize_dynamic_a8(x)
        gate = self._integer_weight(prefix + ".gate_proj")
        up = self._integer_weight(prefix + ".up_proj")
        down = self._integer_weight(prefix + ".down_proj")
        width = gate.shape[0] // TP_SIZE
        partials = []
        for rank in ranks:
            channels = rank * width, (rank + 1) * width
            gate_output = self._integer_product(
                gate, codes, input_scale, channels, (0, gate.shape[1]), torch.float32
            )
            up_output = self._integer_product(
                up, codes, input_scale, channels, (0, up.shape[1]), torch.float32
            )
            activated = functional.silu(gate_output) * up_output
            maximum = activated.abs().amax(-1)
            scale = torch.div(maximum, torch.full_like(maximum, 127))
            factor = torch.div(torch.ones_like(scale), scale)
            quantized = (activated * factor[..., None]).half().round()
            output = self._integer_product(
                down, quantized.to(torch.int8), scale, (0, down.shape[0]), channels
            )
            partials.append(output)
        return tuple(partials)

    def _floating_product(self, prefix, x, rows=None, columns=None):
        fp32_head = self.fp32_head and prefix == "lm_head"
        if fp32_head and x.dtype != torch.bfloat16:
            raise ValueError("FP32 head requires BF16 hidden input")
        weight = self.tensor(prefix + ".weight", rows, columns, x.dtype)
        output = x.float() @ weight.float().T
        if fp32_head:
            return output
        return output.to(x.dtype)

    def input_partials(self, prefix, x):
        self._check_input(x)
        if x.shape[-1] % TP_SIZE:
            raise ValueError("Input width must be divisible by sixteen")
        width = x.shape[-1] // TP_SIZE
        recipe = self.quantization.get(prefix + ".weight", "FLOAT")
        floating = recipe == "FLOAT"
        weight = None if floating else self._integer_weight(prefix)
        if not floating and x.dtype != torch.bfloat16:
            raise ValueError("Quantized projections require BF16 input")
        partials = []
        for rank in range(TP_SIZE):
            columns = rank * width, (rank + 1) * width
            local = x[..., columns[0] : columns[1]]
            if floating:
                output = self._floating_product(prefix, local, columns=columns)
            else:
                if weight.shape[1] != x.shape[-1]:
                    raise ValueError("Input width differs from checkpoint")
                codes, scale = quantize_dynamic_a8(local)
                output = self._integer_product(
                    weight, codes, scale, (0, weight.shape[0]), columns
                )
            partials.append(output)
        return tuple(partials)

    @staticmethod
    def _check_input(x):
        if x.device.type != "cpu" or x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Linear reference input must be CPU BF16 or FP32")
        if x.ndim < 2:
            raise ValueError("Linear reference input needs a token dimension")

    def linear(self, prefix, x, partition):
        self._check_input(x)
        if partition == "input":
            return tp_sum(self.input_partials(prefix, x))
        if partition not in ("replicated", "output"):
            raise ValueError("Partition must be replicated, output, or input")
        recipe = self.quantization.get(prefix + ".weight", "FLOAT")
        floating = recipe == "FLOAT"
        weight = None if floating else self._integer_weight(prefix)
        if not floating and x.dtype != torch.bfloat16:
            raise ValueError("Quantized projections require BF16 input")
        shape = self._metadata(prefix + ".weight")[2]["shape"]
        output_width = shape[0] if floating else weight.shape[0]
        shards = TP_SIZE if partition == "output" or prefix == "lm_head" else 1
        if output_width % shards:
            raise ValueError("Output width is not divisible by shard count")
        width = output_width // shards
        if not floating:
            if x.shape[-1] != weight.shape[1]:
                raise ValueError("Input width differs from checkpoint")
            codes, scale = quantize_dynamic_a8(x)
        outputs = []
        for rank in range(shards):
            rows = rank * width, (rank + 1) * width
            if floating:
                output = self._floating_product(prefix, x, rows=rows)
            else:
                output = self._integer_product(
                    weight, codes, scale, rows, (0, weight.shape[1])
                )
            outputs.append(output)
        return torch.cat(outputs, dim=-1)


def project_codes(weights, prefix, codes, token_scale, *, scale_order):
    if (
        codes.device.type != "cpu"
        or codes.dtype != torch.int8
        or codes.ndim != 2
        or token_scale.dtype != torch.float32
        or token_scale.device.type != "cpu"
        or token_scale.numel() != codes.shape[0]
    ):
        raise ValueError("W8 projection needs CPU INT8 rows and FP32 scales")
    if weights.quantization[prefix + ".weight"] != "W8A8_DYNAMIC":
        raise ValueError("MLA Prolog reference requires W8A8_DYNAMIC")
    integer = weights.tensor(prefix + ".weight")
    channel_scale = weights.tensor(prefix + ".weight_scale").reshape(-1)
    offset = weights.tensor(prefix + ".weight_offset")
    if (
        integer.dtype != torch.int8
        or integer.ndim != 2
        or integer.shape[1] != codes.shape[1]
        or channel_scale.dtype != torch.float32
        or channel_scale.numel() != integer.shape[0]
        or bool(torch.count_nonzero(offset))
    ):
        raise ValueError("Expected symmetric W8 with FP32 channel scales")
    product = (codes.to(torch.int32) @ integer.to(torch.int32).T).float()
    scale = token_scale.reshape(-1, 1)
    if scale_order == "token_first":
        return (product * scale) * channel_scale
    if scale_order == "channel_first":
        return (product * channel_scale) * scale
    raise ValueError("Scale order must be token_first or channel_first")


def rms_fp32(value, gamma):
    if (
        value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim != 2
        or value.shape[1] % 64
        or gamma.device.type != "cpu"
        or gamma.dtype != torch.bfloat16
        or gamma.shape != (value.shape[1],)
    ):
        raise ValueError("RMS needs FP32 rows and their BF16 gamma")
    squares = value.square().reshape(value.shape[0], -1, 64)
    folded = squares[:, 0].clone()
    for block in range(1, squares.shape[1]):
        folded = folded + squares[:, block]
    reciprocal = torch.tensor(1 / value.shape[1], dtype=torch.float32)
    mean = folded.sum(dim=-1, keepdim=True) * reciprocal
    return torch.div(value, (mean + EPSILON).sqrt()) * gamma.float()


def quantize_prolog(value):
    if value.device.type != "cpu" or value.dtype != torch.float32 or value.ndim != 2:
        raise ValueError("Prolog query quantization needs CPU FP32 rows")
    maximum = value.abs().amax(dim=-1)
    denominator = torch.where(maximum > 0, maximum, 1.0)
    full = torch.full_like(maximum, 127)
    factor = torch.div(full, denominator)
    scale = torch.div(maximum, full)
    codes = torch.round(value * factor[:, None]).clamp(-127, 127)
    return codes.to(torch.int8), scale


def rotary_half(value, cosine, sine):
    if (
        value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim != 3
        or value.shape[-1] != ROPE
    ):
        raise ValueError("RoPE needs CPU FP32 [tokens,heads,64]")
    expected = value.shape[0], ROPE
    for angles in (cosine, sine):
        if (
            angles.shape != expected
            or angles.dtype != torch.bfloat16
            or angles.device.type != "cpu"
        ):
            raise ValueError("RoPE angles must be CPU BF16 [tokens,64]")
    cosine = cosine[:, None, : ROPE // 2].float()
    sine = sine[:, None, : ROPE // 2].float()
    even, odd = value[..., 0::2], value[..., 1::2]
    real = even * cosine + (-odd) * sine
    imaginary = odd * cosine + even * sine
    return torch.cat((real, imaginary), dim=-1).to(torch.bfloat16)


def preprocess(weights, prefix, hidden, cos, sin):
    if (
        hidden.device.type != "cpu"
        or hidden.dtype != torch.bfloat16
        or hidden.ndim != 2
        or hidden.shape[1] != HIDDEN
    ):
        raise ValueError("Prolog input must be CPU BF16 [tokens,6144]")
    codes, scale = quantize_dynamic_a8(hidden)
    query = project_codes(
        weights, prefix + ".q_a_proj", codes, scale, scale_order="token_first"
    )
    kv = project_codes(
        weights, prefix + ".kv_a_proj_with_mqa", codes, scale, scale_order="token_first"
    )
    q_norm = rms_fp32(
        query, weights.tensor(prefix + ".q_a_layernorm.weight", dtype=torch.bfloat16)
    )
    latent = rms_fp32(
        kv[:, :KV_LORA],
        weights.tensor(prefix + ".kv_a_layernorm.weight", dtype=torch.bfloat16),
    ).to(torch.bfloat16)
    q_codes, q_scale = quantize_prolog(q_norm)
    expanded = project_codes(
        weights, prefix + ".q_b_proj", q_codes, q_scale, scale_order="token_first"
    )
    expanded = expanded.reshape(hidden.shape[0], HEADS, NOPE + ROPE)
    return {
        "q_nope": expanded[..., :NOPE].to(torch.bfloat16),
        "q_rope": rotary_half(expanded[..., NOPE:], cos, sin),
        "latent": latent,
        "k_rope": rotary_half(kv[:, None, KV_LORA:], cos, sin),
        "q_norm": q_norm,
        "q_codes": q_codes,
        "q_scale": q_scale,
    }


def index_query(weights, prefix, q_norm):
    if q_norm.ndim != 2 or q_norm.shape[1] != Q_LORA:
        raise ValueError("Indexer query input must have 2048 channels")
    codes, scale = quantize_prolog(q_norm)
    projected = project_codes(
        weights, prefix + ".wq_b", codes, scale, scale_order="channel_first"
    )
    return projected.to(torch.bfloat16).reshape(-1, INDEX_HEADS, INDEX_DIM)


class StableTopk(TorchFunctionMode):

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func not in (torch.topk, torch.Tensor.topk):
            return func(*args, **kwargs)
        scores = args[0] if args else kwargs["input"]
        count = args[1] if len(args) > 1 else kwargs["k"]
        dimension = kwargs.get("dim", args[2] if len(args) > 2 else -1)
        largest = kwargs.get("largest", args[3] if len(args) > 3 else True)
        indices = scores.argsort(dim=dimension, descending=largest, stable=True).narrow(
            dimension, 0, count
        )
        return torch.return_types.topk((scores.gather(dimension, indices), indices))


class ReferenceLinear(nn.Module):
    def __init__(self, weights, prefix, partition="replicated"):
        super().__init__()
        self.weights = weights
        self.prefix = prefix
        self.partition = partition

    def forward(self, x):
        return self.weights.linear(self.prefix, x, partition=self.partition)


class ReferenceEmbedding(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, tokens):
        if tokens.shape != (1, 1):
            raise ValueError("The reference consumes one token at a time")
        token = int(tokens.item())
        row = self.weights.tensor(
            "model.embed_tokens.weight", rows=(token, token + 1), dtype=torch.bfloat16
        )
        return row.unsqueeze(0)


def fused_swiglu(gate, up):
    return (functional.silu(gate.float()) * up.float()).to(torch.bfloat16)


def ffn_partials(weights, prefix, hidden):
    gate = weights.linear(prefix + ".gate_proj", hidden, partition="output")
    up = weights.linear(prefix + ".up_proj", hidden, partition="output")
    activated = fused_swiglu(gate, up)
    return weights.input_partials(prefix + ".down_proj", activated)


class ReferenceMlp(nn.Module):
    def __init__(self, weights, prefix):
        super().__init__()
        self.weights = weights
        self.prefix = prefix

    def forward(self, hidden):
        return tp_sum(ffn_partials(self.weights, self.prefix, hidden))


class ReferenceMoe(nn.Module):

    def __init__(self, weights, prefix, moe):
        super().__init__()
        self.weights = weights
        self.prefix = prefix
        self.gate = moe.gate
        self.route = getattr(type(moe), "route_tokens_to_experts", None)
        if self.route is not None:
            for name in (
                "n_routed_experts",
                "n_group",
                "topk_group",
                "norm_topk_prob",
                "top_k",
            ):
                setattr(self, name, getattr(moe, name))
        self.output_scale = 1.0
        self.fused_quantization = False
        self.last_route = None

    @property
    def routed_scaling_factor(self):
        return self.gate.routed_scaling_factor

    def forward(self, hidden):
        with StableTopk():
            router_output = self.gate(hidden)
            if self.route is None:
                _, probabilities, choices = router_output
            else:
                choices, probabilities = self.route(self, router_output)
        self.last_route = choices[0].detach().clone()
        probabilities = probabilities[0].to(torch.bfloat16)
        flat = hidden.reshape(1, -1)
        expert_partials = []
        for expert in choices[0].tolist():
            prefix = f"{self.prefix}.experts.{expert}"
            values = (
                self.weights.fused_expert_partials(prefix, flat)
                if self.fused_quantization
                else ffn_partials(self.weights, prefix, flat)
            )
            expert_partials.append(values)
        shared = ffn_partials(self.weights, self.prefix + ".shared_experts", flat)
        rank_outputs = []
        for rank in range(TP_SIZE):
            products = [
                values[rank].float() * probability.float()
                for values, probability in zip(expert_partials, probabilities)
            ]
            combined = torch.stack(products).sum(dim=0).to(torch.bfloat16)
            if self.output_scale != 1.0:
                combined = combined * self.output_scale
            rank_outputs.append(combined + shared[rank])
        return tp_sum(rank_outputs).reshape_as(hidden)


def rotary_components(x, cosine, sine, head_axis):
    half = x.shape[-1] // 2
    cosine = cosine[..., :half].float().unsqueeze(head_axis)
    sine = sine[..., :half].float().unsqueeze(head_axis)
    even, odd = x[..., 0::2].float(), x[..., 1::2].float()
    return even * cosine + (-odd) * sine, odd * cosine + even * sine


def interleaved_rope(x, cosine, sine):
    first, second = rotary_components(x, cosine, sine, head_axis=2)
    return torch.stack((first, second), dim=-1).flatten(-2).to(x.dtype)


def attention_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    query = rotary_components(q, cos, sin, unsqueeze_dim)
    key = rotary_components(k, cos, sin, unsqueeze_dim)
    return (torch.cat(query, dim=-1).to(q.dtype), torch.cat(key, dim=-1).to(k.dtype))


def install_attention_rope(attention):
    original = attention.forward.__func__
    namespace = dict(original.__globals__)
    namespace["apply_rotary_pos_emb_interleave"] = attention_rope
    adapted = FunctionType(
        original.__code__,
        namespace,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    adapted.__kwdefaults__ = original.__kwdefaults__
    attention.forward = MethodType(adapted, attention)


def prolog_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    position_ids=None,
    prev_topk_indices=None,
    **kwargs,
):
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        eager_attention_forward,
    )

    batch, length, hidden = hidden_states.shape
    cosine, sine = position_embeddings
    prepared = preprocess(
        self._reference_weights,
        self._reference_prefix,
        hidden_states.reshape(-1, hidden),
        cosine.reshape(-1, cosine.shape[-1]),
        sine.reshape(-1, sine.shape[-1]),
    )
    self._prolog = prepared
    query = torch.cat((prepared["q_nope"], prepared["q_rope"]), dim=-1)
    query = query.view(batch, length, self.num_heads, -1).transpose(1, 2)
    expanded = self.kv_b_proj(prepared["latent"].view(batch, length, -1))
    expanded = expanded.view(batch, length, self.num_heads, -1).transpose(1, 2)
    key, value = expanded.split((self.qk_nope_head_dim, self.v_head_dim), -1)
    rope = prepared["k_rope"].view(batch, 1, length, self.qk_rope_head_dim)
    key = torch.cat((key, rope.expand(*key.shape[:-1], -1)), dim=-1)
    if past_key_values is not None:
        key, value = past_key_values.update(key, value, self.layer_idx)
    if self.indexer is not None:
        mask = attention_mask[:, 0] if attention_mask is not None else None
        indices = self.indexer(
            hidden_states,
            prepared["q_norm"].view(batch, length, -1),
            position_embeddings,
            mask,
            position_ids,
            past_key_values=past_key_values,
        )
    else:
        if prev_topk_indices is None:
            raise ValueError("Shared DSA requires the preceding index choices")
        indices = prev_topk_indices
    excluded = indices.new_ones((batch, length, key.shape[2]), dtype=torch.bool)
    excluded.scatter_(-1, indices.long(), False)
    excluded = excluded.unsqueeze(1)
    if attention_mask is None:
        positions = torch.arange(key.shape[2], device=hidden_states.device)
        excluded = excluded | (
            positions[None, None, None] > position_ids[:, None, :, None]
        )
        attention_mask = hidden_states.new_zeros((batch, 1, length, key.shape[2]))
    attention_mask = attention_mask.masked_fill(
        excluded, torch.finfo(hidden_states.dtype).min
    )
    output, weights = eager_attention_forward(
        self, query, key, value, attention_mask, dropout=0.0, scaling=self.scaling, **kwargs
    )
    output = output.reshape(batch, length, -1).contiguous()
    return self.o_proj(output), weights, indices


def prolog_input_norm(self, hidden):
    values = hidden.float()
    variance = values.square().mean(-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + self.variance_epsilon)
    return (normalized * self.weight.float()).to(hidden.dtype)


class PrologIndexerQuery(nn.Module):
    def __init__(self, weights, prefix):
        super().__init__()
        self.weights = weights
        self.prefix = prefix

    def forward(self, normalized):
        return index_query(
            self.weights, self.prefix, normalized.reshape(-1, normalized.shape[-1])
        )


def hadamard_matrix():
    signs = [
        [1 if (row & column).bit_count() % 2 == 0 else -1 for column in range(128)]
        for row in range(128)
    ]
    return torch.tensor(signs, dtype=torch.bfloat16) / math.sqrt(128)


def prefix_length(prefixes):
    if len(prefixes) != LAYERS:
        raise ValueError("The reference needs 78 backbone prefixes")
    length = prefixes[0]["latent"].shape[0]
    for layer, entry in enumerate(prefixes):
        expected = {
            "latent": ((length, 512), torch.bfloat16),
            "rope": ((length, 64), torch.bfloat16),
        }
        full = layer < 3 or (layer - 2) % 4 == 0
        if full:
            quantized = 6 <= layer <= 66
            dtype = torch.int8 if quantized else torch.bfloat16
            expected["index_keys"] = (length, 128), dtype
            if quantized:
                expected["index_scales"] = (length,), torch.float16
        if entry.keys() != expected.keys():
            raise ValueError(f"Wrong prefix regions at layer {layer}")
        for name, (shape, dtype) in expected.items():
            value = entry[name]
            if (
                tuple(value.shape) != shape
                or value.dtype != dtype
                or value.device.type != "cpu"
            ):
                raise ValueError(f"Invalid prefix {layer}.{name}")
    return length


class ReferenceIndexer(nn.Module):

    def __init__(self, weights, prefix, layer, quantized):
        super().__init__()
        self.weights = weights
        self.prefix = prefix
        self.layer_idx = layer
        self.quantized = quantized
        self.rotation = hadamard_matrix() if quantized else None
        self.scales = None
        self.last_key = None
        self.last_scale = None
        self.last_scores = None
        self.query_projection = ReferenceLinear(weights, prefix + ".wq_b")

    def reset(self):
        self.scales = None
        self.last_key = None
        self.last_scale = None
        self.last_scores = None

    def encode(self, x):
        if not self.quantized:
            return x, None
        encoded, scale = quantize_a8(x @ self.rotation)
        return encoded, scale.to(torch.float16)

    def inject(self, cache, keys, scales):
        expected = torch.int8 if self.quantized else torch.bfloat16
        if keys.dtype != expected or keys.ndim != 2 or keys.shape[1] != 128:
            raise ValueError("Invalid encoded index prefix")
        cache.update_indexer(keys.unsqueeze(0), self.layer_idx)
        if self.quantized:
            if (
                scales is None
                or scales.dtype != torch.float16
                or scales.numel() != keys.shape[0]
            ):
                raise ValueError("C8 prefix needs one FP16 scale per key")
            self.scales = scales.reshape(-1).clone()

    def forward(
        self,
        hidden_states,
        q_resid,
        position_embeddings,
        attention_mask,
        position_ids,
        past_key_values=None,
    ):
        if hidden_states.shape[:2] != (1, 1) or past_key_values is None:
            raise ValueError("The reference indexer requires cached M=1")
        query = self.query_projection(q_resid).reshape(1, 1, 32, 128)
        key = self.weights.linear(
            self.prefix + ".wk", hidden_states, partition="replicated"
        )
        gamma = self.weights.tensor(self.prefix + ".k_norm.weight")
        beta = self.weights.tensor(self.prefix + ".k_norm.bias")
        key = (
            functional.layer_norm(
                key.float(), (128,), gamma.float(), beta.float(), eps=1e-6
            )
            .to(torch.bfloat16)
            .unsqueeze(2)
        )
        cosine, sine = position_embeddings
        query = torch.cat(
            (interleaved_rope(query[..., :64], cosine, sine), query[..., 64:]), dim=-1
        ).reshape(32, 128)
        key = torch.cat(
            (interleaved_rope(key[..., :64], cosine, sine), key[..., 64:]), dim=-1
        ).reshape(1, 128)
        encoded_key, key_scale = self.encode(key)
        keys = past_key_values.update_indexer(encoded_key.unsqueeze(0), self.layer_idx)[0]
        self.last_key = encoded_key[0].clone()
        self.last_scale = None if key_scale is None else key_scale.reshape(1)
        query, query_scale = self.encode(query)
        weights = self.weights.linear(
            self.prefix + ".weights_proj", hidden_states, partition="replicated"
        ).reshape(32)
        if self.quantized:
            scale = key_scale.reshape(-1)
            self.scales = (
                scale if self.scales is None else torch.cat((self.scales, scale))
            )
            rounded_weights = weights.to(torch.float16)
            domain = cann_arch22_input_domain(
                query_scale.reshape(32).tolist(),
                self.scales.tolist(),
                rounded_weights.tolist(),
            )
            if not domain["accepted"]:
                raise ValueError(
                    "Unverified CANN FP16 scoring boundary: "
                    + json.dumps(domain, sort_keys=True)
                )
            scores = cann_arch22_index_scores(
                query, keys, query_scale.reshape(32), self.scales, rounded_weights
            )
        else:
            scores = query.float() @ keys.float().T
            scores = (scores.relu() * weights.float()[:, None]).sum(dim=0)
        if attention_mask is not None:
            scores = scores + attention_mask.reshape(-1).float()
        else:
            scores = scores.masked_fill(
                torch.arange(keys.shape[0]) > position_ids.item(), -torch.inf
            )
        self.last_scores = scores.detach().clone()
        indices = scores.argsort(descending=True, stable=True)[:2048]
        return indices.to(torch.int32).reshape(1, 1, -1)


class HfReference:

    def __init__(
        self,
        checkpoint,
        mla_preprocess="native",
        layer_count=LAYERS,
        *,
        fp32_head=False,
        c8_source_directory=None,
    ):
        from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaRotaryEmbedding,
        )

        if mla_preprocess not in ("native", "prolog_v3"):
            raise ValueError("MLA preprocess must be native or prolog_v3")
        if type(layer_count) is not int or not 1 <= layer_count <= LAYERS:
            raise ValueError("The reference layer count must be in [1, 78]")
        self.mla_preprocess = mla_preprocess
        self.weights = ReferenceWeights(checkpoint, fp32_head=fp32_head)
        self.weights.contract["rotary"] = dict(ROTARY_CONTRACT)
        if c8_source_directory is None:
            c8_source_directory = CANN_ARCH22_SOURCE_DIRECTORY
        self.weights.contract["c8_index_scoring"] = cann_arch22_scoring_contract(
            Path(c8_source_directory).resolve()
        )
        raw = json.loads((self.weights.directory / "config.json").read_text())
        config = GlmMoeDsaConfig.from_dict(raw)
        self.weights.contract["lm_head"] = {
            "input_dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "accumulator_dtype": "float32",
            "projection_output_dtype": "float32" if fp32_head else "bfloat16",
            "tensor_parallel_size": TP_SIZE,
            "vocabulary_range": [0, config.vocab_size],
        }
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
        if (
            config.num_hidden_layers != LAYERS
            or config.n_group != 1
            or config.topk_group != 1
            or config.n_shared_experts != 1
            or config.rms_norm_eps != 1e-5
        ):
            raise ValueError("The reference requires the GLM-5.2 TP16 layout")
        config.num_hidden_layers = layer_count
        config.layer_types = config.layer_types[:layer_count]
        with torch.device("meta"):
            self.model = GlmMoeDsaForCausalLM(config).to(torch.bfloat16)
        self.config = config
        self._install_adapters()
        if mla_preprocess == "prolog_v3":
            self._install_prolog()
        with torch.device("cpu"):
            self.model.model.rotary_emb = GlmMoeDsaRotaryEmbedding(config=config)
        self.model.eval().requires_grad_(False)
        remaining = [
            name
            for name, value in (
                *self.model.named_parameters(),
                *self.model.named_buffers(),
            )
            if value.is_meta
        ]
        if remaining:
            raise ValueError(f"Unreplaced HF meta tensors: {remaining}")
        self._attach_trace_hooks()
        self.reset()

    def _install_prolog(self):
        self.weights.contract["mla_preprocess"] = {
            **CONTRACT,
            "input_norm": "Layer0 FP32 norm and gamma; one BF16 output",
        }
        self.weights.contract["expert_combine"] = (
            "Unscaled BF16 probabilities; FP32 weighted sum; BF16 output; "
            "BF16 routed scale; BF16 shared add; FP32 TP sum; BF16 output"
        )
        self.weights.contract["fused_expert_quantization"] = (
            "FP32 GMM1 and SwiGLU; scale=amax/127; reciprocal scale; "
            "FP16 then INT8 nearest-even; BF16 GMM2"
        )
        norm = self.model.model.layers[0].input_layernorm
        norm.forward = MethodType(prolog_input_norm, norm)
        for number, layer in enumerate(self.model.model.layers):
            if isinstance(layer.mlp, ReferenceMoe):
                layer.mlp.output_scale = layer.mlp.gate.routed_scaling_factor
                layer.mlp.gate.routed_scaling_factor = 1.0
                layer.mlp.fused_quantization = True
            attention = layer.self_attn
            prefix = f"model.layers.{number}.self_attn"
            attention._reference_weights = self.weights
            attention._reference_prefix = prefix
            attention.forward = MethodType(prolog_attention_forward, attention)
            if attention.indexer is not None:
                attention.indexer.query_projection = PrologIndexerQuery(
                    self.weights, prefix + ".indexer"
                )

    def _norm(self, module, prefix):
        module.weight = nn.Parameter(
            self.weights.tensor(prefix + ".weight", dtype=torch.bfloat16),
            requires_grad=False,
        )
        module.variance_epsilon = 1e-5

    def _install_adapters(self):
        self.model.model.embed_tokens = ReferenceEmbedding(self.weights)
        self.model.lm_head = ReferenceLinear(
            self.weights, "lm_head", partition="output"
        )
        self._norm(self.model.model.norm, "model.norm")
        for number, layer in enumerate(self.model.model.layers):
            install_attention_rope(layer.self_attn)
            prefix = f"model.layers.{number}"
            attention = prefix + ".self_attn"
            self._norm(layer.input_layernorm, prefix + ".input_layernorm")
            self._norm(
                layer.post_attention_layernorm, prefix + ".post_attention_layernorm"
            )
            self._norm(layer.self_attn.q_a_layernorm, attention + ".q_a_layernorm")
            self._norm(layer.self_attn.kv_a_layernorm, attention + ".kv_a_layernorm")
            for name, partition in (
                ("q_a_proj", "replicated"),
                ("q_b_proj", "output"),
                ("kv_a_proj_with_mqa", "replicated"),
                ("kv_b_proj", "output"),
                ("o_proj", "input"),
            ):
                setattr(
                    layer.self_attn,
                    name,
                    ReferenceLinear(self.weights, attention + "." + name, partition),
                )
            if layer.self_attn.indexer is not None:
                index = attention + ".indexer"
                recipe = self.weights.quantization.get(index + ".quant_type")
                layer.self_attn.indexer = ReferenceIndexer(
                    self.weights, index, number, recipe == "INT8_DYNAMIC"
                )
            mlp = prefix + ".mlp"
            if hasattr(layer.mlp, "gate"):
                gate = layer.mlp.gate
                gate.weight = nn.Parameter(
                    self.weights.tensor(mlp + ".gate.weight", dtype=torch.float32),
                    requires_grad=False,
                )
                gate.e_score_correction_bias = self.weights.tensor(
                    mlp + ".gate.e_score_correction_bias", dtype=torch.float32
                )
                layer.mlp = ReferenceMoe(self.weights, mlp, layer.mlp)
            else:
                layer.mlp = ReferenceMlp(self.weights, mlp)

    def _attach_trace_hooks(self):
        def record_last_merged(module, inputs):
            if inputs[0].ndim == 3:
                self.last_merged_hidden = inputs[0][:, -1].detach().clone()

        self.model.model.norm.register_forward_pre_hook(record_last_merged)
        for number, layer in enumerate(self.model.model.layers):
            def record_attention(module, inputs, output, index=number):
                self.indices[index] = output[2][0, -1].detach().clone()
                self.attention_weights[index] = output[1][0, :, -1, :].detach().clone()
                if module.indexer is not None:
                    self.index_scores[index] = module.indexer.last_scores
                if self.mla_preprocess == "prolog_v3":
                    self.latents[index] = module._prolog["latent"][0].clone()

            def record_latent(module, inputs, output, index=number):
                self.latents[index] = output[0, -1].detach().clone()

            def record_route(module, inputs, output, index=number):
                self.routes[index] = module.last_route.clone()

            layer.self_attn.register_forward_hook(record_attention)
            layer.self_attn.kv_a_layernorm.register_forward_hook(record_latent)
            if hasattr(layer.mlp, "gate"):
                layer.mlp.register_forward_hook(record_route)

    def reset(self):
        from transformers.cache_utils import DynamicCache

        self.cache = DynamicCache(config=self.config)
        for layer in self.model.model.layers:
            if layer.self_attn.indexer is not None:
                layer.self_attn.indexer.reset()

    @property
    def position(self):
        return self.cache.get_seq_length()

    @torch.backends.mkldnn.flags(
        enabled=False, deterministic=None, allow_tf32=None, fp32_precision=None
    )
    def inject(self, prefixes):
        prefix_length(prefixes)
        if self.position:
            raise ValueError("Inject prefixes into an empty reference")
        for number, (layer, entry) in enumerate(zip(self.model.model.layers, prefixes)):
            latent, rope = entry["latent"], entry["rope"]
            expanded = layer.self_attn.kv_b_proj(latent).reshape(-1, 64, 448)
            key, value = expanded.split((192, 256), dim=-1)
            rope = rope[:, None, :].expand(-1, 64, 64)
            key = torch.cat((key, rope), dim=-1).transpose(0, 1).unsqueeze(0)
            value = value.transpose(0, 1).unsqueeze(0)
            self.cache.update(key.contiguous(), value.contiguous(), number)
            if layer.self_attn.indexer is not None:
                layer.self_attn.indexer.inject(
                    self.cache, entry["index_keys"], entry.get("index_scales")
                )

    @torch.inference_mode()
    @torch.backends.mkldnn.flags(
        enabled=False, deterministic=None, allow_tf32=None, fp32_precision=None
    )
    def step(self, token):
        layer_count = len(self.model.model.layers)
        self.indices, self.routes, self.latents = (
            [None] * layer_count for _ in range(3)
        )
        self.index_scores = [None] * layer_count
        self.attention_weights = [None] * layer_count
        self.last_merged_hidden = None
        position = self.position
        output = self.model(
            input_ids=torch.tensor([[token]]),
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        self.cache = output.past_key_values
        hidden = tuple(value[:, -1].clone() for value in output.hidden_states)
        if self.last_merged_hidden is None:
            raise ValueError("HF final norm did not expose its merged input")
        logits = output.logits[:, -1].clone()
        layer_logits = (logits,)
        projected_hidden = hidden[1:-1]
        if layer_count < LAYERS:
            projected_hidden = (*projected_hidden, self.last_merged_hidden)
        if projected_hidden:
            intermediate = torch.cat(projected_hidden)
            normalized = (
                prolog_input_norm(self.model.model.norm, intermediate)
                if self.mla_preprocess == "prolog_v3"
                else self.model.model.norm(intermediate)
            )
            projected = self.model.lm_head(normalized)
            layer_logits = tuple(projected.split(1, dim=0))
            if layer_count == LAYERS:
                layer_logits = (*layer_logits, logits)
        rows = []
        for number, layer in enumerate(self.model.model.layers):
            indexer = layer.self_attn.indexer
            rows.append(
                {
                    "latent": self.latents[number],
                    "rope": self.cache.layers[number].keys[0, 0, -1, 192:].clone(),
                    "index": None if indexer is None else indexer.last_key.clone(),
                    "index_scale": (
                        None
                        if indexer is None or not indexer.quantized
                        else indexer.last_scale.clone()
                    ),
                }
            )
        return {
            "token": token,
            "position": position,
            "committed_position": self.position,
            "logits": logits,
            "hidden_states": hidden,
            "layer_logits": layer_logits,
            "last_merged_hidden": self.last_merged_hidden,
            "index_scores": tuple(self.index_scores),
            "attention_weights": tuple(self.attention_weights),
            "indices": tuple(self.indices),
            "routes": tuple(self.routes),
            "cache_rows": tuple(rows),
        }


def checkpoint_contract(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    description = directory / "quant_model_description.json"
    quant = json.loads(description.read_text())
    index = directory / "quant_model_weights.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    expected = {
        "num_hidden_layers": 78,
        "num_nextn_predict_layers": 1,
        "hidden_size": HIDDEN,
        "vocab_size": VOCAB,
        "num_attention_heads": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "n_group": 1,
        "topk_group": 1,
        "rms_norm_eps": 1e-5,
        "index_topk": TOP_K,
        "index_n_heads": 32,
        "index_head_dim": 128,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"MTP reference requires {name}={value}")
    if not quant.get("is_rot_used") or "compress_ratios" in config:
        raise ValueError("MTP requires rotation and uncompressed embeddings")
    recipe = quant.get(PREFIX + ".self_attn.indexer.quant_type")
    if recipe not in (None, "FLOAT"):
        raise ValueError("The MTP index cache must use BF16")
    shapes = {
        "rot.weight": [HIDDEN, HIDDEN],
        "model.embed_tokens.weight": [VOCAB, HIDDEN],
        PREFIX + ".enorm.weight": [HIDDEN],
        PREFIX + ".hnorm.weight": [HIDDEN],
        PREFIX + ".eh_proj.weight": [HIDDEN, 2 * HIDDEN],
        PREFIX + ".shared_head.norm.weight": [HIDDEN],
        PREFIX + ".shared_head.head.weight": [VOCAB, HIDDEN],
    }
    headers, tensors = {}, {}
    for name, shape in shapes.items():
        filename = weight_map[name]
        if filename not in headers:
            with (directory / filename).open("rb") as source:
                size = struct.unpack("<Q", source.read(8))[0]
                headers[filename] = json.loads(source.read(size))
        entry = headers[filename][name]
        if entry["shape"] != shape or entry["dtype"] != "BF16":
            raise ValueError(f"Unexpected MTP tensor metadata: {name}")
        tensors[name] = {"file": filename, "shape": shape, "dtype": "BF16"}
    return {
        "layer": LAYER,
        "hf_layer_index": 0,
        "tp_size": 16,
        "position_convention": "parent",
        "epsilon": 1e-5,
        "embedding": "model.embed_tokens; PP1 without compression",
        "head": "layer-78 shared_head.head, independently read",
        "index_cache_dtype": "BF16",
        "tensors": tensors,
        "arithmetic": "HF expanded BF16 KV; independent CPU TP16 adapters",
    }


def _shared_indexer(weights):
    class SharedIndexer(ReferenceIndexer):
        def __init__(self):
            super().__init__(weights, PREFIX + ".self_attn.indexer", 0, False)
            self.external_indices = None

        def forward(self, *args, **kwargs):
            selected = super().forward(*args, **kwargs)
            if self.external_indices is None:
                return selected
            return self.external_indices.reshape(1, 1, -1)

    return SharedIndexer()


class MtpReference:

    def __init__(self, checkpoint, capacity=9216):
        from transformers import GlmMoeDsaConfig
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaDecoderLayer,
            GlmMoeDsaRMSNorm,
            GlmMoeDsaRotaryEmbedding,
        )

        if type(capacity) is not int or capacity < 1:
            raise ValueError("Reference capacity must be positive")
        self.capacity = capacity
        self.contract = checkpoint_contract(checkpoint)
        self.weights = ReferenceWeights(checkpoint)
        raw = dict(self.weights.config)
        raw.update(
            num_hidden_layers=1,
            first_k_dense_replace=0,
            mlp_layer_types=["sparse"],
            indexer_types=["full"],
            layer_types=["deepseek_sparse_attention"],
        )
        self.config = GlmMoeDsaConfig.from_dict(raw)
        self.config._attn_implementation = "eager"
        self.config._experts_implementation = "eager"
        with torch.device("meta"):
            self.decoder = GlmMoeDsaDecoderLayer(self.config, 0).to(torch.bfloat16)
        self._install_adapters()
        self.embedding = ReferenceEmbedding(self.weights)
        self.rot = ReferenceLinear(self.weights, "rot")
        self.eh_proj = ReferenceLinear(self.weights, PREFIX + ".eh_proj")
        self.head = ReferenceLinear(
            self.weights, PREFIX + ".shared_head.head", "output"
        )
        for name, prefix in (
            ("enorm", PREFIX + ".enorm"),
            ("hnorm", PREFIX + ".hnorm"),
            ("final_norm", PREFIX + ".shared_head.norm"),
        ):
            norm = GlmMoeDsaRMSNorm(HIDDEN, eps=1e-5)
            self._install_norm(norm, prefix)
            setattr(self, name, norm)
        self.rotary = GlmMoeDsaRotaryEmbedding(self.config, device="cpu")
        self.decoder.eval().requires_grad_(False)
        remaining = [
            name
            for name, value in (
                *self.decoder.named_parameters(),
                *self.decoder.named_buffers(),
            )
            if value.is_meta
        ]
        if remaining:
            raise ValueError(f"Unreplaced HF meta tensors: {remaining}")
        self._attach_hooks()
        self.reset()

    def _install_norm(self, module, prefix):
        module.weight = nn.Parameter(
            self.weights.tensor(prefix + ".weight", dtype=torch.bfloat16),
            requires_grad=False,
        )
        module.variance_epsilon = 1e-5

    def _install_adapters(self):
        layer = self.decoder
        install_attention_rope(layer.self_attn)
        self.weights.contract["rotary"] = dict(ROTARY_CONTRACT)
        attention = PREFIX + ".self_attn"
        for module, prefix in (
            (layer.input_layernorm, PREFIX + ".input_layernorm"),
            (layer.post_attention_layernorm, PREFIX + ".post_attention_layernorm"),
            (layer.self_attn.q_a_layernorm, attention + ".q_a_layernorm"),
            (layer.self_attn.kv_a_layernorm, attention + ".kv_a_layernorm"),
        ):
            self._install_norm(module, prefix)
        for name, partition in (
            ("q_a_proj", "replicated"),
            ("q_b_proj", "output"),
            ("kv_a_proj_with_mqa", "replicated"),
            ("kv_b_proj", "output"),
            ("o_proj", "input"),
        ):
            setattr(
                layer.self_attn,
                name,
                ReferenceLinear(self.weights, attention + "." + name, partition),
            )
        layer.self_attn.indexer = _shared_indexer(self.weights)
        gate = layer.mlp.gate
        gate.weight = nn.Parameter(
            self.weights.tensor(PREFIX + ".mlp.gate.weight", dtype=torch.float32),
            requires_grad=False,
        )
        gate.e_score_correction_bias = self.weights.tensor(
            PREFIX + ".mlp.gate.e_score_correction_bias", dtype=torch.float32
        )
        layer.mlp = ReferenceMoe(self.weights, PREFIX + ".mlp", gate)

    def _attach_hooks(self):
        def attention(module, inputs, output):
            self._stages["attention"] = output[0].reshape(1, HIDDEN).clone()

        def post_attention(module, inputs):
            value = inputs[0].reshape(1, HIDDEN).clone()
            self._stages["post_attention"] = value

        def ffn(module, inputs, output):
            self._stages["ffn"] = output.reshape(1, HIDDEN).clone()

        def latent(module, inputs, output):
            self._latent = output.reshape(1, 512).clone()

        def routes(module, inputs, output):
            self._routes = output[2].reshape(1, 8).clone()

        self.decoder.self_attn.register_forward_hook(attention)
        self.decoder.post_attention_layernorm.register_forward_pre_hook(post_attention)
        self.decoder.mlp.register_forward_hook(ffn)
        self.decoder.self_attn.kv_a_layernorm.register_forward_hook(latent)
        self.decoder.mlp.gate.register_forward_hook(routes)

    @property
    def position(self):
        return self.cache.get_seq_length()

    def reset(self):
        from transformers.cache_utils import DynamicCache

        self.cache = DynamicCache(config=self.config)
        self.decoder.self_attn.indexer.reset()
        self.decoder.self_attn.indexer.external_indices = None

    def check_lengths(self, expected):
        if len(self.cache.layers) != 1 or self.position != expected:
            raise ValueError("HF MLA length differs from parent position")
        keys = self.cache.layers[0].indexer_keys
        length = 0 if keys is None else keys.shape[1]
        if length != expected:
            raise ValueError("HF index cache length differs from MLA cache")

    def inject(self, prefix):
        if self.position:
            raise ValueError("Inject a prefix into an empty MTP reference")
        dimensions = {"latent": 512, "rope": 64, "index_keys": 128}
        if set(prefix) != set(dimensions):
            raise ValueError("MTP prefix requires latent, rope and index_keys")
        length = prefix["latent"].shape[0]
        if length > self.capacity:
            raise ValueError("MTP prefix exceeds capacity")
        for name, width in dimensions.items():
            value = prefix[name]
            if (
                tuple(value.shape) != (length, width)
                or value.dtype != torch.bfloat16
                or value.device.type != "cpu"
            ):
                raise ValueError(f"Invalid MTP prefix {name}")
        if length:
            expanded = self.decoder.self_attn.kv_b_proj(prefix["latent"])
            key, value = expanded.reshape(length, 64, 448).split((192, 256), dim=-1)
            rope = prefix["rope"][:, None, :].expand(-1, 64, 64)
            key = torch.cat((key, rope), dim=-1).transpose(0, 1).unsqueeze(0)
            value = value.transpose(0, 1).unsqueeze(0)
            self.cache.update(key.contiguous(), value.contiguous(), 0)
            self.decoder.self_attn.indexer.inject(
                self.cache, prefix["index_keys"], None
            )
        self.check_lengths(length)

    def _validate(self, tokens, parent, external_indices):
        if not isinstance(tokens, (tuple, list)) or len(tokens) not in (1, 2, 4):
            raise ValueError("MTP tokens must contain M=1/2/4 entries")
        if any(type(token) is not int or not 0 <= token < VOCAB for token in tokens):
            raise ValueError("MTP token is outside the vocabulary")
        rows = len(tokens)
        if (
            tuple(parent.shape) != (rows, HIDDEN)
            or parent.dtype != torch.bfloat16
            or parent.device.type != "cpu"
        ):
            raise ValueError("MTP parents must be CPU BF16 [M,6144]")
        if self.position + rows > self.capacity:
            raise ValueError("MTP rows exceed capacity")
        if external_indices is None:
            return [None] * rows
        if (
            tuple(external_indices.shape) != (rows, TOP_K)
            or external_indices.dtype not in (torch.int32, torch.int64)
            or external_indices.device.type != "cpu"
        ):
            raise ValueError("Shared indices must be CPU INT32/INT64 [M,2048]")
        selections = []
        for row, indices in enumerate(external_indices.tolist()):
            count = indices.index(-1) if -1 in indices else TOP_K
            valid, padding = indices[:count], indices[count:]
            bound = self.position + row + 1
            if (
                not 1 <= count <= min(bound, TOP_K)
                or len(set(valid)) != count
                or any(index < 0 or index >= bound for index in valid)
                or any(index != -1 for index in padding)
            ):
                raise ValueError("Shared indices violate the causal prefix")
            selections.append(external_indices[row, :count].to(torch.int32))
        return selections

    def _step(self, token, parent, external_indices):
        position = self.position
        embedding = self.embedding(torch.tensor([[token]], dtype=torch.int64))
        if position == 0:
            embedding = torch.zeros_like(embedding)
        rotated = self.rot(parent.reshape(1, 1, HIDDEN))
        enorm, hnorm = self.enorm(embedding), self.hnorm(rotated)
        hidden = self.eh_proj(torch.cat((enorm, hnorm), dim=-1))
        self._stages = {
            name: value.reshape(1, HIDDEN).clone()
            for name, value in (
                ("masked_embedding", embedding),
                ("rotated_parent", rotated),
                ("enorm", enorm),
                ("hnorm", hnorm),
                ("eh_proj", hidden),
            )
        }
        positions = torch.tensor([[position]], dtype=torch.int64)
        indexer = self.decoder.self_attn.indexer
        indexer.external_indices = external_indices
        try:
            pre_norm, selected = self.decoder(
                hidden,
                position_ids=positions,
                past_key_values=self.cache,
                use_cache=True,
                attention_mask=None,
                position_embeddings=self.rotary(hidden, positions),
            )
        finally:
            indexer.external_indices = None
        recycle = self.final_norm(pre_norm)
        logits = self.head(recycle).reshape(1, VOCAB)
        indices = torch.full((1, TOP_K), -1, dtype=torch.int32)
        indices[:, : selected.shape[-1]] = selected.reshape(1, -1)
        self.check_lengths(position + 1)
        return {
            "logits": logits,
            "pre_norm_hidden": pre_norm.reshape(1, HIDDEN),
            "recycle_hidden": recycle.reshape(1, HIDDEN),
            "indices": indices,
            "routes": self._routes,
            "stages": self._stages,
            "cache": {
                "latent": self._latent,
                "rope": self.cache.layers[0].keys[0, 0, -1, 192:].reshape(1, 64).clone(),
                "index_keys": indexer.last_key.reshape(1, 128).clone(),
            },
        }

    def forward(self, tokens, normalized_parent_hidden, external_indices=None):
        selections = self._validate(tokens, normalized_parent_hidden, external_indices)
        start = self.position
        self.check_lengths(start)
        with torch.inference_mode():
            rows = [
                self._step(
                    token, normalized_parent_hidden[row : row + 1], selections[row]
                )
                for row, token in enumerate(tokens)
            ]
            result = {
                name: torch.cat([row[name] for row in rows], dim=0).clone()
                for name in (
                    "logits",
                    "pre_norm_hidden",
                    "recycle_hidden",
                    "indices",
                    "routes",
                )
            }
            for group in ("stages", "cache"):
                result[group] = {
                    name: torch.cat([row[group][name] for row in rows]).clone()
                    for name in rows[0][group]
                }
        result.update(position=start, committed_position=self.position)
        return result
