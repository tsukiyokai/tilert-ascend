"""Independent CPU arithmetic for the TP16 GLM-5.2 ModelSlim checkpoint.

Raw safetensors reads bypass the production loader. W4 reproduces CANN
9.2.0 MSD's FP16 partial products and FP32 correction; W8 uses INT32 dots.
TP reduction sums BF16 partials in FP32 rank order and casts once to BF16;
this defines a reference and does not reproduce an HCCL reduction tree.
"""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import sys

import torch
import torch.nn.functional as functional


TP_SIZE = 16
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


def quantize_a8(x):
    """Retain the CPU Div(127,amax) formula for the C8 reference.

    Rounding is nearest-even. Zero rows produce zero codes and zero scales.
    The 128-channel C8 NPU quantization order is not yet calibrated; this
    function specifies the CPU formula rather than asserting C8 parity.
    """
    if x.device.type != "cpu":
        raise ValueError("The reference requires CPU tensors")
    values = x.to(torch.float32)
    maximum = torch.max(torch.abs(values), dim=-1).values
    scale = maximum * (1 / 127)
    denominator = torch.where(maximum > 0, maximum, 1.0)
    # Tensor division avoids Torch scalar rdiv's reciprocal/multiply lowering.
    factor = torch.div(torch.full_like(denominator, 127), denominator)
    codes = torch.round(torch.mul(values, factor.unsqueeze(-1)))
    return codes.clamp(-127, 127).to(torch.int8), scale


def quantize_dynamic_a8(x):
    """Quantize BF16 rows with the calibrated A3 FP32 Div factor.

    Positive amax has exponent [-120,127]; zero rows remain zero. Scale
    uses FP32 Muls(amax,1/127). Codes use the corrected factor, FP32 Mul
    and nearest-even RINT.
    """
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
    corrected = torch.isin(mantissa, torch.tensor(
        DYNAMIC_DIV_BIASED_MANTISSAS, dtype=torch.int32,
    ))
    factor_bits = factor.contiguous().view(torch.int32) - corrected.int()
    factor = factor_bits.view(torch.float32)
    codes = torch.round(values * factor.unsqueeze(-1))
    return codes.clamp(-127, 127).to(torch.int8), scale


def unpack_w4(packed):
    """Decode each byte into adjacent signed output rows, low nibble first."""
    if (packed.device.type != "cpu" or packed.dtype != torch.int8
            or packed.ndim != 2):
        raise ValueError("Packed W4 must be INT8 [output/2, input]")
    unsigned = packed.to(torch.int16).remainder(256)
    low = unsigned.remainder(16)
    high = torch.div(unsigned, 16, rounding_mode="floor")
    output = torch.empty(
        (packed.shape[0] * 2, packed.shape[1]), dtype=torch.int8
    )
    output[0::2] = torch.where(low < 8, low, low - 16)
    output[1::2] = torch.where(high < 8, high, high - 16)
    return output


def w4_msd_product(codes, integer, scales, correction, activation_scale,
                   output_dtype=torch.bfloat16):
    """Emulate CANN 9.2.0 API-dequant MSD with separate FP32 additions."""
    # A3 Fixpipe uses a 1+8+10-bit scale, truncating the low 13 FP32 bits.
    scale_bits = scales.contiguous().view(torch.int32) & -8192
    scales = scale_bits.view(torch.float32)
    high = torch.div(codes, 16, rounding_mode="floor")
    low = torch.bitwise_and(codes, 15) - 8
    high_product = high @ integer.T
    low_product = low @ integer.T
    high_scaled = (high_product.float() * scales).half().float()
    low_scaled = (low_product.float() * scales).half().float()
    result = high_scaled * 16
    result = result + low_scaled
    result = result + correction
    return (result * activation_scale).to(output_dtype)


def tp_sum(partials):
    """Sum in FP32 rank order, then restore BF16 or retain FP32 output."""
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
    """Lazy raw tensors with an active-layer byte limit and expert LRU.

    ``tensor`` accepts half-open row/column ranges and an optional dtype.
    ``linear`` supports replicated, output-concatenated, or input-reduced
    projections. ``input_partials`` exposes the sixteen rank-local results
    for MoE combine before TP reduction. Floating products accumulate in FP32
    and retain the input dtype, except for an explicitly selected FP32 head.
    Global tensors, including embedding/head slices, are never cached.
    """

    def __init__(self, directory, expert_cache_size=8,
                 dense_cache_bytes=256 * 1024**2, *, fp32_head=False):
        if sys.byteorder != "little":
            raise ValueError("The raw reader requires a little-endian host")
        if expert_cache_size < 0 or dense_cache_bytes < 0:
            raise ValueError("Cache limits cannot be negative")
        self.directory = Path(directory)
        filenames = ("config.json", "quant_model_description.json",
                     "quant_model_weights.safetensors.index.json")
        documents = [self.directory.joinpath(name).read_bytes()
                     for name in filenames]
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
                "scale=amax*FP32(1/127)"),
            "a8_source_sha256": (
                "fd0fb30d924b48ed4455cb505d2d530e"
                "822fa48d78b526db663d0bee69ffaefc"
            ),
            "integer_projection": (
                "W8: INT32 dot * FP32 weight scale * FP32 A8 scale; "
                "W4: CANN 9.2.0 API-dequant MSD with FP16 partials "
                "and Fixpipe scale truncated to 1+8+10 bits"),
            "projection_output": "BF16 before TP reduction",
            "tp_sum": "FP32 rank-order sum, final BF16; not bitwise HCCL",
            "w4_scale_bias": "Checkpoint correction added once after 16H+L",
            "w4_msd_source_sha256": (
                "a8b7c446d5373f5f50d3ee22bdf6631a"
                "8468973da420791bb97ccf0e2f351fbb"
            ),
            "c8_quantization": "128-channel NPU operation order uncalibrated",
            "fused_rounding": (
                "W4 FP16 partials; activation boundaries set by HF adapters"),
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
        """Release cached payloads and retain index/header metadata."""
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
        if (rows is not None or columns is not None
                or not self.expert_cache_size):
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
        """Read a vector/matrix slice directly from safetensors bytes."""
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
        if not (0 <= row_start < row_end <= shape[0]
                and 0 <= col_start < col_end <= row_width):
            raise ValueError(f"Invalid tensor slice for {name}")
        offset, end = metadata["data_offsets"]
        if end - offset != shape[0] * row_width * width:
            raise ValueError(f"Unexpected storage size for {name}")
        payload = bytearray()
        with self.directory.joinpath(filename).open("rb") as stream:
            if col_start == 0 and col_end == row_width:
                stream.seek(base + offset + row_start * row_width * width)
                payload.extend(stream.read((row_end - row_start)
                                           * row_width * width))
            else:
                for row in range(row_start, row_end):
                    stream.seek(base + offset + (row * row_width + col_start)
                                * width)
                    payload.extend(stream.read((col_end - col_start) * width))
        count = (row_end - row_start) * (col_end - col_start)
        if len(payload) != count * width:
            raise ValueError(f"Truncated tensor payload for {name}")
        value = torch.frombuffer(payload, dtype=dtype)
        return value if len(shape) == 1 else value.reshape(
            row_end - row_start, col_end - col_start
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
            key = ("moe_intermediate_size" if ".experts." in prefix
                   or ".shared_experts." in prefix else "intermediate_size")
            intermediate = config[key]
            return ((hidden, intermediate) if prefix.endswith("down_proj")
                    else (intermediate, hidden))
        if ".indexer." in prefix:
            return (config["index_n_heads"] * config["index_head_dim"],
                    config["q_lora_rank"])
        dimensions = {
            "q_a_proj": (config["q_lora_rank"], hidden),
            "kv_a_proj_with_mqa": (
                config["kv_lora_rank"] + config["qk_rope_head_dim"], hidden),
            "q_b_proj": (config["num_attention_heads"]
                         * (config["qk_nope_head_dim"]
                            + config["qk_rope_head_dim"]),
                         config["q_lora_rank"]),
            "o_proj": (hidden, config["num_attention_heads"]
                       * config["v_head_dim"]),
        }
        return dimensions[prefix.rsplit(".", 1)[1]]

    @staticmethod
    def _integer_product(weight, codes, activation_scale, rows, columns,
                         output_dtype=torch.bfloat16):
        start, stop = rows
        left, right = columns
        packing = 8 // weight.bits
        if start % packing or stop % packing:
            raise ValueError("W4 output slices must contain complete pairs")
        packed = weight.values[start // packing:stop // packing, left:right]
        integer = unpack_w4(packed) if weight.bits == 4 else packed
        scales = weight.scales[start:stop]
        if weight.correction is not None:
            partitions = weight.correction.shape[1]
            stride = weight.shape[1] // partitions
            if left % stride or right % stride:
                raise ValueError("Input slice splits an MSD correction group")
            actual = weight.correction[
                start:stop, left // stride:right // stride
            ]
            expected = integer.to(torch.int32).sum(dim=1).float() * scales * 8
            if not torch.allclose(actual.sum(dim=1), expected,
                                  rtol=1e-5, atol=1e-8):
                raise ValueError("W4 scale_bias does not match signed weights")
        flat_codes = codes.reshape(-1, codes.shape[-1]).to(torch.int32)
        integer = integer.to(torch.int32)
        if weight.bits == 4:
            result = w4_msd_product(
                flat_codes, integer, scales, actual.sum(dim=1),
                activation_scale.reshape(-1, 1), output_dtype,
            )
            return result.reshape(*codes.shape[:-1], stop - start)
        accumulator = flat_codes @ integer.T
        accumulator = accumulator.reshape(*codes.shape[:-1], stop - start)
        result = accumulator.float() * scales
        return (result * activation_scale[..., None]).to(output_dtype)

    def fused_expert_partials(self, prefix, x, ranks=range(TP_SIZE)):
        """CANN W4 GMM1/SwiGLU/A8 fusion, followed by BF16 GMM2.

        GMM1 and SwiGLU retain FP32. The fused quantizer uses amax/127,
        its reciprocal, then FP16 and INT8 nearest-even conversions.
        ``ranks`` selects the returned partials in logical TP order.
        """
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
                gate, codes, input_scale, channels, (0, gate.shape[1]),
                torch.float32,
            )
            up_output = self._integer_product(
                up, codes, input_scale, channels, (0, up.shape[1]),
                torch.float32,
            )
            activated = functional.silu(gate_output) * up_output
            maximum = activated.abs().amax(-1)
            scale = torch.div(maximum, torch.full_like(maximum, 127))
            factor = torch.div(torch.ones_like(scale), scale)
            quantized = (activated * factor[..., None]).half().round()
            output = self._integer_product(
                down, quantized.to(torch.int8), scale,
                (0, down.shape[0]), channels,
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
        """Apply local A8 quantization before each rank's input-shard dot."""
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
            local = x[..., columns[0]:columns[1]]
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
        if (x.device.type != "cpu"
                or x.dtype not in (torch.bfloat16, torch.float32)):
            raise ValueError("Linear reference input must be CPU BF16 or FP32")
        if x.ndim < 2:
            raise ValueError("Linear reference input needs a token dimension")

    def linear(self, prefix, x, partition):
        """Apply a projection with an explicit TP16 partition contract."""
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
