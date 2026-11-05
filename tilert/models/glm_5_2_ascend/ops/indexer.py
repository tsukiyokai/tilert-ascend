"""Batch-one paged index caches and causal CANN selection on A3."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


INDEX_DIM = 128
INDEX_HEADS = 32
PAGE_SIZE = 128
TOP_K = 2048
VERIFY_ROWS = (1, 2, 4)
CANN_ARCH22_SOURCE_DIRECTORY = (
    "/opt/ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/"
    "ops_transformer/ascendc/quant_lightning_indexer/op_kernel/arch22"
)
CANN_ARCH22_SOURCE_SHA256 = {
    "quant_lightning_indexer_service_cube_arch22.h": (
        "75b676cd308c2b8c573798f9a5ec16636122"
        "af85cb626e59dcefbe8b01147252"
    ),
    "quant_lightning_indexer_service_vector_arch22.h": (
        "55098dd4589260d526c3ede3decf6d0757"
        "fc8458c9a74253849a63dfa6ace1ae"
    ),
}


def cann_arch22_scoring_contract(source_directory=None):
    """Describe captured CANN source and optionally verify identical files."""
    contract = {
        "implementation": "CANN-9.2.0-QuantLightningIndexer-arch22",
        "source_sha256": dict(CANN_ARCH22_SOURCE_SHA256),
        "source_verified": False,
        "reference_module": "tilert.models.glm_5_2_ascend.ops.indexer",
        "reference_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "formula": (
            "FP32_MATMUL(FP16(weight*q_scale), "
            "FP16(ReLU(INT32(QK)/1024))) * FP32(k_scale)"
        ),
        "scale_restoration": "none; scores retain the common 1/1024 factor",
        "cpu_fp16": "IEEE nearest-even with subnormals",
        "device_validation": "pending",
        "unverified_device_boundaries": [
            "FP16 subnormal handling and overflow saturation",
            "FP32 Cube accumulation order",
        ],
        "accepted_fp16_domain": (
            "finite normal values or zero for wire operands and weight*q_scale"
        ),
    }
    if source_directory is not None:
        directory = Path(source_directory)
        observed = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in CANN_ARCH22_SOURCE_SHA256
        }
        if observed != CANN_ARCH22_SOURCE_SHA256:
            raise ValueError("CANN arch22 scoring source hashes differ")
        contract["source_verified"] = True
        contract["source_directory"] = str(directory)
    return contract


def cann_arch22_input_domain(query_scale, key_scale, weights):
    """Return accepted=False for unverified FP16 scalar inputs or products."""
    if len(query_scale) != INDEX_HEADS or len(weights) != INDEX_HEADS:
        raise ValueError("C8 query scales and weights need 32 entries")
    regions = {
        "query_scale": query_scale, "key_scale": key_scale,
        "weights": weights,
        "weight_times_query_scale": [
            float(weight) * float(scale)
            for weight, scale in zip(weights, query_scale)
        ],
    }
    counts = {}
    for name, values in regions.items():
        counts[name] = {
            "nonfinite": sum(not math.isfinite(value) for value in values),
            "subnormal_or_underflow": sum(
                0 < abs(value) < 2**-14 for value in values
            ),
            "overflow": sum(
                math.isfinite(value) and abs(value) > 65504
                for value in values
            ),
        }
    negative_scales = sum(value < 0 for value in (*query_scale, *key_scale))
    return {
        "accepted": not negative_scales and not any(
            any(region.values()) for region in counts.values()
        ),
        "negative_scales": negative_scales, "regions": counts,
    }


def cache_append_range(length, count, capacity):
    """Return the cache slice for one supported query chunk."""
    if type(count) is not int or count not in VERIFY_ROWS:
        raise ValueError("query row count must be 1, 2, or 4")
    if not 0 <= length <= capacity or length + count > capacity:
        raise ValueError("cache capacity exceeded")
    return slice(length, length + count)


def validate_truncate(length, valid_length):
    """Reject cache growth and noninteger rollback positions."""
    if type(valid_length) is not int or not 0 <= valid_length <= length:
        raise ValueError("valid_length must be an integer in [0, length]")


@dataclass(frozen=True)
class VerifyLayout:
    """One TND request whose query rows are the final cached positions.

    CANN receives query lengths [M] and PA key lengths [P + M]. With
    sparse_mode=3, query row i can read keys strictly below P + i + 1.
    """

    query_rows: int
    cache_length: int

    def __post_init__(self):
        if (
            type(self.query_rows) is not int
            or self.query_rows not in VERIFY_ROWS
        ):
            raise ValueError("query row count must be 1, 2, or 4")
        if type(self.cache_length) is not int or (
            self.cache_length < self.query_rows
        ):
            raise ValueError("cache must contain every query row")

    @property
    def key_bounds(self):
        start = self.cache_length - self.query_rows + 1
        return tuple(range(start, self.cache_length + 1))

    @property
    def valid_counts(self):
        return tuple(min(bound, TOP_K) for bound in self.key_bounds)

    @property
    def sparse_shape(self):
        return self.query_rows, 1, TOP_K

    def validate_indices(self, rows, allow_partial=False):
        """Require unique causal indices followed by exactly -1 padding."""
        if len(rows) != self.query_rows:
            raise ValueError("indices must contain one row per query")
        for row, bound, count in zip(
            rows, self.key_bounds, self.valid_counts
        ):
            if len(row) != TOP_K or any(
                type(index) is not int for index in row
            ):
                raise ValueError("each index row must contain 2048 integers")
            if allow_partial:
                used = next(
                    (slot for slot, index in enumerate(row) if index == -1),
                    TOP_K,
                )
                if not 1 <= used <= count:
                    raise ValueError("partial indices exceed the causal size")
                count = used
            selected = row[:count]
            if len(set(selected)) != count or any(
                index < 0 or index >= bound for index in selected
            ):
                raise ValueError("indices must be unique causal positions")
            if any(index != -1 for index in row[count:]):
                raise ValueError("unused index slots must contain -1")


def hadamard128() -> torch.Tensor:
    import torch

    matrix = torch.ones(1, 1, dtype=torch.bfloat16)
    while matrix.shape[0] < INDEX_DIM:
        upper = torch.cat((matrix, matrix), dim=1)
        lower = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((upper, lower), dim=0)
    return matrix / math.sqrt(INDEX_DIM)


def int8_index_scores(query, keys, query_scale, key_scale, weights):
    """Ideal dequantized score, without CANN's intermediate FP16 rounding."""
    import torch

    products = query.to(torch.int32) @ keys.to(torch.int32).T
    scores = products.float() * query_scale.float()[:, None]
    scores = scores * key_scale.float()[None, :]
    return (scores.relu() * weights.float()[:, None]).sum(dim=0)


def cann_arch22_index_scores(query, keys, query_scale, key_scale, weights):
    """CPU source oracle for INT8 QLI with FP16 wire scales and weights.

    The captured arch22 kernel rounds weight*q_scale and ReLU(QK/1024)
    to FP16 before its FP32 head reduction, then multiplies key_scale.
    This matches its source golden; device accumulation and exceptional
    FP16 values require the strict same-input device comparison. Accepting
    a result requires cann_arch22_input_domain(...)["accepted"] to be True.
    """
    import torch

    tensors = (query, keys, query_scale, key_scale, weights)
    if any(value.device.type != "cpu" for value in tensors):
        raise ValueError("The CANN scoring oracle requires CPU tensors")
    if query.dtype != torch.int8 or keys.dtype != torch.int8:
        raise ValueError("The CANN scoring oracle requires INT8 Q and K")
    if any(value.dtype != torch.float16 for value in tensors[2:]):
        raise ValueError("CANN wire scales and weights must be FP16")
    if (query.shape != (INDEX_HEADS, INDEX_DIM) or keys.ndim != 2
            or keys.shape[1] != INDEX_DIM
            or query_scale.shape != (INDEX_HEADS,)
            or key_scale.shape != (keys.shape[0],)
            or weights.shape != (INDEX_HEADS,)):
        raise ValueError("Expected Q[32,128], K[L,128], scales[32]/[L]")
    products = query.to(torch.int32) @ keys.to(torch.int32).T
    positive = (products.float() / 1024.0).clamp_min(0).to(torch.float16)
    weighted_scale = (weights.float() * query_scale.float()).to(torch.float16)
    reduced = weighted_scale.float().unsqueeze(0) @ positive.float()
    return reduced.reshape(-1) * key_scale.float()


class NativeIndexer:
    """Own a replicated 128-dimensional index cache for one request.

    Inputs to encode/select have already received indexer RoPE. The caller
    passes the raw BF16 head-weight projection, matching vLLM-Ascend's
    conversion to the selector dtype before scoring.
    INT8 cache rows use BF16 Hadamard rotation and one FP16 scale. BF16 cache
    rows use their original basis. Shared layers reuse the returned indices.
    """

    def __init__(self, capacity=9216, key_dtype=None, device="npu:0"):
        import torch

        if key_dtype is None:
            key_dtype = torch.int8
        if type(capacity) is not int or capacity < 1:
            raise ValueError("index cache capacity must be positive")
        if key_dtype not in (torch.int8, torch.bfloat16):
            raise ValueError("index cache dtype must be INT8 or BF16")
        self.capacity = capacity
        self.key_dtype = key_dtype
        self.device = torch.device(device)
        self.length = 0
        pages = math.ceil(capacity / PAGE_SIZE)
        self.keys = torch.empty(
            (pages, PAGE_SIZE, 1, INDEX_DIM), dtype=key_dtype, device=device
        )
        self.scales = None
        self.hadamard = None
        if key_dtype == torch.int8:
            self.scales = torch.empty(
                (pages, PAGE_SIZE, 1), dtype=torch.float16, device=device
            )
            self.hadamard = hadamard128().to(device)
        self.block_table = torch.arange(
            pages, dtype=torch.int32, device=device
        ).reshape(1, -1)
        self.query_length = torch.ones(1, dtype=torch.int32, device=device)
        self.key_length = torch.zeros(1, dtype=torch.int32, device=device)

    def encode(self, rows):
        """Encode post-RoPE BF16 rows in the cache's storage format."""
        import torch
        import torch_npu

        if self.key_dtype == torch.bfloat16:
            return rows, None
        rotated = rows @ self.hadamard
        values, scale = torch_npu.npu_dynamic_quant(
            rotated, dst_type=torch.int8
        )
        return values, scale.to(torch.float16)

    def inject(self, keys, scales=None):
        """Install encoded prefix rows with shape [length,128]."""
        import torch

        if keys.ndim != 2 or keys.shape[1] != INDEX_DIM:
            raise ValueError("index prefix must have shape [length,128]")
        length = keys.shape[0]
        if length > self.capacity or keys.dtype != self.key_dtype:
            raise ValueError("index prefix exceeds capacity or dtype differs")
        if self.scales is not None and (
            scales is None or scales.dtype != torch.float16
            or scales.numel() != length
        ):
            raise ValueError("INT8 prefix requires one FP16 scale per key")
        self.keys.view(-1, INDEX_DIM)[:length].copy_(keys)
        if self.scales is not None:
            self.scales.view(-1)[:length].copy_(scales.reshape(-1))
        self.length = length
        self.key_length.fill_(length)

    def append(self, key):
        """Append one post-RoPE BF16 key, retaining the M=1 interface."""
        self.append_many(key.reshape(1, INDEX_DIM))

    def append_many(self, keys):
        """Append M=1/2/4 post-RoPE BF16 keys with shape [M,128]."""
        import torch

        if keys.ndim != 2 or keys.shape[1] != INDEX_DIM:
            raise ValueError("index keys must have shape [M,128]")
        if keys.dtype != torch.bfloat16 or keys.device != self.device:
            raise ValueError("index keys must be BF16 on the cache device")
        target = cache_append_range(
            self.length, keys.shape[0], self.capacity
        )
        values, scale = self.encode(keys)
        self.keys.view(-1, INDEX_DIM)[target].copy_(values)
        if self.scales is not None:
            self.scales.view(-1)[target].copy_(scale.reshape(-1))
        self.length = target.stop
        self.key_length.fill_(self.length)

    def truncate(self, valid_length):
        """Invalidate a suffix without clearing its underlying storage."""
        validate_truncate(self.length, valid_length)
        self.key_length.fill_(valid_length)
        self.length = valid_length

    def reset(self):
        self.truncate(0)

    def select(self, query, weights):
        """Select top-2048 positions for post-RoPE [32,128] query heads."""
        if query.shape != (INDEX_HEADS, INDEX_DIM):
            raise ValueError("index query must have shape [32,128]")
        if weights.shape != (INDEX_HEADS,):
            raise ValueError("index weights must have shape [32]")
        indices = self.select_many(query.unsqueeze(0), weights.unsqueeze(0))
        return indices[0, :min(self.length, TOP_K)]

    def select_many(self, query, weights):
        """Select [M,2048] causal indices for the last M cached positions.

        Query is BF16 [M,32,128], weights are [M,32], and M is 1/2/4.
        CANN returns valid indices first and fills unused slots with -1.
        """
        import torch
        import torch_npu

        if query.ndim != 3 or query.shape[1:] != (INDEX_HEADS, INDEX_DIM):
            raise ValueError("index queries must have shape [M,32,128]")
        layout = VerifyLayout(query.shape[0], self.length)
        if query.dtype != torch.bfloat16 or query.device != self.device:
            raise ValueError(
                "index queries must be BF16 on the cache device"
            )
        if weights.shape != (layout.query_rows, INDEX_HEADS) or (
            weights.device != self.device
        ):
            raise ValueError(
                "index weights must be [M,32] on the cache device"
            )
        values, scale = self.encode(query.reshape(-1, INDEX_DIM))
        self.query_length.fill_(layout.query_rows)
        self.key_length.fill_(layout.cache_length)
        arguments = dict(
            query=values.reshape(layout.query_rows, INDEX_HEADS, INDEX_DIM),
            key=self.keys,
            actual_seq_lengths_query=self.query_length,
            actual_seq_lengths_key=self.key_length,
            block_table=self.block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=TOP_K,
            sparse_mode=3,
        )
        if self.key_dtype == torch.int8:
            indices = torch_npu.npu_quant_lightning_indexer(
                **arguments,
                weights=weights.to(torch.float16),
                query_dequant_scale=scale.reshape(
                    layout.query_rows, INDEX_HEADS
                ),
                key_dequant_scale=self.scales,
                query_quant_mode=0,
                key_quant_mode=0,
            )
        else:
            indices, _ = torch_npu.npu_lightning_indexer(
                **arguments,
                weights=weights.to(torch.bfloat16),
            )
        if indices.shape != layout.sparse_shape or (
            indices.dtype != torch.int32
        ):
            raise RuntimeError("CANN index output must be INT32 [M,1,2048]")
        return indices[:, 0, :]
