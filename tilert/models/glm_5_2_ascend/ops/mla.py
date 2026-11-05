"""Rank-local BF16 compressed MLA through the existing A3 CANN operators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .indexer import VerifyLayout, cache_append_range, validate_truncate

if TYPE_CHECKING:
    import torch


HEADS = 4
LATENT_DIM = 512
NOPE_DIM = 192
ROPE_DIM = 64
VALUE_DIM = 256
PAGE_SIZE = 128
TOP_K = 2048


def _sparse_attention_operator():
    """Register the vLLM-Ascend CANN binding at the first device call."""
    import torch

    namespace = torch.ops._C_ascend
    if not hasattr(namespace, "npu_sparse_flash_attention"):
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError(
                "The A3 sparse flash attention binding is absent"
            )
    return namespace.npu_sparse_flash_attention


@dataclass(frozen=True)
class MlaTrace:
    projected_query: torch.Tensor
    latent_output: torch.Tensor
    output: torch.Tensor


class NativeMla:
    """One request, four TP16 heads, and separate paged latent/RoPE caches.

    kv_b_proj is BF16 [4 * (192 + 256), 512] in checkpoint row order.
    Cache inputs are normalized latent rows and post-RoPE key rows. Query
    RoPE is also applied by the caller. All computation uses NPU operators;
    there is no CPU execution fallback. attend() leaves the cache unchanged.
    """

    def __init__(
        self, kv_b_proj, capacity=9216, device="npu:0", scale=1 / 16,
        validate_indices_on_host=True,
    ):
        import torch
        import torch_npu

        expected = (HEADS * (NOPE_DIM + VALUE_DIM), LATENT_DIM)
        if kv_b_proj.shape != expected or kv_b_proj.dtype != torch.bfloat16:
            raise ValueError(f"kv_b_proj must be BF16 {expected}")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("MLA cache capacity must be positive")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("MLA scale must be finite and positive")
        if torch.device(device).type != "npu":
            raise ValueError("NativeMla requires an NPU device")
        # Register custom ops before projection/indexer kernels populate KV.
        _sparse_attention_operator()
        self.capacity = capacity
        self.length = 0
        self.scale = float(scale)
        self.validate_indices_on_host = bool(validate_indices_on_host)
        weight = torch_npu.npu_format_cast(
            kv_b_proj.to(device).clone(), 2
        ).reshape(HEADS, NOPE_DIM + VALUE_DIM, LATENT_DIM)
        self.key_weight = weight[:, :NOPE_DIM].contiguous()
        self.value_weight = weight[:, NOPE_DIM:].transpose(1, 2).contiguous()
        pages = math.ceil(capacity / PAGE_SIZE)
        self.latent_cache = torch.empty(
            (pages, PAGE_SIZE, 1, LATENT_DIM),
            dtype=torch.bfloat16, device=device,
        )
        self.rope_cache = torch.empty(
            (pages, PAGE_SIZE, 1, ROPE_DIM),
            dtype=torch.bfloat16, device=device,
        )
        self.device = self.latent_cache.device
        self.block_table = torch.arange(
            pages, dtype=torch.int32, device=device
        ).reshape(1, -1)
        self.query_length = torch.ones(1, dtype=torch.int32, device=device)
        self.key_length = torch.zeros(1, dtype=torch.int32, device=device)

    def _check_tensor(self, value, shape, name):
        import torch

        if value.shape != shape or value.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be BF16 {shape}")
        if value.device != self.device:
            raise ValueError(f"{name} must be on {self.device}")

    def inject(self, latent, rope):
        """Replace the valid prefix with BF16 [length,512] and [length,64]."""
        if latent.ndim != 2:
            raise ValueError("latent prefix must have shape [length,512]")
        length = latent.shape[0]
        self._check_tensor(latent, (length, LATENT_DIM), "latent prefix")
        self._check_tensor(rope, (length, ROPE_DIM), "RoPE prefix")
        if length > self.capacity:
            raise ValueError("MLA prefix exceeds cache capacity")
        self.latent_cache.view(-1, LATENT_DIM)[:length].copy_(latent)
        self.rope_cache.view(-1, ROPE_DIM)[:length].copy_(rope)
        self.key_length.fill_(length)
        self.length = length

    def append(self, latent, rope):
        """Append one normalized latent row and one post-RoPE key row."""
        self._check_tensor(latent, (1, LATENT_DIM), "latent row")
        self._check_tensor(rope, (1, ROPE_DIM), "RoPE row")
        self.append_many(latent, rope)

    def append_many(self, latent, rope):
        """Append M=1/2/4 normalized latent rows and post-RoPE key rows."""
        if latent.ndim != 2:
            raise ValueError("latent rows must have shape [M,512]")
        rows = latent.shape[0]
        self._check_tensor(latent, (rows, LATENT_DIM), "latent rows")
        self._check_tensor(rope, (rows, ROPE_DIM), "RoPE rows")
        target = cache_append_range(self.length, rows, self.capacity)
        self.latent_cache.view(-1, LATENT_DIM)[target].copy_(latent)
        self.rope_cache.view(-1, ROPE_DIM)[target].copy_(rope)
        self.length = target.stop
        self.key_length.fill_(self.length)

    def truncate(self, valid_length):
        """Invalidate a suffix without clearing its underlying storage."""
        validate_truncate(self.length, valid_length)
        self.key_length.fill_(valid_length)
        self.length = valid_length

    def reset(self):
        """Invalidate the prefix; subsequent attention requires new rows."""
        self.truncate(0)

    def _sparse_indices(self, indices, layout, allow_partial=False):
        import torch

        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("MLA indices must be INT32 or INT64")
        if indices.shape != (layout.query_rows, TOP_K):
            raise ValueError("MLA indices must have shape [M,2048]")
        # The reference path inspects CANN-generated addresses. The device
        # candidate retains them on the producer's current NPU stream.
        if self.validate_indices_on_host:
            values = indices.detach().cpu().tolist()
            layout.validate_indices(values, allow_partial=allow_partial)
        return indices.to(self.device, dtype=torch.int32).reshape(
            layout.sparse_shape
        ).contiguous()

    def trace(self, q_nope, q_rope, indices):
        """Return the two BF16 projection boundaries and the final output."""
        import torch

        layout = VerifyLayout(1, self.length)
        self._check_tensor(q_nope, (1, HEADS, NOPE_DIM), "q_nope")
        self._check_tensor(q_rope, (1, HEADS, ROPE_DIM), "q_rope")
        count = layout.valid_counts[0]
        if indices.ndim != 1 or indices.numel() != count:
            raise ValueError(f"MLA requires a vector of {count} indices")
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("MLA indices must be INT32 or INT64")
        padded = torch.full(
            (1, TOP_K), -1, dtype=torch.int64, device=self.device
        )
        padded[0, :count].copy_(indices)
        return self.trace_many(q_nope, q_rope, padded)

    def trace_many(self, q_nope, q_rope, indices, allow_partial=False):
        """Return BF16 projection boundaries for the last M cache positions."""
        import torch_npu

        if q_nope.ndim != 3:
            raise ValueError("q_nope must have shape [M,4,192]")
        rows = q_nope.shape[0]
        layout = VerifyLayout(rows, self.length)
        self._check_tensor(q_nope, (rows, HEADS, NOPE_DIM), "q_nope")
        self._check_tensor(q_rope, (rows, HEADS, ROPE_DIM), "q_rope")
        sparse = self._sparse_indices(indices, layout, allow_partial)
        self.query_length.fill_(rows)
        self.key_length.fill_(self.length)
        operation = _sparse_attention_operator()
        projected_query = torch_npu.npu_transpose_batchmatmul(
            q_nope.contiguous(), self.key_weight,
            perm_x1=(1, 0, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2),
        )
        latent_output, _, _ = operation(
            query=projected_query.contiguous(),
            key=self.latent_cache, value=self.latent_cache,
            sparse_indices=sparse, scale_value=self.scale,
            block_table=self.block_table,
            actual_seq_lengths_query=self.query_length,
            actual_seq_lengths_kv=self.key_length,
            query_rope=q_rope.contiguous(), key_rope=self.rope_cache,
            sparse_block_size=1, layout_query="TND", layout_kv="PA_BSND",
            sparse_mode=3, attention_mode=2, return_softmax_lse=False,
        )
        output = torch_npu.npu_transpose_batchmatmul(
            latent_output, self.value_weight,
            perm_x1=(1, 0, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2),
        ).reshape(rows, HEADS * VALUE_DIM)
        return MlaTrace(projected_query, latent_output, output)

    def attend(self, q_nope, q_rope, indices):
        """Return BF16 [1,1024] before the rank-local output projection."""
        return self.trace(q_nope, q_rope, indices).output

    def attend_many(self, q_nope, q_rope, indices, allow_partial=False):
        """Return BF16 [M,1024] for causal M=1/2/4 query rows.

        indices is [M,2048], with min(P+i+1,2048) distinct valid positions
        in row i followed by -1. The cache already includes all M queries.
        allow_partial permits a nonempty earlier MTP selection to be reused.
        """
        return self.trace_many(
            q_nope, q_rope, indices, allow_partial=allow_partial
        ).output
