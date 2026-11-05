"""CANN attention for batch one with a contiguous BF16 MLA cache."""

from types import MethodType


def gathered_fia(
    impl, ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
    actual_seq_lengths_query, actual_seq_lengths_key, block_table=None,
):
    """Attend to 2048 valid causal rows per query, without another mask."""
    import torch_npu

    width = ql_nope.shape[0]
    indices = topk_indices.reshape(-1)
    latent = kv_cache[0].view(-1, 512).index_select(0, indices)
    rope = kv_cache[1].view(-1, 64).index_select(0, indices)
    key = latent.view(width, 1, 2048, 512)
    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        ql_nope.view(width, 4, 1, 512), key, key,
        query_rope=q_pe.view(width, 4, 1, 64),
        key_rope=rope.view(width, 1, 2048, 64), num_query_heads=4,
        num_key_value_heads=1, softmax_scale=impl.scale, input_layout="BNSD",
        sparse_mode=0, block_table=None, block_size=0,
        actual_seq_qlen=[1] * width, actual_seq_kvlen=[2048] * width,
    )
    return output.view(width, 4, 512)


def selection_mask(indices, causal_mask):
    """Encode unique, in-range selected IDs; True excludes a cache row."""
    import torch

    width, capacity = causal_mask.shape
    bitmap = torch.ones(
        (width, capacity), dtype=torch.uint8, device=indices.device,
    )
    bitmap.scatter_(1, indices.reshape(width, -1).long(), 0)
    # A3 supports uint8 scatter and bool logical OR on its vector cores.
    return bitmap.view(torch.bool) | causal_mask


def shared_kv_fia(ql_nope, q_pe, kv_cache, mask, scale):
    """Reuse one full finite KV cache across queries and a prepared mask."""
    import torch_npu

    width, capacity = mask.shape
    latent = kv_cache[0].view(1, capacity, 1, 512)
    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        ql_nope.view(1, width, 4, 512), latent, latent,
        query_rope=q_pe.view(1, width, 4, 64),
        key_rope=kv_cache[1].view(1, capacity, 1, 64),
        atten_mask=mask.view(1, 1, width, capacity),
        num_query_heads=4, num_key_value_heads=1,
        softmax_scale=scale, input_layout="BSND", sparse_mode=0,
        block_table=None, block_size=0,
        actual_seq_qlen=[width], actual_seq_kvlen=[capacity],
    )
    return output.view(width, 4, 512)


class SharedKVAttention:
    """Share selection masks within one ordered backbone forward.

    Each index producer refreshes the mask; consumers reuse that selection
    with their own Q/KV. Other query widths retain the original attention.
    """

    def __init__(self, width, capacity, device):
        import torch

        self.width = width
        self.columns = torch.arange(
            capacity, dtype=torch.int32, device=device,
        ).view(1, capacity)
        self.mask = None

    def bind(self, impl):
        original = impl._execute_sparse_flash_attention_process

        def attend(
            ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
            actual_seq_lengths_query, actual_seq_lengths_key, block_table=None,
        ):
            if ql_nope.shape[0] != self.width:
                return original(
                    ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
                    actual_seq_lengths_query, actual_seq_lengths_key,
                    block_table=block_table,
                )
            if not impl.skip_topk:
                # This backend's contiguous slots are logical token positions.
                causal = self.columns > attn_metadata.slot_mapping.view(-1, 1)
                self.mask = selection_mask(topk_indices, causal)
            return shared_kv_fia(
                ql_nope, q_pe, kv_cache, self.mask, impl.scale,
            )

        impl._execute_sparse_flash_attention_process = attend


GATHER_FIA_MAX_WIDTH = 6


def install_gather_fia(impl, width=4):
    """Bind gather/FIA; widths above six failed serial-equivalence checks."""
    if width > GATHER_FIA_MAX_WIDTH:
        raise ValueError(
            f"gather_fia is unvalidated above width {GATHER_FIA_MAX_WIDTH}; "
            "use the sfa kernel")
    impl._execute_sparse_flash_attention_process = MethodType(
        gathered_fia, impl,
    )
