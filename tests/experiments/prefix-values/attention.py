"""M1 attention with frozen prefix Value and one current-row projection."""


def install(impl, prefix_values):
    import torch

    if (
        prefix_values.shape != (4, 8192, 256)
        or prefix_values.dtype != torch.bfloat16
        or prefix_values.device != impl.W_UV.device
    ):
        raise ValueError("Expected the bound BF16 prefix Value on this device")
    original_attention = impl._execute_sparse_flash_attention_process
    original_value = impl._v_up_proj
    original_hf32 = torch.npu.matmul.allow_hf32
    torch.npu.matmul.allow_hf32 = False
    values = torch.empty(
        (4, 8193, 256), dtype=torch.bfloat16, device=prefix_values.device
    )
    values[:, :8192].copy_(prefix_values)

    def attention(
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table=None,
    ):
        if (
            ql_nope.shape != (1, 4, 512)
            or q_pe.shape != (1, 4, 64)
            or ql_nope.dtype != torch.bfloat16
            or kv_cache[0].dtype != torch.bfloat16
            or actual_seq_lengths_query.cpu().tolist() != [1]
            or actual_seq_lengths_key.cpu().tolist() != [8193]
        ):
            raise ValueError("Requires the frozen BF16 M1/8193 contract")
        table = (
            attn_metadata.block_table if block_table is None else block_table
        )
        positions = torch.arange(8193, device=ql_nope.device)
        pages = table[0].long().index_select(0, positions // 128)
        physical = pages * 128 + positions % 128
        latent = kv_cache[0].view(-1, 512).index_select(0, physical)
        rope = kv_cache[1].view(-1, 64).index_select(0, physical)
        indices = topk_indices.reshape(-1).long()
        if indices.numel() != 2048:
            raise ValueError("Expected the actual2048 selected positions")
        keys, key_rope = (
            latent.index_select(0, indices),
            rope.index_select(0, indices),
        )
        scores = ql_nope[0].float() @ keys.float().T
        scores = scores + q_pe[0].float() @ key_rope.float().T
        scores = (scores * impl.scale).to(torch.bfloat16)
        probabilities = torch.softmax(scores.float(), dim=-1).bfloat16()
        aligned = torch.zeros(
            (4, 8193), dtype=torch.bfloat16, device=ql_nope.device
        )
        aligned.scatter_(1, indices.expand(4, -1), probabilities)
        current = torch.bmm(
            latent[8192:].unsqueeze(0).expand(4, -1, -1).contiguous(),
            impl.W_UV,
        )
        values[:, 8192:].copy_(current)
        return torch.bmm(aligned.unsqueeze(1), values).transpose(0, 1)

    def value(output):
        if output.shape != (1, 4, 256):
            raise ValueError("Cached Value must yield [1,4,256]")
        return output.reshape(1, 1024)

    impl._execute_sparse_flash_attention_process = attention
    impl._v_up_proj = value

    def restore():
        impl._execute_sparse_flash_attention_process = original_attention
        impl._v_up_proj = original_value
        torch.npu.matmul.allow_hf32 = original_hf32

    return restore
