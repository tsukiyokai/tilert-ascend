from copy import copy
from types import MethodType


def fp32_projection(method, layer, hidden, bias=None):
    import torch

    assert bias is None
    return torch.ops.aten.mm.dtype(hidden, layer.weight.t(), torch.float32)


def install_fp32_head(head, processor):
    import torch
    import torch_npu
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    assert isinstance(head.quant_method, UnquantizedEmbeddingMethod)
    assert processor.head_dtype in (None, torch.bfloat16)
    assert head.weight.dtype == torch.bfloat16
    assert torch_npu.get_npu_format(head.weight) == 2
    method = copy(head.quant_method)
    method.apply = MethodType(fp32_projection, method)
    head.quant_method = method
    orig_forward = processor.forward

    def forward(self, lm_head, hidden_states, embedding_bias=None,
                skip_gather=False):
        if getattr(self, "_top_tokens_mode", False) and not skip_gather:
            logits = fp32_projection(lm_head.quant_method, lm_head,
                                     hidden_states, embedding_bias)
            num_pad = lm_head.shard_indices.num_org_vocab_padding
            if num_pad > 0:
                logits = logits[..., :-num_pad]
            local_max, local_idx = logits.max(dim=-1)
            vocab_start = int(lm_head.shard_indices.org_vocab_start_index)
            return torch.ops.tilert.onesided_top_tokens(
                local_max, local_idx, vocab_start,
            )
        return orig_forward(lm_head, hidden_states, embedding_bias, skip_gather)

    processor.forward = MethodType(forward, processor)
