"""FP32 logits from the existing CANN BF16 vocabulary projection."""

from copy import copy
from types import MethodType


def fp32_projection(method, layer, hidden, bias=None):
    import torch

    assert bias is None
    return torch.ops.aten.mm.dtype(hidden, layer.weight.t(), torch.float32)


def install_fp32_head(head, processor):
    """Keep the BF16 TP16 weights and the original gather/vocabulary crop."""
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
