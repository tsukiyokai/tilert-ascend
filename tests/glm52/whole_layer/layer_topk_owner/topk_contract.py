"""Validate the retained selector destination before creating raw records."""


def validate_topk(value, rows, device, groups):
    import torch
    import torch_npu

    if (value is None or value.device != device or value.dtype != torch.int32
            or tuple(value.shape) != (rows, 1, 2048)
            or not value.is_contiguous() or value.storage_offset() != 0
            or value.data_ptr() % 32 != 0
            or torch_npu.get_npu_format(value) != 2):
        raise ValueError("Require a retained aligned INT32 ND [M,1,2048] top-k prefix")
    base = value.untyped_storage().data_ptr()
    if any(owner is not None and owner.untyped_storage().data_ptr() == base
           for group in groups for owner in group):
        raise ValueError("The persistent top-k owner must have separate storage")
