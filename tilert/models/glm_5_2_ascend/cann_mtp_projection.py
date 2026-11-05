"""Shard the two BF16 MTP projections over the existing TP group."""


def shard_mtp_projections(model):
    """Keep complete K reductions and gather outputs before either consumer."""
    import torch
    from vllm_ascend.ops.linear import AscendColumnParallelLinear

    mtp = model.model.layers["78"]
    for owner, name, prefix in (
        (model, "rot", "rot"),
        (mtp, "eh_proj", "model.layers.78.eh_proj"),
    ):
        weight = getattr(owner, name).weight
        outputs, inputs = weight.shape
        with torch.device(weight.device):
            column = AscendColumnParallelLinear(
                inputs, outputs, bias=False, gather_output=True,
                params_dtype=weight.dtype, quant_config=None, prefix=prefix,
                return_bias=False,
            )
        column.skip_weight_nz_conversion = True
        rows = column.weight.shape[0]
        column.weight.copy_(weight.narrow(0, column.tp_rank * rows, rows))
        column.quant_method.process_weights_after_loading(column)
        setattr(owner, name, column)
