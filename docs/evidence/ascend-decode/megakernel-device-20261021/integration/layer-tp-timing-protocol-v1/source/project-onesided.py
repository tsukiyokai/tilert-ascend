def install(engine, endpoint):
    """Point the engine's TP sum at a one-sided reduction.

    The engine already funnels every TP reduction through one override to
    force FP32, so this replaces that override's body and inherits the
    bfloat16 round trip.
    """
    import torch

    reduce = OneSidedReduce(engine.rank, engine.device, 16, endpoint)

    def one_sided(value):
        if value.dtype == torch.bfloat16:
            return reduce.sum(value.float()).bfloat16()
        return reduce.sum(value)

    engine._tp_sum_group._all_reduce_out_place = one_sided
    engine.tp_reduction = "one_sided_float32"
