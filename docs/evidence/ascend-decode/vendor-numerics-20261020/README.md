# The kernel's arithmetic, against the vendor rather than against myself

2026-10-20, a3-84, one die.

The gap identified earlier today was that every correctness result for these
kernels was measured against a host reference of my own, which is evidence
about consistency and not about the format. This closes it.

## The control that earns the conclusion

With random weights the two chains disagree badly:

    vendor final absmax 26.25000
    mine         absmax 18.87500
    max abs difference 30.500000  mean 1.916094  unequal 98214/98304

That alone attributes nothing: the probe guesses how the vendor's int32 word
orders its eight codes, and a wrong guess feeds the two sides different
weights. So set every code to one and the bias to zero -- under a constant
weight every packing permutation is the same weight, and a difference that
survives is arithmetic:

    vendor final (16, 6144) torch.bfloat16 absmax 0.45117
    mine         (16, 6144) torch.bfloat16 absmax 0.45117
    max abs difference 0.001953  mean 0.000117  unequal 14027/98304

Identical magnitude; the residual 0.001953 is one bfloat16 ULP at 0.45. The
INT4 matmul, the dequantisation, the swiglu, the requantisation, the down
projection and the correction's placement all agree with
`grouped_matmul_swiglu_quant_v2` followed by `npu_grouped_matmul`.

So the random-weight disagreement is the probe's vendor-side packing, not the
kernel. Which is the right conclusion to be able to draw, and the control is
what allows it.

## The vendor chain, as it must be called

Six constraints, each found by being refused:

`torch.ops.load_library` on `vllm_ascend_C.cpython-311-aarch64-linux-gnu.so`
-- importing the package does not register `_C_ascend`.

`weight`, `weight_scale` and `weight_assist_matrix` are lists, the correction
one tensor an expert.

The fused call dispatches to `aclnnGroupedMatmulSwigluQuantWeightNzV2` and
takes fractal-NZ weights only, which needs
`torch_npu.npu.config.allow_internal_format = True` -- without it the cast
silently returns a base-format tensor whose storage shape is one-dimensional
and the operator rejects it.

GE will not initialise without the `tbe` Python module, and `python3 -I`
hides it, because `-I` ignores the `PYTHONPATH` that `set_env.sh` sets. My own
isolation flag was the whole of that failure.

The fused operator wants its scale two-dimensional, `[E, N]`. The grouped
matmul wants the same information three-dimensional with the group axis in
the middle, `[E, 1, N]`, which is what `_process_scale`'s `transpose(1, 2)`
produces -- pass `[E, N, 1]` and it reads N as the quantisation group count
and refuses, since K is not divisible by it. Two operators, two conventions,
which is what the comment in `w4a8.py` about keeping the scale unsqueezed is
warning about.

And the scale travels as FP32 bit patterns widened to int64, not as floats.

## What this does not say

Nothing about speed in place, and nothing about the engine. The vendor's gmm1
here is handed sixteen experts, where in the engine it is handed 256 with a
mostly-zero group list. And the comparison is on synthetic codes; the real
weights arrive packed along the output axis and need the repack that
`install` now does, which this probe does not exercise.
