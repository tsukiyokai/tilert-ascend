# The 1935 Casts: what they convert, who they sit between, and why none can just be deleted

From the width-8 capture's `kernel_details.csv`, ordered by stream and start
time. 1935 Cast kernels, 7.520 ms of device time, 3.89 µs each — and about 483
a verify, which at a measured 19.30 µs a launch is 9.32 ms of launch cost for
1.4 ms of work.
[the per-kernel floor](../kernel-floor-20261019/README.md)

## What they convert

| count | conversion |
| --- | --- |
| 869 | `FLOAT → DT_BF16` |
| 622 | `DT_BF16 → FLOAT` |
| 151 | `INT32 → INT64` |
| 96 | `FLOAT → FLOAT16` |
| 50 | `BOOL → INT64` |

The two float conversions are 1491 of 1935, 77%, and 960 of the calls are at
shape `8,6144` — hidden width at eight rows, the layer's main data path.

## 537 of them come in pairs, and 480 of those are one shape

Every run of consecutive Casts has length exactly 2. The commonest is 480
runs of `DT_BF16>FLOAT` followed by `FLOAT>DT_BF16` at `8,6144`, with 471 of
them inside the target graph.

| count | producer | | consumer |
| --- | --- | --- | --- |
| 237 | `QuantBatchMatmulV3` | → [Cast Cast] → | `AddRmsNormBias` |
| 225 | `ForeachAddList` | → [Cast Cast] → | `AddRmsNormBias` |

## They are not a round trip to nowhere, and the gap proves it

Adjacent in kernel order is not adjacent in time. The gap between the two
Casts of a pair: p10 22.250 µs, median 30.500, p90 44.000 — **all 480 pairs
have more than 5 µs between them**, with no kernel in between.

That gap is the reduction. A collective fused into MC2 does not appear as a
kernel, which is why the verify shows zero collectives, so the fp32
accumulation happens in exactly that invisible window.

So the widening is required — 16 bf16 partials accumulated in bf16 lose
precision, which `cann-tp-sum-20261017.json` measured and `b377291` fixed —
and the narrowing is required because the result must return to bf16. Removing
the pair would undo a correctness fix. I was one step from doing it.

## The override is reached by both entry points

`all_reduce` in `vllm/distributed/parallel_state.py:740` branches on
`use_custom_op_call`: true goes to `torch.ops.vllm.all_reduce`, false to
`_all_reduce_out_place` directly. The custom op's implementation at `:196`
ends in `group._all_reduce_out_place(tensor)`, so both paths pass through the
override `_install_tp_sum` puts there — which is consistent with the 480 pairs
being visible at all, and means the conversion's placement is ours to choose.

I checked this because a branch that bypassed the override would have made the
pairs come from somewhere else and the relocation impossible.

## What is actually available

Not deletion but relocation. The widening belongs in the producer's epilogue —
`QuantBatchMatmulV3` already accumulates in fp32 and could emit fp32 instead
of emitting bf16 for a Cast to widen again — and the narrowing belongs in
`AddRmsNormBias`'s prologue, which reads the value anyway.

Both sides lose a kernel and the numerical path is unchanged: 960 kernels,
**18.5 ms a verify**, 47% of it. The price is the same as deletion; the
mechanism is a rearrangement rather than a removal.

The remaining 1455 Casts are a separate question — the `INT32 → INT64` and
`BOOL` conversions are index and mask plumbing, 3.89 µs each as kernels, and
their producers are `Fill`, `GreaterEqual`, `Add` and `Slice`.
