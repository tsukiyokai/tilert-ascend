# Where the 49.92 ms round sits after the rope fix

`a3-84` idle, TP16, source `04a6628` plus
[the rope patch](../../../../tests/glm52/patches/rope-cos-sin-contiguous.patch),
the composed flag set with `HCCL_OP_EXPANSION_MODE=AIV`, `--max-tokens 16
--ignore-eos`, three captured rounds, `--profile-stacks`. Exported with
`torch_npu.profiler.profiler.analyse`. [rank 0](op_statistic-rank0.csv).

`FusedInferAttentionScore` has 244 calls against 81 per round, so the export
covers 3.01 rounds and the per-round column divides by three. The
corresponding profiler-off round is 49.92 ms
([rope fix](../rope-contiguous-20261018/README.md)); a profiled round is
longer and the first one includes capture, so these shares describe the
composition, not the wall clock.

| operator | core | calls/round | per round (ms) | avg (µs) | max (µs) | share |
| --- | --- | --- | --- | --- | --- | --- |
| `SuperKernel` | MIX_AIC | 351 | 22.80 | 64.94 | 195.54 | 53.6% |
| `SuperKernel` | MIX_AIV | 286 | 11.73 | 41.02 | 6047.15 | 27.6% |
| `FusedInferAttentionScore` | MIX_AIC | 81 | 3.04 | 37.42 | 44.26 | 7.2% |
| `Cast` | AI_VECTOR_CORE | 556 | 1.54 | 2.77 | 4.76 | 3.6% |
| `MatMulV3` | AI_CORE | 8 | 0.51 | 63.50 | 150.76 | 1.2% |
| `GroupedMatmulSwigluQuantV2` | MIX_AIC | 3 | 0.34 | 100.50 | 113.05 | 0.8% |

`Slice` has left the table; it was 15.5% before the patch. The operators sum
to about 42.5 ms per round, so roughly 7.4 ms of the round is in no operator
at all.

## What SuperKernel did and did not do

Call count per round fell from about 2,400 to 637 SuperKernel groups plus the
leftovers, and the leave-one-out attribution puts it at 1.807 ms of TPOT
([attribution](../ablate-20261018/README.md)). What it cannot do is visible
here. It does not cross the cube/vector boundary inside a group: 351 AIC
groups and 286 AIV groups per round means roughly 637 handoffs through L2 or
GM still happen, which is architectural. It does not reach outside the
compiled region, so `Cast` is still 556 calls per round and
`FusedInferAttentionScore`, bound onto the attention instance by this project,
stays separate. And it only groups existing operators, so it cannot prefetch
layer k+1's weights during layer k — the one restructuring that would make the
weight stream continuous.

The AIC groups cost 22.80 ms per round against a streaming floor of at most
6.00 ms for all of a round's weights, so about 16.8 ms of that block is not
traffic. Grouping has done what grouping can.

## One AIV group per round takes milliseconds, and it is unidentified

The AIV SuperKernel is sharply bimodal: p50 20.03 µs, p90 54.07, p99 65.03,
and then three calls out of 858 at 2837.06, 6047.15 and 4183.38 µs, together
13.068 ms or 37.1% of the type. All three carry `Block Num` 16 on stream 80,
and all three fall inside the decode window at offsets of 78.48, 214.19 and
345.57 ms across a 410.46 ms window, so one occurs per round and none is a
teardown barrier.

They match no named collective — the nearest `communication.json` entry is
9.9 to 24.0 ms away and 0.03 ms long — because under AIV expansion with
SuperKernel the collectives leave the operator table altogether, with both
`allreduceAicpuKernel` and `allGather` absent. `trace_view.json` names them:

```
sk_1277_full_model_start_aiv_all_reduce_float_end_aiv_all_reduce_float
Model Id 48, Task Type MIX_AIV, Physic Stream Id 80
```

So it is a collective after all, embedded in a SuperKernel group whose first
and last operators are both an AIV FP32 all-reduce.

## One grouping boundary costs 4.36 ms per round

Grouping the trace's `sk_*` events by name gives 637 distinct groups totalling
103.587 ms over three rounds, so 34.53 ms per round. Of those, 157 name an
`all_reduce` and total 29.921 ms, which is 28.9% of all SuperKernel time or
9.97 ms per round. The count matches the 157 reductions a round performs, so
each reduction sits in its own group.

The distribution inside that set is the finding. `sk_1277` alone takes
13.068 ms of the 29.921, which is 4.36 ms per round, while the other 156
groups share 16.853 ms and average about 36 µs each. One group costs 121
times what its siblings cost, and the names say why: `sk_1277` begins and
ends with an all-reduce, while the others begin with a compute kernel, for
example
`sk_667_full_model_start_static_kernel_AddRmsNormBias_...`.

A group that begins with a collective cannot start until all sixteen ranks
arrive, and it holds no preceding compute to cover the skew. A group that
begins with compute hides its reduction behind that compute. This is the same
mechanism the draft stages showed, where three reductions after a host
broadcast waited about 1.2 ms each
([draft collectives](../draft-collectives-20261018/README.md)), expressed here
as a single grouping boundary worth 8.7% of the round.

The next experiment is to move that boundary: `cann_mtp.py` already sets
`compilation_config.static_all_moe_layers`, so the grouping is configurable
from this project, and the test is whether forcing a compute kernel ahead of
that reduction collapses the 4.36 ms.
