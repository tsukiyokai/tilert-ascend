# The five-flag configuration, profiled

The first profile of a configuration close to the current best. `a3-84`
idle, TP16, source `04a6628`, the frozen 8K prefix, three captured rounds at
M = 4 with `--target-graph --draft-graph --draft-projection-tp
--attention-kernel gather_fia --fp32-head` and
`TORCHINDUCTOR_AUTOGRAD_CACHE=0 VLLM_DISABLE_COMPILE_CACHE=1`. Only
`--compile-target` and `--super-kernel` are missing, so this is the 19.496 ms
case of [the composed comparison](../compose-20261018/README.md), not the
16.640 ms one. `--compile-target` still cannot be profiled: it fails at
`vllm_ascend/attention/sfa_v1.py:1779` with an output buffer sized
`[1, 6144]` against a `[4, 6144]` tensor.

The capture receipt records `draft_graph_state: {captures: 1, replays: 28}`,
so the draft really was graphed. An earlier reading of a receipt's
`draft_execution` field as `eager` was a reporting bug: that string was
hardcoded in `profile_cann_decode.py` and is now derived from `draft.graph`.

Raw profiling was exported with `torch_npu.profiler.profiler.analyse`, which
produces the CANN tables `op_statistic.csv`, `kernel_details.csv`,
`task_time.csv`, `step_trace_time.csv`, `communication.json` and
`trace_view.json`. [rank 0](op_statistic-rank0.csv),
[rank 14](op_statistic-rank14.csv), [configuration](config-rank0.json).

`MlaPrologV3` has 244 calls against 81 per round (78 verify layers plus one
per draft stage), so the export covers 3.01 rounds and the per-round column
below divides by three.

| operator | core | calls | per round (ms) | avg (µs) | max (µs) | share |
| --- | --- | --- | --- | --- | --- | --- |
| `allreduceAicpuKernel` | AI_CPU | 501 | 44.28 | 265.16 | 3706.71 | 47.8% |
| `GroupedMatmulSwigluQuantV2` | MIX_AIC | 235 | 7.90 | 100.88 | 126.17 | 8.5% |
| `Slice` | AI_VECTOR_CORE | 190 | 6.85 | 108.15 | 366.26 | 7.4% |
| `GroupedMatmul` | MIX_AIC | 235 | 5.46 | 69.74 | 84.85 | 5.9% |
| `MlaPrologV3` | MIX_AIC | 244 | 3.16 | 38.88 | 49.19 | 3.4% |
| `QuantBatchMatmulV3` | MIX_AIC | 564 | 3.14 | 16.70 | 21.48 | 3.4% |
| `FusedInferAttentionScore` | MIX_AIC | 244 | 3.13 | 38.48 | 44.66 | 3.4% |
| `GatherV3` | AI_VECTOR_CORE | 488 | 2.48 | 15.23 | 23.34 | 2.7% |
| `allgatherAicpuKernel` | AI_CPU | 34 | 2.06 | 181.56 | 393.30 | 2.2% |
| `MatMulV3` | AI_CORE | 249 | 1.94 | 23.41 | 177.86 | 2.1% |

## gather_fia works, and an earlier objection to it was wrong

`SparseFlashAttention` does not appear at all; `FusedInferAttentionScore`
with two `GatherV3` calls per layer replaces it. Per layer that is
38.48 + 2 × 15.23 = 68.94 µs against SFA's 90.28 µs in
[the 10-17 capture](../cann-mtp-profile-20261017/summary.json), so
`gather_fia` saves 21.3 µs per layer and about 1.7 ms per round, close to
what [the reuse screen](../sfa-reuse-20261018/README.md) predicted from
89.05 → 69.15 µs.

This retracts a claim made from the ablation. Dropping `gather_fia` shortens
the round by 0.95 ms ([attribution](../ablate-20261018/README.md)), which was
read as the screen failing to transfer. The attention kernel is not the cause:
the two cases deliver different trajectories, 307 rounds at A = 3.332 against
298 at 3.433, so they touch different expert sets, and the round difference is
trajectory-coupled rather than kernel-coupled. The synthetic measurement that
found `gather_fia` slower used a base-format cache, which the production path
does not.

## Two targets this capture puts on the board

`allreduceAicpuKernel` takes 47.8% of device time: 501 calls averaging
265.16 µs with a 3706.71 µs maximum, 44.28 ms per round. This is the AICPU
kernel's own wall time, which includes its wait loop and overlaps compute —
the 10-17 capture showed `hccl_aicpu_union_ms` of 39.078 ms against a 5.313 ms
envelope in verify — so it is not all exposed. But a 265 µs mean for a 12 to
96 KiB message is far from the 33.8 µs envelope the same reductions showed,
and the gap between kernel time and envelope is where the draft stages' waits
live.

`Slice` is now the third largest operator at 108.15 µs mean, 366.26 µs max
and 6.85 ms per round. In the 10-17 verify it was 48 calls at 45.7 µs; the
per-call cost has risen 2.4 times and the count per round has risen from 48
to 63. It was not on any earlier list. Which flag introduced it is not
established here; `--draft-projection-tp` adds six AllGathers whose shards
need slicing, and the `gather_fia` path slices its gathered rows, so either
could account for it.
