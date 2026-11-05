# Native reduced greedy head

Direct reuse of installed vLLM `LogitsProcessor.get_top_tokens` loses this TP2
screen. Keep the current head. Reopen with a fused selection mechanism or TP16
evidence; the TP2 result does not establish TP16 latency.

Both arms project BF16 `[M,6144]` against real checkpoint `[9680,6144]` shards
with the existing FP32-output `aten.mm.dtype`. The control gathers logits and
argmaxes; the candidate selects locally and gathers FP32 `(value, global_id)`
pairs. At TP16/M32 the nominal gathered tensor shrinks from19,824,640 to4096 B.
Projection work remains unchanged.

## Correctness and measurement

- Server CPU: the installed method passes36 cases across16 simulated ranks,
  including ties, signed zero, infinities, NaNs and FP32 near ties.
  [Script and result](cpu/result.json).
- a3-194 dies14/15, real STANDARD TP2 HCCL, CANN9.2.0 B114. Each rank has two
  address-distinct identical weight banks; no weight copies occur in timing.
  Hidden inputs are seeded BF16 fixtures, not model activations. Logical vocab
  is19360 with padding1, retaining9680 columns/rank.
- M1/8/32 eager A/B/A/reset and Graph A/B/reset token IDs match exactly.
  Zero-input ties return token0. Warmup≥10 communications; eager10 and Graph30
  ABBA pairs,10 calls/Graph, profiler off, dedicated stream. Rank MAX is taken
  for each aligned sample before paired statistics. Shared load, unlocked clocks.
- Peak reserved memory360MiB/rank, within512MiB quota. Source and input hashes,
  versions, p90/p99/std and raw samples are in
  [measured-summary.json](measured-summary.json),
  [paired-summary.json](paired-summary.json), [rank0](rank-0.json) and
  [rank1](rank-1.json). Reproduce with [run.sh](run.sh).

| M | Eager wall p50, full→reduced (µs) | Paired delta (µs) | Graph wall p50/call, full→reduced (µs) | Paired delta (µs), wins |
| --- | --- | --- | --- | --- |
| 1 | 432.992→589.494 | +163.144 | 135.863→150.043 | +14.179,2/30 |
| 8 | 367.370→538.587 | +170.729 | 148.364→169.199 | +20.737,0/30 |
| 32 | 462.254→585.528 | +113.342 | 180.121→206.889 | +29.852,0/30 |

Paired deltas are medians of within-pair differences, not differences between
pooled medians. Eager wins are0/10 at all widths. Graph device-event paired
deltas are+14.256/+21.206/+28.372µs,0/30 wins each. Smaller communication payload
alone does not justify this additional local selection path. No production
head change or whole-model TPOT gain follows from this experiment.
