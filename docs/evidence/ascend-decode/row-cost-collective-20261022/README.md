# The TP16 reduction supplies 2.3% of the round model's row cost

2026-10-22, a3-180, sixteen ranks, bf16, captured graph, 156 dependent
in-place `all_reduce` calls -- the whole round's collective term at one verify
width. Rows 1 and 4 repeat the previous chain's configuration as the control
arm. Thirty samples a case after ten warmups; the reported spread is the
rank-maximum p50 across ranks.

| rows | payload | chain p50 | p90 | std | per call |
| --- | --- | --- | --- | --- | --- |
| 1 | 12 KB | 5.590 ms | 5.644 | 0.037 | 35.83 us |
| 4 | 48 KB | 5.085 | 5.183 | 0.062 | 32.60 |
| 8 | 96 KB | 5.127 | 5.204 | 0.071 | 32.87 |
| 12 | 144 KB | 5.125 | 5.248 | 0.062 | 32.85 |
| 13 | 156 KB | 5.163 | 5.231 | 0.068 | 33.09 |
| 16 | 192 KB | 5.168 | 5.265 | 0.079 | 33.13 |
| 32 | 384 KB | 5.505 | 5.558 | 0.064 | 35.29 |
| 64 | 768 KB | 6.593 | 6.676 | 0.078 | 42.27 |

Slopes: **0.019 ms a row** over 12 to 32, which is the span the round model was
fitted on, and 0.028 over 12 to 64. The round model's term is **0.814**, so the
collective supplies 2.3 to 3.5 percent of it.

## The arithmetic bound needs no slope

The fitted row term at the engine's width is 0.814 x 12 = **9.77 ms**, and the
entire collective chain at that width is **5.125 ms**. A term cannot supply
more than its own total, so the collective was already excluded before the
sweep -- even at perfect proportionality it is short by 4.6 ms.

## No token bucket, at any width

12 to 13 moves 0.038 ms against a 0.062 std, and 13 to 16 moves 0.002. The
collective has no step. The engine has none either: `cann_graph.py:49` captures
at an exact width from the whitelist `(4, 6, 8, 12, 16, 24, 32)` and
`:92` refuses metadata whose `num_actual_tokens` differs, so width 13 -- the
discriminator `plan.md` proposed -- is not capturable without changing both
that whitelist and the vendor's graph parameters.

## What is left to carry 0.814

The single-die layer probe that measured 0.026 ms a row
([row-cost-mismatch](../row-cost-mismatch-20261021/README.md)) timed norm, the
fused input projection, q_b, o_proj, the shared down projection, the dynamic
quant and the grouped matmul. It did not time SFA, the indexer or routing,
which are 7.0, 5.5 and 12.1 ms of the round. Those are the only width-dependent
stages left, and each draft row is one more query against an 8K cache.

Non-monotonic at the small end: one row costs 0.46 ms a chain *more* than four.
The smallest payload is not the cheapest, so a per-row model fitted through
rows below 4 inherits that curvature.
