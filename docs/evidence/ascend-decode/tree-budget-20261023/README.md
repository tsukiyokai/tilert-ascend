# The tree's optimum budget is 16, so the width caps are not worth raising

2026-10-23, CPU, the engine's own two 78-layer streams (`lever-20261020`
`tail32` and `tail8`, three measured requests each, 1024 output tokens, the
same 8192-token capture prompt).

The tree was the only drafter still climbing past width 32, and both the
capture whitelist and `route_metadata`'s `MAX_TOKENS` stop there. This prices
the climb before either is touched.

| budget | accepted | nodes used | TPOT now | with shared read | and eleven passes |
| --- | --- | --- | --- | --- | --- |
| 12 | 5.491 | 11.3 | 10.14 | 9.36 | 3.91 |
| 16 | 5.989 | 14.9 | **9.84** | 8.89 | **3.89** |
| 24 | 6.544 | 22.2 | 10.00 | **8.69** | 4.12 |
| 32 | 6.966 | 29.5 | 10.33 | **8.69** | 4.40 |
| 48 | 7.161 | 44.1 | 11.86 | 9.48 | 5.30 |
| 64 | 7.447 | 58.6 | 13.16 | 10.10 | 6.08 |
| 96 | 7.642 | 87.8 | 16.23 | 11.76 | 7.84 |
| 128 | 7.787 | 116.8 | 19.27 | 13.42 | 9.58 |

TPOT from the round model's measured terms: `45.88 + 0.814 x budget`, then with
attention's 0.356 a row removed by the shared read, then with the pass budget's
fixed 15.97.

Acceptance rises monotonically and sublinearly -- 42 percent from budget 12 to
128 -- while the row cost rises by a factor of 10.7. Every cost model bottoms
out at a budget of 16 to 32, and 64 is 56 percent worse than 16 even under the
most favourable one.

So `MAX_TOKENS = 32` and the capture whitelist are not the constraint, and
raising them is negative. This is the second confirmation that widening is
spent, and the first one taken on a tree and on the engine's own output rather
than on a chain or a constructed corpus.

What it leaves: the acceptance leg cannot be closed by budget or by tree shape.
2.9 ms at the measured 6.97 accepted at budget 32 wants a round of 20.2 ms
against today's 71.9, so the fixed term is the only term left with that much
room in it.
