# MoE at the unions the routes actually produce, and the row cost closes to 87 percent

2026-10-23, a3-180, one die, W4A8 grouped matmul at the expert unions the
teacher routes measured: width 8 gives 44.35 active experts, width 16 gives
68.07, width 32 gives 99.36.

| union | gate_up | down | pair | ms a round |
| --- | --- | --- | --- | --- |
| 44 | 90.67 us | 95.04 | 185.71 | 14.49 |
| 68 | 116.96 | 105.26 | 222.22 | 17.33 |
| 99 | 143.40 | 119.78 | 263.18 | 20.53 |

## Two independent measurements meet

MoE's marginal cost a verify row is **355 us a round** from width 8 to 16 and
**200** from 16 to 32. The round model's row term is 814 us, and this session
measured attention's share at 356. Together 711 against 814 -- **87 percent of
the row cost is the scattered read plus the expert union**, from two
measurements that share no code and no fixture.

The deceleration explains the budget sweep: acceptance rises sublinearly while
the row cost, though linear overall, slows on the MoE side, so TPOT bottoms out
between budgets 16 and 32 rather than at 64.

## And it raises the floor, so 2.x is not reachable this way

The previous ledger used a single 68-expert union and put the irreducible floor
at 21.59 ms a round. Per width it is:

| width | MoE + collective | tree accepted | TPOT floor |
| --- | --- | --- | --- |
| 8 | 19.62 ms | -- | -- |
| 16 | 22.45 | 5.989 | **3.75 ms** |
| 32 | 25.66 | 6.966 | **3.68** |

So even with attention and routing driven to zero, this configuration bottoms
at 3.68 ms, not 2.90. The earlier 2.90 came from holding the union fixed while
raising the width, which the routes do not do.

What that leaves for 2.x: the floor is MoE bytes, so it falls only if a round
touches fewer experts or reads fewer bytes for each -- a smaller union per row,
weight residency across rounds, or a narrower expert dtype. Acceptance alone
cannot do it: at width 32 it would need 8.8 accepted against the tree's
measured 6.966, and the tree is the best drafter this project has on its own
streams.
