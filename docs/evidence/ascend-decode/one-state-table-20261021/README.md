# The same operators cost 15 ms in a clean process and 53 in the engine

2026-10-21, a3-84, one die, one process, decode shape -- twelve tokens, 96
sorted rows, 78 layers.

| item | a layer | a round | recorded |
| --- | --- | --- | --- |
| MoE matmul | 62.578 us | 4.88 ms | 12.3 |
| routing permute | 32.549 us | 2.54 ms | part of 12.1 |
| routing combine | 30.076 us | 2.35 ms | part of 12.1 |
| norm and quant | 47.536 us | 3.71 ms | <= 5.5 |
| TP reduction, local sum | 24.382 us | 1.90 ms | 9.9 at sixteen ranks |
| sum | | 15.38 ms | about 40 of the 53 |

Gating is excluded: it takes the vector core down in this probe at both
float32 and bfloat16, and its cost is known separately at 20.798 us.

## What this overturns

The round's 45.88 ms fixed term was being treated as a floor -- the thing that
makes 2.x ms unreachable at width twelve. It is not a property of the
operators. The same operators, at the same shape, cost 15.38 ms a round here
and about 40 ms in the engine's accounting. The difference is the process
state, already measured at 2.5x on the MoE matmul alone: 61.6 us clean against
153.3 mixed.

If the engine ran in the clean state, a round would be near 18 ms rather than
55.424, which is a TPOT near 3.3 ms at the acceptance already achieved -- with
no kernel written and no cross-layer fusion.

## What has been ruled out as the cause

Loading a custom kernel library (60.309 us), running the kernel once (61.161),
building a communication domain (66.478), a second resident fractal-NZ weight
bank (61.300), the full two-stage vendor chain (60.936), and constant versus
random INT4 codes (62.102 against 63.581). Six candidates, all innocent.

## Why this is now the largest lever on the board

Measured levers, a round: one-sided reduce 6.5 ms, routing absorption at best
4.15, MoE kernel -16.3. The state difference is worth about 25 ms. It is also
the only one that needs no new kernel.
