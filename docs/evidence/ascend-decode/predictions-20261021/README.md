# What the arms should show, written before they run

The baseline: a round of 50.710 ms at a TPOT of 9.3794, so 5.407 accepted --
which the offline corpus put at 5.441, so the drafting measurements transfer to
this arm within a percent.

| arm | expected saving | round | TPOT |
| --- | --- | --- | --- |
| narrow | 22.20 ms | 28.51 | **5.27** |
| hoist-rope | 0.13 | 50.58 | 9.36 |
| aicpu | negative | 50.71+ | 9.38+ |
| all | 22.33 | 28.38 | 5.25 |

## Why each number

`narrow` is the bank's device-side price: sixteen experts of 18.9 MB cost
160.53 us at the checkpoint's wider projection and 80.21 at the narrower, twice
a layer, against about 17 us each when two experts are handed over. 285 us a
layer, 22.2 ms a round.

`hoist-rope` is a floor, not an estimate. The metadata rebuild's host half
measures 130.62 us a round and the vLLM objects it also builds are not counted,
so the arm should beat 0.13 ms by an unknown margin. The rope graph contributes
only what hoisting introduces.

`aicpu` should be slower, and by how much is the one quantity three methods
disagreed on today -- the native HCCL device unit says 10.74 us a call, torch
timed around a barrier says 177 to 296, the profiler's kernel Duration says
452. This arm measures the difference AIV makes to a round, which is the only
scope that settles it.

## What the prediction does not claim

None of these reaches 2.9 ms. The gap between 5.25 and the target is what the
device account cannot explain: 78 layers of compute at 228.67 us profiled is
17.84 ms, and a round is 50.710, so two thirds of it is collectives and host
path that only `steady` and `--round-phases` can attribute.

Writing this down first is the discipline the day earned: every figure that had
to be withdrawn was withdrawn after being explained, and an explanation that
arrives before the measurement cannot be fitted to it.
