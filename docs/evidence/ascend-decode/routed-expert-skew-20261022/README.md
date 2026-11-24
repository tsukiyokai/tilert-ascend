# The expert skew ported into the routed kernel, and its real ceiling

`ROUTED_EXPERT_SKEW=n` cuts the expert index into `n` chunks and runs the
gate/up of step `s` beside the activate of step `s-1`. The experts are already
one matmul invocation each, so interleaving them costs barriers rather than
reduction walks -- the criterion measured separately.

## It is correct and it wins a little

Two alternating passes, two weight banks, four calls, twelve rows, 24 blocks,
one die. Every arm passes all four changing-route replays bitwise, at `max_abs`
0.0, with the same active-expert counts of 8, 8, 87 and 81.

| chunks | us a call | median | against unskewed |
| --- | --- | ---: | ---: |
| off | 169.14, 168.32 | 169.14 | — |
| 2 | 166.20, 166.21 | 166.21 | **−2.94** |
| 4 | 171.00, 167.99 | 171.00 | +1.86 |

With the skew compiled out the kernel matches the untouched baseline: 170.12
against 167.08 microseconds a call, inside the spread, and bitwise equal on
every replay.

## Why 2.94 and not 26, and the estimate it corrects

This pairs `gate_up`, which is 40.35 microseconds of cube work, with
`activate`, which is 4.04 of vector work. The overlap available to that pair is
`min(40.35, 4.04)`, about 4 microseconds, and 2.94 of it was taken.

So an earlier estimate of 26 to 33 microseconds was wrong in its ceiling. It
took the available overlap as the window less the cube work, 136.40 − 71.25 =
65.15, but that difference is mostly barrier waiting and skewing does not
remove waiting -- it hides **vector work** under cube work. The FFN's whole
vector half is 23.40 microseconds, so that is the ceiling for a complete skew,
and at the 40 to 50 percent capture rate measured on the front stage it is
nearer 10.

The vector half's bulk is not in `activate` either: `store_routes` is 9.87,
`metadata_and_split` 6.26 and `combine` 2.12. Those depend on `down_matmul` and
precede `gate_up` respectively, so a complete pipeline has to skew all five
stages, not the pair coupled here. That is what the remaining seven of the ten
microseconds would come from.

## The FFN's floor, and why the imbalance cannot be fixed

Per cube core at twelve rows, from the same trace:

| stage | work min | median | max | spread | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate_up_matmul | 38.28 | 40.36 | 50.28 | 1.31x | 43.16 |
| down_matmul | 28.32 | 29.65 | 35.31 | 1.25x | 30.89 |

A stage ends when its slowest core does, so the cube's critical path is
50.28 + 35.31 = 85.59 microseconds. The 1.31 spread is exactly 4/3: 81 active
experts over 24 cores gives nine cores four experts and fifteen three, and the
cube's cost is per expert -- about 12.8 microseconds, the reduction walk --
rather than per row.

That imbalance is not fixable. Levelling it means splitting one expert's rows
across two cores, which is another invocation and so another walk of a 6144
reduction, about 13 microseconds each. Twenty-four such splits would cost three
hundred.

So the body's window decomposes as

    136.40 = 85.59 cube critical path
           + 23.40 vector work, none of it hidden today
           + 27.4 residue, the cube waiting on the cube

and a complete skew removes the middle term only, landing near 113. The whole
FFN side is worth about 23 microseconds, of which this entry takes 2.94.

It also qualifies a framing from earlier today. The layer's roughly 500
microseconds of waiting is not all recoverable: waiting on the other half is,
bounded by the vector work there is to hide, while waiting on one's own half is
locked by the walk cost of dividing the work more finely.

## Scope

Synthetic routing, two banks, one die, no tensor-parallel reduction. The
numerical gate is the kernel's own: four changing-route replays against the
stock CANN canonical NZ chain, bitwise. Timing is the median of two passes of
three pairs.

[`run.sh`](run.sh) is the comparison; the JSON files are its reports, and
`skew-off.json` is the compiled-out arm against the baseline.
