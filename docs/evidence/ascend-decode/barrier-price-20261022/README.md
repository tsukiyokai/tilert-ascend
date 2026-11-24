# A grid barrier costs 2.29 microseconds, which prices the skewed pipeline

A skewed expert pipeline pays one grid barrier a step, so the step's price
decides whether the schedule is worth its overlap. Issuing `SyncAll<false>()` a
sweepable number of times in the front stage, one die, 24 cube cores, five
repeats:

| barriers | us | above none | us each |
| ---: | ---: | ---: | ---: |
| 0 | 26.37 | — | — |
| 1 | 28.49 | +2.12 | 2.120 |
| 2 | 29.67 | +3.30 | 1.650 |
| 4 | 34.48 | +8.11 | 2.027 |
| 8 | 45.18 | +18.81 | 2.351 |
| 16 | 62.79 | +36.42 | 2.276 |
| 32 | 99.60 | +73.23 | 2.288 |

Linear at 2.29 microseconds a barrier from eight upward. For comparison, AMD
reports about 1.7 microseconds for the grid-wide synchronization at its
per-layer megakernel's only boundary.

## It corrects the FFN estimate downward

The routed body's two halves idle against each other for 61 microseconds of a
136.40 window, and the earlier reading took that as the available saving. It
ignored what the schedule costs.

A round touches 81 experts spread over 24 cube cores, so a core owns three or
four. Skewing gate/up against activate takes about five steps and skewing down
against the store and combine about five more, so ten barriers at 2.29 is 22.9
microseconds. Against the cube's 71.25 microsecond floor that is roughly 94,
not 71 -- a saving near 42 microseconds a layer, or 3.3 milliseconds a round,
rather than 61 and 4.76.

A shallower pipeline trades the other way: two steps a phase is six barriers at
13.7 microseconds but covers less of the vector half's 23.40. Both arrangements
land near 93 to 95 microseconds, so the depth is not the lever and the estimate
is not sensitive to it.

[`run.sh`](run.sh) is the sweep; the JSON files are its output.
