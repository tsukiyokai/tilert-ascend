# The FFN's vector half is idle 82 percent of its window

The routed body's own stage trace already holds the decomposition, and it says
the two halves of the machine take turns idling. At twelve rows, 24 cube cores
and 48 vector cores, medians over the per-core records:

| half | work us | barrier us | idle share |
| --- | ---: | ---: | ---: |
| cube | 71.25 | 67.52 | 49% |
| vector | 23.40 | 111.74 | 82% |

The window's median is 136.40 microseconds, and each half's work plus barrier
accounts for it: 138.77 for the cube and 135.14 for the vector. Total work is
94.65 microseconds in a 136.40 window.

## Where each half waits

| stage | cube work | cube wait | vector work | vector wait |
| --- | ---: | ---: | ---: | ---: |
| restore_dispatch | 0.17 | 6.90 | 1.07 | 2.28 |
| metadata_and_split | 0.26 | 15.30 | 6.26 | 9.29 |
| gate_up_matmul | 40.35 | 10.94 | 0.02 | 51.31 |
| activate | 0.28 | 11.54 | 4.04 | 7.79 |
| down_matmul | 29.65 | 6.52 | 0.02 | 36.15 |
| store_routes | 0.28 | 12.93 | 9.87 | 3.33 |
| combine | 0.27 | 3.38 | 2.12 | 1.60 |

The vector cores lose 87.46 microseconds waiting out the two matmuls, and the
cube loses 39.77 waiting out the vector stages. The remaining 27.75 of cube wait
is cube waiting on cube -- the expert loop's imbalance.

## What closes it

Within one expert the chain is strict: metadata feeds gate/up, which feeds
activate, which feeds down, which feeds the store and the combine. So the
vector work cannot hide under the cube work it produces. But a round touches
about five experts a rank, and expert `e`'s activate is independent of expert
`e+1`'s gate/up. Pipelining the loop puts each half's work under the other's.

The cube's 71.25 microseconds is the floor. Perfect pipelining would leave
roughly that plus the schedule's own barriers. A grid barrier costs 2.29
microseconds and a skew over three or four experts a core needs about ten, so
the floor with the schedule is near 94 against today's 136.40 -- about 42
microseconds a layer, or 3.3 milliseconds a round. See
[`barrier-price-20261022`](../barrier-price-20261022/README.md). It is the largest single
mechanism still on the table, and the shared expert already demonstrates it:
the `ROUTED_SHARED_PIPELINE` variant places shared cube projections beside
routed vector stages and measures 187.17 against 219.24 microseconds at twelve
rows, a 32 microsecond saving.

This also sizes the capacity measured separately. A cube stage hides about 10.5
microseconds of vector work for nothing, and the FFN's whole vector half is
23.40 -- so two cube stages' worth of capacity covers all of it.

## The cube half is already at bandwidth, so this is the FFN's last lever

The same trace's `measured_routing` corrects a count used elsewhere in this
project. Its array holds 256 entries with 81 active, so a rank carries **all**
256 experts -- tensor parallelism shards each expert along its intermediate
dimension rather than giving ranks different experts -- and a round at twelve
rows touches 81 of them, not five of sixteen.

At 302 MB a rank for the whole bank that is 95.6 MB read a layer, and the cube
half's 71.25 microseconds covers it at 1341 GB/s, or 83 percent of the 1607 the
spread arm measured on this die. The cube work is bandwidth-bound and cannot be
made shorter without reading fewer bytes.

So 71.25 is the FFN body's floor, today's window is 136.40, and the 61
microseconds between them is the last mechanism on this side. It also moves the
layer's traffic floor: 11.9 MB of attention weights, 9.4 of KV cache and 95.6 of
touched experts is 116.9 MB a rank, or 81 microseconds at 1442 GB/s, against a
235 microsecond budget for 2.9 milliseconds. The ratio is 2.9, not the 5.7 this
project recorded from the wrong expert count.

[`decompose.py`](decompose.py) reads the trace; [`width12.txt`](width12.txt) and
[`width32.txt`](width32.txt) are its output. The trace is
`megakernel-device-20261021/a3-84/routed-stage-profile.json`, taken with the
instrumented library, so the absolute window carries its instrumentation cost;
the split between work and waiting is what this page uses.
