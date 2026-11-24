# The layer is mostly waiting, and a skewed schedule reaches the target

Two of the layer's stages have per-core traces, and both say the same thing:
the two halves of the machine alternate, each idling while the other works, and
the idling costs as much as either half's work.

| stage | cube work | cube barrier | vector work | vector barrier | window |
| --- | ---: | ---: | ---: | ---: | ---: |
| routed FFN body | 71.25 | 67.52 | 23.40 | 111.74 | 136.40 |
| MLA tail | 13.02 | 19.84 | 12.32 | 19.52 | ~32.8 |

In the tail the alternation is visible stage by stage: the cube spends 9.41
microseconds on `output_matmul` while the vector cores wait 10.10, then the
vector cores spend 10.10 on `output_dequant` while the cube waits 11.06. In the
FFN body the vector half waits 87.46 across the two matmuls and the cube waits
39.77 across the vector stages.

## What the layer's floor is under a skewed schedule

A walk of a reduction is nothing but streaming that slice of the weight, so the
cube's work is the layer's reads: 116.9 MB a rank -- 11.9 of attention weights,
9.4 of KV cache, 95.6 of touched experts -- which is 73 microseconds at the 1607
GB/s measured on this die. The two traces put the vector half at 0.33 and 0.95
of the cube half, so the vector work is of the same order and perfect skewing
leaves `max(cube, vector)` plus the schedule's barriers.

At 2.29 microseconds a grid barrier:

| barriers a layer | layer us | round ms | TPOT ms |
| ---: | ---: | ---: | ---: |
| 10 | 95.9 | 7.48 | 1.375 |
| 20 | 118.8 | 9.27 | 1.703 |
| 30 | 141.7 | 11.05 | 2.031 |
| 40 | 164.6 | 12.84 | 2.360 |
| today | 650.4 | 50.73 | 9.324 |

So the 2.x millisecond target is reachable arithmetically with up to about
thirty grid barriers a layer, and the whole distance from today's 9.324 is the
waiting. Of the layer's 650 microseconds, roughly 146 is work -- 73 of reads and
a comparable amount of vector -- and the remaining 500 is barrier and idle.

## What this does and does not establish

It is an upper bound on what skewing can give, not a plan that has been run,
and one of its assumptions has since been measured false: the cube's work cannot
be cut into pipeline steps for free, because each step is another invocation and
each invocation walks the reduction again. Cutting the front stage eight ways
added 70 microseconds of walks to hide 25 of vector work. So the bound holds
only for the part of the layer whose cube work is **already** divided into
invocations -- the FFN's expert loop and the tail's head loop. See
[`skew-needs-separate-invocations-20261022`](../skew-needs-separate-invocations-20261022/README.md).

Three further gaps:

The stage traces are a native FFN body and a standalone MLA tail, taken with
instrumented libraries; the 650 microseconds a layer is the vendor round divided
by 78. Those are different scopes, and the vector-to-cube ratio is taken from
two samples that disagree by threefold.

Skewing needs independent items to interleave, and the two stages supply them:
the FFN's 81 touched experts spread three or four to a core, and the tail's
heads. Whether the attention front's selector and top-k offer the same is not
measured.

And the schedule is unbuilt. One attempt at a cross-half handoff deadlocked on
`SyncAll` and a second on an atomic counter; the pattern that does work in this
repository is co-scheduling independent work inside one barrier region, which
`ROUTED_SHARED_PIPELINE` uses for 32 microseconds.

[`decompose_mla.py`](decompose_mla.py) reads the tail's trace;
[`mla-tail.txt`](mla-tail.txt) is its output. The FFN body's decomposition is
[`ffn-idle-halves-20261022`](../ffn-idle-halves-20261022/README.md).
