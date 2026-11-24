# The host enqueues a round in 2.14 ms and then waits 50.45: the round is device work

2026-10-23, a3-180, sixteen dies, 78 layers, `shared_fia` at width 8 with the
metadata hoisted -- an arm that passes its gate and reads TPOT 14.605 ms.

`RoundPhases` now resets each request, so this covers 277 measured rounds and
not the graph warmup, and it reports how often each step was marked:

| step | ms a round | marks a round |
| --- | --- | --- |
| propose, the host drafter | 0.037 | 1.0 |
| verify, enqueueing every kernel | **2.138** | 1.0 |
| host_tokens, the round's one synchronisation | **50.454** | 1.0 |
| resolve, commit and truncate | 0.090 | 1.0 |
| sum | 52.720 | |

One mark each makes the partition exact, and the sum matches the round the
timings give independently: 14.605 x 3.693 accepted = 53.9 ms.

## What this settles

The host issues the entire round -- 78 layers of kernels -- in **2.138 ms**,
about 1.1 microseconds a kernel, and then blocks for 50.454 waiting for the
device. So the host is 4 percent of the round.

That ends the reading this project carried for days: that the round's fixed
term was host enqueue at tens of microseconds a kernel, and that fusing to cut
kernel count would therefore pay. An earlier entry had already measured the
same scalar multiply at 13.931 us back to back and 16.696 as host enqueue and
concluded the fixed term "reads the host"; at the round's own scale that is
wrong, because the host's whole contribution is 2.1 ms of 53.9.

The target is device time. 50.454 ms over 78 layers is **647 microseconds a
layer**, against the 555 this session's single-die ledger accounts for -- MoE
at the roofline plus its per-call tiling, the collective's rendezvous,
attention and routing. The ledger is the right shape; what it prices is real
work, and 2.x needs that work to fall, not the kernel count.

It also explains the ladder's one reading: hoisting the metadata and capturing
the rope refresh moved TPOT from 14.743 to 14.605, 0.9 percent, with the
output identical in all three of its measures. `handoff.md` priced that lever
at 7.28 ms a round. It is 0.14.
