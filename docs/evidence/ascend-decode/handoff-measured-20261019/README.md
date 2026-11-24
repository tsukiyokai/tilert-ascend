# The GM round trip costs 0.87 ms a round, so the missing on-chip path is not the lever

One die, no HCCL, no model. `tests/glm52/handoff`.

| mode | per round trip |
| --- | --- |
| the read depends on the write | 1.364 µs |
| same bytes, no dependency | 1.381 µs |

A 32 KiB tile over 20 blocks, written and read back. **The dependency costs
−0.017 µs — nothing.** So the figure is transfer, 895 GB/s in both directions
together, which is an L2-served rate.

At 637 cube-to-vector handoffs a round that is **0.87 ms**: 2.4% of the
verify's 36.02 ms fixed cost and 1.7% of a 52.26 ms round.

## This refutes a premise I had been building on

The chip difference table gives `L0C → UB` as absent for this generation
while David carries 128 to 256 bits, and `L1 → UB` is absent too, so a cube
result reaches the vector unit only through global memory. That much stands on
three independent sources.
[the path](../cube-vector-path-20261019/README.md)

What does not stand is the conclusion I drew from it — that the absence is
what makes a layer-scale kernel unreachable here, because AMD's design rests
on the on-chip handoff. Restoring that path would be worth at most 0.87 ms of
a 52 ms round. The absence is real and almost free.

## What is still unmeasured

This is a vector-only round trip: one core writes global memory and reads it
back through its own pipeline. A real cube-to-vector handoff adds an AIC-to-AIV
rendezvous, which is not measured here and was what made the first version of
this benchmark deadlock — under `core_ratio(1, 2)` one AIC block's
`CrossCoreSetFlag` faces two AIV blocks' `CrossCoreWaitFlag`, and the kernel
hangs. `shmem/fused_proj_reduce_kernel.cpp` carries the same never-run
pattern.

So the open question narrows from "what does a handoff cost" to "what does the
cross-core rendezvous cost", with the transfer half now priced at 2.4% of the
target.

## Where that leaves the 36 ms

The verify's fixed cost is 36.02 ms, 91% of it at width 8, and 9.3× the whole
round's weight floor. It is not the rows — the verify costs 39.496 ms at six
rows and 39.677 at eight. It is not the GM transfer. Over roughly eighty
layers it is 450 µs a layer, and what a layer spends that on is now the
question.
