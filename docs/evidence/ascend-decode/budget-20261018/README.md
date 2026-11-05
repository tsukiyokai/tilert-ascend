# The round's real budget, and what has to hide inside the weight stream

Counted from the checkpoint's safetensors headers rather than from the
architecture, so packed int4 and per-expert scales are counted as stored.
Activated experts use the measured union of 27.121 of 256 over a four-row
verify. [script](../../../../tests/glm52/bench_weight_budget.py).

```
checkpoint total            376.97 GiB over 235842 tensors
  routed_experts            350.91 GiB
  attention                  13.45
  dense_and_shared            7.22
  embedding_head_and_norm      5.39

read per round, all ranks    63.24 GiB
read per round, one rank      3.95 GiB  (4.24 GB)
streaming floor at 1190.77 GB/s   3.56 ms/round
```

| acceptance | budget at 2 ms/token | floor's share |
| --- | --- | --- |
| 3.433 (today) | 6.87 ms/round | **52%** |
| 5.0 | 10.00 | 36% |
| 6.0 | 12.00 | 30% |

## This retracts the 84% figure

Earlier records put the weight read at 6.83 GB per rank per round and a
streaming floor of up to 6.00 ms, which made 84% of a 2 ms budget look
unavoidable and left 1.1 ms for everything else. Counted properly it is
4.24 GB and 3.56 ms, 52% of the budget, leaving 3.31 ms. The programme's
shape changes with it: the target is not "clear the tax and hope", it is
**drive the round down to roughly the weight-stream time and hide everything
else inside it**.

Today's round is 49.07 ms. Of that, 3.56 ms is irreducible, so 45.5 ms has to
become 3.31 ms — and not by running faster but by overlapping with the weight
stream that must happen anyway.

## What that implies for the collective

The round spends 11.29 ms on collectives — 157 reductions at 9.974 ms, the
entry reduction at 4.356, the wrapping `Cast` at 1.320 — and overlaps them
with compute by exactly 0.000000 ms, cube and vector being serial on stream
81 ([audit](../stack-selection-20261018/README.md)).

63.5 µs for a 48 KiB message across sixteen ranks is 11 GB/s, two orders off
any bandwidth limit, so the cost is rounds and rendezvous rather than
transfer. A one-shot push moves 15 × 48 KiB out of each rank — 48 KiB per
link if the links are independent — and reduces 768 KiB locally, which is
under a microsecond of vector work. The physical floor per reduction is
single-digit microseconds.

At a conservative 10 µs, 157 reductions cost 1.57 ms instead of 9.97, and
they can overlap the weight stream, which they currently cannot at all. That
is 8 to 10 ms per round, the largest single item on the board.

## And it retracts the prerequisite argument

The earlier reasoning — 157 × 63.5 µs exceeds the budget, therefore the
*number* of collectives must fall, therefore EP and MTP replication are
prerequisites — treated 63.5 µs as a constant. It is not: the count can stay
and the per-reduction cost can fall by an order of magnitude. The measured
failure of EP16 ([EP](../ep-megamoe-20261018/README.md), +1.09 ms per round
with spread growing 0.252 → 0.756 ms) is consistent with that correction:
cutting the count was the wrong lever.

## Why it must live inside the kernel

Both external routes are closed on A3: `MatmulAllReduce` has no
`ascend910_93` binary and `MatmulReduceScatter` demands the token axis divide
by the world size, which four rows over sixteen ranks cannot
([vendor fusions](../fused-collective-ops-20261018/README.md)). And a
standalone one-shot push, even if written, still waits for its producer to
finish writing GM — it removes rounds but not serialisation.

Inside a layered kernel the second gain appears: while AIC works layer k's
tiles, AIV pushes and reduces layer k−1's output, and MTE2 streams layer
k+1's weights. That is the only shape in which the round can approach the
weight-stream time.

## Ranked, with the ordering corrected

| item | worth per round | vehicle |
| --- | --- | --- |
| collective inside the kernel, one-sided | 8–10 ms | hand-written MIX kernel |
| amortise the 8 µs per-call floor | a few ms | same |
| on-chip cube/vector handoff | ≤3.3 ms net | same |

This inverts the earlier ordering, which put the handoff first and the
collective last.

## The reservation

The single-digit microsecond figure is derived from physical quantities, not
measured, and this project has already hit 507015 attempting a TP16 one-shot
push with two PEs surviving. HyperMPMD's source explains that failure and
names the recipe — sender never DCCIs a remote address, waiters spin on local
flags, host barrier before first remote visibility, 64-byte signal isolation
([dissection](../hyper-mpmd-20261018/README.md)). Its graph-capture guard
does not apply to us, since it exists for online heap rebuild and our message
size is fixed.

So the next step is not the layer kernel. It is a minimal sixteen-rank
one-sided reduction that drives 63.5 µs down, built to that recipe. Until
that number exists, everything above is arithmetic rather than result.
