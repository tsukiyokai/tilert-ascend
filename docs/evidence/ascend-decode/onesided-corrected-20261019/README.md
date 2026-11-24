# The 11.2 µs was not an all-reduce, and the bf16 path is 16.571

## A brace, hidden by indentation

`} else if (stage == 10) {` was attached to a chain nested inside
`else if (stage != 2)` and was missing a closing brace. The indentation read
as a function-scope chain; the compiler saw a nested one. So stage 10 executed
`TpPushPublish` — the library's float32 `put_nbi` — followed by
`TpPushReduce`'s float32 sum, with no synchronisation at all, and the host read
those float32 sums as pairs of bf16. `TpPushPublishBf16` and
`TpPushReduceBf16` appeared only at their definitions and were never called.

Established by extracting the forty lines verbatim, replacing each call with a
print, and compiling with `g++` to see the control flow — not by reading the
source again.

## The same defect invalidates the figure quoted all night

Stages 4, 5, 6, 7 and 9 cannot reach the `quiet`/`sync_vec_all` chain either,
for the same reason, and stage 9's `TpPushReducePipelined` is unreachable —
it falls through to `TpPushReduce`. `main.cpp` only checked the payload for
stage 0, so none of it showed.

**So 11.2 µs was a floor without cross-PE synchronisation, not a correct
all-reduce.** On one protocol, 400 rounds in a single launch, 50 warmup, 16
blocks, passes interleaved in different orders with a fresh port each:

| path | µs | against HCCL's 32.340 in-graph |
| --- | --- | --- |
| library wrapper, stage 0 — already synchronised | 22.363 | 1.45× |
| float32 direct store with named events, stage 11 | 19.981 | 1.62× |
| **bf16, stage 10** | **16.571** | **1.95×** |
| float32 without sync, stage 7 — the old figure | 11.195 | — |
| bf16 without sync, stage 12 | 8.582 | — |
| `quiet` + `sync_vec_all` alone, stage 3 | 5.142 | — |

So the direct-store and named-event work is worth **11% over the library
wrapper**, not the 2× that the missing synchronisation manufactured. The bf16
path is worth a further 3.41 µs, 17.1%, and 2.61 µs (23.3%) when both sides
skip the sync.

## Why the transfer model over-predicted

The model 0.85 + 0.1175·KiB predicted 12.1 µs at 96 KiB falling to 6.5 at 48,
a saving of 5.64; the measurement gives 2.61 without sync and 3.41 with. The
shortfall is in the reduction half and the barrier: halving the bytes halves
the load, but the vector side now runs sixteen `Cast` calls per chunk and the
`Add` width is unchanged, while `quiet` + `sync_vec_all` does not shrink with
the payload — 5.142 µs measured alone and 8.79 in place, the difference being
what it spends waiting for the slowest rank, which is skew rather than
transfer.

## The correctness check that was still not sharp enough

All three payloads pass 400 rounds on all sixteen ranks with every one of the
24576 elements compared each round, and zero non-zero exit codes.

But the cancelling construction I designed does **not** prove float
accumulation. With the negation at rank 1, the accumulator returns to zero
before the small terms arrive, so a bf16 accumulator and a float one both give
14, 28, 56. It tests slot coverage and ordering, nothing more.

Moving the negation to the last rank does prove it: the accumulator must then
carry 4096 through all fourteen additions, and at 4096 the bf16 step is 32, so
a bf16 accumulator loses every `e` and lands on 0.0 while float gives 14. A
host model confirms the discrimination (`e=1 negative_rank=15 fp32_acc=14.0
bf16_acc=0.0`), and that payload passes 400 of 400.

So my "order-sensitive" input was order-sensitive and not
precision-sensitive, and the distinction matters precisely for the claim the
bf16 path makes.

## What the collective is worth, corrected

Against HCCL's measured 32.340 µs in-graph, the bf16 one-sided reduction saves
15.77 µs per reduction, or **2.48 ms a round** over the round's 157
reductions. Against the correct float32 one-sided path it saves 0.54 ms.

The 1.54 ms of `Cast` is **not** added to this. That item is 94% inside the
target graph with its producer unattributed, so nothing establishes that a
bf16 reduction removes it.
[Cast census](../cast-census-20261019/README.md)

## Two operational notes

`libshmem.so` is not on the default search path; `build/lib` must be added
explicitly. And a port composed as `2972$stage` overflows 65535 at a two-digit
stage — which is what killed one all-rank run, not the code.
