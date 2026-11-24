# What the expert GMM actually spends its microseconds on

Measured on a3-84, device 7, one binary per arm selected by an input bit that
is arithmetically inert, paired arms interleaved A,B,B,A in one process, in a
captured graph rather than eager, with six weight banks rotated so each batch
of 32 calls touches 202.1 MB of distinct addresses against the die's 192 MiB
L2. Shape is the TP16 `gate_up`: K=6144, N=256, M=8 — batch one's top-8 gives
each active expert exactly one row.

## The measurement had to move into a graph first

In eager mode **this operator is host-bound and an event pair measures the
host**. The host enqueue floor is 109.90 µs per call against an event-pair
reading of 113.04; inside a replayed graph the host drops to 0.66–1.48 µs and
the device time appears at 98.47. One of this investigation's own earlier
conclusions — that the cost is independent of table length — was an artefact
of that and is withdrawn; re-measured in a graph the slope is 0.182 µs/slot
rather than 0.037.

## The call, decomposed

**This is not a decomposition and an earlier version of this page presented it
as one.** A call's duration is the critical path after the AIV and AIC paths
join at two mixed barriers, so removing one core type's work shortens the call
only when that core type was the critical path, while shortening the table
shortens both paths. The ablation figures are therefore lower bounds.

| item | µs | status |
| --- | --- | --- |
| **cost linear in `group_list` length** | **37–50** | mechanism certain, 4.157 attributed |
| kernel launch floor | ~23 | by subtraction, so inherits the above |
| cube matmul | ≥12 | ablation lower bound |
| mandatory mixed barriers | 4.6 | priced directly, additive |
| removable barrier pair | 3.5 | priced directly, landed |
| vector tail | ≥5.9 | ablation lower bound |
| weight stream's HBM increment | 3.4 | measured directly |
| MSD activation split | 0.1 | measured directly |

## There is no universal floor: dispatch is 1.2 µs and prologues are per operator

Measured in one process, one set of captured graphs, 32 calls per graph, 30
rounds taking the minimum then the median, host 0.44–1.17 µs per call:

| operator | µs per call in a graph |
| --- | --- |
| `torch.add`, 8 float32 | 1.127–1.310, floor 1.192 |
| `npu_dynamic_quant`, 4×32 bf16 | 2.797–3.590 |
| `npu_quant_matmul`, 4×32 by 32×32 | 8.183–8.808, floor 8.303 |

So `fixed-cost-20261018`'s 8 µs reproduces exactly and **is that operator's own
per-call cost, not a dispatch floor**. The dispatch floor is 1.2 µs.

This GMM's own prologue — enter, read the one base tiling member it needs to
branch, leave — is **16.4 µs** (16.424 and 16.428 over two runs). Sliced by
moving the cut later, paired graph replays within one binary, σ ≤ 0.26, 30 of
30, 256 slots:

| segment | µs, 256 slots | µs, 8 slots |
| --- | --- | --- |
| enter + base tiling | 16.43 | 16.42 |
| + two tiling copies (one `TCubeTiling`) + `mm.Init` | 4.54 | 5.18 |
| + `op.Init` (global buffers, sum the table) | **38.55** | ~0 |
| `op.Process` (two barriers, three stages) | 30.67 | 21.48 |

The 67–70 µs was never a floor. It is 16.4 of launch, 4.5 of `TCubeTiling` and
`mm.Init`, 38.6 of the table's first traversal, and 4.6 of barriers. Of the
three things I had guessed, the tiling copy is 4.54 and the barriers 4.60
while `preProcess` is 0.1; the large one was the table, which I had not
listed.

**The round's arithmetic**: 2,416 calls at the real 1.2 µs floor is 2.9 ms,
20% of the current 14.247. The 8.3 µs figure cannot be multiplied uniformly —
2,416 × 8.3 = 20.1 ms already exceeds the round — so most calls must be
cheaper than `npu_quant_matmul`. The megakernel's prize is the dispatch floor
plus the sum of per-operator prologues, and this GMM's share of the latter is
78 × 16.4 = 1.28 ms.

**The operator is not weight-bound at batch one.** Thirty-two windows crossing
L2 against one window inside it is 98.665 versus 95.301 µs, a 3.36 µs
difference against the 4.09 µs that 6.29 MB at 1540 GB/s predicts. The weight
stream is three microseconds of a hundred.

## The empty-slot scan is the largest device-side item in the round

Same binary, same process, same weights, same activation; the two graphs
differ only in whether the `group_list` tensor is 256 long or 8. Both arms
read only experts 0–7, 37.9 MB and fully L2-resident, and the first call's
output codes are bit-identical. Long table 95.301 µs, short table 55.387,
paired difference **39.870 µs**, σ 0.195, 30 positive of 30, host 0.72 µs.
That is 0.1608 µs per empty slot.

Three consumers walk the dense table: `mid.h:191-223` takes one
global-memory scalar read per slot and only reaches its `continue` at 209-211;
`InitWorkSpaceSplitConfig` (pipeline.h:148-152) sums the whole table on every
core when `groupListType == 1`; `UpdateWorkSpaceSplitConfig` walks it again.
With mLimit 4681 against 8 total rows, `rightMatrixExpertEndIndex` always
reaches `groupListLen - 1`, so where the active experts sit does not matter —
only the table's length does.

**78 layers × ~40 µs = 3.11 ms per round, 21.8% of the current 14.247 ms** —
larger than the weight stream, the MSD round trip, the barriers and the vector
tail combined.

But **only 4.157 µs of it is attributed to code and removable**, and the rest
is reproducibly caused by table length while being localised to nothing. The
two walks that can go are `InitWorkSpaceSplitConfig`, which sums the table
when it needs only the row total that tiling already holds
(`base_tiling.cpp:201,434`, read by `post.h:106`), worth 0.908 and 1.517 µs;
and `UpdateWorkSpaceSplitConfig`, whose span is closed-form at one split,
worth 3.574 and 1.566. Together 4.157 µs, additive, bit-identical.

`midProcess` genuinely needs per-slot counts, because its weight addressing
accumulates `wBaseOffset` (`mid.h:141-153`) rather than indexing by
`groupIdx`, so an empty slot must still be walked. A fourth consumer walks it
twice more: `postProcess` sums the table at `post.h:271-273`, reads it per
group at 279-283, and runs to the table's end at 338.

Batching the table into UB cannot reach the rest. The 769 global scalar reads
the two patches remove cost 5.4 ns each — about ten cycles at 1.85 GHz, so
already cached — while the per-slot length term is 0.16 µs, about 296 cycles.
Thirty times apart. And `midProcess` has no `TPipe` at all (`mid.h:88-113`),
so giving it UB staging is not a local change.

## The table cost belongs to whoever touches it first

Eliminating both pipeline-level traversals moved the cut-point cost from 59.52
to 23.1 µs while `op.Process` rose from 30.67 to 69.6 — and **the total did not
move, 88–93 µs** (σ 0.108 and 0.100, 30 of 30). The bill changed hands rather
than being cancelled, which is exactly why the three elimination patches
measured only 0.9, 1.6–3.6 and 4.16 µs.

The mechanism: a 2 KB table pulled by 24 cube and 48 vector cores, each doing
blocking scalar global reads of one int64 at a time — 72 unshared request
streams against the same 32 cache lines, serialising. 38.55 µs over 248 extra
slots is 0.155 µs per slot, matching the 0.1556 measured directly.

**So the earlier conclusion that UB batching cannot help is withdrawn**; it
rested on the moving-house illusion, inferring from "deleting one consumer is
worth 4 µs" that each read costs 5.4 ns and is already cached, when the cold
cost simply passed to the next consumer. One `DataCopy` of 2 KB is a single
MTE2 bulk transfer of 32 lines against 256 serial scalar round trips. It is a
kernel-internal change: no operator prototype, no index tensor, no full
package build. Not yet built — the four consumers span three classes and
`midProcess` has no `TPipe` — so 38.55 µs is a measured size with an
identified but unmeasured fix.

Localisation used three cuts at the kernel entry crossed with two table
lengths, each cut a paired graph replay within one binary — the same
discipline as everything else here, and neither msprof nor `CrossCoreSetFlag`
was needed.

## The straight-line path at one split: 0.50 ms per round, ready to land

`loopCount = Ceil(M, mLimit)` with mLimit 4681 is 1 for every decode call
(M=8 at batch one, M=32 at batch four; it reaches 2 only at M=4682). Confirmed
at runtime, not just in arithmetic: a binary with 14 extra `SyncAll<false>`
inside the `loopCount == 1` branch measures +32.490 µs, which would have to be
zero if the branch were not taken.

Both guards return early — `pre.h:125-130` on `idx >= loopCount`,
`post.h:361-366` on `idx >= loopCount || idx < 0` — so the loop's
`preProcess(idx+1)` and `postProcess(idx-1)` are pure cost, as is the barrier
pair around them. `tempWorkspaceSplitConfig` is zero rather than
indeterminate, since `WorkSpaceSplitConfig` carries NSDMI on every field; what
saves it is the `< 0` clause, not the loopCount comparison.

Replacing the loop with pre(0), sync, mid(0), sync, post(0) when
`loopCount == 1`, leaving the general path untouched, measures **−6.43 µs per
call** (three runs: −6.426, −9.171, −4.865; σ 0.442/0.194/0.274; 0 positive of
40 in all three; p = 1.8e-12), with codes and scales bit-identical. Batch four
gives −8.729 and −10.35.

That is more than the 3.54 µs the two barriers cost, because the two no-op
stage calls, a whole-struct assignment, an extra `pipe->Reset()` and the loop
control also disappear; the 2.9 µs difference is measured but not attributed
further.

Barriers were priced separately and additively: mixed `SyncAll<false>` 2.30 µs
and AIV-only `SyncAll<true>` 1.25, summing to 3.55 against 3.54 measured for
the pair.

**`down` is out of reach**: `w4a8.py:621,643` call
`torch_npu.npu_grouped_matmul`, a CANN built-in outside this vendor package,
and rebuilding it would mean touching /opt/ascend.

## The shipped binary is non-deterministic at this shape

The factory object — sha256 `2444c7e7…`, byte-identical to a no-patch rebuild —
given bit-identical inputs four times produces calls 0 and 1 agreeing while 2
and 3 each differ, codes by up to 114 and scales by 0.83 and 1.02. All five
inputs were checked for drift before and after every call: 0.0.

Adding the repository's one-line `PipeBarrier<PIPE_V>()` at `post.h:207`,
after the SwiGLU merge copies `workspaceLocal` into `_inMMLocal`, makes four
consecutive calls agree. So `tests/glm52/patches/gmm1-swiglu-barrier.patch`
fixes a live epilogue race that ships in the vendor package, in the exact
shape production runs — which matters because the trajectory gate compares
tokens bit for bit.

## Warming the table on chip: refuted as a cheap fix

Three variants, each one binary with two arms selected by the inert input bit,
graph replay, paired:

| where the bulk read happens | result |
| --- | --- |
| both core types | **hangs** — 4m53s at 100% AICore, killed |
| vector cores only | runs, bit-identical, **5.512 µs slower** (σ 0.271, 25 of 25) |
| cube cores only | **hangs**, killed by a 150 s watchdog |

Cube cores cannot do `TBuf<TPosition::VECCALC>` plus a GM→UB `DataCopy` plus
an `MTE2_S` flag, which matches `cube_vector_combine=split` in
`Ascend910_9391.ini`: UB and the vector unit sit on the side separate from
cube. So "each core does its own `DataCopy`, nothing needs sharing" does not
work here — the 24 cube cores have no UB to copy into.

And the vector side alone is not where the time is. Warming only AIV should
have made the call much faster if the 38.55 µs lived there; it made it slower
by exactly the batching machinery's own cost. **Both sides pay for the first
traversal**, and removing one leaves the other holding the first barrier.

The shape that could work, not built: the side with UB compresses the table
into a short list of expert id and cumulative count in workspace, publishes it
through the pre→mid barrier that already exists, and `midProcess` walks that
list instead of 256 slots. Still kernel-internal with no package build and no
host index tensor, but it needs a workspace allocation and `midProcess`'s
weight addressing to change from `UpdateMnConfig`'s accumulation
(`mid.h:141-153`) to offsets computed from the list. That is a real refactor,
and now it is clear why it cannot be reduced to one copy per core.

After the two hangs the straight-line control was re-run to confirm the die
was unharmed: 97.888 against 92.079 µs, saving 5.838 (σ 0.156, 0 of 20),
bit-identical, within the 4.865–9.171 range measured before.

## Eager measurement is safe for `npu_grouped_matmul`, and why

Its eager host enqueue floor is 28.128 µs with a median of 28.613, against an
eager event pair at 59.589/60.104 and graph replay at 58.876/59.097 with a
host cost of 0.576–0.602. The host is under half the device and the two
readings differ by 1.7%, so `gmm-l2-20261018`'s 60–63 µs fixed cost and
1138 GB/s convergence stand without re-measurement. An independent check: 8
active experts measure 59.1 µs on device here against gmm-l2's 65.9 and 65.2
at 1 and 4 groups.

The custom fused operator differs because its host floor is 98.97–109.90 µs
through `torch.ops._C_ascend`, 3.5× higher, which is what pushed it past its
own device time. The discipline that follows: before measuring any operator,
print its eager host cost beside the event pair; if they are close, the event
pair is reading the host.

## The prologue list so far

| operator | in-graph per call | prologue |
| --- | --- | --- |
| `torch.add`, 8 float32 | 1.127–1.310 µs | — (this is the dispatch floor) |
| `npu_dynamic_quant` | 2.797–3.590 µs | not separated |
| `npu_quant_matmul` | 8.183–8.808 µs | not separated |
| `npu_grouped_matmul` | 59.097 µs | not separated, no source |
| this W4A8 GMM | 88–93 µs | **16.4 µs**, sliced |

Six-fold spread across operators, which is why no single figure can be
multiplied by the round's 2,416 calls.

## Provenance

A rebuild with no patches gives sha256 `2444c7e7…`, byte-identical to the
shipped object. The control with only the SwiGLU barrier gives `2eb81546…`,
matching `gmm1-tiling-20261018/single-chunk-audit.json`'s recorded control.
The straight-line candidate is `a63d5899…`. The second proof that our binary
ran is the barrier slope above: 14 barriers exist only inside the
`loopCount == 1` branch, so an unselected overlay could not produce 32.5 µs.
