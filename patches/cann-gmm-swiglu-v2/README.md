# Two patches for the W4A8 expert GMM at batch one

## Verified in a round, 2026-10-19: −1.468 ms a round, −0.326 a token

The three kernel patches below were priced per call, about 8.5 µs over 78
calls, roughly 0.66 ms a round. In a round they are worth more than twice
that, and the round is the quantity that matters.

Width 8, one tree, `FIX_ARM=cache-off` so neither arm installs a stale
static-kernel package, each arm's gmm1 object read back from its own recorded
search path:

| arm | gmm1 object | round p10 | TPOT |
|---|---|---|---|
| barrier only | `a9863296` | 57.110, 56.875, 56.839 | 13.1049, 12.9081, 12.8858 |
| + these three | `0a20fe78` | 55.749, 55.343, 55.327 | 12.6554, 12.6037, 12.6427 |

Both arms emit the same 1024 tokens, digest `68f2bc184553`. The effect is 7.3×
the 0.201 ms between-run floor.
[the measurement](../../docs/evidence/ascend-decode/kernelpatch-remeasured-20261019/README.md)

An earlier reading of these same patches was +3.15 ms a token. That comparison
shared a working directory, and so a static-kernel cache index whose key omits
the vendor path, which means at the captured width both arms ran whichever
compiled first. It measured nothing about the patches.

`w4a8-compact.patch` is **not** part of this result and is not verified: its
bit-exactness gate varied the declared expert count while holding the layout
fixed, so it never compared leaving `w2` in ND against converting it to
FRACTAL_NZ.


Both apply with `patch -p1` at the root of the `vllm-ascend-unified` checkout,
against `csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/`, in the order
listed. They live here because this repository is where the measurements are;
they belong upstream.

Both are gated on `loopCount == 1`, which is every decode call and no prefill
call: `mLimit = (64 MiB / 2) / (k·1 + 2·n·2)` is 4681 for the TP16 `gate_up`
shape, so `Ceil(M, mLimit)` first reaches 2 at M = 4682 while batch one gives
M = 8 and batch four gives M = 32. Confirmed at runtime rather than by
arithmetic alone: a binary with 14 extra `SyncAll<false>` inside that branch
measures +32.490 µs, which would be zero if the branch were not taken. The
gate is what keeps prefill byte-identical, and it is the line between a
batch-one fast path and a rewrite.

## `gmm1-straight-line.patch` — 6.43 µs per call

At one split the three-stage loop runs `preProcess(idx+1)` for a split that
does not exist and `postProcess(idx-1)` for an index of −1. Both return on
their own guards — `pre.h:125-130` on `idx >= loopCount`, `post.h:361-366` on
`idx >= loopCount || idx < 0` — so the two calls and the barrier pair around
them are pure cost. Replacing the loop with pre(0), sync, mid(0), sync,
post(0) measures −6.43 µs per call over three runs (−6.426, −9.171, −4.865;
σ 0.442, 0.194, 0.274; 0 positive of 40 in each; p = 1.8e-12) with codes and
scales bit-identical. Batch four gives −8.729 and −10.35.

The saving exceeds the 3.54 µs the two barriers cost — priced separately and
additively at 2.30 for a mixed `SyncAll<false>` and 1.25 for an AIV-only
`SyncAll<true>` — because two no-op stage calls, a whole-struct assignment, an
extra `pipe->Reset()` and the loop control also disappear.

## `gmm1-chunk-span.patch` — a further 2.1 µs per call

`UpdateWorkSpaceSplitConfig` derives the group-slot span this split covers by
walking the table, and at one split the answer is closed-form: every
cumulative row count is below `mLimit` by construction, so none of the three
breaks can fire, the loop always reaches the table's end, and the tail clamps
it to `groupListLen - 1`. Writing that answer saves `2·groupListLen + 1`
global scalar reads per core: 2.118, 2.414 and 1.611 µs over three runs
(σ 0.227, 0.300, 0.313; 0 positive of 40 each), bit-identical.

Measured on a toggle binary one arm of which executes exactly the submitted
code, so the figure is for the shipped form rather than an approximation.

## `gmm1-tiling-rowcount.patch` — a further 2 to 4 µs

`InitWorkSpaceSplitConfig` sums the whole table when it needs only the row
total, which tiling already holds (`base_tiling.cpp:201,434`) and
`postProcess` already reads (`post.h:106`). Gated on one chunk; other loop
counts keep summing. Measured 3.882, 0.663 and 2.127 µs (σ 0.207, 1.530,
0.264; the middle run had a neighbour on the die), bit-identical — read it as
2 to 4 µs rather than as a mean.

The invariant is not newly assumed: feeding a `group_list` whose sum differs
from tiling's M is silently accepted today and gives a wrong answer, because
the activation indexing and the output scale buffer already size themselves
from M while the chunking is derived from the table. The change picks which of
two quantities that must already agree gets read.

## What these do not fix

78 layers of `gate_up` at 8.5 µs is 0.66 ms per round against a 6.866 ms
budget. `down` is out of reach: `w4a8.py:621,643` call
`torch_npu.npu_grouped_matmul`, a CANN built-in outside this package.

And the large item is untouched, though it is now understood. A call's cost is
linear in `group_list`'s length at 0.1556–0.1825 µs per slot, worth 37–50 µs
for a 256-slot table, and **the cost belongs to whichever consumer touches the
table first**: eliminating both pipeline-level traversals moved 38 µs from the
prologue into `op.Process` while the total stayed at 88–93 µs. These three
patches move the bill; they do not cancel it, which is why each is worth only a
few microseconds.

The mechanism is 24 cube and 48 vector cores each pulling a 2 KB table by
blocking scalar global reads, one int64 at a time — 72 unshared streams against
32 cache lines. The fix is one bulk `DataCopy` per consumer instead of 256
serial round trips, which is kernel-internal and needs no prototype change. It
is measured in size, 38.55 µs per call or about 3.0 ms per round, and
unmeasured as a fix.

## `w4a8-compact.patch` — 2.39 ms per round, and it changes where the fix goes

Applies at the root of the `vllm-ascend-unified` checkout. Two changes, both
behind the module constant `_COMPACT_EXPERT_DECLARATION`: `w4a8.py:379` stops
converting `w2` to FRACTAL_NZ, and `apply_gmm2` calls a new module-level
`_compact_expert_declaration` when `group_list_type == 1`.

The route changed because **`index_select` does not work on an NZ tensor**, and
the earlier claim that it did was a false positive: the probe's `attempt()`
lacked `torch.npu.synchronize()`, and TransData failures are asynchronous, so
every expression read as passing. Re-measured one expression per process with
the synchronisation inside the `try`:

| expression on production `w2` (format 29) | result |
|---|---|
| narrow view `w2[32:40]` | passes, stays format 29, contiguous |
| narrow view plus `.contiguous()` | passes |
| `index_select` | **fails** (TransData) |
| per-expert `copy_` into a preallocated NZ destination | **fails** |
| `npu_format_cast(w2, 2)`, NZ back to ND | **fails** |
| the built-in accepting ND (format 2) weights | passes |

So a scattered gather is only expressible if `w2` never becomes NZ, which
means the change belongs at load time. That is not a cost — ND is faster here
even before compression. Down shape, K=128 N=6144, three banks, 32 calls per
graph, 25 rounds, median:

| arm | µs per call |
|---|---|
| NZ, declared 256, scattered — **today's production** | 76.644 |
| NZ, declared 8, contiguous view | 31.240 |
| ND, declared 256, scattered | 71.041 |
| ND, declared 8, gathered | 29.530 |

ND at full declaration already beats NZ by 5.603 µs. End to end, 76.644
against 29.530 plus the 15.190 gather is **31.924 µs saved per layer, 2.39 ms
per round** over the round's ~75 built-in grouped calls. Charging the gather in
full against the ND declaration effect alone gives the conservative 26.3 µs
per layer, 1.97 ms per round. Both exceed the 1.01 ms figured earlier, because
that baseline was ND with a contiguous window while production is NZ with a
scattered one.

### The gates it passed

Real checkpoint, layer 3, rank 0's eight experts' `down_proj` through the real
`process_weights_after_loading`: 49,152 output elements bit-identical to the
full-bank call, `torch.equal` true. The dead tail beyond `sum(group_list)` is
all-zero and bit-identical in both arms, so nothing downstream indexing by
global expert id can see a difference.

The patch's own expression — `torch.sort(torch.topk(group_list, active).indices).values`
plus `index_select` into a resident buffer — was validated separately across
four routing shapes, each run twice to cover buffer reuse: ascending scattered,
contiguous window, unsorted counts, and four at each end. All bit-identical.
It is device-side with static shapes and no read-back, so capture survives.

An operator property found on the way: `npu_grouped_matmul` **overwrites its
activation in place** (max int8 drift 251), identically in both arms;
`per_token_scale` is untouched. A first gate run reported a 1.11 mismatch
purely because both arms shared one `x`.

### What is deliberately not in it

No view path. Selecting it requires testing the group list's contents, which
inside a captured graph is a device-to-host read that would break capture and
insert a synchronisation into the decode stream. The view itself works and
remains available to eager callers.

`index_select(..., out=)` into a resident buffer is kept but is worth nothing
in a graph: 16.439 µs allocating against 16.477 resident, a difference inside
the noise, because graph allocation is decided at capture and replay never
reaches the allocator. It still matters on the eager path.

### A NameError that `ast.parse` could not see, and a test that hid it

The first version of this patch referenced `ACL_FORMAT_ND`, a name that does
not exist anywhere in `vllm_ascend` — the tree defines only
`ACL_FORMAT_FRACTAL_NZ`. `ast.parse` passing says nothing about unbound names,
so "applies cleanly and parses" was not evidence the patch could run, and the
branch that would have raised is the guard returning the weight unchanged when
it is not ND: the first decode under the NZ production configuration.

It survived its own test because `helper_check.py` locally defined
`ACL_FORMAT_ND = 2`. The harness supplied the binding the patch was missing,
validating the logic while masking the error. **A test that provides what the
code under test forgot to import converts a crash into a pass.**

Fixed by spelling it out as a module constant with the reason, and the check
upgraded from "it parses" to resolving every `Name` node against local
bindings, module-level definitions, imports and builtins.

### Open before landing

`maybe_trans_nz` on `w2` may be the right choice at prefill shapes; ND was
only measured faster at decode shapes. The module constant exists for this —
if NZ wins clearly at prefill, the switch has to become per-phase rather than
global. And the merged run putting the patch's expression and the real
checkpoint together in one pass was stopped mid-load; the two halves are
covered separately but not in one.

## A third patch this repository already carries

`tests/glm52/patches/gmm1-swiglu-barrier.patch` is not an optimisation. The
shipped object — sha256 `2444c7e7…`, byte-identical to a no-patch rebuild —
given bit-identical inputs four times produces calls 0 and 1 agreeing while 2
and 3 each differ, codes by up to 114 and scales by 0.83 and 1.02, with all
five input tensors verified for zero drift around every call. Adding one
`PipeBarrier<PIPE_V>()` at `post.h:207` makes four consecutive calls agree,
giving `2eb81546…`. That is a live epilogue race in the exact shape production
runs, and the trajectory gate compares tokens bit for bit.
