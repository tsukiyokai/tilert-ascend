# Width 8 is not a valid result yet: the trajectory diverges run to run

## Located: the W4A8 routed-expert MoE段 under TP, and a race that re-ignites every forward

**The narrowest bracket.** Two separate processes, identical input: the first
divergence is in **round 0**, in the target's `predictions` alone. Candidates
match (`[6500, 382, 262, 9897, 721, 3467, 284, 656]`), accepted tokens match
(`[382, 262, 576]`), and predictions split from the fourth position. Identical
up to it: prefix injection, the eager width-1 seed draft
(`mtp_round.py:95-99`), six M1 captured-graph replays (`mtp_round.py:106-110`),
and the eight host token ids handed to `verify` (`mtp_round.py:111`). So the
first divergence is the width-8 verify's output, `cann_model.py:492-529`.

**It is not setup.** Six consecutive `target.verify` calls on identical input
in one process give six different row-wise digests, all eight rows differing,
with |Δ|max 0.297–0.605 against |ref|max 3.45–4.19 — 7.1% to 17.5% relative.
That is corrupted data, not reduction noise, and the same magnitude class as
the int8 code difference of 114 seen in the vendor epilogue race. Captured-graph
replay diverges identically, so the graph is not a necessary condition.

**Bisected.** Per-layer, 795 module hooks, four repeats with identical module
order: layers 0, 1 and 2 are bit-identical including their six TP all-reduces,
as are layer 3's norms, attention and gate, and the first difference is
`model.layers.3.mlp.experts`. Per-operator, 644 instrumented calls:
`apply_gmm1_act_quant`, `npu_grouped_matmul`, `apply_gmm2` and
`npu_moe_token_unpermute` are clean and the first difference is a following
`npu_dynamic_quant`.

**But the ignition point is random.** Three independent bisections put it in
three different places. What is stable across all three is that the dense
layers plus the first one or two MoE layers are bit-identical and everything
after differs — 73 of 75 MoE layers. So this is a race that re-ignites
somewhere in the MoE段 on every forward, not a fixed line and not
accumulation.

**The confirming differential.** `--expert-parallel` is the only change that
restores reproducibility: one stream of three, 249/249/249 rounds, TPOT range
0.110 ms, round-0 predictions unmoved across five samples. It replaces the
whole dispatch, GMM, SwiGLU and combine path and removes the MoE TP reduction
(`cann_model.py:144-147`). Five independent width-8 baselines give three
streams each.

It is a control and not a workaround: w8-ep's median TPOT is 15.612 ms against
width 4's 15.609, so EP16 eats the whole of width 8's gain — consistent with
the earlier finding that EP16 costs more per round than TP16.

**Two candidates remain, both with a file:line.** The shared expert's
`npu_dequant_swiglu_quant` (`shared_experts.py:546`), where
`dequant_swiglu_quant.h:793` and `:797` have only a non-ordering
`SetDeqScale` between two `Cast` calls while the two pairs that follow both
carry `PipeBarrier<PIPE_V>`. Or `apply_gmm2`'s built-in `GroupedMatmul` A8W4
split-K with `SetAtomicAdd<half>`, whose split dimension is chosen by host
tiling from M — and M is 32 at width 4 against 64 at width 8.

## A consequence nobody has stated: width 4 passes on margin, not on correctness

Two measured facts, placed together.

The vendor epilogue race is **worst at M = 32** — four distinct results over 64
consecutive calls, three or four of them wrong, int8 codes differing by 115 on
a range of ±127. And M = 32 is exactly width 4's verify shape, since batch one
at top-8 gives M = rows × 8.

Width 4 is nevertheless bit-reproducible end to end, across seven independent
processes, every one at 319 rounds with a single token stream.

The only way both hold is that the corrupted values **do not flip any argmax**
at width 4. That is margin, not correctness. The same mechanism explains why
width 6's round-0 predictions agree while its round 1 diverges: that round's
top-2 margins happen to be wide.

So the barrier patch is needed at width 4 as well — not for reproducibility,
which already holds, but because the arithmetic is wrong by up to 114 int8
codes and the release gate is passing on the distance between the top two
logits rather than on the computation being right. A gate that compares tokens
bit for bit cannot see this, because the tokens are identical.

This also sets the standard for the width-8 fix: making width 8 reproducible
is not the same as making it correct, and a change that widens margins would
do the first without the second.

## The threshold is between 4 and 6, not at 8

Width 4 is deterministic **across processes**, not only across requests: seven
independent processes each give one token stream and 319 rounds, and a
same-session control gives 307/307/307 with a 0.286 ms range and round-0
predictions unmoved.

But width 6 gives three streams of three, 232/230/241 rounds, first divergence
in round 1. Its round-0 predictions hold only because that round's top-2
margins are wide, not because width 6 is clean. So the asymmetry is real and
its boundary is 4 to 6.

## Eliminated by measurement, so that nobody re-tests them

The down-stepping width hypothesis I had put first: refuted by counting. All
879 width-8 rounds ran at width 8, `room` never fell below 129, and
`mtp_state.py:166-169`'s down-step branch was never taken — which also clears
the `torch.empty` input buffers, since one run uses one width and
`positions`, `slots` and `input_ids` are fully written every round.

SFA attention and the indexer: two independent differentials. Layer 3's
attention is bit-identical before the first difference, and swapping the whole
attention kernel for `gather_fia` at width 6 still gives three streams. **A
hypothesis that fit perfectly and is nonetheless wrong** is worth recording:
`sparse_flash_attention_kernel_mla.h:645` sets
`info.mSizeV = (info.mSize <= 16) ? …` and with `gSize = 64/16 = 4` the
threshold lands exactly between width 4 and width 6, matching the observation —
and at width 4 the odd AIV of each pair early-exits throughout because
`vecDealM == 0`. Replacing that kernel does not fix the divergence, so the fit
is coincidence.

`_int64_kv_slots`: follows from the attention output being identical.

ACL graph, static kernels and SuperKernel: not necessary conditions, since the
divergence reproduces with `compile_target=False` and `super_kernel=False`.

A second missing barrier in `grouped_matmul_swiglu_quant_v2`: built and tested.
`post.h:126`'s `Cast` and `:133`'s `Muls` have only `mmOutQueue`'s MTE2→V event
between them, which cannot order V→V, while the two pairs that follow carry
explicit barriers. Patched, rebuilt and placed first: still six differing
hidden states and three streams. Not it.

HCCL reduction order, both directions: unset and set, with and without the
barrier active, three streams each. And the dense layers' six all-reduces are
bit-identical, which removes the collective directly.

## The vendor-path trap, read out of my own runs

Every one of my width-8 records resolves the same way, from each run's own
`runtime_env`:

| index | path | provides the operator |
| --- | --- | --- |
| 0, 2, 3 | `fla_npu` vendors | no |
| **1** | the hoist overlay's own `_cann_ops_custom/vendors/custom_transformer` | **yes, unpatched — wins** |
| 4 | my barrier overlay | yes |
| 5 | the in-tree vendor | yes |

So the factory object ran in `barrier/w8`, `barrier/w8-hoist`, `det/w8`,
`det/w8-hoist` and `isolate/run1`.

The trap's shape is more specific than either half of the story suggested.
`activate.sh` already ships `ASCEND_CUSTOM_OPP_PATH` containing a
barrier-patched owner overlay
(`/workspace/offline/cann-workspace-owner-20261017/kernel-overlay/…`, holding
`2eb81546…`) at index 3, ahead of the in-tree vendor at 4 — and
`_prepend_env_path`'s guard (`utils.py:449`) finds the in-tree path already
present and therefore never moves it to index 0. **Inheriting the variable is
safe.** Setting it from scratch is what breaks: it drops the owner overlay,
and the bootstrap then prepends the path derived from
`dirname(vllm_ascend.__file__)` — which, with a Python overlay first on
`PYTHONPATH`, is that overlay's own unpatched copy.

Five completed width-4 records from 09:03–09:28 show the inherited order with
the barrier at index 3, so those runs executed the patched object.
`onesided-20261018` records no `ASCEND_CUSTOM_OPP_PATH` at all, which is the
other dangerous case: with the variable unset the guard does not fire and the
bootstrap hands index 0 to the unpatched in-tree vendor.

## Margin absorption, demonstrated — and scoped to my runs, not to production

`barrier/w4` resolved to the unpatched object at index 1 — **it ran the factory
object** — and produced 307 rounds in all three measured requests, a single
token stream and a 0.141 ms range. So in one run, an operator measured as
non-deterministic at exactly that configuration's verify shape (M = 32: four
distinct results over 64 calls, int8 codes differing by 115) produced
bit-reproducible tokens end to end. The corrupted values never flipped an
argmax.

**But this is not a statement about production, and an earlier version of this
page made it one.** Read out of a production run's own record, the resolution
is: indices 0–2 are `fla_npu` vendors that do not provide the operator, index
3 is the owner overlay carrying `2eb81546…` — the barrier-patched object — and
index 4 is the in-tree vendor carrying `2444c7e7…`. **The winner is index 3.**
Production has been running the patched object all along; the factory object
sits behind it unused.

So the correct scope is narrower and more useful. The margin absorption is
demonstrated for runs that export `ASCEND_CUSTOM_OPP_PATH` from scratch, which
drops the owner overlay — mine did, and that is why they executed the factory
object. Production inherits the variable and is unaffected. What survives as a
general lesson is that an operator's non-determinism can be invisible
end-to-end because the margins absorb it, which is a reason not to trust a
token-comparing gate as evidence of arithmetic correctness; it is not a live
defect in the deployed path.

## My own barrier evidence was void

**The barrier overlay never took effect in any of my width-8 runs.**
`utils.py:84-86` takes `_CUSTOM_OP_BASE_DIR` from
`dirname(vllm_ascend.__file__)`, and `bootstrap_custom_op_env`
(`utils.py:454-458`) prepends the vendor path derived from it at index 0.
Because `PYTHONPATH` put the hoist overlay first, what got prepended was that
overlay's **own unpatched vendor copy** — byte-identical to in-tree — ahead of
the barrier at index 4. Spelling the in-tree path second does not help,
because the bootstrap does not prepend that string.

With the barrier genuinely first, width 8 still gives three streams, so the
elimination stands. But the evidence I published for it did not exist, and the
lesson is the one already in this project's record in a different form: an
override installed in the wrong position fails in silence.

## A probe that cannot be used on a collective

Wrapping the TP sum to call it twice on one input and compare gave 314 of 314
unequal with |Δ|max 0.575 against |ref|max 0.038 — a difference fifteen times
the value measured. That is not non-determinism: two calls on the same
symmetric window without a barrier let the second overwrite what peers are
still reading. "Call twice and compare" is unusable for HCCL.


Three arms, one tree, one frozen prefix, 1024 output tokens, three warmups and
three measured requests each, `--target-graph --draft-graph
--draft-projection-tp --compile-target --super-kernel --attention-kernel sfa
--fp32-head`:

| arm | median TPOT | range | rounds per request | token hashes |
| --- | --- | --- | --- | --- |
| width 4 | 14.932 ms | 0.169 | 307, 307, 307 | `df10d06f911c` ×3 |
| width 8 | 13.276 ms | 0.940 | 220, 236, 238 | three distinct |
| width 8 + hoist | 13.041 ms | 4.080 | 287, 246, 212 | three distinct |

**Width 4 is deterministic and width 8 is not.** The same binary, prefix and
flags give three different token streams at width 8, with acceptance ranging
4.655, 4.339, 4.303 and the round count varying by 8%. The hoist arm is worse:
3.568 to 4.830.

So the 4.08 ms spread is not timing noise — it is different trajectories
taking different numbers of rounds. Which means two things.

The width-8 TPOT is confounded: its difference from width 4 mixes a real
speed change with an acceptance change, and the project's own ablation
established that acceptance changes are not speed. And the hoist's −0.235 ms
here is unreadable, sitting far inside its own arm's spread, against the
−0.7967 it measured at width 4 with tight ranges.

More seriously, **width 8 fails the trajectory gate**, which compares tokens
bit for bit.

## The attribution, and it is already patched in this repository

The installed kernel is the original: the source at
`csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h`
has no barrier after the `DataCopy` at line 209, and the vendor binary is
timestamped 16:20, before any of tonight's builds.

That is exactly the epilogue race measured independently: the shipped object,
given bit-identical inputs four times, produces calls 0 and 1 agreeing while 2
and 3 each differ, codes by up to 114 and scales by 0.83 and 1.02, with all
five input tensors verified for zero drift around every call. Adding one
`PipeBarrier<PIPE_V>()` after that copy makes four consecutive calls agree.
[decomposition](../gmm-fastpath-20261019/README.md)

`tests/glm52/patches/gmm1-swiglu-barrier.patch` carries that line and is not
installed.

## The barrier works, and my M-threshold guess was wrong

Installed as a vendor overlay and measured at 64 consecutive calls per cell,
two independent processes per cell, inputs rewritten from a frozen copy before
every call, and **a scratch allocation churned between calls**:

| object | M=8 | M=32 | M=64 |
| --- | --- | --- | --- |
| factory `2444c7e7` | 2 results, 1 of 64 wrong, max diff 68 | **4 results, 3–4 of 64, 115** | 2 results, 1 of 64, 114–115 |
| barrier `2eb81546` | 1 result, 0 wrong | 1 result, 0 wrong | 1 result, 0 wrong |

384 calls with the barrier, zero disagreements. The factory object mismatches
10 of 384, and a difference of 114–115 on an int8 range of ±127 is an element
written wrong rather than rounding drift.

The load order was measured rather than inferred, because installing it in the
wrong order fails silently: barrier first gives eight identical results;
pristine first gives two distinct results with call 4 differing by 99. CANN
takes the first matching vendor, so the prepend is

```
ASCEND_CUSTOM_OPP_PATH=<overlay>/vendors/custom_transformer:/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
```

with the second element spelled exactly so that `bootstrap_custom_op_env`
(`utils.py:454-458`) finds it already present and leaves it in second place.

**But my hypothesis that the race needs a large M is refuted.** The factory
object is worst at M=32 — four distinct results, three or four of 64 wrong —
and M=32 is width 4's verify shape, the arm I observed as clean. M=64 is only
one in 64. So width 4's reproducibility is not because its shape fails to
trigger the race, and the attribution behind this withdrawal is not yet
established.

Two candidates remain, and the decisive test is re-running the three arms with
the overlay prepended. Either the trajectory's sensitivity differs — a
corrupted expert row only probabilistically flips an argmax, and at width 4 a
polluted draft the verifier rejects still yields the same accepted token,
while at width 8 the measured quantity *is* the acceptance rate, so one
flipped decision changes rounds, hash and TPOT together — or width 8 has a
second source, with the `_int64_kv_slots` identity caching as the named
candidate.

## A tight loop is not a determinism test

The first attempt at this matrix fired eight identical calls inside one
process and read **all ten cells as deterministic, including the known-bad
factory object**. A tight loop does not change the workspace the operator
receives, so reads of regions it never writes return the same stale bytes
every time and the race is hidden. Adding an allocation churn between calls
exposed it immediately in three shapes.

The churn is not adversarial, it is faithful: a real round allocates and frees
hundreds of tensors between two expert matmuls.

This pairs with the earlier finding that a harness locally defined a constant
the patch had failed to import, turning a crash into a pass. Both are the same
failure: **the test supplied something the thing under test was missing** —
there a name, here a static environment it had no right to depend on.

## With the barrier installed, width 8 still diverges

Three arms re-run with the overlay prepended, same tree and prefix:

| arm | median | range | rounds | distinct token streams |
| --- | --- | --- | --- | --- |
| width 4 | 14.533 ms | 0.141 | 307, 307, 307 | **1** |
| width 8 | 12.688 ms | 2.267 | 218, 255, 221 | **3** |
| width 8 + hoist | 13.030 ms | 0.610 | 247, 238, 243 | **3** |

**The barrier is not the cause.** Width 4 stays deterministic and width 8
stays at three streams, so this withdrawal's attribution was wrong and the
caution against accepting it was right. There is a second source at width 8.

The barrier should still be installed — 384 calls with zero disagreements
against the factory object's 10 — but it buys correctness on that operator
rather than reproducibility on the round.

## Three hypotheses eliminated, and the divergence is process-level

| hypothesis | test | verdict |
| --- | --- | --- |
| the vendor epilogue race | barrier overlay installed, 384 calls clean | **not it** — width 8 still gives three streams |
| HCCL reduction order | `HCCL_DETERMINISTIC=true` | **not it** — rounds 231/227/223, three streams |
| state leaking across requests in one process | three separate processes, one request each | **not it** — two completed runs gave 228 and 280 rounds, two streams |

So it is process-level non-determinism with identical inputs, and the question
narrows to what differs across processes at width 8 and not at width 4.

What remains, as leads rather than conclusions. Something read before it is
written whose content depends on allocation history: `cann_model.py:372-384`
allocates `_input_buffers` per width with `torch.empty`, and a width-8 run also
executes widths 6, 4, 2 and 1 because `mtp_state.begin()` steps down through
`WIDTHS` as the token budget shrinks, while only width 8 has a captured graph.
A buffer sized or filled for one width and read at another would produce
exactly this signature. And `_int64_kv_slots` (`sfa_v1.py:459-474`)
invalidates by tensor identity alone, with three safeties and no assertion
among them.

One control still missing: width 4 has been shown reproducible across requests
within a process but not across processes. If it diverges across processes too,
the asymmetry this page is built on does not exist and the real question
becomes why the round count is so much more sensitive at width 8.

## The candidate that was in my own script



`HCCL_DETERMINISTIC` is unset on this machine, and the benchmark only records
it into the report rather than setting it. With HCCL free to vary its
reduction order, a TP all_reduce's float sum differs run to run, which flips a
marginal `argmax` occasionally — and width 8 has more draft positions for one
to flip in, which is exactly the asymmetry observed. That is one variable and
it is being measured now.

If that is not it either, the remaining named candidate is the
`_int64_kv_slots` identity caching: it invalidates by tensor identity only
(`sfa_v1.py:459-474`), so a reused metadata object makes the check always hit
and `cache_index` keeps the first round's int64 copy. Three independent
safeties prevent it today and none of them is an assertion.

## Why both measurements of width 8 can be right

An earlier investigation reported five runs at width 8 emitting bit-identical
tokens; this one reports three streams from three requests. A race is
sensitive to timing, so both can hold — and that is the reason a determinism
check has to run in the same session as the number it licenses, rather than
being inherited.

## What has to happen before width 8 counts

Build and install the barrier-patched kernel, then re-run width 8 and check
that the three measured requests agree bit for bit and on their round count.
Only then is the median a speed claim. Until that is done, **width 8's
13.276 ms is withdrawn as a TPOT result**, and the width-4 figure of 14.932 ms
on this tree stands as the only deterministic point measured here.

Note what this does not touch: the hoist's −0.7967 ms was measured at width 4
with all six arms token-identical over 319 rounds, so it stands on its own
configuration.
