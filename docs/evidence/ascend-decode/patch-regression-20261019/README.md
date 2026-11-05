# Four patches that passed per-call bit-exactness cost 3.15 ms a token

Paired, one session, one tree, one frozen prefix, width 4, the composed flags,
each arm its own cache root, each figure read from that arm's own record.

## 2026-10-19: this comparison cannot attribute anything to the patches

The static-kernel package cache's index lives under the process's working
directory, and only `VLLM_CACHE_ROOT` was separated per arm. Its key omits the
vendor path and the header contents, so the arm that ran second installed the
package the arm that ran first compiled. At the captured width — the only
width that has static kernels — both arms therefore executed identical
kernels, whatever their overlays said.

So the +3.15 ms is not a measurement of the four patches at the captured
width: the patches were not there. What it could still be measuring is their
effect at the non-captured widths, or the compact declaration, which is a
Python-level change and applies at every width — and the page below already
localises the cost to that declaration.

Which arm ran first, and hence whose source both arms ran, is not recorded.
[the cache](../w4a8-race-closed-20261019/README.md)

| | baseline | four kernel patches + compact declaration |
| --- | --- | --- |
| TPOT | 14.9121 / 14.4877 / 14.3187 | 17.6350 / 17.8901 / 17.5922 |
| median | 14.4877 ms | **17.6350 ms** |
| range | 0.5934 | 0.2979 |
| rounds | 311, 311, 311 | **326, 335, 326** |
| token stream | one, `df10d06f911c` | one, **`df10d06f911c`** |
| winning vendor index | 3, barrier only | 1, the four-patch object |

**+3.15 ms a token, a 21.7% regression, against an expected −0.9.** The
overlay resolved to the patched object, so this is a measurement of the
patches and not of nothing.

## The two facts worth more than the number

**The token stream is bit-identical to baseline** — the same digest across all
three requests in both arms. A 1024-token generation agreeing token for token
is a stronger correctness statement than any of the four per-call gates, and
it covers the kernel patches and the compact declaration together. Whatever is
wrong, it is not correctness.

**The round count moved and stopped being constant within the arm**, 311 to
326/335/326, while the tokens did not. Decomposed: per-round time goes
14.4877 × 1023 / 311 = 47.6 ms to 17.6350 × 1023 / 326 = 55.3 ms, so the round
is 7.7 ms slower *and* there are fifteen more of them. The per-call bench
predicted 3 ms a round saved. **Both halves are wrong, which means the
per-call bench does not reach whatever this is.**

## The gate's hole, found by its own author

The compact-declaration patch bundles two changes: compressing the declared
expert count, and leaving `w2` in ND instead of converting it to FRACTAL_NZ.
The bit-exactness gate compared ND-declaring-256 against ND-declaring-8 — the
declaration varied, the layout held fixed. **It never compared NZ against ND.**
Leaving `w2` in ND was justified on speed alone, 5.603 µs a call, and never on
numerical equivalence.

If the two layouts tile the reduction differently they accumulate in a
different order, so a bf16 down-projection output can differ in its low bits.
That changes no value materially but it moves draft acceptance, and acceptance
sets the round count. **Identical final tokens with a different accept/reject
path is exactly that signature.**

## What this does to the rest of the board

Every expectation on this project's list derived from a per-call measurement is
now suspect, because here a per-call prediction of −3 ms a round produced
+7.7. That includes the kernel three's 0.66 ms, the compact declaration's 2.39,
and the one-sided reduction's 2.48 — the last of which at least was measured
as a whole protocol rather than as a per-call delta, but still outside a round.

Nothing from this set should land on today's evidence.

## A unifying observation

The draft head's argmax is the sensitive element and the target's is not. Width
8 diverges run to run because it has more draft positions for a marginal
decision to flip in; this regression moves acceptance because the draft sees
different low bits from a different reduction order. In both cases the final
tokens are stable while the accept/reject path is not — which is why round
count, not token identity, is the quantity that detects a numerical change
here.
[width 8](../width8-nondeterminism-20261019/README.md)

## The third arm separates them, and the cause is not what I recorded

| | baseline | kernel three only | all four |
| --- | --- | --- | --- |
| median | 14.4877 | **14.7117** | 17.6350 |
| range | 0.5934 | 0.3517 | 0.2979 |
| rounds | 311, 311, 311 | **311, 311, 311** | 326, 335, 326 |
| winner | index 3, barrier only | index 1, four-patch | index 1, four-patch |

All three arms emit the same 1024-token stream.

**The kernel three are clean but unresolvable.** Round count and stream match
baseline, so the comparison is unconfounded, and the median moves +0.224 ms a
token against a predicted −0.2 — inside both arms' ranges. Three requests at a
0.35–0.59 ms range cannot see a 0.2 ms effect in either direction. Not harmful,
not demonstrated.

**The compact declaration is the whole regression**, +2.92 ms a token against
the kernel-only arm, and it is what moves the round count off 311 and makes it
non-constant within the arm.

### Three hypotheses refuted, including the one recorded here

**ND against NZ is bit-identical**, not merely close: zero unequal elements of
49,152 at M = 8 and of 196,608 at M = 32, maximum absolute difference 0.0. So
leaving `w2` in ND is numerically neutral and the "different tiling
accumulates in a different order, moving low bits and therefore acceptance"
story — which an earlier version of this page gave as the likely cause — is
dead.

**`topk` tie-breaking is stable.** The production case the gates never covered
is `active = hidden_states.shape[0]`, the expanded row count, which exceeds
the number of experts carrying rows, so `topk` must break ties among equal
zeros. At active/live of 32/32, 32/8, 64/8 and 32/16, six calls each, the
derived index is identical every time and every output matches the full-bank
reference bit-exactly.

**The compaction did fire**: `_maybe_convert_group_list` always returns type 1.

### So the patch is an instrument, not an optimisation

The arithmetic is sound at every constructible shape, yet in situ the arm's
accept/reject path became non-deterministic. The economical reading is that
the compaction introduces no error of its own and instead perturbs timing in
the W4A8 routed-expert MoE section enough to expose **the race already
localised there** — with `apply_gmm2`'s built-in `GroupedMatmul` A8W4 split-K
among the named candidates, and this patch perturbing precisely that call.
[width 8](../width8-nondeterminism-20261019/README.md)

That makes the patched overlay a **width-4 reproducer** for a
non-determinism previously visible only at width 8, in the one configuration
that is otherwise bit-reproducible. For closing the race that is worth more
than the 2.39 ms the patch was meant to buy, because it moves the
investigation into a deterministic baseline.

## What lands and what does not

`w4a8-compact.patch` does not land: it costs 2.92 ms a token, destabilises the
round count, and the 2.39 ms it was supposed to buy does not survive contact
with a round — the per-call bench measured the declaration's saving but not
the gather sized by the expanded row count, nor the extra launches, nor the
cost of the perturbation.

The three kernel patches and the barrier patch did no harm and changed no
token. Their case rests on the per-call evidence and on correctness, which is
where the barrier's case already was.

## Every gate held fixed something production varies

The layout, the declared-slot count, the tie structure, the call's
neighbours — each gate was correct and each answered a narrower question than
the one that mattered. The end-to-end run is the only thing that found this,
and it found it in an afternoon of machine time against a day of per-call work.

## In flight

A third arm with the four kernel patches and `_COMPACT_EXPERT_DECLARATION`
false, which also restores `w2` to NZ, separating the kernel three from the
declaration-and-layout pair. At 311 rounds and a median near 14.49 the kernel
three are clean and the declaration is the problem; if it also regresses, all
four need re-examining inside a round rather than per call.
