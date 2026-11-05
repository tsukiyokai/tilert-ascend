# Where a round's 35 ms of non-arithmetic device time lives

The profile's compute union for a round is 40.7 ms while the arithmetic it
requires — the weight stream — is 3.886 ms at the die's marginal read rate.
This census locates about 20 ms of the difference. All per-call figures are
in-graph, since the cheapest families are entirely host-bound in eager mode;
single die, 32 or 64 calls captured per graph, median, host 0.40–0.75 µs.

Caveat on the totals: per-call costs measured on a near-idle single die,
multiplied by the round's family counts from a 16-rank profile, account for
about 27 of the 40.7 ms. The remainder is inside SuperKernel fused spans,
which a per-operator method cannot reach.

| family | calls | in-graph per call | per round | intercept |
| --- | --- | --- | --- | --- |
| grouped GMM `gate_up`, custom | ~75 | 88.1–90.2 µs | 6.6–6.8 ms | 16.43 prologue + 4.60 barriers + 39.9 table |
| grouped GMM `down`, built-in | ~75 | 54.4–59.4 µs | 4.1–4.5 ms | 62.23, slope 0.160 µs per active group |
| MLA prolog + attention | 234 | 37.42 µs | 8.8 ms | 29.4, of which ~19 device-side |
| quantised projections | 255 | 8.3 µs | 2.1–2.6 ms | 7.92 / 8.30 |
| cast, quant, norm | 941 | 2.0–3.9 µs | 1.9–3.7 ms | Cast 1.589 … LayerNorm 4.356 |
| MoE routing, combine, other, indexing, RoPE | 836 | 0.88–14.3 µs | ~2.1 ms | as above |

Intercepts sum to about 15.4 ms — `gate_up` 1.23, `down` 4.67, attention 4.45
by the project's own device-side attribution, projections 2.02, elementwise
~1.69, the remaining 836 calls ~1.34 — plus 4.27 ms of redundant table
traversal that is neither prologue nor arithmetic.

## The elementwise family is the surprise

Shape scans give slopes of 0.0085 ns per element for `Cast`, 0.0065 for
int8→bf16, 0.0017 for `Slice`, 0.0142 for `DynamicQuant`, 0.0175 for
`RmsNorm`, 0.0734 for `LayerNorm`. At the model's 8×6144, `Cast` costs 2.005 µs
of which **1.589 is intercept and 0.418 is element work**; `LayerNorm` costs
7.963 of which 4.356 is intercept.

So across 941 calls, **1.5–2.9 ms is intercept against 0.4–0.9 ms of actual
work**. A resident kernel that does this family without re-entering saves more
than the entire `gate_up` prologue, and it is the clearest case anywhere in the
round for fusion as opposed to faster operators.

## Removable by fusion, or intrinsic, by evidence

**Graph entry, 1.079–1.19 µs × 2,416 = 2.61 ms — removable.** Measured at zero
work: `torch.add` on eight float32, and the project's own marginal of
1.01–1.10 µs across chains of 8 to 256 kernels.

**MIX_AIC_1_2 core startup, 16.43 − 1.19 = 15.2 µs — removable.** The arm
behind the entry cut does nothing but read one tiling member and is thirteen
times a pure-vector entry; a resident kernel starts its cores once per round
rather than once per call.

**Tiling copy plus `mm.Init`, 4.54 µs — removable.** A resident kernel holds
`TCubeTiling` instead of copying it out of global memory per call.

**Two mixed barriers, 4.60 µs — intrinsic.** Not a classification but a
structure: one separates the AIV pre-processing from the AIC matmul and the
other separates the matmul from the AIV epilogue, and fused work doing the same
three stages still needs two cross-core rendezvous. The third barrier, the
`SyncAll<true>` already removed, was the redundant one.

**Group list traversal, 39.9 µs custom and ~17 built-in — neither.** It is
redundant work, removable without fusion: linear in the declared length at
fixed content, and it changes hands between consumers rather than disappearing.

**Attention's 19–29.4 µs intercept — not sliced.** The project's own scan puts
it inside the fused kernel's fetch loop at 27.9 ns per row and insensitive to
query heads, so it is not an entry cost and fusion alone will not remove it.

**Quantised projections' 7.92–8.30 µs — not sliced.** 1.08 is graph entry and
roughly 7 is the operator's own prologue, presumed removable by analogy but
unverified.

## The built-in operator pays for empty slots too

Same process, two graphs, four weight banks each, eight active experts at one
row: declaring 256 experts costs 54.372 µs per call against 33.964 when only
the eight used are declared — paired difference **20.424 µs** (σ 0.167, 25 of
25, host 0.75 µs), or 0.0824 µs per empty slot. Half the custom operator's
0.155 but the same disease.

### The footprint control, and the measured net

Repeated with identical allocations on both arms — each four banks of the full
256 experts' weight, scale and bias — reading the same eight experts' share
with the same rotation and reuse distance, in three arms that separate the
declaration from read locality:

| shape | declared 256, active contiguous | declared 256, active scattered | declared 8, gathered | declaration | locality |
| --- | --- | --- | --- | --- | --- |
| K=6144 N=256 | 59.836 | 69.014 | 33.695 | 26.141 | 9.178 |
| K=128 N=6144 | 60.073 | 65.654 | 30.401 | 29.672 | 5.581 |

A third run on a different bank count gave 30.400 and 0.835. So the
declaration effect is **28.7 ± 2.2 µs** and survives the control; the earlier
17 µs attribution was conservative rather than an L2 artefact. The locality
arm swings from 0.835 to 9.178 and is deliberately excluded from the figure
that lands.

**The gather costs 15.3 ± 0.4 µs, not the 4.1 that bytes over bandwidth
predicts.** `index_select` of weight, scale and bias moves 12.6 MB read plus
written at 824 GB/s, 54% of the die's marginal rate, on the matmul's own
stream — and at batch one there is no other work at that point to overlap it
with. Dividing bytes by a peak figure was off by 3.7×, the third time tonight.

Net **13.4 µs per layer, 1.01 ms per round** across the round's ~75 built-in
grouped calls, host-side only, bit-identical.

### Half of it can avoid the copy entirely

A narrow view `w[32:40]` with a length-8 group list **is accepted**, stays in
npu format 29, and returns output bit-identical to the full-bank call reading
the same experts (max absolute difference 0.0). So when the active window is
contiguous the copy disappears and the whole 26–30 µs is saved. The scattered
gather is bit-exact too, so the compression is semantically sound rather than
merely fast.

The production tensor permits it: `w2` is (256, 128, 768) int32 in format 29 at
100,663,296 B, and narrow view, view plus contiguous, `index_select` and
`clone` all succeed on it. An earlier refusal under format 29 belonged to the
(256, 6144, 32) `w13` tensor specifically, not to NZ.

Landing point is `AscendW4A8DynamicFusedMoEMethod.apply_gmm2`
(`w4a8.py:637-645`): compress the group list to the active experts and replace
weight, scale and bias with an `index_select` on those indices, or a view when
the window is contiguous. Both must change together, since the built-in
rejects a group list shorter than the weight's expert dimension —
`size of groupList(tensor) 8 should be equal to weight dim 0 256 with
groupType 0`.

Limits to carry: per-call figures come from a single die with neighbours
present, multiplied by a family count from a 16-rank profile; the gather
allocates per call, and an `out=` buffer plus the 46% of peak it leaves on the
table are untested headroom of roughly 7 µs; `apply_gmm1` has a built-in path
not currently in use, worth 10.5 µs per layer if it is ever enabled. Still
outstanding before landing: a bit-exact comparison on the real checkpoint
rather than on synthesised weights.
