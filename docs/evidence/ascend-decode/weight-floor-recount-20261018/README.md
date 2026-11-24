# The weight floor, recounted, and why 2 ms is arithmetically reachable

## Both sides of the old figure were wrong, and nearly cancelled

The bandwidth was not a ceiling. `1190.7723` GB/s came from clamping an event
pair around each single read, so it measured one launch latency as well as the
transfer. Batching thirty reads inside one event pair, same die, same shapes:
257 MiB in 185.858 µs, 385 in 274.106, 513 in 360.585, 641 in 449.145 —
marginal rates 1532.8, 1564.1 and 1527.4 GB/s, asymptotic 1498.8, intercept
about 12 µs. The old method's own best sample was already 1246.5 GB/s, above
the "maximum" it reported.

The byte count was not a round. `bench_weight_budget.py` counted only the
verify forward, missing the two chained drafts and the conditioned draft, and
then divided replicated tensors by sixteen although every rank reads them
whole — the replicated set inside verify alone is 1.7120 GB. A round at one
rank really reads 6.8330 GB, which agrees to the byte with
`execution-gap-20261018/route-audit.json`'s own
`logical_bytes_per_round.mean`. The composed configuration's
`--draft-projection-tp` removes 637,009,920 B, giving 5.9752 GB of weights.

So the floor is **3.880 ms at 1540 GB/s, 56.5% of the 6.866 ms a round may
cost** — against the old 3.56 ms and 52%. A numerator 1.41× low and a
denominator 1.29× low left the conclusion 9% out.

## Residency is real, and useless here

The gap between resident and streaming is 2.07–2.21×, and the criterion is
total footprint rather than per-call size: at or below 0.5×L2 the dividend is
full, at 1.0×L2 partial, at or above 1.9×L2 zero. Three depths give one curve.

But the model's own expert GMM does not care. Sweeping the footprint from
0.105×L2 to 32×L2 moves `npu_grouped_matmul` by 0.73%, 0.16%, 0.55%, −0.05%
and −0.46%, while in the same harness a bf16 matmul at 0.5×L2 shows 2.208×.
The operator is not bandwidth-bound: `gate_up` at g256 runs at 798 GB/s and
`down` at 470, so each layer spends 116.2 µs reading 31.85 MB that the
hardware reads in 20.7 — **5.6× of slack**. The attribution is three
device-level `SyncAll` calls in `a8w4_msd_pipeline.h` that survive even at
`loopCount == 1`.

And there is no second read to cache anyway. Only the verify forward walks all
78 layers; the three drafts walk layer 78 alone. Reordering to layer-major
wins 1.555× on a synthetic bf16 stream and *loses* 7% on the real GMM, and
even a working reorder's 4.328 ms is worse than the true single pass at
1.68 ms. MTP's one-pass design already beats any cache-assisted schedule.

## No residency API exists on this part

Compile-verified rather than read: the same vector kernel built four times
with only `CacheMode` changed gives byte-identical objects for `NORMAL`,
`PERSISTENT` and `LAST` (md5 `ec5ab20b3ebd2087106be8dc2cc0c8c6`), and a
different one only for `DISABLE`. The implementation in
`kernel_utils_macros.h:241-254` is gated on `__NPU_ARCH__` values that exclude
A3's 2201. The vendor's own tree uses `CACHE_MODE_DISABLE` 176 times and
`NORMAL` 21, `PERSISTENT` and `LAST` never. The API that could pin pages,
`rtSmData_t` with `L2_preload` and `remap[64]`, exists in
`pkg_inc/runtime/rt_external_kernel.h:274-297` — matching the platform's
`l2PageNum=64` — but ACL does not expose it and `libruntime.so` carries no
implementation. What exists is prefetch, not residency, and a prefetched line
is an ordinary first victim that 2.4 GB of passing traffic evicts.

## What can be cut: 1.449 GB, worth 0.941 ms

| change | bytes | ms at 1540 | risk |
| --- | --- | --- | --- |
| attention to W4A8 | 0.985 GB | 0.640 | numerical, and the budget is thin |
| stop reading the head three extra times | 0.357 GB | 0.232 | none |
| narrow the scale carrier off the vendor path | 0.107 GB | 0.069 | leaves the vendor kernel |

The head is read four times per round and three of those exist only to take a
draft's top-1, yet `compute_logits` fetches the whole vocabulary even with
`full_logits=False`. That is the one free item.

W4A8 on attention is not free: at W8 the model's hidden NRMSE already peaks at
0.170898 against a self-imposed 0.01, the dense sensitivity decomposition puts
all of a layer's output error in the input-perturbation term with a 32×
amplification, and acceptance has 16% of headroom at M=4. For scale, the
vendor int4 path requires uint64 carriers in all ten int4 rows of
`GroupedMatmul`; half of each is zero by construction, so narrowing means
leaving that path.

## Why 2 ms is now arithmetically reachable

A round may cost 6.866 ms. Device-side work is 37.7 ms, which looks fatal
until the slack is counted: the operators read their weights at 18% of what
the die delivers, and 37.7 / 5.6 ≈ 6.7 ms. The target is therefore not a
fantasy but a statement about operator efficiency — it is reachable if and
only if the weight stream runs at hardware bandwidth and the host disappears.

| term | ms | route |
| --- | --- | --- |
| weight stream at 1540 GB/s | 3.880 | irreducible; 2.939 after the cuts above |
| the round's 157 reductions at the measured 11.2 µs | 1.758 | one-sided, measured |
| everything else | 1.228 | what the megakernel must fit in |

The 5.6× slack and the host's 5.861 ms are the same disease: synchronisation
that exists because work is chopped into separately launched pieces. A
megakernel is the treatment for both, which is why it is not an optimisation
of the second-largest item but the only route that reaches the largest two.

## Two residual uncertainties, named

1540 GB/s is the vector unit's read rate; the cube path reaches 1185–1255
today. If a perfect kernel cannot reach 1540 either, the floor is 4.761 ms
rather than 3.880, and settling that needs a cube-side weight-stream kernel
that actually saturates. The head's three repeated measurements were taken
while the die was shared and only the idle one (45.800 against 95.019 µs) is
trusted.
