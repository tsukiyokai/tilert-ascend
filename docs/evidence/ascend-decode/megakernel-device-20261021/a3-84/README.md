# Component validation on a3-84

These runs used physical die 4 only, exposed as `npu:0`, with CANN 9.2.0,
torch/torch_npu 2.12.0 and Ascend910_9392 (24 AIC, 48 AIV, 192 MiB L2).
[Host inspection](environment.txt) records driver 25.5.5 and device mapping.
All fixtures are synthetic. JSON files identify the actual library and
harness hashes. Reports now distinguish zero-tolerance numerical equality
from raw-byte equality, including signed zero.

The routed reports also preserve a standalone-DynamicQuant diagnostic.
Its codes and scales can differ from the dispatch quantizer. The accepted
native chain retains `MoeInitRouting` codes and scales; that diagnostic is
outside the chain's equality checks.

## Numerical and graph results

Every accepted case below has zero numerical error and zero differing bytes
at its checked output boundaries. Changing-input graph tests use widths
1, 2, 12 and 32 with four replays at each width. They include zero/sign
changes and changing routes or cache pages where applicable.

| Stage | Comparison | Reports |
| --- | --- | --- |
| NZ routed experts | Native NZ reads versus stock CANN ND-weight chain; all 256 experts available | [M=1](routed-nz-m1.json), [M=2,12,32](routed-nz-replay.json) |
| Routed experts using dispatch metadata | Same dispatch codes, inverse permutation, scales and final BF16 output versus original native chain and stock CANN ND chain | [M=1](routed-dispatch-m1.json), [M=2,12,32](routed-dispatch-replay.json) |
| Composable MLA tail | Same-input CANN value-up, DynamicQuant and output projection; queued input producer | [Replay](mla-queued-composable.json) |
| Prolog plus SFA | CANN Prolog V3 and SFA; Q, RoPE, full caches and attention output | [Eager M=1](attention-m1.json), [Replay](attention-replay.json) |
| Prolog/SFA plus projection tail | Separate native attention and tail; latent, caches and final projection | [Eager M=1](attention-full-m1.json), [Replay](attention-full-replay.json) |
| Residual/norm/A8 | Separate same-source norm body followed by CANN DynamicQuant; five outputs, with and without beta | [Eager M=1](norm-source-m1.json), [Replay](norm-source-replay.json) |
| Residual/norm through output projection | Separate native norm/A8 and full attention; retained residual, latent, caches and projection; beta absent | [Eager M=1](attention-norm-m1.json), [Replay](attention-norm-replay.json) |
| Indexer front | Q/K projections, BF16 key norm, interleaved RoPE, Hadamard, dynamic quantization and head weights; C8 and BF16 modes | [Eager M=1](indexer-front-m1-v3.json), [Replay](indexer-front-replay-v3.json) |
| Prolog query-norm export plus SFA | Stock Prolog V3 with query-norm output enabled; INT8 Qnorm, FP32 scale, main Q/RoPE, both complete caches and SFA output | [Eager M=1](attention-export-m1.json), [Replay](attention-export-replay.json) |
| Balanced routed entry with the engine ABI | Stock canonical NZ chain, changing routes, widths 1/2/12/32 | [16 groups](routed-balanced-promoted16.json), [24 groups](routed-balanced-promoted24.json) |

The norm reference is source replay, not an installed vLLM custom-op binary.
The composed attention consumes existing indexer selections and returns the
local projection before TP reduction. It excludes routing and the FFN.

The indexer front uses the unmodified vLLM RoPE source and stock CANN
projections as its reference. The first build produced zero head weights
despite correct BF16 projection values. A vector-pipeline barrier before the
following BF16-to-FP32 cast fixes that dependency. The accepted
`4a24dabb187b204fd08e27880b9acf75aa6be4f6be3f3abfcbd76196f42a0fc1`
library preserves every production boundary above as raw bytes. Its diagnostic
FP32 norm statistics differ by at most `2.384185791015625e-7` at M=12/32;
the BF16 normalized key and all downstream outputs remain byte-identical.

The [first indexer attempt](indexer-m1.json) failed in reference host tiling:
CANN requires rank-three `key_dequant_scale`; the harness passed a rank-two
cache view. After correcting that descriptor, [M=1 eager](indexer-m1-v2.json)
passed four changing-input checks in both INT8 and BF16 modes.
[Captured INT8](indexer-replay-v2.json) passed M=1 and M=2, then failed at
M=12. The [failure detail](indexer-m12-diagnostic.json) identifies a swap at
row 9, positions 750/751: native `[7363,4978]`, reference `[4978,7363]`.
The full selected multiset and both caches match. The exact index-order
check remains failed in that eight-group configuration.

The [device-score snapshot](indexer-m12-score-device.json) recovers both
scores from live merge records whose row, causal length and output offset
match the failed query. Both FP32 bit patterns are `1049140327`.
The [CPU score reconstruction](indexer-m12-score-cpu.json) gives the same
value, but the device snapshot establishes the actual tie. With
[24 groups](indexer-core24.json), matching the stock tiler's core count,
all four widths and both modes pass exact index-order and cache comparisons.
Changing the core count changes the stable merge tree's ordering of ties.
The corrected default (`blocks=0`) selects all physical AIC groups and
[passes the same widths and modes](indexer-default24.json), including queued
producers. The explicit core-count override remains available for diagnostics.

## Paired component timings

All entries are raw microseconds per captured graph replay, with 30 replays
per timing pair. They reuse one synthetic weight
bank and the final input from the changing-input check. No timing floor is
subtracted, and no result is a layer time or model TPOT.

The [matched MLA comparison](mla-paired-matched.json) puts an identical input
clone inside both graphs and uses three alternating pairs. Eight groups gave the lowest native median among
the tested 4, 8 and 12 groups at all three widths:

| Width | Native, 8 groups | CANN chain |
| --- | ---: | ---: |
| 1 | 30.05 | 37.81 |
| 12 | 42.03 | 37.58 |
| 32 | 65.62 | 38.87 |

Attention stages use five alternating pairs, sequence length 8192, 2048
selected keys and eight core groups. The first table compares the native
Prolog/SFA entry with stock CANN operators:

| Width | Native Prolog/SFA | CANN Prolog/SFA |
| --- | ---: | ---: |
| 1 | 97.92 | 125.24 |
| 12 | 147.23 | 130.94 |
| 32 | 264.63 | 196.95 |

The [raw comparison](attention-paired-timing.json) checks query, RoPE, full
caches and attention outputs. The following compositions compare with their
constituent native entries; these rows do not use a stock CANN reference:

| Width | Prolog/SFA/tail fused | Separate attention + tail | Norm/attention fused | Separate norm + attention |
| --- | ---: | ---: | ---: | ---: |
| 1 | 121.32 | 124.79 | 131.43 | 133.51 |
| 12 | 180.96 | 184.14 | 191.63 | 192.94 |
| 32 | 297.06 | 303.20 | 311.31 | 311.30 |

[Attention/tail results](attention-full-paired-timing.json) and
[norm/attention results](attention-norm-paired-timing.json) retain every
sample and workspace size. Combining the norm stage gives no measured
median gain at M=32 in this configuration.

At [24 groups](attention-core24.json), Prolog/SFA takes 121.79 us at M=12
and 185.22 us at M=32, versus 132.39 and 199.43 us for the same-run CANN
reference. The [composed attention/tail](attention-full-core24.json) takes
156.33 and 246.57 us, versus 169.01 and 262.43 us for the separate native
stages. These core-count comparisons use three alternating timing pairs.

The MLA [local-counter probe](mla-profile-smoke.json) preserves the original
arithmetic and validates its output before reading traces. Its instrumentation
adds pipeline drains and a final grid barrier. At M=32, output dequantization
accounts for 53.6% of the slowest dequantizing AIV's local measured window.
This is an instrumented local-window fraction, not a model-time fraction.
The [30-sample probe](mla-profile-timing.json) includes five alternating
event-timing pairs. Ordinary/instrumented medians are 29.56/33.57 us at M=1,
38.40/42.64 us at M=12 and 56.13/61.81 us at M=32. This records the probe's
overhead separately from the stage counters.

The batched output epilogue preserves each row's arithmetic order while
loading and writing up to eight rows per DMA batch. It passes
[changing queued inputs](mla-dequant-smoke.json) at M=1,2,12,32 against both
the ordinary native entry and stock CANN, including intermediate zero inputs.
Six alternating [timing pairs](mla-dequant-timing.json), with the same input
clone in all three arms, give:

| Width | Ordinary native us | Batched epilogue us | CANN us |
| --- | ---: | ---: | ---: |
| 1 | 31.57 | 31.39 | 36.18 |
| 12 | 40.69 | 34.25 | 37.79 |
| 32 | 58.15 | 39.85 | 39.31 |

These are single-bank component results from the independently built
`4a9304fe9342733d3078943592f2671083400d086a1b6eecb54baca2794beabb` library.
[INT32 bias](mla-dequant-bias-int32.json) and
[FP32 bias](mla-dequant-bias-float32.json) also pass at M=1,12,32.
A [random non-power-of-two channel-scale fixture](mla-dequant-random-scale.json)
passes M=1,2,12,32 with four changing queued inputs against both ordinary
native and CANN entries. This exercises scale rounding beyond the original
power-of-two fixture; the MLA epilogue retains channel-then-token multiplication.
The [promoted standalone entry](mla-dequant-promoted.json) retains its original
Torch ABI and passes all four widths.

The helper also passes in the
[24-group attention composition](attention-full-dequant24.json). Its medians
at M=12/32 are 148.95/214.98 us, versus 167.18/254.49 us for separate attention
and the original native tail. Both arms compare caches, query and output
bytes. At [24 groups with norm](attention-norm-dequant24.json), fused versus
separate norm plus the updated attention entry takes 163.68/159.77 us at M=12
and 237.89/239.00 us at M=32. Fusion alone does not explain the performance
ordering; the core configuration and stage schedule remain material.

The MLA weight-reuse test captures 64 independent projections in each graph,
with no input copy inside either arm. One bank has 7 MiB of weights per arm;
64 distinct allocations contain equal values and occupy 448 MiB per arm.
The last output passes four changing-input byte comparisons in every run.
Five alternating pairs each replay the graph ten times:

| Width | Weight addresses | Native us/projection | CANN us/projection | Report |
| --- | --- | ---: | ---: | --- |
| 12 | One reused bank | 35.92 | 28.51 | [Reuse](mla-banks-m12-reuse-v3.json) |
| 12 | 64 banks | 39.45 | 28.44 | [Distinct](mla-banks-m12-distinct-v3.json) |
| 32 | One reused bank | 53.65 | 30.12 | [Reuse](mla-banks-m32-reuse-v3.json) |
| 32 | 64 banks | 62.97 | 29.96 | [Distinct](mla-banks-m32-distinct-v3.json) |

The graph has no inter-layer activation dependencies. This tests sensitivity
to the weight address working set and amortizes host replay gaps; it does not
directly measure L2 misses, HBM traffic or decoder-layer latency.

The [initial timing](diagnostics/mla-paired-timing.json) and
[initial core sweep](diagnostics/mla-core-sweep.json) cloned only the native
input. They are retained as diagnostics and excluded from performance claims.

The initial routed timings use three alternating pairs, include dispatch and the native path's two recovery gathers,
but exclude native route-metadata construction. The reference uses ND weights;
the deployed custom NZ reference is still required. These values therefore
do not establish the complete replacement's saving:

| Width | Native groups | Native | Stock ND reference | Report |
| --- | ---: | ---: | ---: | --- |
| 1 | 8 | 79.86 | 175.79 | [8 groups](routed-nz-paired-timing.json) |
| 12 | 8 | 314.19 | 278.53 | [8 groups](routed-nz-paired-timing.json) |
| 12 | 12 | 238.78 | 300.36 | [12 groups](routed-nz-core12.json) |
| 12 | 16 | 216.14 | 287.93 | [16 groups](routed-nz-core16.json) |
| 12 | 24 | 180.32 | 310.94 | [24 groups](routed-nz-core24.json) |
| 32 | 24 | 312.99 | 391.75 | [24 groups](routed-nz-core24.json) |

The dispatch-metadata entry consumes the vendor dispatch outputs directly.
These medians of five alternating pairs include dispatch, all route-metadata
work, row recovery and the final combine. The original native chain builds
metadata and recovers rows outside its kernel, inside the timed graph.
Router/top-k, shared experts and TP reduction remain outside all three arms.

| Width | Groups | Dispatch-metadata entry | Original native chain | Stock ND reference | Report |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 8 | 84.93 | 174.24 | 177.95 | [M=1](routed-dispatch-m1-timing.json) |
| 12 | 24 | 182.10 | 294.67 | 291.96 | [M=12,32](routed-dispatch-wide-timing.json) |
| 32 | 24 | 307.80 | 461.73 | 362.81 | [M=12,32](routed-dispatch-wide-timing.json) |

The canonical stock CANN NZ reference uses the original packed weight bytes
with an explicit INT32 physical descriptor. [M=1](routed-canonical-nz-m1.json)
and [M=2,12,32](routed-canonical-nz-replay.json) pass the final, routing and
intermediate byte checks. It is the SDK WeightNz implementation, distinct
from the deployed vLLM custom library. Five alternating timing pairs give:

| Width | Groups | Native complete chain | Stock NZ complete chain | Report |
| --- | ---: | ---: | ---: | --- |
| 1 | 8 | 81.75 | 196.33 | [M=1](routed-canonical-nz-m1-timing.json) |
| 12 | 24 | 179.33 | 265.95 | [M=12,32](routed-canonical-nz-wide-timing.json) |
| 32 | 24 | 303.34 | 354.78 | [M=12,32](routed-canonical-nz-wide-timing.json) |

Timings reuse the final replay's routes: 81 distinct experts at M=12 and 158
at M=32. Real-model weights and route distributions remain required.

The routed weight-reuse test captures 16 independent complete routed chains,
each including dispatch. Both arms share the same immutable NZ weight
allocations. One full bank occupies 288 MiB; four allocations contain equal
values and occupy 1152 MiB. The minimum selected weight bytes at M=12 are
91.125/364.5 MiB for one/four banks, and 177.75/711 MiB at M=32. Every run
passes four changing-route comparisons of the last output. Five alternating
pairs each replay the graph ten times:

| Width | Weight addresses | Native us/call | Stock NZ us/call | Report |
| --- | --- | ---: | ---: | --- |
| 12 | One reused bank | 173.58 | 259.21 | [Reuse](routed-banks-m12-reuse.json) |
| 12 | Four banks | 181.82 | 264.34 | [Distinct](routed-banks-m12-distinct.json) |
| 32 | One reused bank | 300.55 | 336.81 | [Reuse](routed-banks-m32-reuse.json) |
| 32 | Four banks | 333.80 | 351.39 | [Distinct](routed-banks-m32-distinct.json) |

These are independent calls with synthetic weights and routes. Their address
working sets expose reuse sensitivity, without measuring cache misses, HBM
traffic or decoder-layer dependencies directly.

The [routed stage probe](routed-stage-profile.json) verifies instrumentation
against the ordinary entry before collecting 30 local-counter samples.
Ordinary/instrumented graph medians are 183.45/187.20 us at M=12 and
303.84/310.95 us at M=32. At M=32, the owner assigned 17 routes has the longest
input split, activation and route-store work; other owners receive 6–14 routes.
The longest gate/up and down work occurs on different AICs. Counter differences
are local SYS_CNT ticks; waiting time on idle cores is not added to active
stage work as an independent cost.
The raw stage-1 activity label omits odd AIV metadata loads. Their counters
are present; the harness label was corrected after this report.

The [Fixpipe partial variant](routed-fixpipe-numerics.json) passes M=1,12,32
with four changing-route replays at 24 groups. It changes partial storage
from INT32 to FP16 and performs the channel scaling in CUBE Fixpipe. The
remaining FP32 correction, activation-scale multiplication, activation and
BF16 boundaries retain their order. Its library SHA256 is
`d39bb965f4e0a91cab712e7032e58e3a5cfceb1725db96263f259e549b82cfdc`.
M=32 scratch falls from 35,232,384 to 28,678,784 bytes.

The subsequent three-arm experiment uses four independent weight banks,
16 calls per graph and all six arm-order permutations. All three arms pass
four changing-route comparisons against the same canonical NZ reference.

| Rows | Original INT32, us/call | Rebuilt INT32 control, us/call | Fixpipe FP16, us/call | Report |
| --- | --- | --- | --- | --- |
| 12 | 183.62 | 188.28 | 197.26 | [Three arms](routed-partials-three-arm-m12-bank4.json) |
| 32 | 334.03 | 336.98 | 345.10 | [Three arms](routed-partials-three-arm-m32-bank4.json) |

The FP16 partials reduce allocated storage but lose latency against both
INT32 controls. These measurements favor INT32 partial storage.

The [balanced AIV candidate](routed-balanced-numerics.json) preserves the
INT32 products and CUBE expert assignment, while distributing input splitting
and activation across all 48 AIVs by global route row. Route-output stores
retain two AIVs per row, each owning half the hidden dimension. It passes
M=1,12,32 with four changing-route replays at every retained boundary.
The same four-bank, six-order timing protocol gives:

| Rows | Original INT32, us/call | Same-source INT32 control, us/call | Balanced INT32, us/call | Report |
| --- | --- | --- | --- | --- |
| 12 | 181.75 | 180.57 | 176.21 | [Three arms](routed-balanced-m12-bank4.json) |
| 32 | 334.95 | 335.58 | 292.14 | [Three arms](routed-balanced-m32-bank4.json) |

All six measurements favor the balanced candidate over both controls.
Its SHA256 is
`592ae06565af86fe38c6574e7db8a397c5c04289a5228bf88c3f0401e64f161f`;
the same-source control is
`2c040a9db24416525514efd045e2f61650892bd608c9e41b8c69f702ee5f254f`.

The same event-loop [calibration](replay-floor.json) measured an empty graph
at 8.46 us/replay and a 32-byte-copy graph at 8.66 us/replay with 30 replays.
Those controls expose host/replay overhead; they are not assumed to be an
additive term in a longer graph. The [calibration script](replay_floor.py)
also records 1, 10 and 100 replay cases.

## Shared W8A8 expert

The standalone entry covers the shared TP16 shard from quantized input
through local BF16 down projection. Its reference is the stock CANN
INT32 QuantMatmul, mode-0 DequantSwigluQuant and BF16 QuantMatmul chain.
The [initial vector-quant attempt](shared-w8-smoke-v2.json) differs in one
FP32 activation scale; scalar division and reciprocal restore the installed
mode-0 quantizer's order. The [next failure](shared-w8-down-diagnostic.json)
has exact INT32 products, INT8 codes and FP32 scales but one different BF16
output. An [independent arithmetic reconstruction](shared-w8-down-order-audit.md)
locates it at a BF16 midpoint and establishes that this small-K output
projection combines the two scales before multiplying the accumulator.

The [corrected entry](shared-w8-scale-first.json) passes M=1,2,12,32 with
eight changing queued-input replays and raw-byte equality at all four
boundaries. Its library SHA256 is
`f3027866ab33efd9a9f92379897ec3c74214148e00675a184f9a705000384d13`.
Six alternating [timing pairs](shared-w8-scale-first-timing.json) give:

| Rows | Native, us | Stock CANN chain, us |
| --- | --- | --- |
| 1 | 38.11 | 36.31 |
| 12 | 40.89 | 38.98 |
| 32 | 43.06 | 40.65 |

The standalone native chain is slower. The vLLM custom mode-1 quantizer
has a different contract: vector scale multiplication, vector division,
then INT32/FP16/INT8 casts. A [separate grouped variant](shared-w8-grouped-source.json)
passes all four widths with eight changing queued-input replays against the
unmodified source kernel, between the same two stock CANN projections.
The wrapper supplies explicit fixed-shape tiling and no host operator dispatch.
Candidate/reference library SHA256 values are respectively
`3c6bd9dd66c0d84bf9ff7ffb32aa2094109ba96f75ceac739036eec92379c04a` and
`7221f01c4fea66293ee2ddc4223533283c845cd8ec78030a9568ff0347c0caf4`.
This establishes the tested source arithmetic; deployed custom-op dispatch
and full-engine acceptance remain separate checks.

## Two-rank communication

Physical dies 4 and 5 ran one MIX launch containing BF16 all-reduce, a
dependent 16x16 CUBE projection and a second BF16 all-reduce. The corrected
entry passed M=1,12,32 with 12 changing-input replays per width, alternating
the delayed rank. Both ranks' first and final outputs match the ordered CPU
oracle and installed HCCL comparison byte for byte:
[rank 0](communication-two-rank-reuse-fence/communication-rank-0.json),
[rank 1](communication-two-rank-reuse-fence/communication-rank-1.json).
The accepted library SHA256 is
`1be5fa375d0d6e7a2a6dcf7f5265cf886317108398aeec940e2b1ce0de04e22f`.
The same M=1,12,32 checks with 12 changing inputs also pass at 24 resident
core groups: [rank 0](communication-two-rank-core24/communication-rank-0.json)
and [rank 1](communication-two-rank-core24/communication-rank-1.json).

The [original rank-0 report](communication-two-rank-replay/communication-rank-0.json)
and [rank-1 report](communication-two-rank-replay/communication-rank-1.json)
passed M=1 but failed M=12/32. Its reduction loop waited for `MTE3_V` after
writing the shared narrow UB buffer to GM. That does not order the next
tile's MTE2 overwrite. Replacing that event with `MTE3_MTE2` closes the buffer
reuse dependency; no arithmetic, transport or collective protocol changed.
M=1 assigned at most one tile per AIV, so it did not exercise the faulty reuse.

These checks establish the two-rank probe's numerical and replay behavior.
They do not establish TP16 behavior or whole-layer performance. The single-die
container was stopped during these tests: mapping the same physical die into
both live containers caused driver initialization to reject the second one.
