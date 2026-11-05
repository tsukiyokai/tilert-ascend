# Native decode composition on a3-162

The [environment snapshot](environment.txt) records physical die 4, exposed
alone as `npu:0`. The host uses driver 25.5.5 and Ascend910_9392. The experiment
container uses CANN 9.2.0 and torch/torch_npu 2.12.0; each report fingerprints
the actual libraries. Its loaded `libascendcl`, `libopapi`, `libruntime` and
`libtorch_npu` hashes match the preceding a3-84 runs.

The [first interleaved comparison](routed-shared-m1-16.json) includes independent
DynamicQuant for the shared input and MoeInitRouting for routed inputs. At
M=1 with 16 core groups, four changing-route cases pass all retained boundaries
against both separate native stages and the CANN chains. The following zero
input case fails final routed output bytes in both native arms: all numerical
values are equal, but 857 of 6144 BF16 values have different zero signs.
Shared outputs and routed values before combine remain byte-identical.

The [failure tensors](routed-shared-m1-16-m1-r4.pt) contain the routed BF16
values, probabilities, intermediate native buffers and both native outputs.
Exactly 857 columns contain eight negative-zero routed values. Initializing
the sequential FP32 weighted sum from its first product preserves 857 negative
zeros; the current native sum starts at positive zero and preserves none.
The full zero-input byte gate remains failed for these recorded binaries.

The installed CANN `moe_token_unpermute.h` confirms the accumulation contract:
slot 0 initializes the accumulator with its weighted value when its probability
is nonzero; otherwise it initializes positive zero. Slots 1–7 then add their
weighted values in order, skipping zero probabilities. This is not initialization
from the first nonzero-probability slot. The source SHA256 is
`d7d7eb7687e2aeab5b2e365909ae983a0522a3bce660adcbc04bba03236878b0`.

The corrected interleaved library
`d311bcffc7b64ccc102537ec8743f9accc8239edd261d93e0cff67a2165e165a`
and separate balanced library
`c6e0116ffca2c15567f6741bfc45789e9930c20af3e2243319f73d2e80c29200`
pass M=1,2,12,32 at both 16 and 24 core groups. Each width has eight changing
graph replays, including zero input, large signed values, zero probability in
slot 0, and zero probabilities in odd slots. All 34 checked boundaries match
as raw bytes, including routed intermediate codes, scale, per-route BF16
outputs, metadata, both final outputs, and the shared gate/up INT32 sum.

- [16 groups, M=1/2](routed-shared-zero-m1-16.json)
- [16 groups, M=12/32](routed-shared-zero-wide16.json)
- [24 groups, all four widths](routed-shared-zero-replay24.json)

At M=32, fused scratch is 36,811,392 bytes versus 53,588,608 bytes for the
separate native bodies. The 16 MiB difference is the shared Matmul system
workspace. Both branches retain separate BF16 outputs; their addition, routed
scaling and TP reduction are outside this comparison.

## Four-bank timings

Each graph calls the body 16 times, cycling through four equal-valued weight
banks at distinct addresses. All arms include standalone DynamicQuant and
MoeInitRouting. The two native arms share the same routed NZ and shared ND
weight allocations; the vendor shared branch uses equivalent NZ allocations.
The graph contains no input clone or dependency between successive calls.
Six timing pairs cover all three-arm orders, with ten replays per pair.
Every arm's last routed/shared output passes four changing-input byte checks.

| Width | Interleaved us | Separate native us | CANN chain us | Report |
| --- | ---: | ---: | ---: | --- |
| 12 | 187.17 | 219.24 | 300.44 | [M=12](routed-shared-zero-m12-bank4.json) |
| 32 | 312.36 | 347.33 | 427.04 | [M=32](routed-shared-zero-m32-bank4.json) |

The interleaved entry is faster in every measured order. Relative to separate
native execution, the medians decrease by 32.08 us at M=12 and 34.96 us at M=32.
These are local MoE-body measurements, before output addition and TP reduction.

The corrected standalone entry with the original `tilert` namespace also
passes strict byte comparisons at all four widths and eight changing replays
with [16 groups](routed-balanced-zero-promoted16.json) and
[24 groups](routed-balanced-zero-promoted24.json).

The [TileRT adapter body comparison](routed-shared-adapter-replay.json) uses
the production `run_body`, `pack_shared_weights` and workspace pool with the
same native binaries. Eager warmup uses a different stream from graph capture;
capture claims preallocated buffers without preparing new ones. M=1,2,12,32
each pass eight changing-input replays at 24 core groups, with all 34 retained
boundaries byte-identical. This test covers independent input quantization,
dispatch and both local outputs. It excludes the vLLM runner hook, top-k
selection, routed scaling, output addition and TP reduction.

The [BF16-scale adapter comparison](routed-shared-adapter-bf16-scales.json)
also passes every width and replay. The vendor QMM receives the original BF16
scale, while native projections and SwiGLU receive its exact FP32 conversion.
The [FP16-scale probe](routed-shared-adapter-fp16-scales.json) is rejected by
the vendor gate/up QMM with error 161002: this installed NZ entry does not
accept `DT_FLOAT16` scales. The adapter therefore accepts FP32 and BF16 scales.

## Gate/up K-axis load grouping

The independent `CFG_MDL` gate/up variant has library SHA256
`7f48678afcd186f3bb286d7230abfe8c36c9685f49520fb23edc8354e9564a7a`.
It preserves the balanced AIV schedule, INT32 partials, down projection and
tiling. Its [strict comparison](routed-gu-mdl-numerics.json) passes M=1,2,12,32
with eight changing-route replays each, including zero and zero-probability
cases. Four-bank graphs use the same weights and dispatch, 16 calls per graph,
ten replays per order and all six three-arm orders.

| Width | GU MDL us | Balanced control us | CANN chain us | Report |
| --- | ---: | ---: | ---: | --- |
| 12 | 172.01 | 173.98 | 266.36 | [M=12](routed-gu-mdl-m12-bank4.json) |
| 32 | 288.80 | 289.26 | 354.38 | [M=32](routed-gu-mdl-m32-bank4.json) |

All M=12 pairs favor MDL. One M=32 pair favors the control; its median
difference is only 0.46 us. These binaries remain separate from the
interleaved routed/shared implementation.

The [source snapshot](gu-mdl-build/source-snapshot.json) verifies all fourteen
files against the original build identity. Ten are matching retained files;
four were reconstructed by reversing the padding changes and match the
original SHA256 exactly.

## Gate/up M16 padding

The independent padding library
`59c0df20b3c05953657ee900d89df633e6790ca41f3f63f829603d2217f731aa`
rounds each expert's actual gate/up M dimension to a multiple of 16. This
tests the SDK's small-M MMAD barrier condition while retaining the balanced
AIV schedule and original down projection. The
[strict comparison](routed-gu-pad16-numerics.json) passes all four widths and
eight changing replays, including packed high/low inputs, gate/up and down
INT32 partials, middle codes/scales, per-route outputs and final bytes.

Four-bank graphs use 16 calls, 30 replays per order and all six three-arm
orders. Padding is slower than the control in every measured order.

| Width | GU padded us | Balanced control us | CANN chain us | Report |
| --- | ---: | ---: | ---: | --- |
| 12 | 175.20 | 174.64 | 268.53 | [M=12](routed-gu-pad16-m12-bank4.json) |
| 32 | 300.00 | 291.78 | 353.37 | [M=32](routed-gu-pad16-m32-bank4.json) |

Padding adds 5.25 MiB of scratch at M=12 and 14 MiB at M=32. These results do
not support replacing the balanced implementation. The
[build manifest](gu-pad16-build/build-manifest.json),
[source identities](gu-pad16-build/source-identity.sha256) and
[verification log](gu-pad16-build/source-verification.log) retain the frozen
sources used for compilation.

## Indexed attention

The indexed-attention library
`ff5525b8a73ad20d48f35625be3c7a3b383d0bd874dee57ba1d41d10ba599a89`
combines Prolog/Qnorm export, indexer front, index-cache write and selection,
SFA, and the local output projection. Its [eager M=1 comparison](attention-indexed-m1.json)
and [queued graph comparisons](attention-indexed-replay.json) pass all production
boundaries as raw bytes for M=1,2,12,32, with INT8 and BF16 index caches and
four changing inputs per width. The main and indexer cache families have
different physical page counts and mappings. Full caches and ordered top-k
indices are compared. FP32 normalization statistics remain a diagnostic-only
boundary; they are also byte-identical in these composition runs.

The reference uses stock CANN Prolog/SFA and separately validated native
indexer-front, selector and projection-tail stages. The [timing report](attention-indexed-timing.json)
uses one weight bank, sequence length 8192, 24 core groups, six alternating
pairs and 30 graph replays per pair. Both arms contain the same queued input
copies. The composed entry is faster in every measured pair.

| Index cache | Width | Composed us | Separate stages us |
| --- | ---: | ---: | ---: |
| INT8 | 1 | 232.37 | 265.25 |
| INT8 | 12 | 288.42 | 307.31 |
| INT8 | 32 | 406.48 | 419.64 |
| BF16 | 1 | 219.66 | 258.94 |
| BF16 | 12 | 274.69 | 307.86 |
| BF16 | 32 | 373.63 | 411.48 |

These timings end at the rank-local BF16 output, before TP reduction. The
prototype retains independent stage scratch arenas for differential inspection;
its M=32 allocations are 130,220,544 bytes for INT8 index caches and 156,922,368
bytes for BF16. The reference's explicit buffer count excludes internal CANN
workspaces, so it is not a complete memory comparison.

The [adapter M=1 check](attention-adapter-m1.json) and
[all-width check](attention-adapter-replay.json) call the production
`attention_native.run_body`, indexer packing and workspace pool. M=1,2,12,32
pass four changing replays in each indexer mode: all 25 C8 or 20 BF16 retained
production boundaries match raw bytes. Main and indexer use independent
INT32 slot sources and page maps. Default-stream warmup leaves capture
reservations intact; the capture stream claims separate workspaces and slot
buffers. Every case consumes its reservations and retains four workspaces and
four slot buffers across the two streams.

The probe copies ordered top-k into separate persistent storage and runs a
shared-indexer consumer using the same hidden states, weights and main cache.
That consumer also matches the reference output. These are local-body and
storage-lifetime checks, not two dependent decoder layers or a vLLM
ACLGraphWrapper/TP result. The adapter source hash is
`cf0384cfb5d72c86552250115e789b141a11ca4e85f015a0555dff281cef7271`;
the shared-attention library is
`dce289d8fa391921348b50505f5eaf96a496546f0f359d7b38c39ee3ab4b0d06`.

## Original BF16 projection scales

The attention composition and adapter reports above use original FP32 WQ and
output scales. The [BF16 WQ probe](indexer-front-bf16-scales.json) gives stock
QMM the original BF16 tensor and the native front its exact FP32 conversion.
M=1/2 pass four changing C8 replays. At M=12, replay 0 differs at one of 49,152
Q projection elements by 0.00390625. It propagates to 31 Hadamard values and
one INT8 query code. The key path, query/key scales and head weights match
bytes. The run stops at that failure. The
[retained failure identity](indexer-front-bf16-scales-failure-artifact.json)
records the complete input/intermediate dump's host paths, size and SHA256.

The independent [BF16 output-scale check](mla-tail-bf16-scales.json) passes
M=1,2,12,32 with eight changing graph replays each and 24 core groups. Seed 830
produces varied channel mantissas and exponents; stock QMM receives BF16 scales
and the native tail their FP32 conversion. Every final output matches bytes.
The unchanged tail library is
`06a5511c26741b7fd74b6107013ee080dab6d3cc8c8faba1cc5688c4cc6421a1`.

Extended FP32 checks also expose one Q projection mismatch at M=12, replay 4.
The [unchanged original library](indexer-front-fp32-extended-control.json)
reproduces that mismatch. Exact integer reconstruction from the saved
[FP32 witness](indexer-front-fp32-scale-witness.json) and
[BF16 witness](indexer-front-bf16-scale-witness.json) agrees with every native
INT32 accumulator. Both failures come from scale multiplication order at a
BF16 rounding boundary: this K=2048, N=4096 QMM combines channel and token
scales before multiplying the accumulator.

The corrected front library is
`477d20ecd6cd2ec5e9e9cec20546801615d5fc89a4be7259818a54f23096b62c`.
The [FP32](indexer-front-fp32-scale-order.json) and
[BF16](indexer-front-bf16-scale-order.json) reports each pass M=1,2,12,32,
both C8 and BF16 indexer modes, and 32 changing graph replays per case.
All production boundaries match raw bytes. Internal FP32 normalization
statistics remain diagnostic-only. Original WQ scale dtype is retained in
the binding; both dtypes use the combined-scale arithmetic for this shape.

The corrected indexed library is
`6588e1dcde9dd843c10479ce1276ae069d89a9e60c16bc296f20ba4afb7c1c34`.
Its actual TileRT adapter body passes the same widths and both indexer modes
with eight changing graph replays for original
[FP32](attention-adapter-fp32-scale-order.json) and
[BF16](attention-adapter-bf16-scale-order.json) projection scales. Independent
INT32 slot sources, persistent ordered top-k and the shared-indexer consumer
are retained. The [executed harness sources](attention-scale-order-harness/source-identity.json),
[front build](indexer-front-scale-order-build/build-manifest.json) and
[indexed build](attention-indexed-scale-order-build/provenance.json) retain
their exact identities. These tests still end before the vendor forward
hook, ACLGraphWrapper and TP reduction.

The corrected library's [eight-bank timing summary](attention-indexed-scale-order-bank-summary.json)
retains raw reports for M=12/32 at sequence length 8192. The eight equal-valued
weight banks occupy 280.66 MiB at distinct addresses, exceeding the reported
192 MiB device L2 capacity. Each graph contains 16 independent calls;
an event measures 30 graph replays, normalized per call. Thirty alternating
event pairs are retained for each row below.

| Indexer cache | Width | Indexed p50 us | Separate p50 us | Indexed p90 us |
| --- | ---: | ---: | ---: | ---: |
| C8 | 12 | 297.034 | 308.674 | 297.109 |
| C8 | 32 | 402.488 | 423.165 | 402.613 |
| BF16 | 12 | 285.267 | 306.318 | 285.371 |
| BF16 | 32 | 386.310 | 405.884 | 386.398 |

All 30 pairs in each row favor the composed launch. Both arms include input
quantization and omit queued diagnostic clones. The numerical gate checks
the last call's retained boundaries over eight changing replays; it does
not inspect every bank independently. These graphs do not form dependent
decoder layers, and bank rotation does not guarantee every access misses L2.

## Indexed-attention phase profile

The [profile summary](attention-indexed-profile-summary.json) retains two
compressed raw reports, the A3 50 MHz SYS_CNT source identity and ordinary
versus instrumented event measurements. The instrumented library has a
separate namespace and records six timestamps on each of 24 AICs and 48 AIVs.
It keeps the existing phase barriers and stores traces after the last mark.
The [numerical gate](attention-indexed-profile-numerics.json) passes
M=1,2,12,32, both indexer-cache modes and eight changing graph replays per
case. All production boundaries match raw bytes; FP32 normalization
statistics are diagnostic-only.

The following values are maximum per-worker median intervals over 30 samples,
in microseconds. Each interval includes its original synchronization. The
maxima may belong to different workers and must not be added.

| Cache | Width | Prolog | Indexer front | Indexer | SFA | Projection tail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C8 | 12 | 42.38 | 70.37 | 54.84 | 76.97 | 56.55 |
| BF16 | 12 | 42.61 | 45.28 | 70.57 | 77.29 | 56.13 |
| C8 | 32 | 47.89 | 109.20 | 68.25 | 131.59 | 55.01 |
| BF16 | 32 | 47.25 | 69.57 | 76.35 | 134.16 | 56.95 |

The profiles use eight distinct weight banks, 16 independent calls per graph
and sequence length 8192. Each trace sample retains the final call after a
full graph replay. Thirty alternating event pairs measure 30 replays per arm;
the median paired instrumentation overhead is 7.26/11.24 us for C8/BF16 at
M=12 and 8.07/5.35 us at M=32. The ordinary event medians in those comparisons
are 294.68/283.25 us and 409.97/381.53 us. Local windows measure only differences
on the same worker; no common clock origin across workers is assumed.

The [build manifest](attention-indexed-profile-build/provenance.json) identifies
the library as `e34088c8d12f6c00831d871886b59959cb59fdfe56a1a3954088db06386d1987`.
All 55 recorded source/artifact hashes and 58 frozen snapshot files were
verified. The [executed harness](attention-indexed-profile-harness/source-identities.json)
records the six checker dependencies used in these runs.

## Parallel indexer projections

The candidate assigns the 16 query tiles and five key tiles to disjoint AIC
groups when at least 21 groups are available. Fewer groups retain the
sequential assignment. The candidate also skips initialization for empty
projection assignments; these changes have not been timed separately.
The [BF16-scale gate](attention-indexed-parallel-numerics-bf16.json) covers
M=1,2,12,32, both cache modes, 16 and 24 groups, and eight changing replays.
The [FP32-scale gate](attention-indexed-parallel-numerics-fp32.json) covers
the same widths, modes and replays at 24 groups. All production boundaries
match raw bytes; FP32 normalization statistics remain diagnostic-only.

The [paired timing summary](attention-indexed-parallel-bank-summary.json)
uses eight distinct weight banks, 16 independent calls per graph and 30
alternating ordinary/candidate pairs, with 30 replays per event. Values below
are microseconds per call. Savings are medians of paired differences.

| Cache | Width | Ordinary median | Candidate median | Paired savings | Candidate wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| C8 | 12 | 298.40 | 291.23 | 7.16 | 30/30 |
| BF16 | 12 | 285.45 | 284.08 | 1.37 | 30/30 |
| C8 | 32 | 403.89 | 402.92 | 0.91 | 30/30 |
| BF16 | 32 | 375.22 | 385.21 | -9.98 | 0/30 |

The original M=32 BF16 comparison regresses in every pair.
The post-timing final outputs also match raw bytes. These graphs share input,
metadata and scratch; retained numerical boundaries belong to the final
call, and TP reduction remains outside the measurement.
The [build manifest](attention-indexed-parallel-build/provenance.json)
identifies library
`d400c0527969ed2d365c4a3698794e0571988c282ff787cf50dd6d3166806c5e`;
56 source/artifact identities and 59 frozen files were checked.
The [executed harness](attention-indexed-parallel-harness/source-identities.json)
retains the five source files used in both timing runs.

The independent front passes [FP32](indexer-front-parallel-v2-fp32-r32.json)
and [BF16](indexer-front-parallel-v2-bf16-r32.json) scale checks over four
widths, both cache modes, 16/24 groups and 32 changing replays per case:
1,024 replays in total. The [front build](indexer-front-parallel-build/build-manifest.json)
retains 19 verified compiled sources and library identity `5556aa90e797e020d4b546bdfc9cdc8b245fe3aaadeef65e181e4e5b96b3a884`.
The [paired front comparison](indexer-front-parallel-v2-paired.json) uses
shared weights/input and separate scratch, with producer clones outside
timing. All eight cases favor the candidate in all 30 pairs. With BF16
channel scales, M=12/32 paired savings are 3.56/8.83 us for C8 and 4.82/6.01 us
for BF16 cache mode. Each case also passes 32 changing-input comparisons and
post-timing production-byte checks.

The [parallel phase profile](attention-indexed-parallel-profile-summary.json)
passes M=32 numerical checks, but its BF16 instrumented event median is
383.25 us versus 386.79 us for the ordinary binary. Instrumentation changes
execution beyond its final stores; its event difference cannot be treated as
a fixed additive cost, and separate profile runs do not isolate the schedule.

The [placement comparison](attention-indexed-parallel-placement-summary.json)
captures four graphs: ordinary and candidate each use arena A and arena B.
Within an arena, both kernels use identical scratch, tiling, output and
complete cache addresses. All graphs share input, weights and metadata.
Every arm restores identical cache values and warms three replays before
its timed event. Both arenas pass all retained production boundaries and
complete cache bytes before and after timing.

| Cache at M=32 | Arena | Ordinary median (us) | Candidate median (us) | Median paired savings (us) | Candidate wins |
| --- | --- | ---: | ---: | ---: | ---: |
| C8 | A | 409.82 | 406.94 | 2.38 | 30/30 |
| C8 | B | 409.06 | 407.83 | 1.38 | 24/30 |
| BF16 | A | 378.66 | 378.84 | -0.19 | 10/30 |
| BF16 | B | 384.55 | 383.75 | 0.94 | 30/30 |

This narrows the earlier regression: its magnitude cannot be attributed
solely to Q/K scheduling. The result depends on arena placement, without
identifying a specific cache mechanism. The sequential schedule remains
the default because the BF16 paired effect changes sign across arenas.
The [executed placement harness](attention-indexed-parallel-placement-harness/source-identities.json)
records all six source identities.

## Native BF16 routed input

The [numerical report](routed-input-front-numerics.json) compares one native
launch from BF16 input through routed combine against both the retained
balanced-dispatch entry and stock CANN NZ projections. M=1,2,12,32 each pass
eight changing graph replays at 24 core groups. All 17 checked boundaries
match raw bytes, including input codes/scales, stable route metadata,
intermediate INT32 products, quantized activation, per-route outputs and
queued-producer output.

The native front computes each token's `amax/127`, divides before quantizing,
and retains CANN's integer/half/INT8 cast order. AIV0 builds stable counts and
route metadata; token rows are distributed across AIVs. The existing first
barrier publishes both results to the unchanged balanced projection body.
The library SHA256 is
`0afd051c2f6088adc04744a3cc0a0d910abc68738aae6069c7cabd1fd636a136`.
The [build manifest](routed-input-front-build/build-manifest.json) records
17 frozen sources, SDK header identities and the 76,320-byte front UB budget.

The [M=12](routed-input-front-m12-bank4.json) and
[M=32](routed-input-front-m32-bank4.json) performance checks use four distinct
NZ weight banks and 16 calls per captured graph. Each bank first passes all
17 boundaries over eight changing replays. Timing uses route case 3, 30 graph
replays per event and 30 event samples per arm; all six three-arm orders
repeat five times. Input copies and diagnostic clones are outside timing.

| Width | Active experts | Native front p50 us | Balanced dispatch p50 us | CANN p50 us |
| --- | ---: | ---: | ---: | ---: |
| 12 | 81 | 150.891 | 172.972 | 261.284 |
| 32 | 158 | 269.867 | 285.950 | 353.885 |

All 30 paired samples at each width favor the native front. Its p90 is
151.821/270.057 us and standard deviation is 0.435/0.315 us at M=12/32.
Both native final outputs still match the CANN output bytes after timing.
The [timing harness identity](routed-input-front-timing-harness/source-identity.json)
records the per-bank checks added after the initial single-bank report.

## BF16 routed front with shared-expert interleaving

The combined entry uses the routed front above and the existing shared W8A8
interleaving. Shared input quantization remains an independent DynamicQuant
operation. The [16-group](routed-pipeline-front-numerics16.json) and
[24-group](routed-pipeline-front-numerics24.json) checks pass M=1,2,12,32,
eight changing replays and all 37 raw-byte boundaries. Routed and shared
inputs are distinct. The production adapter's original
[FP32](routed-pipeline-front-adapter-fp32.json) and
[BF16 shared scales](routed-pipeline-front-adapter-bf16.json) also pass those
widths and replays, including queued INT64 route-ID conversion and capture
workspace ownership. These calls end before the vendor runner and TP finalize.

The [four-bank summary](routed-pipeline-front-bank-summary.json) links the
raw M=12/32 reports. Every bank independently passes eight changing-input
checks at all 37 boundaries. Each timing graph contains 16 calls; each event
contains 30 graph replays. Thirty samples per arm cover all six execution
orders five times, using distinct nonzero routed/shared inputs and BF16
shared scales.

| Width | Combined front p50 / p90 us | Prior interleaved dispatch p50 us | CANN p50 us |
| --- | ---: | ---: | ---: |
| 12 | 161.111 / 161.173 | 187.218 | 310.011 |
| 32 | 285.147 / 285.280 | 306.638 | 411.279 |

All 30 comparisons at each width favor the new entry. Its standard deviation
is 0.271/0.346 us at M=12/32, and outputs still match raw bytes after timing.
All arms include shared input DynamicQuant and return separate BF16 routed
and shared outputs. Final addition, routed scaling and TP reduction are
outside timing. These measurements use synthetic rank-local weights and routes.

The library SHA256 is
`b0a87a13cf07ce653e6e602c4dfad9eadbbe4ace91cf6f65b0ef673efce74016`.
The [build manifest](routed-pipeline-front-build/build-manifest.json) retains
28 frozen sources; [executed harness identities](routed-pipeline-front-harness/source-identities.json)
retain the checker and adapter versions used on the device. The shared
Matmul system workspace remains 16 MiB. Total explicit scratch is 24,442,288
bytes at M=12 and 36,811,392 bytes at M=32, unchanged from the prior composition.
The adapter selects `build-pipeline-front` and native input-front execution
by default.

## Balanced routed-stage profile

The [summary](routed-balanced-profile-summary.json) and
[compressed raw report](routed-balanced-profile.json.gz) compare the retained
balanced-zero library with an instrumented build of the same kernel source.
The [source identity](balanced-profile-build/source/source-identity.json)
records the kernel/header reconstruction and matching original SHA256 values.
The diagnostic enables the balanced row schedule. M=12/32 pass eight changing
route comparisons, followed by 30 local-counter samples and 30 alternating
event samples per arm, each containing 30 graph replays.

| Width | Active experts | Ordinary p50 us | Instrumented p50 us | Maximum per-core median window us |
| --- | ---: | ---: | ---: | ---: |
| 12 | 81 | 170.357 | 176.243 | 141.58 |
| 32 | 168 | 266.455 | 271.673 | 233.40 |

These are single-bank timings for the last changed routing fixture. The
instrumentation adds 5.89/5.22 us. A3 SYS_CNT runs at 50 MHz according to the
versioned AscendC API source recorded in the summary. No alignment of different
cores' counter origins is assumed. Maximum per-core median work intervals are:

| Stage | M=12 us | M=32 us |
| --- | ---: | ---: |
| Restore dispatch | 13.45 | 11.84 |
| Metadata and split | 9.74 | 19.12 |
| Gate/up | 50.90 | 81.30 |
| Activate | 7.20 | 14.44 |
| Down | 34.44 | 53.03 |
| Store routes | 11.16 | 30.52 |
| Combine | 1.18 | 2.31 |

The largest work and barrier intervals can belong to different cores and
overlap; their maxima cannot be added. The difference between the event and
local-window measurements includes external input routing, launch and
instrumentation work. It does not isolate dispatch latency.

## Packed shadow KV with planar SFA workspace

The [build manifest](attention-indexed-shadow-build/provenance.json) records
library `32eba988e4356074a77ca863a24552ac5a23f0adb635d12b895cf6d3f2ebd430`
and the private SFA vector-header overlay. Each cache row packs 512 latent
and 64 RoPE BF16 values. Current INT64 slots are copied during the first
indexer projection, using its existing barrier. SFA gathers packed rows and
deinterleaves them into the unchanged planar workspace.

The [numerical gate](attention-indexed-shadow-numerics.json) passes four widths,
both index-cache modes and 16/24 groups over 128 changing-input replays.
The [negative-slot gate](attention-indexed-shadow-negative-slots.json) passes
another 48 replays. Both compare complete original caches and the entire
shadow as raw bytes, independently of selected attention rows.

The [timing summary](attention-indexed-shadow-summary.json) links the raw
M=12/32 reports and includes p50, p90, p99 and standard deviation. Eight
equal-valued weight banks occupy distinct addresses. Each event contains
30 replays of a 16-call graph. Both kernels are recaptured on each of two
arenas, sharing the original caches, scratch, tiling and output addresses
within an arena. Thirty alternating pairs per arena give these median
ordinary-minus-candidate differences:

| Width | C8 arena A / B us | BF16 arena A / B us |
| --- | ---: | ---: |
| 12 | -17.53 / -14.87 | -21.65 / -20.86 |
| 32 | -13.60 / -11.12 | -16.52 / -14.39 |

The candidate loses all 240 same-arena comparisons and is not selected.
Pre/post timing checks retain exact production boundaries, original caches
and shadow bytes. Per-call shadow updates are timed; full initialization
is outside the events. The 68-page fixture adds 10,027,008 bytes (9.5625 MiB)
of persistent cache. [Executed harness sources](attention-indexed-shadow-timing-harness/source-identities.json)
are retained separately from the frozen compile inputs.

The separate [initialization probe](sfa-shadow-pack.json) directly concatenates
into preallocated ND shadow storage. All bytes, including signed-zero and
NaN payload witnesses, remain unchanged through initial, changing-input graph
and post-timing checks. For 68 pages, prewarmed single-operation event medians
are 62.60 us for the direct API and 77.06 us for one graph replay. These events
can include host enqueue or replay gaps; compilation, allocation and capture
are excluded. They measure initialization-operation cost separately from the
native per-decode slot update. The logical read-plus-write payload is 19.125 MiB,
which is not a measurement of HBM traffic.

## Packed shadow KV with packed SFA workspace

The [v2 manifest](attention-indexed-shadow-v2-build/provenance.json) records
library `a81d1987b617d255d96c4eff90221d382ac9c9bf41bd2f6d9a14514a304db005`.
Its 61 wrapper source files match v1. The private overlay changes the merge
workspace to contiguous 576-element rows and changes CUBE source addresses
and pitches, preserving the original L1 operands and six QK accumulation
steps. [Transfer verification](attention-indexed-shadow-v2-build/transfer-verification.json)
checks all 58 manifest files on the test host, the 61 archived wrapper
sources and four overlay headers.

The [numerical report](attention-indexed-shadow-v2-numerics.json) passes 128
changing replays across four widths, both index-cache modes and 16/24 groups.
The [negative-slot report](attention-indexed-shadow-v2-negative-slots.json)
passes another 48. Complete original/shadow caches and all production
boundaries match raw bytes.

The [timing summary](attention-indexed-shadow-v2-summary.json) uses the same
eight-bank, 16-call, two-arena protocol as v1, with 30 event pairs per arena
and 30 graph replays per event. Median ordinary-minus-candidate differences:

| Width | C8 arena A / B us | BF16 arena A / B us |
| --- | ---: | ---: |
| 12 | -17.45 / -14.05 | -20.00 / -18.60 |
| 32 | -36.96 / -37.33 | -25.93 / -28.10 |

All 240 pairs regress, so v2 is not selected. Pre/post timing production
and full-cache byte gates pass. Per-call updates remain included and full
shadow initialization remains excluded. The unchanged
[executed harness](attention-indexed-shadow-v2-harness/source-identities.json)
retains the final call's boundaries in each bank sequence. Neither packed
variant establishes a performance benefit for the complete indexed path.

## Shared-eight-expert routing

The [summary](routed-pipeline-front-shared8-summary.json) measures the selected
native input-front composition with every token using the first token's
ordered eight expert IDs. Routed/shared hidden states and original-slot
probabilities retain their case-3 values. The
[target CPU witness](routed-pipeline-front-route-pattern-harness/target-cpu-fixture-identities.json)
checks equality between the mixed and shared-eight input bytes without
initializing NPU. Its hashes match the tensors recorded by the device runs.

Each event contains 30 replays of a 24-call graph over 24 independent,
equal-valued weight banks. Thirty samples per arm cover all six execution
orders five times. The selected native entry, previous dispatch entry and
CANN reference include shared input DynamicQuant and return the two BF16
branch outputs before their addition, routed scaling and TP reduction.

| Width | Selected native p50 us | Previous dispatch p50 us | CANN p50 us |
| --- | ---: | ---: | ---: |
| 12 | 113.48 | 138.25 | 220.77 |
| 32 | 162.33 | 184.59 | 243.38 |

The selected entry wins all 30 samples against each comparison at each
width. Every bank passes eight changing-input replays, giving 192 replays
and 7,104 raw-byte boundary comparisons per width. Post-timing outputs also
match. The executed
[harness](routed-pipeline-front-route-pattern-harness/source-identities.json),
raw samples and input identities are retained.

The [working-set calculation](routed-pipeline-front-route-pattern-harness/working-set.json)
counts 216 MiB of selected routed weights and 54 MiB of shared weights per
arm. The recorded CANN 9.2
[platform configuration](routed-pipeline-front-route-pattern-harness/Ascend910_9392.ini)
sets L2 capacity to 192 MiB. These payload counts describe addresses touched;
they do not measure cache misses or HBM traffic. The full routed NZ bank
allocations occupy 6.75 GiB. The earlier 81/158-expert results use four banks,
so the two sets are separate fixture measurements. Full-model route
distribution and round timing remain unmeasured.


## Three-stripe Down projection

The [candidate summary](routed-pipeline-front-striped-v2-summary.json) records
three N2048 slices per expert when 24 groups serve at most eight active
experts. The original K128 INT32 reduction, other projection tilings and
synchronization remain unchanged. Four widths with shared-eight or mixed
routing pass 96 changing replays, covering both the striped path and the
16/24-group fallback paths. The two 24-bank timing fixtures add 384 passing
replays; all 17,760 checked production boundaries match raw bytes.

| Width | Three-stripe median | Selected median | CANN median | Median paired saving | Wins |
| --- | --- | --- | --- | --- | --- |
| 12 | 116.63 us | 118.01 us | 224.58 us | 1.36 us | 30/30 |
| 32 | 159.92 us | 163.08 us | 246.33 us | 3.19 us | 30/30 |

Each arm includes the independent shared-input DynamicQuant and returns
separate routed/shared outputs before addition, routed scaling and TP
reduction. Each event averages 30 replays of a 24-call, 24-bank graph. The
30 samples cover all six three-arm orders. These initial measurements use
separate native workspace allocations in the same process. The candidate
remains unselected after the placement comparison below. The
[build manifest](routed-pipeline-front-striped-v2-build/build-manifest.json)
and [executed harness](routed-pipeline-front-striped-v2-harness/source-identities.json)
identify the artifacts used.


The [shared-native-workspace comparison](routed-pipeline-front-striped-v2-placement-summary.json)
aliases all 17 writable native buffers, including Matmul workspace and both
outputs. The three tiling tensors retain each library's own preparation
identity; their contents and the device sync base are checked for equality.
Shared DynamicQuant remains in each graph, with graph-specific temporaries.

| Width | Native workspace owner | Candidate median | Selected median | Median paired saving | Wins |
| --- | --- | --- | --- | --- | --- |
| 12 | Candidate | 115.19 us | 112.70 us | -2.51 us | 0/30 |
| 12 | Selected | 117.32 us | 113.48 us | -3.83 us | 0/30 |
| 32 | Candidate | 160.22 us | 162.60 us | 2.35 us | 30/30 |
| 32 | Selected | 161.97 us | 164.15 us | 2.15 us | 30/30 |

The four configurations pass another 768 changing replays at all 37
boundaries. Numerical checks use independent arenas. After timing, each
native graph is replayed, synchronized and copied before executing the next
arm, so shared output storage cannot mask differences. All post-timing output
bytes match the CANN reference. M12 regresses on both arenas; M32 retains a
small gain in the shared-eight fixture. The default library remains unchanged.


The same M32 comparison with 158 active experts exercises the original
Down loop in the candidate. Hidden/shared inputs and probabilities retain
the shared-eight fixtures' exact bytes; only the expert IDs change.

| Native workspace owner | Candidate median | Selected median | Median paired saving | Wins |
| --- | --- | --- | --- | --- |
| Candidate | 288.60 us | 285.63 us | -2.85 us | 0/30 |
| Selected | 291.96 us | 289.50 us | -2.33 us | 0/30 |

Both configurations pass 192 changing replays and separate post-timing
output snapshots. Together with the earlier numerical and timing fixtures,
this candidate passes 1,632 changing replays and 60,384 raw-byte boundary
comparisons. The shared-eight M32 gain does not justify enabling this
candidate for all M32 calls.


The independent [GU split-K candidate](routed-pipeline-front-splitk-v1-summary.json)
uses three K2048 products per expert when 24 core groups serve at most eight
active experts. The three INT32 results are summed before the original scale
and FP16 boundary; Down and the shared branch retain their original arithmetic.
The [build records](routed-pipeline-front-splitk-v1-build/build-manifest.json)
and [numerical harness](routed-pipeline-front-splitk-v1-harness/source-sha256.json)
identify the exact artifacts. Four widths, changing hot/mixed routes and the
16-group fallback pass 104 numerical replays at all 37 boundaries.

The [placement comparison](routed-pipeline-front-splitk-v1-placement-summary.json)
shares all 17 writable native buffers. Candidate GU storage contains three
planes; the control uses a contiguous view of its first plane. Each library
retains its three prepared tiling objects, and both tensor views remain alive
until replay and separate output snapshots finish. Two independent backing
allocations are tested. Shared DynamicQuant temporaries remain graph-specific.

| Route pattern | Width | Arena owner | Candidate median | Selected median | Median paired saving | Wins |
| --- | --- | --- | --- | --- | --- | --- |
| Shared eight | 12 | Candidate | 114.44 us | 115.86 us | 1.39 us | 30/30 |
| Shared eight | 12 | Selected | 111.68 us | 113.14 us | 1.48 us | 30/30 |
| Shared eight | 32 | Candidate | 162.91 us | 163.21 us | 0.27 us | 29/30 |
| Shared eight | 32 | Selected | 159.92 us | 161.37 us | 1.46 us | 30/30 |
| Mixed, 158 experts | 32 | Candidate | 294.53 us | 289.25 us | -5.27 us | 0/30 |
| Mixed, 158 experts | 32 | Selected | 296.25 us | 289.93 us | -6.20 us | 0/30 |

All six timing fixtures pass 192 changing-input replays each, followed by
separate raw-byte output snapshots. Together with the initial gates, this
candidate passes 1,256 numerical replays and 46,472 boundary comparisons.
The mixed-route regression exceeds the shared-eight benefit. The selected
library remains `b0a87a13cf07ce653e6e602c4dfad9eadbbe4ace91cf6f65b0ef673efce74016`.
