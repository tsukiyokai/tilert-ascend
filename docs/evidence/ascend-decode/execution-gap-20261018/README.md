# Execution boundary experiments

A3 `a3-gs:trt-asc`, GLM-5.2-W4A8C8, TP16, batch one, frozen 8K
prefix. The target remains request TPOT ≤2 ms.

## Small-packet greedy selection

The candidate replaces full-vocabulary gather followed by argmax with the
pinned vLLM `LogitsProcessor.get_top_tokens`. Projection, quantization,
FP32 TP sums, AI_CPU HCCL and the M4 target graph stay unchanged; head
selection remains eager. Four head calls per steady round send 193,600
versus 80 logical bytes per rank. Collective count remains four.

[Precision and timing audit](top1-audit.json): 2,176 same-hidden head
checks and 4,512 token IDs agree, including M1/M2/M4 projections. Sixty-four
serial/speculative/graph/reset snapshots match the previous execution at
the tensor-bit level. Both long runs preserve all sixteen ranks' six
1024-token trajectories, proposals, acceptance and cache states.

| Sequential run | Three measured TPOT values (ms) | Median (ms) |
| --- | --- | --- |
| Small packet | 22.062, 22.117, 22.292 | 22.117 |
| Matched full vocabulary | 20.674, 21.060, 20.229 | 20.674 |

Each run has three warmups, 303 rounds/request and actual acceptance
1023/303. The matched sources differ only in the two head implementations.
Small packets are 6.98% slower in this control and are not the default.
Fewer bytes do not establish lower latency; additional eager device work
has not been separately timed. These three-request measurements exclude
HTTP/HIXL and do not satisfy teacher or formal performance acceptance.

## Compiled target comparison

The precision runner and session benchmark accept
`--compile-target --target-graph`; the precision runner requires width four.
The target uses the installed AscendCompiler. Add `--super-kernel` to enable
static kernels and SuperKernel; both are disabled by default. MTP keeps
both model configs eager and separate registration state. Static binary
selection is process-wide, including matching draft operator shapes.
An uncaptured forward compiles before manual M4 ACL capture.
Benchmark reports retain compiler counters and cache-artifact hashes.

[Configuration precheck](compiled-config-audit.json) uses the actual
installed packages in fresh server CPU processes: eager 14/14, compiled
15/15 and six CLI cases pass. Guarded device calls remain zero, with NPU
uninitialized. The checked eager-config snapshot matches the prior HEAD;
the compiled variant changes compilation/graph settings, including the
platform's `enable_npugraph_ex`.

[Resident-layer execution](compiled-resident-audit.json) retains the original
decorated backbone and AscendCompiler, with only layer0 resident. Real M4,
8K KV, TP16 and FP32 TP sum execute under a 2 GiB Torch quota. All sixteen
ranks compile one artifact; 848 driver checks and an independent audit of
80 saved phase tensors pass. Graph/compiled and both variant resets are
bit exact; compiled/eager hidden NRMSE is 0.00540412, cosine 0.99998540,
maxabs 0.03125. Selected indices and new KV are exact between variants.
The hidden boundary includes final norm after the truncated layer0.

The audit binds 392 files, including 32 compiler artifacts and 16 original
prefixes. Full prefixes were not saved per phase; their 480 exact checks
belong to the driver. AOT function serialization warns on all ranks;
in-process compilation/Graph pass, disk reload is unverified. This run
supplies no latency. Full-model serial/M4/reset validation precedes
1024-output timing.

The [paired hot-graph measurement](compiled-resident-timing-audit.json)
uses independent wrappers, graph handles and memory pools. Ten warmups
precede 30 AB/BA pairs, each event covering ten replays. Rank-MAX event
p50/p95 are 4112.45/4302.42 µs for eager and 3968.07/4171.79 µs for
compiled. Compiled wins 20/30 pairs; the paired difference has median
50.25 µs and mean 78.46 µs. The difference between the two p50 values,
144.38 µs, is not the paired median gain. All 32 saved timed outputs
match their own eager/compiled references bit for bit. This fixed-input
shared-device graph excludes metadata, head and session work. Profile
the remaining layer cost before selecting a fusion; do not extrapolate
these timings to 78-layer TPOT.

The [short profile](resident-profile-audit.json) captures three replays per
case on ranks 0/14. This graph contains embedding, layer0 and final norm:
three AllReduce calls, including embedding. The helper's frozen two-role
annotation omitted that first collective. Compiled execution replaces
AddRmsNormBias plus DynamicQuant with one AddRmsNormDynamicQuant; compute
task count falls from 45 to 44.

| Replay 1 | Rank | Graph span (ms) | Compute union (ms) | O wait (ms) | Prolog span (µs) |
| --- | --- | --- | --- | --- | --- |
| Eager | 0 | 4.445 | 0.377 | 3.657 | 30.96 |
| Eager | 14 | 4.444 | 0.394 | 3.648 | 39.74 |
| Compiled | 0 | 4.572 | 1.865 | 2.255 | 1495.80 |
| Compiled | 14 | 4.573 | 2.907 | 1.200 | 2553.00 |

These are individual profiled samples. Exported O transit is 7.62–7.84 µs
in these four samples; its envelope includes waiting. All three replay
APIs finish before the first graph completes on each profiled rank.
Kernel-uncovered intervals are 6.64–20.78 µs and contain exported runtime
tasks; they do not establish launch gaps. RoPE tasks with missing CSV model
IDs are bound to this graph through trace model/stream/task identifiers.
No PMU or cross-rank device-clock alignment was captured. The shared
4 ms graph span is not a layer compute floor; the slowest of all sixteen
ranks and the source of individual task stalls remain unmeasured here.

## Static kernels and SuperKernel

The [activation audit](static-super-activation-audit.json) exercises the
installed AscendCompiler → npugraph_ex path on the same resident layer0,
M4 and 8K prefix. Both static and super options are enabled. The scheduler
limit is four so the official shape filter retains bucket `[4]`; actual
metadata still describes one request with four verification rows.

Rank 0 compiles 34 operators and installs the generated package into a
private directory. Sixteen distinct worker PIDs select that same directory
through the path-specific API. The guard records 146 installed-file hashes
before the SDK removes its directory; source, package and receipts remain
on the server. No global operator installation changes.

Independent CPU comparison passes 2,959 checks. Every rank's graph and
reset tensors agree with its active configuration bit for bit. Static
compiled/graph outputs also match the frozen plain compiled/graph outputs:
569,584 elements per comparison, including indices, new KV and metadata.
Compiled/eager hidden NRMSE remains 0.00540412, with all 2048 indices per
row shared; that comparison is diagnostic. Peak Torch allocation is
939,961,344 bytes/die.

The [short trace](static-super-profile-audit.json) independently passes
1,315 checks. On ranks 0/14, each of three replays executes six SuperKernel
tasks: four groups have different first/last operators, while LightningIndexer
and final AddRmsNormBias are single-operator wrappers. There are also 28
standalone compute tasks and three separate AllReduce AICPU tasks. This
graph includes embedding. Using the same model/stream/task counting rule,
compute tasks fall from the plain compiled trace's 44 to 34; MIX AIC/AIV
is counted once. Ten observed endpoint names bind to package
metadata and binary hashes; Level0 does not enumerate all interior members.
Host launches link to complete MODEL_EXECUTE/WAIT spans and final sync.
These profiled shared-device spans do not establish a latency gain.

Both precision and timing runners expose
`--target-graph --compile-target --super-kernel`. The
[product audit](static-super-product-audit.json) uses the actual
`create_config(super_kernel=True)` and reuses the approved static package.
On sixteen ranks, all five saved phases match the frozen static and plain
controls bit for bit. CPU checks cover the three configuration modes, six
CLI cases and eager MTP isolation. The device scope is one resident dense layer.

The [MoE activation audit](static-super-moe-audit.json) extends the actual
product configuration to resident layer3, M4 and TP16. Input A uses frozen
hidden/residual/shared indices; B applies documented synthetic row/index
reorders. Ten phases include eager, compiled, Graph and dirty resets.
The [plain/static comparison](static-super-moe-comparison.json) finds all
5,611,520 compared elements bit identical; each configuration also passes
its own replay/reset checks. Peak allocation is 999,919,104 bytes/die.

The static process privately installs its 37-operator package and records
158 installed-file hashes; sixteen processes reselect it. Every rank's
actual ELF-open trace retains the patched GMM1. The process-local static
blacklist preserves that implementation.

The [MoE short trace](static-super-moe-profile-audit.json) closes three
actual replays on ranks 0/14. Plain compilation executes 27 compute tasks
per replay. Static execution has six SK tasks and eight standalone compute
tasks, totaling 14; both paths retain two separate AllReduce tasks. Three
SK groups have distinct first/last endpoints; three are single-operator
wrappers. Patched GMM1 remains standalone. Endpoints bind to the exact
private package and binaries, including rank-specific Unpermute names.
Final outputs from all sixteen participants match their graph references.
Complete teachers and matched model timing follow separately. Raw artifacts
stay in `/home/shanshan/trt-asc/offline/cann-resident-moe-static-super-20261018-01/`.
Trace artifacts stay in
`/home/shanshan/trt-asc/offline/cann-resident-moe-profile-20261018-01/`.

## Symmetric-buffer reduction

[SHMEM audit](shmem-audit.json) binds source, binaries, launch parameters
and logs for SHMEM `24c15d21`. The frozen push implementation passes two
PEs on dies 0/1: 12 KiB BF16 payload, four tiles, fixed-rank FP32
accumulation and one BF16 output rounding. Each rank checks 1,204,224
values through epoch 196. Thirty samples give rank-MAX event p50 27.77 µs
and wall p50 90.32 µs. Inputs rotate independently; this is not the
156-dependent-sum HCCL workload or the TP16 ≤6 µs acceptance.

TP16 driver mapping, ACL mapping and peer-pull variants all terminate
with runtime error 507015 before timing. The push ELF's first fault is a
DCCI after remote MTE3 output; its L2 hint is present. The precise peer
access failure is unresolved. The separate built-in barrier has an
FFTS parameter-offset inconsistency; replacing that barrier enables the
two-PE run but does not fix TP16. Keep working HCCL for decode controls.

## Observed expert unions

[Route audit](route-audit.json) checks 2,431 calls/rank from one graph
prewarm round, one warmup and one diagnostic 1024-output request. All
sixteen JSONL files are byte-identical; request trajectories and states
match the full-vocabulary control. A captured device copy refreshes every
layer's final expert IDs on replay, after poisoning the buffers with −1.
This instrumentation supplies no performance measurement.

For the diagnostic's 303 M4 rounds, target expert unions across four rows
average 27.121 per layer (median 28, range 14–32); conditioned MTP averages
29.729. Applying these IDs to the fixed 8K logical-read formula gives
6.833–7.481 GB/round for one/four cache passes. At measured A=1023/303 and
die 0 stream bandwidth, the HBM scenarios are 1.753–1.919 ms/token, leaving
0.247–0.081 ms for work that cannot overlap. Actual HBM reads, cache reuse
and model bandwidth remain unmeasured; the trajectory grows toward 9K.

## Communication decision

Keep the verified complete peer protocol for idle and model comparisons;
the publication-fusion screen below supplies no net gain. The
[open A3 deterministic executor](https://gitcode.com/cann/hcomm/blob/2b4f2a4ed628cbeb48d76eead015ef220ea5400b/src/legacy/ascend910/algorithm/base/alg_aiv_template/all_reduce/aiv_all_reduce_91093_deter.h)
already performs peer pull and reduction within one AIV kernel. This is
the source model; the installed trace proves AIV execution, without binding
it to that exact algorithm. Full
replication can remove its result-gather stage and GM reduction tree, but
raises per-rank network bytes from `2(P−1)B/P` to `(P−1)B`: eightfold at
TP16 for equal dtype, fourfold for BF16 transport versus current FP32.
At 48 KiB BF16, one-shot sends 720 KiB/rank; an 8 µs window requires at
least 92.16 GB/s of outgoing payload bandwidth before protocol work.

The measured 156-sum FP32 96 KiB chain is 5.044 ms. A hypothetical
156×8 µs scheme would use 1.248 ms, or release 1.124 ms/token of
communication budget at the observed acceptance. This is a budget
scenario, not a predicted TPOT gain. It still consumes 0.370 ms/token,
more than the unoverlapped budget above. Fusion/overlap and reduced reads
are therefore part of this route's acceptance, alongside standalone speed.

The [initial resource queries](hccl-query-audit.json) both hit the 60 s
timeout. In the phased second run, all sixteen ranks first pass the
6144-element FP32 control, then enter the native shim without returning.
The registered-workspace variant then passes on all sixteen ranks under
concurrent external load: same-tag 64 KiB scratch, successful registration,
resource/context copies. All 256 public remote-buffer queries return zero
status but null addresses and sizes; they establish no peer mapping.
The device reader uses the graph context's peer table. Installed ELF
and copied context agree on 336 bytes, field offsets and ranks 0–15; scratch
starts at the caller's allocation. All process groups are destroyed, the
external PIDs remain and every die reports healthy. This functional run
has no communication latency measurement.

The [device-read and reduction audit](peer-reduction-audit.json) binds the
subsequent TP16 runs. The 32-byte reader checks all 256 rank/peer pairs
and 2048 INT32 words exactly. Ordered BF16 peer pull then checks 393,216
output elements against an independent integer round-to-nearest-even
oracle; 17,526 positions/rank require final BF16 rounding. Accumulation
uses FP32 in ascending rank order and rounds to BF16 once.

| Constant-input 48 KiB peer pull | Rank-MAX p50 (µs) | p95 (µs) |
| --- | --- | --- |
| Wait after each peer load | 21.815 | 32.368 |
| Issue sixteen peer loads, then wait once | 14.334 | 25.621 |

Each graph contains ten calls; thirty batches supply 480 raw rank timings.
Sixteen AIV workers each handle a 3 KiB tile. The burst variant reserves
64,768 UB bytes and retains the same ordered arithmetic. These separate
shared-device hot runs exclude input publication, ready/done and overwrite
protection. They are not complete AllReduce latency or decoder speedups.
At observed acceptance, 156×14.334 µs corresponds to 0.662 ms/token before
those missing costs. The current pull implementation does not meet 8 µs.

The registered scratch extent follows the installed executor's allocation
formula, not the caller's allocation size: FP32 count 6144 exports 50,308
bytes. Increasing the count to 131072 under a new tag exports a formula
extent of 1,049,732 bytes inside a 1,114,112-byte owned allocation.
All sixteen ranks read both endpoint regions correctly: 4096 independently
encoded words. All groups close and subsequent snapshots preserve the same
external PIDs and sixteen healthy dies; the burst timing run has no
immediate postflight snapshot.

## Complete one-shot push

The [full-protocol audit](full-protocol-audit.json) checks actual TP16 peer
writes, sender-indexed mailboxes, READY, ordered local FP32 sum, one BF16
rounding, ACK and retirement before reuse. Both implementations pass
changing inputs, delayed consumption, dirty mailboxes and repeat replay;
each has 1,572,864 raw output elements matching an independent integer
RNE oracle. Initial epoch 37 reaches 441 after four functional and 400
zero-chain calls. The single-kernel variant uses the installed public
`aclrtLaunchKernelWithHostArgs`, batch schedule mode one and sixteen AIV
blocks; binary load, graph capture, replay and unload complete successfully.

| Shared TP16, ten dependent sums/graph | Rank-MAX p50 (µs/call) | p95 |
| --- | --- | --- |
| Full push, separate producer/consumer kernels | 43.618 | 51.955 |
| Full push, one batch-scheduled kernel | 38.059 | 47.747 |
| Full push, double payload/READY slots, no ACK | 35.716 | 41.519 |
| HCCL, BF16→FP32→SUM→BF16 | 59.183 | 74.837 |
| Bare FP32 HCCL, 96 KiB | 56.826 | 64.903 |
| Native BF16 HCCL, 48 KiB | 48.429 | 59.512 |

Thirty batches supply 480 rank timings per case. Push includes payload,
flags, arithmetic and reuse; the matched HCCL contract includes both dtype
conversions. Timing uses an in-place zero dependency chain without rescale;
functional checks use nonzero changing inputs. Runs are consecutive under
the same sixteen external PIDs, with healthy postflight on all dies.
These are shared-device batch averages, not 156-call-chain or decoder
measurements. The earlier 14.334 µs read-only number is not the complete
protocol's latency.

The double-slot variant also passes the changing-input, dirty-buffer,
delayed-consumer and reuse checks: 1,572,864 oracle output elements and
480 raw rank timings. Both READY banks finish at their expected epochs;
external PIDs and sixteen healthy dies are preserved. For strictly ordered
per-tile P→C epochs, double payload/READY slots replace ACK: every `P_i(e+2)`
follows `C_i(e+1)`, which waits for all `P_j(e+1)`, each following
`C_j(e)`. Thus every consumer has retired slot `e mod 2` before its reuse.
Errors must stop later publication, and epoch counters must not wrap.
The shared p50 improves from 38.059 to 35.716 µs but remains far above
the 8 µs idle budget. Stop standalone flag-loop and tile-size tuning;
the next mechanism must fuse the producer or overlap useful work.
Idle calibration and a 156-call/model comparison remain separate measurements.

The [paired sender-parallelism screen](sender3-pair-audit.json) assigns three
AIV groups to disjoint peers, using all 48 allowed vector workers. Local
sender-done flags protect the in-place input before group zero writes back.
Five changing-input/delay cases supply 1,966,080 exact integer-oracle output
elements; all sixteen ranks reach epoch 842 and close both libraries.
Thirty same-process AB/BA pairs give baseline/sender3 rank-MAX p50
32.476/32.537 µs. The paired baseline-minus-candidate median is −0.196 µs;
the candidate wins 14/30. Its positive mean difference is dominated by one
182 µs baseline sample. Retain all samples and stop this branch.

## Actual O producer

The captured layer0 M4 call uses INT8 `[4,1024]` activations, NZ INT8
`[1024,6144]` weights, BF16 channel scales and FP32 token scales. The
installed dispatch is QuantBatchMatmulV3 key 24, CV 1:1, 24 cores. Its
376-byte tiling selects one M/N wave: each core completes K=1024 and
produces a BF16 `[4,256]` tile. The full output has row stride 6144.
There is no later N wave to hide communication in this shape.

An independent INT32 dot followed by the actual FP32 scale order matches
all sixteen frozen partials: 393,216 BF16 elements. The
[isolated rebuild](o-producer-audit.json) reuses the installed pertoken
implementation unchanged, with the same template and tiling. Two single-die
executions match the original partial and integer reference bit for bit
(49,152 output elements, 23.6 MB peak allocation). The audit rehashes 854
source/artifact bindings. No global operator installation changes.

The original local projection's separate hot graph measures p50/p95
19.229/108.962 µs over 30 samples of ten calls, under external load. This
excludes dynamic quantization and TP reduction; the roughly 6 MB weights
remain hot. These single-die values are not the TP16 pipeline timings below.

## O publication and complete reduction

The [complete-pipeline audit](o-publication-audit.json) preserves the CANN
BF16 production point, original output writeback and existing TPipe. Each
of 24 producer tiles publishes payload/READY; a separate AIV consumer waits
for all ranks and sums in ascending rank order in FP32, rounding once to
BF16. Push writes every peer's mailbox; pull publishes locally and lets
consumers read remote payloads. Both use double slots with causal retirement.
The strongest unfused path uses the installed CANN projection followed by
the existing 16-tile single-kernel push. It and fused publication plus
consumer each use two kernels; fusion saves no launch here.

Changing-input, delayed-consumer and dirty-buffer checks pass. The
installation-control run checks 320 actual partials against an independent
INT32 dot/FP32-scale reference. Five separate graph pools then execute ten
complete pipelines per graph, with ten warmups and thirty cyclic blocks.
Every timed last-output snapshot agrees with its own warm reference;
ordered paths also agree with the fixed-order CPU sum. Resets and Gloo
barriers occur outside timing. The native stream handle is read after the
installed Torch matmul, draining its host submission queue before native
publication without disabling TaskQueue.

| Complete O + sum, TP16 | Rank-MAX p50/p95 (µs) | Paired installed-minus-case median (µs) | Wins/30 |
| --- | --- | --- | --- |
| Installed CANN + push16 | 162.141 / 167.688 | — | — |
| Rebuilt CANN + push16 | 171.980 / 180.745 | −10.609 | 1 |
| Fused push24 + consumer | 162.781 / 169.121 | −0.328 | 13 |
| Fused pull24 + consumer | 164.485 / 168.010 | −2.129 | 9 |
| Installed CANN + FP32 HCCL | 183.439 / 188.582 | −21.994 | 1 |

The earlier rebuilt-only comparison reported a 9.025 µs paired median gain
for fused push. The installed control removes that gain: the same-round
rebuilt-minus-installed median is 10.609 µs. The rebuild's additional cost
is not localized to a specific instruction or compiler change. Stop this
publication-fusion branch and retain the unfused protocol for idle
calibration and model comparison. Shared hot-graph gains over HCCL are not
decoder TPOT gains. Next measure existing static kernels and SuperKernel
on the actual resident layer before developing another fusion mechanism.

## Raw artifacts

Under `/home/shanshan/trt-asc/offline/` on a3-gs:

- `cann-small-top1-20261018-01/`: precision reports, snapshots, frozen
  candidate source, patch and CPU audit.
- `cann-small-top1-long-20261018-01/`: candidate timing and CPU audit.
- `cann-full-vocab-control-20261018-01/`: matched control and CPU audit.
- `cann-route-unions-20261018-02/`: final expert IDs and CPU audit.
- `cann-compiled-config-20261018-02/`: frozen compiler candidate and CPU
  configuration/CLI checks; reuses the prior HEAD check in `-01/`.
- `shmem-oneshot-20261017-01/`: both source variants, binaries, per-rank
  logs, CSV samples and ELF audit. The failed prototype stays here.
- `hccl-peer-probe-20261018-01/`: installed-header query shim, sources,
  bounded runs, complete push variants, matched HCCL samples and audits.
- `cann-resident-profile-20261018-01/`: raw ranks 0/14 traces, CPU exports
  and all sixteen frozen O projection calls, captured after profiling.
- `cann-o-producer-20261018-01/`: actual dispatch dump, integer references
  and the existing projection's hot-graph baseline.
- `cann-o-key24-20261018-01/`: isolated build, exact tiling, generated ABI,
  two device outputs and independent CPU audit.
- `cann-o-publish-20261018-01/`: publication sources, finite protocol
  checks, both complete-pipeline timing runs and independent raw audits.
- `cann-resident-static-super-20261018-01/`: private static package,
  sixteen selection receipts, saved phase tensors and activation audit.
- `cann-super-product-20261018-01/`: actual product-source snapshot,
  configuration checks, resident-layer execution and raw comparisons.
