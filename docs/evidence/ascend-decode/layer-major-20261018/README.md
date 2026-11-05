# Layer-major frozen-teacher replay

The [runner](../../../../tests/glm52/run_cann_layer_major.py) keeps one
decoder layer resident and executes its teacher tokens sequentially at
M=1/DecodeOnly. A3 `a3-gs:trt-asc`, TP16, W4A8C8, frozen 8192-token
prefix; sixteen external jobs remain present. The Torch quota is 2 GiB/die.

For fixed teacher inputs, node `(layer, token)` depends on that token's
preceding-layer hidden/residual, earlier KV rows in the same layer and the
same token's latest full-index selection. Layer order followed by token
order satisfies all dependencies. Carry hidden and residual separately;
save each token's top-2048 and restore it at every shared-index layer.
This preserves M1 shapes and numerical configuration during the reorder.

| Device control | Raw tensor comparisons | Elements compared |
| --- | --- | --- |
| [2 layers × 2 tokens](dense-audit.json), sixteen ranks | 416 | 1,355,776 |
| [7 layers × 2 tokens](mixed-index-audit.json), sixteen ranks | 1472 | 5,716,000 |

All comparisons are bit exact against frozen token-major CANN tensors.
They include merged layer boundaries, new KV and selected indices; the
separate hidden/residual components also match their following-layer
inputs. Layers 3–5 restore shared selections per token. Layer 6 covers
INT8 index keys and FP16 scales. Peak allocated memory is 940,005,376
bytes/die, with layer ownership released between iterations.

The indexed loader removes the C8 quant class's unused `q_rot/k_rot`
parameters before checking weight coverage. The installed class creates
empty placeholders and its postprocess is a no-op; actual C8 execution
uses the indexer's `q_hadamard/k_hadamard` buffers. The layer-6 bit control
includes the resulting keys, scales, indices and hidden boundaries.

Teacher inputs and successor labels come directly from
[teacher-32.json](../vllm-baseline-20261017/teacher-32.json). Its prompt hash
identifies the original JSON file; the captured prefix hash identifies
compact token IDs. The runner checks both representations against the
same original prompt. Final norm and LM head execute once per M1 row.
Complete 78-layer/32-step agreement remains a separate end-to-end check;
the controls above establish the reorder, not full teacher acceptance.
There is no latency or TPOT measurement in this execution order.

## Complete 78-layer teacher execution

The [full execution audit](full-teacher-audit.json) checks 1,264 artifacts,
2,464 exact hidden/residual handoffs covering 484,442,112 elements, and
81,788,928 unique causal index entries. All sixteen final logit tensors
are bit identical. Peak allocation is 940,374,016 bytes/die. Execution
takes 1,171.65 seconds in this residency order; it supplies no decode timing.

Every rank matches 31/32 frozen successors. The only mismatch is zero-based
step 19, position 8211, input 323: prediction 279 versus label 64019.
Both BF16 logits equal 16.75, with zero top-1 margin.

The [exact production-weight diagnostic](head-production-ties-exact-audit.json)
first converts checkpoint FP32 head rows to the deployed BF16 dtype, then
forms exact rational dot products with the frozen BF16 normalized hidden.
Scores for 279 and 64019 are 16.6934759437 and 16.7554145380: both correctly
round to 16.75. All three tied-maximum rows, seven candidates in total,
select the frozen labels at this higher output resolution. This is a
CPU diagnostic. Earlier raw-FP32-weight dot products in the server artifacts
are superseded by this dtype-correct calculation.

The [CANN FP32-head audit](head-fp32-audit.json) then executes the existing
`aten.mm.dtype(BF16, BF16, FP32)` on those frozen normalized outputs.
All sixteen actual `[9680,6144]` vocabulary shards run sequentially on one
die; weights remain BF16. M1 and M4 each match 32/32 successor labels,
with 4,956,160 FP32 logits bit identical between query widths. All winners
are unique; the smallest margin is 0.00261688. Peak allocation is
142,149,120 bytes. Seven exact production-weight dot products differ by
at most 1.1684e-5, six FP32 ULP; this is not a correctly-rounded FP32 claim.
The production constructor uses default global padding 64 and needs no
extra rows at vocabulary 154880. Earlier manually padded 9728-row probes
are superseded for product-shape comparisons.

The target and independent MTP head expose `--fp32-head` through the
precision/timing runners, retaining BF16 weights and the original gather
and vocabulary crop. The default remains BF16. The [actual TP16 M1/M4
processor, gather and crop](head-tp-audit.json) match the same-shape frozen
FP32 reference on all ranks, including all 32 teacher successors. Each width
checks 79,298,560 logits against the reference and between ranks.

The [local head Graph audit](head-graph-audit.json) replays all sixteen
shards sequentially on one die at M1/M4. Thirty-two graphs consume the
32 changing normalized hidden rows through 640 replays. All 9,912,320
FP32 logits match their eager artifacts bit for bit; both widths preserve
32/32 successors. Peak allocation is 141,264,896 bytes.

The [TP16 head Graph audit](head-tp-graph-audit.json) executes the official
processor, HCCL gather and vocabulary crop with one actual weight shard per
die. M1/M4 eager, changing-input Graph and recaptured reverse-order Graph
produce 96 saved matrices: all 475,791,360 FP32 values match the canonical
concatenated reference bit for bit. Within-rank, cross-rank and M1/M4
comparisons also agree; each variant retains 32/32 successors. Sixty-four
captures supply 1,280 replays, with output poisoned to NaN before each
replay and every graph reset before shutdown. Peak allocation is
143,805,952 bytes/die. These head-only runs supply no TPOT.

The process exits normally; sixteen dies remain healthy and external PIDs
unchanged. The log retains one CPU shared-memory warning. Raw artifacts stay in
`/home/shanshan/trt-asc/offline/cann-fp32-head-tp-graph-20261018-01/`.

## MTP own head

The [MTP head audit](head-mtp-audit.json) loads the actual layer-78 model,
its separate BF16 head and norm, then shares the target embedding. One real
M4 forward produces four pre-norm inputs. Its 9,912,320 initial logits and
118,947,840 M1/M4 head outputs across sixteen ranks match the independent
raw-shard references bit for bit. Each head call normalizes once.
Sixty-four head captures supply 320 changing-input/reset replays; peak reserved memory is
1,868,562,432 bytes/die. M1/M4 differences remain diagnostic.

The [first probe](head-mtp-load-failure.json) omitted the target constructor's
packed quantization mapping and failed during loading. The corrected probe
uses the actual constructor with empty decoder slots; product loading is unchanged.
Graphs here cover norm, projection, gather and crop. Artifacts
remain in `/home/shanshan/trt-asc/offline/cann-mtp-own-head-20261018-02/`.

## Complete MTP forward Graph

The [full-forward audit](mtp-forward-graph-audit.json) checks seed M1,
shared-index chain M1 and target-conditioned M4 with the real layer-78
loader, FP32 output head and frozen 8K prefix. Six changing-input cases
have eager, Graph and reverse-order recapture results on all sixteen ranks.
The 64,979,328 compared output elements and all saved new KV match bit for
bit. Every case also gives identical logits when the head processor uses
the already-normalized recycle hidden directly.

Each rank performs six captures and twelve replays with one graph resident
at a time. Peak allocated/reserved memory is 1,605,115,904/2,134,900,736
bytes/die under the 2 GiB Torch quota. The first three-graph residency
attempt exhausted that quota before saving numerical results; its failure
did not identify the bucket. Serial bucket lifetime supplies the results
above. These tests use frozen parents; the product check below exercises
live device-output chaining. Joint bucket residency is a separate check.

Artifacts stay in
`/home/shanshan/trt-asc/offline/cann-mtp-forward-graph-20261018-{01,02}/`.

## Product chain graph

`--draft-graph` in the precision/timing runners captures shared-index M1
chains through the actual
[CannMtp](../../../../tilert/models/glm_5_2_ascend/cann_mtp.py) interface.
The graph includes the MTP layer, its own head, gather, argmax and index
snapshot. Seed and target-conditioned M4 calls execute eagerly. Returned
graph storage is reused; the next call stages its parent in a separate
persistent buffer before replay overwrites the previous recycle output.

The [product audit](mtp-chain-graph-audit.json) runs an eager seed, two live
chain calls, then truncates to P and repeats the first chain with dirty
tail bytes retained. Graph, dirty request reset and recapture all match
the eager sequence: 64,979,328 output elements and 405,504 new KV elements
are bit exact. Sixty-four pointer records confirm direct recycle-output
aliasing and a distinct staged parent. Each rank performs two captures and
nine replays, closes both graphs and retains eager seed/M4 dispatch.
Peak allocated/reserved memory is 1,606,456,832/2,061,500,416 bytes/die.

CPU contracts pass 15/15 and 163/163 checks plus six CLI cases. Two old
cache-checker assertions now follow the established cross-width diagnostic
policy; production comparisons are unchanged. Nonfinite cache values and
same-configuration reset mutations still fail. Complete target/session
validation remains pending.
Artifacts stay in
`/home/shanshan/trt-asc/offline/cann-mtp-chain-product-20261018-01/`.

The [paired stage measurement](mtp-chain-timing-audit.json) covers two
dependent chain calls, including live recycle feedback and both token
reads. Each sample restores the same prefix and runs its eager seed
outside the timing window. Thirty AB/BA pairs first take the maximum
of sixteen rank times, then compute paired statistics.

| Two-chain wall time | p50 (ms) | p95 (ms) |
| --- | --- | --- |
| Eager | 29.51294 | 29.71705 |
| Graph | 28.88505 | 30.92198 |

Paired savings have median 0.468161 ms, p10/p90 −2.51656/+2.97847 ms;
Graph wins 19/30 pairs and has a worse tail. This shared-device result
does not establish a stable gain. All 960 samples match the frozen eager
C2 outputs (162,448,320 elements) and three new KV rows (2,027,520 elements)
bit for bit. Seed equality and prefix preservation are driver checks.
This stage excludes target verification, conditioned MTP and delivery;
it is not TPOT. Raw data stay in
`/home/shanshan/trt-asc/offline/cann-mtp-chain-bench-20261018-01/`.

## MTP output projection sharding

`--draft-projection-tp` in the precision/timing runners replaces replicated
`rot` and `eh_proj` with existing CANN column-parallel linear modules.
Each rank keeps 384 output rows and the complete K dimension, then gathers
the BF16 result before its consumer. Weights stay BF16 ND. The two matrices
occupy 212,336,640 fewer bytes/die; two M1 chains plus conditioned M4 avoid
637,009,920 logical weight bytes per steady round and add six AllGathers.
M4 also restores the gathered output layout. Defaults remain unchanged.

The [projection-path measurement](mtp-projection-module-audit.json) includes
rot, norms, concatenation, eh, AllGather and layout restoration on real
checkpoint weights and frozen parents. Thirty AB/BA pairs take rank-MAX
device-event times before aggregation:

| Projection path | Original p50 (ms) | Sharded p50 (ms) | Paired median saved (ms) | Wins |
| --- | --- | --- | --- | --- |
| M1 | 2.98914 | 0.56152 | 2.38907 | 30/30 |
| M4 | 2.95538 | 0.58270 | 2.36528 | 30/30 |

Own-configuration changing-input/reset outputs match bit for bit in
9,830,400 compared elements. The weighted `2*M1+M4` result is an isolated
budget proxy; it does not time a decode round.

The [complete MTP audit](mtp-projection-graph-audit.json) uses the actual
product loader and opt-in. Eager seed, two dependent chain calls, rollback,
dirty reset, recapture and conditioned M4 compare 64,979,328 output and
405,504 valid KV elements bit for bit within the new configuration. The
old/new seed outputs and seed-written KV agree exactly. Across projection
configurations, recycle NRMSE is 0.017567 at C1 and 0.080302 at C2; C2 logits
NRMSE is 0.061923 and KL is 0.011759. All eight saved top-1 positions agree.
The [fixed-C1 experiment](mtp-projection-fixed-c1-audit.json) restores the
same parent, token, full valid prefix and ordered indices for four choices:
original, rot-only, eh-only and both. Plain/hooked results and saved full
prefixes agree on all sixteen ranks. Rot-only changes one BF16 value by
one ULP; that difference disappears at eh output. Eh-only changes four
BF16 values by one ULP, then attention output NRMSE becomes 0.011584 and
final recycle NRMSE 0.017567. Both-sharded matches eh-only from eh output
onward. All choices select the same eight experts; their weights change.

An [exact integer/rational dot oracle](mtp-projection-dot-oracle.json)
checks the four changed eh rows against the frozen BF16 input and actual
checkpoint weights. All four sharded values are correctly rounded; each
original value is one ULP away. This establishes those four local results,
not full-projection or trajectory accuracy. Keep cross-configuration drift
diagnostic and measure proposal coverage and eventual accepted tokens.

The [teacher proposal screen](mtp-teacher-proposal-audit.json) restores one
frozen trajectory at thirty overlapping positions. Each configuration
commits one teacher-conditioned M1 row, drafts two scratch rows, then
truncates them before the next position. Labels come from teacher JSON;
the BF16-head trace's predictions are not substituted for those labels.
Five proposal slots change across three windows, but all thirty continuous
prefix lengths agree. Their L=1/2/3/4 histogram is 9/8/9/4; L includes the
known input token. This is proposal coverage, not actual M4 acceptance or
an end-to-end teacher check. The audit verifies 2,880 raw argmax results,
prefix preservation and rollback. Artifacts stay under
`/home/shanshan/trt-asc/offline/cann-mtp-teacher-coverage-20261018-01/`.

The [complete-chain timing](mtp-projection-timing-audit.json) retains the
actual original `nn.Linear` objects and switches them with the new modules
on one model. Each variant captures one graph, runs three unmeasured
two-chain warmups, then one measured sample. Seed, restore, capture,
validation and cleanup are outside the window; both token reads are inside.

| Two-chain Graph wall time | p50 (ms) | p95 (ms) |
| --- | --- | --- |
| Original projections | 28.30291 | 30.93028 |
| Sharded projections | 27.42619 | 29.92008 |

Thirty paired rank-MAX differences have median 0.522182 ms and p10/p90
−2.34552/+3.06099 ms; sharding wins 19/30 pairs. AB/BA median savings are
0.535997/0.453317 ms. The microbenchmark gain has not established a stable
complete-chain gain. All 960 saved C2 samples match their own frozen
references; each rank closes 60 captures after 480 replays. Peak reserved
memory is 1,788,870,656 bytes/die. Target verification, conditioned M4 and
delivery are outside this measurement. The short trace below identifies
task elapsed times and collective arrival skew.

Raw data and reproducible auditors stay under
`/home/shanshan/trt-asc/offline/cann-mtp-projection-{shard,product,chain-bench}-20261018-01/`.

## Complete-chain short trace

The [independently audited trace](mtp-chain-profile-summary.json) covers
two two-chain replays per variant on all sixteen ranks: 128 chain windows
and 640 collectives. Profiling is enabled; these are diagnostic durations.

| Head TASK elapsed, 64 samples/variant | p50 (µs) | p90 (µs) | max (µs) |
| --- | --- | --- | --- |
| Original | 144.683 | 3129.756 | 3510.240 |
| Sharded | 148.252 | 3222.114 | 3858.640 |

All heads use AI_CORE MatMulV3, 24 blocks, BF16 ND input and weights with
FP32 output. The preceding norms produce BF16 directly; the later Cast
converts ArgMax integers. Similar millisecond tails occur in MLA prolog
and the small router matmul. TASK spans do not separate useful arithmetic,
memory stalls, device scheduling, active co-load or profiler effects.

Empirical causal clock envelopes retain a head collective entry-spread
lower bound of 2.931–3.101 ms for original and 3.093–3.526 ms for sharded
using only the corresponding window. They do not prove microsecond
completion spread or always identify the last rank. All 640 collective
entries follow their local producer by 0.50–0.90 µs; this does not observe
remote payload visibility. HCCL child waits cross operation boundaries in
625/640 cases, so summing them with parent durations double-counts time.
Raw profiles, exports, join keys, causal bounds and auditors stay under
`/home/shanshan/trt-asc/offline/cann-mtp-projection-chain-profile-20261018-01/`.

### PMU capture

The [PMU audit](mtp-chain-pmu-summary.json) matches all 7,936 compute tasks
in 128 chain windows to exported PMU data and original context records.
The 640 AICPU collective wrappers are outside this counter coverage.
Raw integer counters reproduce all 896 AICORE time/Cube/MTE conversions.

| Same-shape instance | TASK elapsed (µs) | Counter-derived AICORE time (µs) |
| --- | --- | --- |
| Original head, fastest / slowest | 128.02 / 3579.44 | 103.09 / 3466.16 |
| Sharded rotation, fastest / slowest | 10.52 / 115.34 | 4811.09 / 4.78 |

The rotation reverses the apparent relation between counters and elapsed
time. These values survive raw stream/task/occurrence and timestamp
checks; they cannot establish exclusive counter ownership or attribute
the head tail to arithmetic, memory stalls or preemption. Context counters
are aggregated across blocks. All eighty sparse frequency samples report
1850 MHz; they do not measure frequency continuously. This capture runs
under the recorded external NPU load and a concurrent own CPU build.
Retain the unresolved attribution and use profiler-off matched execution
to select optimizations. Raw data and CPU reproducers stay under
`/home/shanshan/trt-asc/offline/cann-mtp-chain-pmu-20261018-01/`.

## Two chain calls in one Graph

The [unrolled Graph prototype](mtp-two-chain-graph-audit.json) keeps the
original projections and captures C1, device argmax/recycle feedback, and
C2 in one graph. Two private metadata/RoPE/slot sets advance independently;
one host read returns both candidates. A/B change positions from 8192/8193
to 8193/8194, then dirty reset restores A. On TP16, 16,244,832 output and
191,488 valid KV elements match the corresponding eager runs bit for bit.
Saved device-bridge tensors also match C1 recycle/token and seed indices.
Each rank captures once and replays three times, with peak reserved memory
1,835,008,000 bytes/die. Artifacts stay in
`/home/shanshan/trt-asc/offline/cann-mtp-two-chain-20261018-01/`.

The [sharded-model paired measurement](mtp-two-chain-timing-audit.json)
compares two single-chain replays/two token reads against one two-chain
graph/one candidate read. Both wall windows include dynamic metadata and
device feedback. Each sample follows three full-chain warmups; capture,
seed, restore and tensor snapshots stay outside the window. One graph is
resident at a time.

| Two dependent draft calls, rank-MAX wall | p50 (ms) | p95 (ms) |
| --- | --- | --- |
| Two replays | 27.20328 | 30.23225 |
| One two-chain Graph | 27.16949 | 28.97017 |

Thirty AB/BA pairs save a median 0.089471 ms; p10/p90 are
−1.696436/+3.369580 ms, with 15/30 wins. AB and BA medians have opposite
signs, −0.164832/+0.280664 ms. This shared-load screen does not establish a
stable gain; stop default integration and investigate device execution
tails. Capture-context rank-MAX median rises from 552.18 to 1051.22 ms.
Full target verification, conditioned M4, acceptance and delivery are
excluded, so this is not TPOT.

An independent CPU audit compares A/B/dirty/reset/recapture, metadata,
device bridge snapshots and all 960 timed C2 outputs/valid KV: 405,199,504
element comparisons are bit exact. Sixteen ranks each close 64 captures
after 375 replays; reported peak reservation is 1,788,870,656 bytes/die.
Memory peaks, graph counters and full-prefix preservation are driver
records; saved tensors and paired clocks are independently reconstructed.
Reproducers and raw outputs stay in
`/home/shanshan/trt-asc/offline/cann-mtp-two-chain-paired-20261018-01/`.

## Reference scope

The [reference configuration audit](reference-config-audit.json) binds
the baseline's original vendor, MTP3/graph4/async settings, and the
separate MTP1/eager prefill request that supplied this KV. Prompt IDs and
first token agree; the baseline request's KV was not saved. The prior
same-configuration native/TileRT/reset paths also choose 279 at position
8211. These observations distinguish the wrapper comparison, configuration
changes and prefix provenance; they do not identify a specific cause for
the original service's nondeterminism. HF remains the independent
diagnostic. Release acceptance remains eight prompts with 32/32 successors
each, plus the long trajectory check.

```bash
torchrun --standalone --nproc-per-node=16 \
  tests/glm52/run_cann_layer_major.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --capture-case /workspace/offline/prefix8k-allocation-20261017/capture/case-00 \
  --teacher-json docs/evidence/ascend-decode/vllm-baseline-20261017/teacher-32.json \
  --prompt-json /workspace/experiments/glm52/prompt-8192-code-review.json \
  --layers 78 --steps 32 --attention-kernel sfa --output /workspace/offline/teacher
```

Use the validated SwiGLU overlay, deterministic AI_CPU HCCL and 200 MiB
HCCL buffers, as in the frozen controls. Server artifacts reside under
`/home/shanshan/trt-asc/offline/cann-layer-major-20261018-{01,02}/`:
frozen source, launch/environment records, raw per-layer tensors and audits.
The command above retains the BF16 head and its 31/32 result. Add
`--fp32-head` to exercise the new output dtype through the complete runner;
the head-only evidence does not substitute for that execution.
