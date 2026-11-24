# A3 decode: implementation and remaining gaps

Snapshot: 2026-10-16. Target: one A3 host, TP16, GLM-5.2-W4A8C8, batch one, 8K context,
TPOT ≤2 ms. Development uses existing CANN operators. The target container
remains stopped; the checks below execute on the a3-gs host CPU.
[The 20:21 CST target check](prefix-capture/availability-target.json) finds
all sixteen dies occupied on a3-gs. The earlier
[19:43 inventory check](measurement/availability-20261016-1943.json) finds
six listed A3 hosts occupied and three unreachable.
[The CPU regression](prefix-capture/offline-regression.json) passes all
eighteen suites with the repository explicitly on `PYTHONPATH`.

## Runtime budget

The [calculator](../../../tests/glm52/estimate_traffic.py) follows the current
tensor representation: replicated FP32 routers, INT64 W4 scale carriers,
FP32 MSD compensation, BF16 norms/head, and discarded zero offsets.
[Results](runtime-traffic.json) give 3.844 GB of active weights plus 0.212 GB
of KV/index reads per rank. A streaming M=1 pass needs 2.535 ms at
1.6 TB/s, or 3.621 ms at 70% bandwidth, before communication and launches.
The earlier 3.828-GB header-based estimate undercounted runtime metadata
and the FP32 router conversion.

M=4 amortizes common weights but can activate 32 experts per layer.
For that union, the present main/MTP layout reads 7.54–8.19 GB per steady
round, depending on KV reuse. At 70% bandwidth and 3.2 committed tokens,
HBM alone costs 2.10–2.29 ms/token. Each round also issues 166 reductions
and four greedy all-gathers. The 2-ms route therefore requires less
replicated traffic, greater reuse/bandwidth, or greater acceptance as well
as communication overlap. These are analytical budgets; L2 effects,
MSD scratch, actual expert/KV unions and exposed HCCL time need profiling.
The steady-round budget excludes the initial MTP seed; request TPOT includes
that seed after first-token delivery.

## Implemented paths

| Component | Code | Current evidence |
| --- | --- | --- |
| Main verification | [NativeDecodeEngine](../../../tilert/models/glm_5_2_ascend/native_engine.py) consumes M=1/2/4 rows, keeps M1 trace shapes, and truncates valid KV lengths | CANN source contracts; device numerical execution pending |
| Causal index/MLA | Query lengths `[M]`, key lengths `[P+M]`, row bound `P+i+1`; selected IDs are `[M,2048]` with trailing `-1` | [21 CPU checks, 51 layouts](verify-layout.json) |
| C8 scoring reference | [arch22 source oracle](c8-arch22/README.md) rounds weight×query scale and positive QK/1024 to FP16 before head reduction | [6 source/scalar checks](c8-arch22/scoring-stdlib.json), including two strict 8K top-k counterexamples; device scoring/encoding comparison pending |
| MoE grouping | [MoePlan](../../../tilert/models/glm_5_2_ascend/moe_plan.py) groups selected experts and preserves token/slot inverse maps; CANN unpermute combines original slots | [26 CPU checks](moe-plan.json), including 210 exact rational-sum comparisons; CANN grouping parity pending |
| MTP state | [MtpState](../../../tilert/models/glm_5_2_ascend/mtp_state.py) commits main/MTP lengths, correction/bonus tokens, EOS and output budgets | [State oracle](check-mtp-state.json), including 10,000 alternating rejection/acceptance rounds |
| MTP computation | [NativeMtp](../../../tilert/models/glm_5_2_ascend/native_mtp.py) loads layer 78, applies rot→hnorm, shares the target embedding, retains its own head | [25 source/header/manifest checks](mtp-serving/mtp-contract-metadata.json); CANN stage parity pending |
| MTP oracle | [MtpReference](../../../tilert/models/glm_5_2_ascend/mtp_reference.py) retains HF decoder/expanded attention/cache and uses independent CPU TP16 weights | [Metadata checks](mtp-reference-metadata/manifest.json); actual CPU numerical generation pending |
| MTP orchestration | [MtpRoundExecutor](../../../tilert/models/glm_5_2_ascend/mtp_round.py) seeds, chains, verifies, recomputes target-conditioned KV and commits | [9 CPU tests](mtp-round.json): 11,664 boundary fixtures and 9,424 cross-product rounds, plus longer decode/cache comparisons |
| Request control | [AscendPDEngine](../../../tilert/models/glm_5_2_ascend/session.py) emits the prefill token before consumption; [TP16 control](../../../tilert/models/glm_5_2_ascend/tp_runtime.py) orders inject/step/reset across rank-local storage | [20 CPU tests](session.json); HCCL control execution pending |
| MTP serving | [AscendMtpPDEngine](../../../tilert/models/glm_5_2_ascend/mtp_session.py) uses the production serving factory and TP16 start/advance/finish commands; trims undelivered cache rows | [17 TP checks](mtp-serving/check-mtp-tp.json), [21 session checks](mtp-serving/mtp-session.json): 364 sequence/cache comparisons and 724 rounds; NPU/HCCL execution pending |
| Ascend PD | [V1 connector and TP16 receiver](pd-ascend.md) bind actual cache groups, synchronously WRITE main P/MTP P-1 rows, and retain the slot through decode | [Connector](check-ascend-connector.json) and [control](ascend-pd-control.json) CPU checks; real HIXL and NPU injection pending |
| Greedy head | Local max plus two FP32 scalars per rank; lower vocabulary ID wins ties | Device checker compares against gathered logits and HF tokens |
| Device MoE candidate | [Expert bank](../../../tilert/models/glm_5_2_ascend/ops/expert_bank.py) keeps route sorting/counts on NPU and calls two CANN W4A8 GMMs around SwiGLU | [Main layer](check-expert-bank-metadata.json) and [MTP layer](check-expert-bank-mtp-metadata.json): 13 metadata/source checks each; device parity/timing pending |
| Measurement | [Precision, capture and PD timing entries](measurement.md) bind the original prompts and record separate producer/client token times | [22 report checks](prefix-capture/precision-metadata.json), [22 timing checks](prefix-capture/decode-timing.json), [10 capture checks](prefix-capture/profiling-control.json); device/HTTP timing pending |
| Frozen inputs | [Capture receiver/client](measurement.md#capture-original-vllm-prefixes) preserve original main P/MTP P-1 rows, prompt IDs and the prefill token without decode weights | [8 TP16 control checks](prefix-capture/prefix-capture-control.json), [8 HTTP checks](prefix-capture/prefix-capture-http.json) with 33 CLI cases; live HIXL/NPU export pending |

The CPU checks import no Torch. Scalar cache/oracle execution establishes
control behavior, not NPU arithmetic or speed. The main path still uses a
host expert-group plan by default. `--expert-bank` selects the device
candidate in verify/MTP/round runners. Its bank occupies 321,650,688 B per
routed layer/rank and replaces that layer's per-expert weight storage.
The optional device index path avoids address-validation transfers.
[Current source checks](prefix-capture/source-checks.json) cover Python 3.11
syntax and ordinary control-module imports without Torch; neither candidate
has device timing yet.

The C8 reference now follows the installed arch22 scoring boundaries, with
the ideal formula retained as a diagnostic. The historical 419 key and one
query encoding differences remain unresolved. HF/native manifest checks
bind the current scorer and verified CANN sources, and require completed
numerical execution. [HF](c8-arch22/hf-reference-metadata.json) and
[MTP](c8-arch22/mtp-contract-metadata.json) metadata checks exercise those
conditions; no actual Torch reference has run.

## MTP position and hidden-state contract

MTP row p consumes normalized target hidden[p] and embedding(token[p+1]);
its RoPE and KV position are p. At main prefix length P, the MTP prefix
has length P-1 or P. A missing seed is computed at P-1. A full MTP prefix
without its output token/recycle state/indices also needs that row replayed.

The target conditioning state is after the target final RMSNorm. MTP applies
`rot.weight` before hnorm, concatenates normalized embedding before hidden,
and applies its own final norm once for logits and recycle state. Chain
steps reuse the first step's selected indices while writing fresh keys.

Each round verifies main rows `[P,P+M)`, then recomputes MTP rows over the
same range from target predictions and target hidden. Both retain P+A rows.
The final emitted token is pending outside main KV. Chained MTP KV cannot
be retained merely because its token was accepted: its parent hidden differs.
The [contract report](check-mtp-contract.json) records fixed vLLM/vLLM-Ascend source
locations, metadata hashes, embedding/head decisions and normalized roles.

## Device execution after the environment is free

Use `/opt/trt-asc/venv` in `a3-gs:trt-asc`, with source at
`/workspace/src/TileRT-Ascend`. Each device runner uses torchrun with
`--nproc-per-node=16`, CANN 9.2.0 and the same checkpoint. Run in this order:

1. Follow the [M1/HF guide](cann-operators.md#correctness-execution), first
   with a short prefix, then with frozen 8K prefixes and teacher tokens.
2. Run [run_native_verify.py](../../../tests/glm52/run_native_verify.py)
   with `--checkpoint`, `--reference`, `--output`, `--source-commit`, and
   `--prefix` for nonempty prefixes. It compares M=1/2/4 layer logits,
   selections/routes and new cache rows against the same HF traces;
   exercises truncation by 0–4 rows; and checks device-only indices/top1.
3. Run [run_native_mtp.py](../../../tests/glm52/run_native_mtp.py) with explicit
   normalized parent inputs and an independent oracle generated first by
   [run_mtp_reference.py](../../../tests/glm52/run_mtp_reference.py), using
   the same `--input` and `--prefix`. The native runner binds the oracle's
   implementation, input/checkpoint/prefix/rank hashes. Without an oracle
   it reports capture/replay only.
4. Run [run_native_mtp_rounds.py](../../../tests/glm52/run_native_mtp_rounds.py)
   to compare full speculative rounds with serial CANN AR and
   target-conditioned MTP teacher forcing. It accepts a cold prompt, or
   paired main/MTP prefixes plus the last prompt token. This checks device
   execution consistency separately from HF parity.
5. Run [run_native_session.py](../../../tests/glm52/run_native_session.py)
   over a JSON request list (`request_id`, `seq_len`, `prefix`,
   `first_token_id`, `max_tokens`, optional greedy `sampling`). It checks
   rank-zero delivery and all-rank reset with file-backed frozen prefixes.
   This entry does not perform HIXL transfer or OpenAI HTTP serving.
   Then run [run_native_mtp_session.py](../../../tests/glm52/run_native_mtp_session.py)
   with the round runner's CLI. It uses the actual serving factory and
   TP16 command channel, checking normal delivery, cancellation before/after
   outputs and repeated reset against serial CANN tokens and all valid
   main/MTP rows. Frozen prefix rows compare exactly; replay/new rows compare
   separately, so an 8K prefix cannot dilute their numerical error.
6. After numerical acceptance, use the [capture and timing entries](measurement.md)
   for complete M1/M4 decode. Each rank captures its own execution thread;
   request timing runs separately with the profiler off and records both
   callback completion and client token arrival. Then measure
   HBM and 12/48-KiB HCCL on the selected host, reconcile actual traffic with
   the budget, and compare each CANN candidate with its baseline.

The precision preflight binds the eight original prompts, frozen prefix
hashes, numerical reports, checkpoint, implementation and Torch/NPU versions
to these entries. A copied case or an unrelated 8K prompt cannot satisfy
the eight-input measurement requirement.

[check_expert_bank.py](../../../tests/glm52/check_expert_bank.py) defaults to
header/source checks. `--mode npu` exercises real rank-0/rank-15 partitions,
empty groups, shared/disjoint routes and local K128 quantization.
`--bench` runs only after those checks pass and records device/wall times
for both the device bank and the current host per-expert baseline. It does
not include shared FFN or HCCL. Full-model checks use `--expert-bank`.

## Remaining work

| Gap | Next measurable result |
| --- | --- |
| Full W4A8C8/TP16 precision | 78-layer HF parity; C8 encoding/scales, arch22 Cube accumulation and fused SwiGLU boundary resolved |
| MTP precision | Execute the independent layer-78 stage/logit/cache oracle against source-matched inputs and MTP prefixes |
| Host scheduling | Validate/profile the device GMM candidate, then implement fixed-stage graph capture from measured launch/synchronization costs |
| Traffic and communication | Real route/KV unions, HBM counters, exposed HCCL time, MTP acceptance on AgentX inputs |
| Live PD | Execute the TP16 connector/receiver/launcher against real vLLM caches and HIXL; compare injected rows, prompt replay and first token |
| Final performance | 8K/1024-token requests with TPOT p50/p95 ≤2 ms, including accept/sampling/delivery |

The [Ascend PD path](pd-ascend.md) uses a dedicated sixteen-rank receiver and
launcher. It does not depend on the GPU profile's eight-rank default.
Its CPU checks cover control and address planning; actual DMA, callback order
and NPU arithmetic still require the target environment. Serving now selects
serial M1 or MTP widths 2/4 with `--mtp-width`; the default is the serial
precision baseline. MTP round/control/delivery CPU coverage does not establish
acceptance rate or the device time needed by the speculative budget.
