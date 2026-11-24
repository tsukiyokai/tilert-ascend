# CANN MTP execution

The [CANN target](../../../tilert/models/glm_5_2_ascend/cann_model.py)
and [drafter](../../../tilert/models/glm_5_2_ascend/cann_mtp.py) use the
pinned vLLM-Ascend model implementations and existing CANN operators.
TileRT owns prefix injection, greedy MTP rounds and request delivery.
The target has no vLLM scheduler or speculative proposer.

## Boundary and execution

| Component | Contract |
| --- | --- |
| Main | Full-P cache, position P; M1/2/4 causal forwards return normalized hidden and next-token predictions. |
| Draft | Layer 78 cache contains P-1 rows; row p consumes token[p+1] and normalized, unrotated main hidden[p]. |
| Parent | Layout v3 transfers BF16 `[1,6144]` hidden[P-1]. Injection clones it before the receive slot can be reused. |
| First token | All sixteen ranks project the parent through the target LM head. This changes no KV position and applies no additional norm or rotation. HTTP decode checks the prefill token before delivering it. |
| Round | Seed and chained drafts precede target verification. Draft rows are then recomputed from verified target hidden; both caches retain only the accepted prefix. |

The drafter has its own configuration, static forward context and index
buffer. It uses the target's TP16 group and reserved layer-78 cache memory.
The official loader selects checkpoint shards 91-93 plus `rot.safetensors`,
performs weight postprocessing, retains the MTP head, and shares the target
embedding. Logits consume pre-norm MTP output; chained parents consume
normalized recycle output. Chaining explicitly reuses the seed indices.

[CannRankEngine](../../../tilert/pd_vllm/ascend_cache.py) connects this
boundary to the existing TP command loop and MTP session. Injection does
not call a main-model forward. Reset clears the owned parent; HTTP releases
the receive slot after the decode worker exits and reset succeeds.

## Serving and timing

[serve_cann_pd.py](../../../tests/glm52/serve_cann_pd.py) selects serial
decode with `--mtp-width 1` and MTP with `--mtp-width 2/4`. M1 uses layout
v2; MTP requires v3 and `TRT_MTP_PARENT_HIDDEN=1` on the prefill launcher.
The target and optional drafter load before the HTTP service starts.

`--timing-dir` saves callback completion times, accepted spans and TP
commands. Head projection occurs during injection, before g0 delivery.
The first draft seed occurs after g0 and is included in callback TPOT.
Client arrival time is measured separately by the PD benchmark. A stopped
prefill followed by decode model loading must withhold g0 until decode is
ready to keep that loading interval within TTFT.

## CPU acceptance

All checks below ran in isolated server source directories, with no NPU
initialization. They do not measure model precision or performance.

| Evidence | Passed | Executed boundary |
| --- | ---: | --- |
| [MTP API](cann-mtp-cpu-20261017.json) | 14 | Real speculative config, loader file selection, cache binding and metadata; model creation, payload loading, postprocessing and arithmetic use explicit fixtures. |
| [PD integration](cann-mtp-pd-cpu-20261017.json) | 6 | Full-P injection, parent ownership, TP16 scalar rounds, actual ASGI streaming and decode thread, timing persistence, bad-g0 rejection, reset and next-generation admission. |
| [PD regression](cann-mtp-regressions-20261017/ascend-pd.json) | 33 | Receiver ownership and existing serial/replay lifecycle. |
| [Parent regression](cann-mtp-regressions-20261017/parent-hidden.json) | 9 | Torch hook, v3 metadata, sixteen TCP senders and a HIXL WRITE double. |
| [Session regression](cann-mtp-regressions-20261017/mtp-session.json) | 21 | Acceptance, EOS, cancellation, capacity tails and retained positions. |

The [14 static and CLI checks](cann-mtp-static-20261017.json) pass,
including default M1 selection and explicit M2/M4 options.

Device acceptance requires real weight loading, an 8K v3 HIXL capture,
serial-versus-MTP token/cache comparison, and independent HF numerical
comparison. P/D coexistence memory remains unmeasured; the current staged
accuracy route is described in the [capacity audit](pd-capacity-20261017.json).

The [device consistency entry](../../../tests/glm52/run_cann_mtp_rounds.py)
loads one target/drafter pair and compares serial continuation, speculative
rounds and an exact reset run. Use a v3 `--capture-case`; its defaults are
`--width 4 --max-tokens 16`. Every rank saves new main/MTP KV, actual round
commits, input identities and source hashes. Logical outputs include EOS:
for T=1, main/draft lengths are P/P-1; for T>=2 both are P+T-1, with the
last output still pending. Frozen prompt rows must remain exact.
KV comparison checks each token row. Serial and the first speculative run
start with cleared storage; the reset run retains the speculative tail
and must reproduce the fresh speculative result exactly.

The [163 host checks](cann-mtp-rounds-cpu-20261017.json) exercise synthetic
v3 file validation and scalar schedules across acceptance modes, EOS,
page boundaries and capacity tails. They run no official model arithmetic
or TP collective and do not import Torch NPU. Real device consistency and
HF parity remain separate acceptance steps.
