# GLM-5.2 vLLM P+D baseline

Measured 2026-10-17 on `a3-gs:trt-asc`: one A3 host, 16 dies,
GLM-5.2-W4A8C8, TP16/DP1, batch one. vLLM performs both prefill and decode.
MTP uses three drafts; ACL Graph captures four target verification rows.

| Statistic | Result |
| --- | --- |
| TPOT p50 / p95 | 16.390 / 18.295 ms |
| TPOT mean / p99 | 16.514 / 18.542 ms |
| Token-weighted decode rate | 60.555 tok/s |
| Measured requests / warmups | 32 / 3 |

Each serial request consumes the same 8192 input IDs and returns exactly
1024 output IDs. TPOT is the interval between the first and last streamed
token arrivals divided by 1023; prefill and final usage/DONE events are
outside that interval. MTP events retain their complete token-ID spans.
The [raw requests](benchmark.json) contain IDs, text, arrival timestamps and
usage. The [result](result.json), [command](command.txt),
[runtime](environment.json), [startup](startup-summary.txt) and
[occupancy snapshots](occupancy.before.txt) identify the measurement.

The runtime is Torch/torch-npu 2.12.0, Transformers 5.14.1 and CANN 9.2.0
master20260917. vLLM is `c4df37df`; vLLM-Ascend is `f558b58c`.
[Build patches](source-patches.json) adjust dependencies and custom-op
compilation. `TORCHINDUCTOR_AUTOGRAD_CACHE=0` preserves npugraph compilation
and ACL Graph replay while avoiding the AOT OutputCode cache assertion.
The [32 recorded compiler entries](compiler-options.json) all enable
npugraph_ex and disable static kernels and SuperKernel.

Output sequences were not reproducible in this MTP3/graph4 service run:
all 32 saved 1024-token sequences are unique despite `temperature=0` and
`seed=0`. The [client](../../../../tests/glm52/bench_vllm_decode.py), whose
source hash matches `benchmark.json`, reuses one 8192-ID prompt and checks
the returned prompt IDs on every response. All responses report 8192
prompt tokens, 1024 completion tokens and `finish_reason="length"`.

All sequences start with `[6500, 382, 262, 576]`. Against the first saved
request, `cmpl-9aa86ab6d6f99327`, the other 31 sequences first differ at
these one-based output positions (position: request count):
`5:17, 8:1, 12:3, 21:1, 33:3, 74:2, 82:1, 84:2, 110:1`.
[teacher-32.json](teacher-32.json) freezes that request's first 32 tokens
and their next-token sequence as a trajectory anchor for teacher forcing.
HF remains the primary independent numerical diagnostic; this frozen
trajectory supplies fixed teacher inputs and greedy successor labels for
end-to-end acceptance. A new request to the running service does not
supply a reproducible token oracle or intermediate layer reference.

This repeatability result applies to the recorded MTP3/graph4 service
configuration. Fixed eager M1/FP32 comparisons establish only their own
recorded trajectory. TileRT did not execute these P+D requests, so their
output divergence cannot be attributed to TileRT.

At p50, reaching 2 ms requires an 8.20-fold reduction from this baseline,
which already includes MTP and decode graph replay.
