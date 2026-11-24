# CANN M1 execution and latency

Environment: `a3-gs:trt-asc`, TP16, GLM-5.2-W4A8C8, batch one,
8192 captured prefix rows, eager M1, FP32 TP sums. CANN 9.2.0 and the
existing Div/SwiGLU vendor fixes are retained. HTTP and HIXL are outside
these measurements; the input comes from the completed
[live HIXL prefix capture](../pd-hixl-20261017/README.md).

## Same-configuration execution

The [native comparison](../cann-vllm-32-20261017.json) runs the installed
vLLM scheduler, worker, model runner and greedy sampler, then TileRT with
the same loaded weights and separate caches. All sixteen ranks execute
32 steps and a reset replay. Independent adjacent-float checks find all
344,064 native-comparison floating tensors and all 172,032 reset-comparison
floating tensors bit-exact. Choices, new cache rows and prefix retention
also pass. The [two-step receipt](../cann-vllm-two-20261017.json) records the
preceding short run; the [current comparator receipt](same-config-helper.json)
checks the committed helper against an actual saved forward.

These checks establish execution agreement for the same configuration.
They do not replace the independent HF diagnostics or the frozen
teacher-32 end-to-end acceptance. The native M1 trajectory first differs
from teacher-32 at row 19, position 8211: both consume token 323, but M1
predicts 279 and the frozen label is 64019. Subsequent free-running rows
have different histories and are not a teacher-forced accuracy score.

## Profiler-off short measurement

[Rank-zero callback clocks](short-bench-rank-0.json),
[all-rank identities](short-bench-summary.json) and the
[independent audit](short-bench-audit.json) bind the measured outputs,
source hashes, command, checkpoint and completed reset on all sixteen ranks.

| Request | Decode steps | Internal TPOT (ms) |
| --- | --- | --- |
| Warmup | 32 | 253.9804 |
| Measured 1 | 32 | 230.7091 |
| Measured 2 | 32 | 260.7323 |
| Measured 3 | 32 | 289.6761 |

Median is 260.7323 ms; mean is 260.3725 ms. Each request delivers the
prefill token followed by 32 decode tokens. Timing is
`(last_callback_ns-first_callback_ns)/32`; prefix copies, their NPU/Gloo
readiness barrier, reset and model loading are outside that window. All
four 33-token outputs equal the native M1 trajectory. No profiler or
forward-capture hooks are active. This is a short internal session result,
not the 1024-token HTTP or AgentX acceptance result.

Entry: [bench_cann_decode.py](../../../../tests/glm52/bench_cann_decode.py).
The exact command is in rank zero and the audit's `launch.command`.
The source snapshot is based on `44bf12c` plus the four hashed overrides
in `short-bench-audit.json`. Preserve the activated CANN `PYTHONPATH` when
prepending the snapshot. Capacity is 9344, with 73 cache pages and
`gpu_memory_utilization=0.85`.

## Two-step device profile

[Profile summary](profile-summary.json) covers all sixteen ranks after
three warmup requests. Each of the two steps contains 2,681 kernel rows,
78 MLA prologs, 78 sparse attention calls, 157 all-reduces and one
all-gather. Every owned flow closes, no kernel is unassigned to a step,
and device tails finish before capture completion and profiler disable.
These checks cover the observed RECORD-stop warning for this capture.

| Interval union or span (ms) | Position 8192 | Position 8193 |
| --- | --- | --- |
| Rank 0 linked device span | 267.906 | 264.333 |
| Rank 0 compute union | 36.886 | 36.035 |
| Rank 0 collective union | 215.240 | 214.455 |
| Rank 14 collective union | 23.297 | 20.394 |
| Rank 14 gap inside device span | 185.144 | 184.008 |

Rank 14 is the last host collective submitter in 140/158 and 158/158
ordinals. Same-ordinal device-start skew has medians 1.531 and 1.476 ms.
Arrival skew and submission scheduling are the first optimization
hypothesis; the trace does not establish their cause.

The collective envelope includes waiting. `hccl_wait_sum_ms` comes from
the CANN `communication.json` parser: master-plane union of `Notify_Wait`
minus overlap with transit, with synchronization included in wait.
It is not `kernel_details.csv`'s `Wait Time(us)`, and neither collective
duration nor transit slices are standalone network latency. Source paths,
hashes and the parser definition are in `profile-summary.json`.

The raw profile and analysis remain on a3-gs under
`/home/shanshan/trt-asc/offline/cann-native-vllm-20261017-03/`.
Run its `profile-analysis-01/analyze.py` and `summarize.py` to regenerate
the summary. The profiler-off measurement remains under
`/home/shanshan/trt-asc/offline/cann-short-bench-20261017-02/`, including
the audit script and all rank reports.

The profile uses logical capacity 8320, 66 physical cache pages, a
65-column block table and memory utilization 0.4. The short benchmark uses
the 9344 configuration above and adds a prefix-readiness barrier. Their
similar scale does not make them a controlled profiler-overhead comparison.
Use the fixed short-benchmark configuration for the next affinity/ACL Graph
comparison, retaining the same kernels, FP32 sums and M1 trajectory.
