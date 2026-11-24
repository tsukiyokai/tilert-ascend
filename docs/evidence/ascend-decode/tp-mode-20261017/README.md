# A3 small-message HCCL modes

Environment: `a3-gs:trt-asc`, TP16, CANN 9.2.0, Torch/Torch-NPU 2.12.0,
`HCCL_DETERMINISTIC=true`, `HCCL_BUFFSIZE=200` and task queue 1.
Four sequential runs compare AI_CPU/AIV with Python submission and ACL
graph replay. Each dtype/size/call-count case has ten warmups and thirty
samples; clocks remain at stock settings. The [audit](audit.json) checks
all 64 rank reports and recomputes each sample's rank maximum before
percentiles. Loading, correctness and Gloo barriers are outside timing.

## Profiler-off graph results

FP32, 156 dependent in-place SUMs on one buffer. Values below are the
rank-MAX event p50 for the complete chain, in milliseconds.

| Payload | AI_CPU | AIV |
| --- | --- | --- |
| 24 KiB | 10.92500 | 4.97617 |
| 96 KiB | 10.93072 | 5.04398 |

For 96 KiB, wall p50 is 11.13768/5.20180 ms. AIV reduces the event chain
time by 53.85%. This supports the small-message expectation on this host;
the totals include graph replay and dependency scheduling, not bare
collective latency. Python-submitted 96 KiB chains give event p50
25.04660/22.47804 ms but wall p50 26.97140/27.97542 ms. Event intervals
and end-to-end wall intervals must remain separate.

Each rank separately checks `sum(rank+1)=136` and zero-buffer retention.
The zero chain avoids overflow across repeated in-place sums. This
validates the benchmark's operations, not full-model numerical accuracy.

## Actual execution

After timing, each graph run profiles two 96 KiB chain replays. The
[trace extraction](trace-summary.json) copies and exports rank zero's
raw capture, retaining hashes. It finds 312 `allreduceAicpuKernel` tasks
for AI_CPU and 312 `AivKernel` tasks with block count 16 for AIV; neither
run contains the other's kernel. Both log `RING-RING-NHR`, so that
algorithm string alone does not identify the execution engine.

The AIV communication parser reports zero transit/wait and puts elapsed
time under idle; those fields cannot establish zero transfer or waiting.
Profiler durations do not replace the table above.

The existing [1024-output session](../cann-graph-20261017/README.md#ai_cpu-versus-aiv-control)
instead has median TPOT 20.44/21.47 ms. Its result does not negate the
communication-only gain. Diagnose the real decode trace before assigning
the difference to resource contention, arrival skew or another stage.

Run `bench_tp_sum.py` and `bench_tp_sum_graph.py` with each explicit mode.
The graph script's optional `--profile-output` captures only the separate
post-timing replays. Raw runs, launch, source, `audit.py` and CPU export
scripts remain on a3-gs under
`/home/shanshan/trt-asc/offline/tp-mode-control-20261017-02/`.
Source is `aa547f1` plus the two benchmark overrides in the audit.
The first attempt, `tp-mode-control-20261017-01`, lacks framework markers
in its diagnostic capture and is excluded from these results.
