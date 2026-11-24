# The composed optimisation set on an idle machine

`a3-84`, whole machine idle, TP16, source `04a6628`, the same frozen 8K
prefix and checkpoint for every case. `--max-tokens 1024 --capacity 9344
--warmup 3 --requests 3`, profiler off. Cases run back to back after a wait
for the driver to reclaim HBM.

| case | flags beyond `--target-graph` | TPOT (ms) | tok/s | rounds | A | round (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| default | — | 21.210 | 47.1 | 303 | 3.376 | 71.61 |
| no-compile | `--draft-graph --draft-projection-tp --attention-kernel gather_fia --fp32-head` | 19.496 | 51.3 | 319 | 3.207 | 62.52 |
| composed | the above plus `--compile-target --super-kernel` | 16.640 | 60.1 | 298 | 3.433 | 57.12 |

Medians of the three measured requests. The three requests of a case agree
on round count, so each trajectory is deterministic, but the spread is not
uniformly small: composed is tight at 16.637, 16.640, 16.704 and no-compile
at 19.484, 19.496, 19.575, while the default case runs 20.491, 21.210,
22.461 — rising monotonically over a 1.970 ms range, 9.3% of its median and
the same order as the 2.856 ms attributed to the compiled pair. The default
case also carries a `created_at` 79 minutes before the other two, so "back
to back" describes only the latter pair.
[default](idle-default-rank0.json), [no-compile](idle-no-compile-rank0.json),
[composed](idle-composed-all-rank0.json); token dumps and per-round records
were dropped from the archived reports.

The composed set is 21.5% below the default and the round falls 20.2%, from
71.61 to 57.12 ms. The compiled target with SuperKernel accounts for
19.496 → 16.640, that is 2.856 ms or 14.6%, and this is the first measured
latency gain for SuperKernel: the earlier screen could only report a task
count falling from 44 to 34 and stated that it did not establish a latency
gain. Against the vLLM-Ascend baseline p50 of 16.390 ms this is a ratio of
1.015, where `--target-graph` alone was 1.247 — but these are different
scopes and parity is not established: 16.390 ms is a client p50 over 32
requests, while 16.640 ms is this bench's internal delivery with HTTP and
HIXL excluded. Note also that the `a3-gs` 20.441 ms reference set
`HCCL_OP_EXPANSION_MODE=AI_CPU` explicitly while these runs record it as
`null`, and 21.210 ms sits 1.2% from the `a3-gs` AIV comparison of
21.473 ms, so machine identity and expansion mode are not separated here.
The round column is TPOT × A, a derived quantity; of the 4.570 ms between
default and composed, 0.278 ms comes from acceptance rising 3.376 → 3.433
and 4.292 ms from the round.

Distance to the target is now 8.32×, from 10.61× at the default.

## The compiled target needed a cache workaround

`--compile-target` failed on this stack before this run: `AOTAutogradCache`
asserts its entry is an `OutputCode`, while the npugraph FX compiler returns
`npugraph_ex.npu_fx_compiler._CompiledFxGraph`, so `aot_stage2_inference`
raised `AssertionError: expected OutputCode, got <class
'...._CompiledFxGraph'>` during model construction. Setting
`TORCHINDUCTOR_AUTOGRAD_CACHE=0` bypasses `_cache_inference_info` and the
compiled path runs. Every compiled measurement needs that variable; without
it `--compile-target` and therefore `--super-kernel` are unreachable.

## What this does not establish

`precision_verified` is false in all three cases, and acceptance moves with
the configuration (3.376, 3.207, 3.433), so the numerics changed: `--fp32-head`,
`--draft-projection-tp` and `gather_fia` each alter rounding, the draft
proposals differ, and the delivered trajectories are not the same tokens.
TPOT remains comparable because it is time per delivered token over a fixed
1024 outputs, but the composed configuration still owes the frozen-teacher
and eight-input acceptance of the plan's section 4.
