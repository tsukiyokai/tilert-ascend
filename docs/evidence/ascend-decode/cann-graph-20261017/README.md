# CANN decode execution

Environment: `a3-gs:trt-asc`, GLM-5.2-W4A8C8, TP16, batch one,
8192 captured prefix rows, capacity 9344, memory utilization 0.85, CANN
kernels and FP32 TP sums. Requests use the production rank command
loop; prefix readiness, reset, HTTP and HIXL are outside internal TPOT.

## CPU thread affinity

[Audited results](affinity.json) record two sequential runs from the same
source snapshot. Each uses one warmup and three 32-step requests.

| Request TPOT (ms) | Original affinity | Bound threads |
| --- | --- | --- |
| 1 | 222.156 | 227.715 |
| 2 | 256.843 | 246.645 |
| 3 | 301.381 | 286.644 |
| Median | 256.843 | 246.645 |

The medians differ by 3.97%; the ranges overlap. These short sequential
runs do not establish a causal speedup. Thread affinity does not explain
the roughly 250 ms eager cost.

The benchmark's `--cpu-affinity` uses the installed
`CpuAlloc.build_cpu_pools()`, `allocate()` and `bind_threads()` after model
loading. Actual per-thread masks match the plan on all sixteen ranks.
It does not call `run_all()`, change IRQ masks or migrate NUMA pages.
All eight 33-token outputs are identical. Rank report hashes, checkpoint,
source, runtime settings and clocks are checked in the receipt.

Raw runs remain on a3-gs under
`/home/shanshan/trt-asc/offline/cann-affinity-20261017/{bound,original}/`.
The source is `12ae8e2` plus the hashed benchmark override in the receipt.

## ACL Graph input updates

The optional `CannModel.enable_graph()` captures M1 or M4 target model
forward with the installed `ACLGraphWrapper`; head and argmax remain
outside. Eager warmup precedes capture. Before each replay, the original
metadata builders update token, position, sequence length, slot mapping
and the persistent RoPE buffers. A separate persistent INT64 slot tensor
is also refreshed: the SFA warmup cache otherwise retains an old cast.
Full indexer layers generate and copy top-k indices within the graph;
shared layers read that graph-updated buffer.

In the installed rope64/BF16 path, SFA and indexer pass device sequence
length tensors to the kernels. Python `max_seq_len` is not an argument of
this branch. The pinned source paths and hashes are retained by each run.

The first wrapper call records a graph and returns uncomputed output;
actual replay is the subsequent branch of `acl_graph.py:294-302`.
The initial probe returned zero hidden/logits and token 0 instead of 382
at position 8192. The adapter now replays before exposing the first output.
The checker also prefills unwritten KV rows with NaN or an INT8 sentinel,
so retained eager rows cannot conceal missing graph writes. Numerical
diagnostics, reset, prefix retention and actual ACL graph entries are
checked before any profiler-off timing request.

## M1 device result

[Independent CPU audit](graph-audit.json) rehashes all 1536 snapshots
(725,712,384 bytes). Graph/eager, reset/eager and reset/graph each compare
100,352 tensors across sixteen ranks; every tensor has identical bits.
These tensors contain final logits, final normalized hidden, token IDs
and all layers' new KV rows. They do not contain every intermediate hidden.
Each rank also executes 579 full-prefix checks across the three phases.

The actual wrapper holds one `UpdatableGraph` with the M1 descriptor and
the recorded token/position addresses. Numerical diagnosis records one
capture and 64 replays; the four subsequent requests add 128 forwards.
Every rank completes command serial 137 and finishes reset at position 0.

| Profiler-off request | Internal TPOT (ms) |
| --- | --- |
| Warmup | 44.8040 |
| Measured 1 | 44.7035 |
| Measured 2 | 44.6755 |
| Measured 3 | 44.6391 |

Median is 44.6755 ms. Each output equals the 33-token eager M1 trajectory;
the [rank-zero receipt](graph-rank-0.json) retains callback clocks. This
compares with 256.8426 ms in the preceding unbound eager short run, about
5.75 times lower. Both are internal 32-step results, without HTTP/HIXL.
They are not the MTP3/graph4 vLLM baseline or 1024-token acceptance.

Run [check_cann_graph.py](../../../../tests/glm52/check_cann_graph.py)
with sixteen ranks, the checkpoint and capture-case paths, `--steps 32`,
`--requests 3 --warmup 1`, and a fresh `--output`. The exact launch and
source snapshot remain on a3-gs at
`/home/shanshan/trt-asc/offline/cann-graph-20261017-02/`.
Its `audit.py` regenerates `graph-audit.json` on CPU. The prior failed
capture and zero-output snapshots remain in sibling `cann-graph-20261017-01`.

The M4 target graph results follow below. The MTP draft layer remains
eager. HF remains the independent numerical diagnostic; the eight-input
frozen teacher and 1024-token release checks are separate.

## M4 serial and speculative execution

The [MTP audit](mtp-audit.json) covers the same full-P 8K case, real target
and MTP weights, width four and sixteen output tokens. All sixteen ranks
produce the same tokens in serial, speculative and dirty-reset runs.
The audit rehashes 48 snapshots (70,818,384 bytes). Serial/speculative and
speculative/reset each compare 3136 tensors containing 47,088 new cache
rows; all bits agree. Full-prefix retention is checked on the device;
the CPU snapshots contain only new rows.

The eight rounds have natural acceptance counts `[3,1,1,3,1,2,2,3]`, mean
2.0. The output budget trims the final count from three to two, so the
actual fifteen committed rows average 1.875 per round. Including prefill's
first token, sixteen tokens are emitted; main and MTP lengths finish at
8207, with the last token pending. This short trace does not establish
the acceptance distribution for 1024 outputs.

This run uses eager M4, without timing. At a sustained acceptance of two,
the 2 ms/token goal permits only 4 ms per round.

Run [run_cann_mtp_rounds.py](../../../../tests/glm52/run_cann_mtp_rounds.py)
with sixteen ranks, the checkpoint and capture-case paths, `--width 4`,
`--max-tokens 16 --capacity 9344 --ignore-eos`, memory utilization 0.85
and a fresh output directory. Launch, frozen source, rank reports,
snapshots and `audit.py` remain on a3-gs under
`/home/shanshan/trt-asc/offline/cann-mtp-rounds-20261017-01/`.

## M4 target graph and session timing

The [target graph audit](mtp-target-graph-audit.json) covers sixteen
outputs on all sixteen ranks. Serial, eager speculative, target graph
and graph dirty-reset runs produce identical tokens. Across 64 snapshots
(94,436,800 bytes), each adjacent comparison checks 3136 new-KV tensors
and 47,088 rows; every bit agrees. Device checks also preserve the full
prefix and parent hidden. Each rank records one M4 graph capture and
sixteen replays. These snapshots cover new KV and request state;
intermediate layer activations are outside this comparison.

The [session benchmark audit](mtp-bench-audit.json) checks the real
rank command loop, seed, chained drafts, target verification,
target-conditioned MTP recomputation, acceptance and token delivery.
Only the target uses an ACL graph; draft execution stays eager. Each run
has one warmup request, followed by the measured requests below.

| Target execution | Requests | Median TPOT (ms) | P95 TPOT (ms) |
| --- | --- | --- | --- |
| Eager | 3 | 152.461543 | 155.3382743 |
| M4 graph | 30 | 38.7026239 | 39.0243213 |

Each request emits 32 tokens including prefill's first token. Its 17
rounds naturally accept 33 tokens; the output budget commits 31, giving
actual acceptance `31/17 = 1.8235` versus natural `33/17 = 1.9412`.
Main and MTP caches end at 8223, with the last output pending. All requests
have the same output sequence. Graph round wall time has median
69.86422 ms across 510 measured rounds, including Gloo dispatch; the first
round also includes the seed. Target/draft kernel time is not separated.

Internal TPOT is `(last_callback - first_callback) / 31`, counting token
IDs within bursts. Loading, prefix readiness, HTTP, HIXL, reset and finish
after the last callback are excluded. The sequential 3-request/30-request
runs give a median ratio of 3.9393; this is a short-session measurement,
not the 1024-token acceptance workload.

Run [bench_cann_mtp.py](../../../../tests/glm52/bench_cann_mtp.py) with
`--max-tokens 32 --warmup 1`, and `--target-graph --requests 30` for the
graph case. Raw runs, launch commands and frozen source remain on a3-gs
under `/home/shanshan/trt-asc/offline/`:

- `cann-mtp-target-graph-20261017-01/`: numerical snapshots and audit.
- `cann-mtp-eager-bench-20261017-01/`: three measured eager requests.
- `cann-mtp-target-graph-bench-20261017-01/`: thirty graph requests and audit.

Both benchmark runs use source `d11505bf6769990aa1b0a870bc5ca3bbea72e2c7`
with the recorded overrides: graph `a1b6ab356c20`, model `c3c8294b275d`,
and benchmark `d38609fbd685`. The receipt preserves their full hashes,
checkpoint metadata, input identities and all 32 rank-report hashes.
The copied benchmark audit SHA256 is
`384313a9e37b4187d0b7d2e5eb3b0510e6c88f7dd9206a315a82313c91e4cd55`.

## M4 1024-output diagnostic

The [long-session audit](mtp-long-bench-audit.json) uses the same frozen
source, checkpoint, full 8K prefix and runtime configuration as the
32-output graph benchmark. Three warmups precede three measured requests;
all six produce the same 1024 tokens. All sixteen ranks agree on command
history, acceptance and reset. Main and MTP positions finish at 9215 and
reset to zero; one target graph is captured and replayed 1818 times.

| Measured request | Internal TPOT (ms) | Rounds | Actual tokens/round |
| --- | --- | --- | --- |
| 1 | 20.441157 | 303 | 3.376238 |
| 2 | 20.422166 | 303 | 3.376238 |
| 3 | 20.957581 | 303 | 3.376238 |

Median TPOT is 20.441157 ms, mean 20.606968 ms. Each request commits
1023 tokens in 303 rounds; natural acceptance is 1025/303 = 3.382838,
with the last round trimmed by two tokens. Backend round wall means,
including Gloo, are 68.997/68.933/70.741 ms. The first 32 tokens and first
17 proposal sequences match the earlier short request, whose lower
acceptance was 31/17. The TPOT change from
38.70 to 20.44 ms comes from the measured trajectory length, with no
kernel change. It is not an optimization speedup.

The original vLLM baseline has median client TPOT 16.390 ms over 32
1024-output requests. The observed median ratio is now 1.247, rather
than the 2.361 ratio obtained from the short TileRT request. The
[baseline log extraction](../cann-mtp-profile-20261017/baseline-comparison.json)
records backbone compilation, ACL replay and aggregate natural acceptance
3.268 over 69 windows including warmup. The windows cannot be assigned to
individual requests. Its compiled model, AIV/native reduction, scheduler,
custom kernels and generated trajectories differ from this run; the ratio
does not isolate any of those costs. Both launch configurations set draft
`enforce_eager=true`; the baseline also logs eagle-head compilation.

The internal callback formula is `(last - first) / 1023`; HTTP/HIXL and
reset remain excluded. This three-request diagnostic establishes neither
teacher parity nor the 32-request performance acceptance. Raw reports,
launch, source and the CPU audit remain on a3-gs under
`/home/shanshan/trt-asc/offline/cann-mtp-target-graph-long-bench-20261017-01/`.

## AI_CPU versus AIV control

The [paired audit](mtp-aiv-bench-audit.json) changes only
`HCCL_OP_EXPANSION_MODE=AI_CPU` to `AIV`. Source, input, FP32 TP sums,
custom kernels, graph and request lengths stay fixed. All sixteen ranks'
six 1024-token trajectories, proposals, predictions, acceptance and cache
positions agree exactly between runs. Each run has three warmup and
three measured requests.

| Profiler-off metric | AI_CPU | AIV |
| --- | --- | --- |
| Median internal TPOT (ms) | 20.441157 | 21.473259 |
| Mean internal TPOT (ms) | 20.606968 | 21.316405 |
| Backend round wall median (ms) | 69.518428 | 72.029132 |
| Actual tokens/round | 3.376238 | 3.376238 |

AIV does not improve this session; its measured median is 5.05% higher.
This sequential three-request control does not measure bare collective
latency or vLLM's native BF16 reduction. Retain AI_CPU for the next
compiled-target comparison. Raw AIV reports, launch and `paired-audit.py`
remain on a3-gs under
`/home/shanshan/trt-asc/offline/cann-mtp-long-aiv-20261017-01/`.
