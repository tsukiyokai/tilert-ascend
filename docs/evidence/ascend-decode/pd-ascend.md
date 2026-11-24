# Ascend prefill/decode handoff

This path connects a TP16 vLLM-Ascend producer to rank-local CANN decode
engines through HIXL. It serves one fresh greedy request at a time, with the
whole prompt scheduled in one step.
The [CANN model entry](../../../tests/glm52/serve_cann_pd.py) supports
serial M=1 and MTP width 2/4 with full-P injection. A
[live M1 check](pd-live-20261017/README.md) completes an 8K prompt and eight
HTTP output tokens. M4 target graph and eager draft have separate
[offline device comparisons](cann-graph-20261017/README.md).

## Ownership and completion

- Each decode rank owns one registered receive allocation described by
  [layout v2](../../../tilert/pd_vllm/ascend_layout.py): 196 regions,
  including 79 MLA latent/RoPE pairs, 22 index keys and 16 FP16 scales.
- A [central receiver](../../../tilert/pd_vllm/ascend_receiver.py) publishes
  sixteen rank-specific HIXL endpoints. Source rank r writes destination r;
  on this single host, destination rank r uses die `(r+1)%16` to avoid
  HIXL self-device connections. TP shard ownership remains logical rank r.
  Equal virtual addresses in different processes are separate allocations.
  Each endpoint port also selects that instance's internal HCCL listener;
  P and D ports must be distinct on every shared die.
- The [V1 connector](../../../tilert/pd_vllm/ascend_connector.py) binds the
  actual cache views and scheduler block tables. Its `wait_for_save` returns
  after producer synchronization, synchronous WRITE and the matching
  request/rank/generation acknowledgement. Scheduler blocks remain live
  throughout the read. Metadata hashes identify the checkpoint recipe and
  tensor-to-file manifest on both sides.
- Main caches transfer P rows; MTP transfers P-1 target-conditioned parent
  rows. The receiver publishes ready only after all sixteen ranks finish
  the same generation. The slot remains occupied throughout decode.
- [CannModel injection](../../../tilert/models/glm_5_2_ascend/cann_model.py)
  keeps all P main rows. Its next input is the prefill first token at
  position P. The older Native rank engine injects P-1 rows and replays the
  final prompt token, then checks its prediction against prefill.
- [Optional layout v3](mtp-parent-handoff.md) adds the unrotated target
  final-norm hidden[P-1], BF16 `[1,6144]`, to the same completed WRITE.
  CANN MTP consumes it for one-row seeding. Online MTP continuation still
  needs its own live check.
- Cancellation retains admitted writers until their WRITE completions.
  A disconnect without completion quarantines the slot. Stop both sides
  before replacing registered allocations or restarting an endpoint.
  A cancelled prefill also retires its slot when no HTTP decode was started;
  a later HTTP announcement cannot revive that generation.

With vLLM `c4df37df` and vLLM-Ascend `f558b58c`, `max_tokens=1` still runs
the first MTP forward. The scheduler passes the fixed draft count without
capping it to the request's output budget (`scheduler.py:1446–1451`). The
Ascend runner proposes after sampling, provided prompt length plus the draft
count fits the drafter (`model_runner_v1.py:2779–2791,2981–2999`). Its
`llm_base_proposer.py:1491` calls the draft model before the one-draft early
return at line 1634. The first pass shifts input IDs while retaining parent
positions (`llm_base_proposer.py:1793–1806`). Connector finalization then
calls `wait_for_save` (`kv_connector_model_runner_mixin.py:53–61`); the
scheduler checks the output limit only after receiving model output
(`scheduler.py:2443–2448`). The connector requires all 79 MLA save callbacks.
This excludes a skipped MTP pass, prefix hits and partial prompts.

## MTP delivery and cache retention

`--mtp-width 2` or `4` selects
[AscendMtpPDEngine](../../../tilert/models/glm_5_2_ascend/mtp_session.py).
The decode side emits the prefill token once; it counts toward `max_tokens`.
Only requests needing a successor start MTP. Every rank completes a whole
propose/verify/recompute/commit round before rank zero delivers its accepted
tokens. Cancellation is checked between deliveries and before the next round.

For prompt length P and N delivered tokens, normal completion and
cancellation retain main length `P + max(0,N-1)`. Discovering an EOS retains
`P+N`; that EOS is not delivered. Cancellation during an accepted span
discards computed but undelivered rows on every rank. Draft length is capped
by main length; a request cancelled before the first round still has P-1
draft rows. A callback that raises has not completed delivery of that token.

`last_stats` separates completed rounds, verified positions, logical accepted
tokens and delivered `output_tokens`. Logical acceptance includes EOS and
tails withheld after cancellation; it is not response usage. Each span also
records its delivered count. The final emitted token stays pending outside
main KV unless it was consumed to discover EOS.

The TP commands are inject, start, advance, finish and reset, ordered by
serial, request ID and main position. All ranks create a separate CPU Gloo
group before HTTP starts; its command receive timeout is 24 hours.
Workers wait there while rank zero handles HTTP. Model computation and
cache receipts use the original HCCL group. A command failure after broadcast can
leave ranks at different serials. The backend then rejects further commands
locally; HTTP cleanup retains the slot and lock. Restart all sixteen ranks
instead of broadcasting a recovery reset.

## PD configuration

Use the existing `/opt/trt-asc/venv` and CANN 9.2.0 in `trt-asc`. Start the
decode service with sixteen ranks after device precision passes and the
host is free:

```bash
source /workspace/environment/activate.sh
cd /workspace/src/TileRT-Ascend
PYTHONPATH="$PWD:${PYTHONPATH:-}" TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  PYTORCH_NPU_ALLOC_CONF=expandable_segments:False \
  HCCL_OP_EXPANSION_MODE=AI_CPU HCCL_BUFFSIZE=200 \
  ACL_OP_INIT_MODE=1 OMP_NUM_THREADS=1 HCCL_NPU_SOCKET_PORT_RANGE=auto \
  /opt/trt-asc/venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=16 tests/glm52/serve_cann_pd.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --hixl-host "${TRT_HIXL_IP:?set the business IP}" \
  --hixl-library /workspace/experiments/hixl-build/build-sync-contract/libtilert_hixl.so \
  --capacity 9344 --gpu-memory-utilization 0.4 \
  --ctrl-port 5556 --http-port 18080 \
  --output /workspace/experiments/glm52/pd-serving
```

This entry loads the pinned vLLM-Ascend model and CANN weights, registers
one HIXL endpoint per rank at ports 26100–26115, and gathers their
descriptors before starting HTTP. For single-machine sequential execution,
add `--load-model-after PATH` with an absent file. This M1 mode registers
the receive allocation first and loads weights inside the pending inject
after the file exists. Complete all sixteen WRITEs, stop P and verify its
exit, then create the file; received KV stays on device. The measured
co-resident startup fails during P warmup, so the
[live check](pd-live-20261017/README.md) uses this sequence. The
[Native launcher](../../../tilert/pd_vllm/ascend_server.py) retains
`--mtp-width 2/4`, `--timing-dir` and `--expert-bank` for its separate
numerical and performance experiments.

Set the vLLM producer to TP16, PP/DP/PCP/DCP=1, expert parallel off,
`max_num_seqs=1`, prefix caching off, block size 128,
and a fixed positive MTP draft count with `parallel_drafting=false`.
Prompt length plus that count must fit
the draft model length; the scheduler token budget must fit the whole
prompt. Chunking may be enabled, but the connector requires the actual
scheduled token count to equal the complete fresh prompt and rejects a
partial prefill. Use the same local ModelSlim checkpoint and mixed C8 index
recipe. The [a3-gs producer launcher](../../../tests/glm52/serve_vllm_prefill.sh)
uses an 8192-token budget, MTP1, eager execution and synchronous scheduling;
its default destination is the capture receiver on control port 5557.
Set `TRT_HIXL_IP` to the prefill host's business IP. `TRT_DECODE_HOST`
defaults to that address; set `TRT_CTRL_PORT=5556` for the decode service.
Build the current HIXL bridge and set `TRT_HIXL_LIBRARY` to its output;
pass the same library to the receiver's `--hixl-library`. The
[co-resident transport check](pd-hixl-20261017/hixl-listen-port-20261017.json)
verifies the listener isolation required by shifted P/D placement.
Use `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False` on both sides.
The transport resolves each Torch storage to its allocator block for IPC
registration and retains the original logical ranges for metadata and
WRITE validation. Bound tensors stay alive until transport close. The
[suballocation check](pd-hixl-20261017/hixl-allocation-20261017.json)
verifies block registration, deduplication and logical bounds.
The [launcher argument checks](pd-hixl-20261017/prefill-launcher.json)
exercise both destinations and a changed host without starting vLLM.
The [prefix capture receiver](measurement.md#capture-original-vllm-prefixes)
uses the same producer configuration for frozen numerical inputs, before
the decode service is started.
The V1 connector configuration is:

```json
{
  "kv_connector": "AscendTileRTConnector",
  "kv_connector_module_path": "tilert.pd_vllm.ascend_connector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "tilert_host": "DECODE_BUSINESS_IP",
    "tilert_ctrl_port": 5556,
    "tilert_max_seq_len": 9344,
    "hixl_host": "PREFILL_BUSINESS_IP",
    "hixl_port_base": 26000,
    "hixl_library": "/workspace/experiments/hixl-build/build-sync-contract/libtilert_hixl.so"
  }
}
```

The existing router can use the decode HTTP endpoint; requests explicitly
use `temperature=0`. Prefill produces the first token. CANN M1 emits that
token and consumes it at P only when a successor is needed. The final
emitted token stays outside main KV.
During measured decode, the prefill service must be quiescent.

## CPU evidence and remaining device work

[Connector checks](check-ascend-connector.json) exercise actual local TCP
admission/completion on sixteen ranks with synthetic source addresses and
fake HIXL WRITE. They cover P=1/128/129/8192, mixed C8 scales, distinct
cache-group block tables, page strides and producer block lifetime.
[Control checks](ascend-pd-control.json) cover request ownership, cancellation,
generation checks, prompt replay and individual token wakeups.
[MTP TP checks](mtp-serving/check-mtp-tp.json) and
[MTP session checks](mtp-serving/mtp-session.json)
exercise the real command executors, rank engines and round controller with
independent scalar caches. They compare accepted/rejected spans, EOS,
capacity tails, callback cancellation and repeated requests with serial AR.
[Runner metadata checks](mtp-serving/native-runner.json) reject missing
layers/fields and corrupted prompt/new rows, and verify 322 prefix/new
comparison scopes, including INT8 keys and FP16 scales. Tensor arithmetic is
replaced in this metadata check.

Streaming uses [TokenQueue](../../../tilert/pd_vllm/token_queue.py), waking
the response on every token instead of the former 200-ms poll. A decoder
thread still alive after a cancellation join timeout retains its request
slot and HTTP lock until it terminates; a failed reset retains ownership.

The scalar checks execute on the a3-gs host CPU without Torch. The
[HIXL device checks](pd-hixl-20261017/README.md) verify a 1 KiB WRITE from
die 0 to an independent process on die 1, with exact receiver readback.
The same-die control fails during Connect with status 103900. A subsequent
[live TP16 8K HIXL capture](pd-hixl-20261017/README.md) completes all sixteen
rank transfers and freezes 32 main/MTP files. Separate CANN M1 runs use
those files for NPU continuation and internal timing. Live HIXL handoff
followed immediately by online decode, HTTP delivery timing and final
numerical acceptance remain to be verified. Follow the
precision→profiling→calibration order in
[the development record](offline-development.md).

After main and MTP HF parity, run
[run_native_mtp_session.py](../../../tests/glm52/run_native_mtp_session.py)
on sixteen ranks. It uses the production serving factory, CPU command
channel and rank engine; compares normal/cancelled/repeated requests with
serial CANN AR plus all valid target-conditioned MTP cache rows; and checks
every rank before reset. Frozen prefix rows compare exactly; replay/new rows
compare separately to preserve their numerical error scale. Its CLI matches
`run_native_mtp_rounds.py`. A cold prompt is
given with `--tokens`; an 8K case uses paired `--prefix`/`--mtp-prefix` and
the last prompt token. This runner checks device execution consistency;
HIXL and HTTP delivery timing are separate acceptance measurements.
