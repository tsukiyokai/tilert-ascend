# A3 precision, capture and request timing

These entries use the CANN TP16 decoder and production request/command loop.
[Eighteen server CPU suites](prefix-capture/offline-regression.json) exercise
their control and report handling; [source checks](prefix-capture/source-checks.json)
cover 102 Python files and imports without Torch.
The [TP16 8K HIXL capture](pd-hixl-20261017/prefix-8k-20261017.json)
is complete in `a3-gs:trt-asc`. Decode numerical acceptance and request
TPOT remain pending.

## Capture original vLLM prefixes

Use the [Ascend producer configuration](pd-ascend.md#pd-configuration)
with `parallel_drafting=false` and `--return-tokens-as-token-ids`, and start
this capture receiver when the host is available. It registers HIXL buffers
without loading decode weights:

```bash
source /workspace/environment/activate.sh
cd /workspace/src/TileRT-Ascend
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ACL_OP_INIT_MODE=1 OMP_NUM_THREADS=1 \
  PYTORCH_NPU_ALLOC_CONF=expandable_segments:False \
  python -m torch.distributed.run --standalone --nproc-per-node=16 \
  tests/glm52/capture_pd_prefix.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --hixl-host "${TRT_HIXL_IP:?set the business IP}" \
  --hixl-library /workspace/experiments/hixl-build/build-sync-contract/libtilert_hixl.so \
  --output /workspace/experiments/glm52/prefix-capture
```

In another shell, use the same activation and working directory:

```bash
python tests/glm52/capture_pd_inputs.py \
  --prefill-url http://127.0.0.1:18000 \
  --capture-url http://127.0.0.1:18081 \
  --capture-host "${TRT_HIXL_IP:?set the business IP}" \
  --prompt-ids /workspace/experiments/glm52/prompts-8x8192.json \
  --output /workspace/experiments/glm52/prefix-capture
```

The prompt file accepts one short prompt for initial numerical work or up
to eight distinct prompts of at most 8192 token IDs. Final 8K precision
and timing still require eight 8192-token inputs. The client first verifies
that port 18081 runs `PrefixCaptureEngine`, then sends requests serially.
The Ascend connector carries all original IDs independently of sampling.
The [pinned source audit](prefix-capture/source-audit.json) confirms that
positive fixed K still executes the first MTP pass for `max_tokens=1`,
before the connector finalizes; P+K must fit the drafter's model length.

Each `case-NN/` contains `main-prefix/rank-N.pt`, `mtp-prefix/rank-N.pt`,
`manifest.json` with sixteen file receipts and the prefill first token, and
the client's `prefill-response.json`. `inputs.json` indexes the cases.
All ranks finish copying before the first-token response; the HIXL slot is
retained through reset. Tensors clone into CPU storage before serialization,
preserving INT8/BF16/FP16 contents and excluding padded capacity. Main rows
are captured before prompt replay, so the last prompt row remains original.
Capture output records no decode inference or TPOT.

Stop the capture receiver after the batch completes. Use a fresh output
directory for a restarted capture service; existing case files are retained.
Run the HF/native comparison commands below with each captured main prefix
and `--tokens FIRST_TOKEN_ID --generate 32`. The HF generator derives the
original prompt identity from the saved bundle.

## Inputs and numerical reports

Freeze eight different 8192-token prompts. A main `rank-N.pt` bundle contains
`metadata_sha256`, the full `prompt_token_ids` list and 78 `layers` accepted
by `NativeDecodeEngine.inject`. Every tensor contains 8192 prefix rows.
The HF generator checks that prompt length matches the frozen rows and that
all sixteen files contain the same prompt and replicated cache contents.
It records `prompt_length`, `last_prompt_token` and `prompt_sha256`, defined
as SHA256 of `json.dumps(prompt_token_ids, separators=(',', ':')).encode('ascii')`.

The paired layer-78 bundle contains matching `metadata_sha256` and a `layer`
dictionary with BF16 `latent`, `rope` and `index_keys`. The initial MTP prefix
has 8191 parent rows. All four MTP runners use this bundle format.

For the M4 path, freeze the three MTP inputs from one actual native round:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ACL_OP_INIT_MODE=1 OMP_NUM_THREADS=1 \
  python -m torch.distributed.run --standalone --nproc-per-node=16 \
  tests/glm52/capture_native_mtp_inputs.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --capture-case /workspace/experiments/glm52/prefix-capture/case-00 \
  --output /workspace/experiments/glm52/mtp-inputs/case-00 \
  --source-commit f554fa57d0a69e7a44ac4f3134d0134b4964e848
```

Run this in the activated unified environment after the main numerical
check. The [capture runner](../../../tests/glm52/capture_native_mtp_inputs.py)
replays the last prompt token, checks the prefill first token, and observes
the real seed, first chain and target-conditioned M4 calls. Each of
`seed/`, `chain/` and `verify/` contains `input.pt` and `prefix/rank-N.pt`.
Pass each pair to `run_mtp_reference.py`, then to `run_native_mtp.py` with
that oracle's output as `--reference`; write the resulting native reports
to `mtp_seed`, `mtp_chain` and `mtp_verify` below. The seed prefix preserves
the original file bytes. Capture itself does not establish numerical parity.

Place numerical reports under `precision/case-00` through `case-07`:

| Directory | Required execution |
| --- | --- |
| `main` | TP16 `run_native_decode.py`; 32 teacher rows after the frozen 8K prefix, HF layer/logit/cache parity and reset replay |
| `verify` | `run_native_verify.py` against the same HF manifest; M1/2/4, device indices and distributed greedy selection |
| `mtp_seed` | HF/native MTP comparison at parent position 8191, width 1, target final-normalized parent |
| `mtp_chain` | HF/native MTP comparison at parent position 8192, width 1, MTP recycle parent and external indices |
| `mtp_verify` | HF/native MTP comparison at parent position 8192, selected width 2/4, target final-normalized parents |
| `session` | `run_native_mtp_session.py --ignore-eos`, selected width, at least 16 output tokens and two rounds; serial cache agreement and TP16 reset |

Every directory contains sixteen `rank-N.json` reports; the three MTP
comparison directories also contain the recorded `rank-N.pt` device traces.
M1 requires only `main` and `verify`. MTP reports use the same expert backend
as the selected session. The [precision preflight](../../../tests/glm52/precision_gate.py)
checks current source/checkpoint/package identities, report status, numerical
checks, positions, prefix/trace hashes and the original prompt identity.
The benchmark requires eight distinct case identities. See the
[numerical execution guide](cann-operators.md#correctness-execution) and
[MTP execution order](offline-development.md#device-execution-after-the-environment-is-free).

## Short device capture

After one case passes, run in the unified environment:

```bash
source /workspace/environment/activate.sh
cd /workspace/src/TileRT-Ascend
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ACL_OP_INIT_MODE=1 OMP_NUM_THREADS=1 \
  python -m torch.distributed.run --standalone --nproc-per-node=16 \
  tests/glm52/profile_native_session.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --precision-dir /workspace/experiments/glm52/precision/case-00 \
  --prefix /workspace/experiments/glm52/case-00/main-prefix \
  --mtp-prefix /workspace/experiments/glm52/case-00/mtp-prefix \
  --last-prompt-token "${TRT_LAST_PROMPT_TOKEN:?set the original last token}" \
  --mtp-width 4 --max-tokens 6 --source-commit "$(git rev-parse HEAD)" \
  --output /workspace/experiments/glm52/profile-m4
```

Use `--mtp-width 1` for the first serial capture; `--expert-bank` selects
the independently accepted bank configuration. Experts load before three
unprofiled warmup requests. Each rank starts its profiler in the thread
executing its command loop, then captures one request with at most five
post-first-token steps/rounds. The fixed Level1 CPU/NPU configuration includes
HCCL tracing; MSTX/CPU ranges label prompt replay, seed, chain, verification,
conditioned MTP, commit and delivery. Source identities are pinned to the
installed TorchNPU 2.12.0 profiler implementation.

Outputs are `rank-N.json` and `raw-rank-N/`. Check nonempty framework/device
data and analyze the raw timeline for event coverage, stream dependencies,
host gaps and exposed communication. Capture stops synchronize the device.
The entry defers trace analysis; request TPOT is measured in a separate run.

## Profiler-off PD timing

Start the [Ascend service](pd-ascend.md#pd-configuration)
with the accepted `--mtp-width`/expert backend and
`--timing-dir /workspace/experiments/glm52/pd-timing`. Keep the vLLM prefill
configuration from that guide; add `--return-tokens-as-token-ids` so its
one-token response identifies the exact vocabulary ID.

```bash
python tests/glm52/bench_pd_decode.py \
  --prefill-url http://127.0.0.1:18000 \
  --decode-url http://127.0.0.1:18080 \
  --decode-host "${TRT_HIXL_IP:?set the business IP}" \
  --prompt-ids /workspace/experiments/glm52/prompts-8x8192.json \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --precision-dir /workspace/experiments/glm52/precision \
  --timing-dir /workspace/experiments/glm52/pd-timing \
  --mtp-width 4 --warmup 3 --requests 32 \
  --output /workspace/experiments/glm52/tpot-m4
```

The prompt file contains JSON lists of token IDs from the eight accepted
cases. Requests run serially: vLLM produces one token and hands off KV through
HIXL, then `/pd/decode` delivers 1024 exact token IDs with EOS ignored.
Producer callback completion and client NDJSON arrival are recorded separately.
Tokens delivered in one client read share its timestamp; bursts are retained.
Each request must finish and reset with identical producer/client sequences,
matching runtime configuration and complete M1 steps or MTP accepted counts.

Per-request TPOT is `(t_last-t_first)/(1024-1)`. It includes the initial MTP
seed after the first callback, later rounds, sampling and token delivery.
`summary.json` reports both clocks' request p50/p95, token-weighted speed,
token intervals/bursts, TTFT, round times and actual acceptance. `passes_2ms`
requires at least three completed warmups, 32 measured requests covering all
eight prompts, and both clocks' request p50/p95 at most 2 ms. Shorter or
subset runs remain diagnostics. Full OpenAI router/SSE and AgentX scoring
use their own delivery and workload acceptance.
