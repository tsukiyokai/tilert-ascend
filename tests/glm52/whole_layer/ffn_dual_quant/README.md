# Shared input-quantization prefix

This candidate passed the numerical gates but regressed in matched full-FFN timing. Retain the Down-N512 parent `c90b3666a1697070eda19d5b3e133e9ed58f36fe429abefceae6ffe102a6a032`. The candidate loads each normalized BF16 row once, widens it once, and computes one absolute maximum. Routed quantization retains its division by `amax / 127`; shared quantization retains its multiplication by `127 / amax`, zero guard, clamps, casts, and reciprocal-multiply scale. Neither codes nor scales are shared between the two branches. [The design](DESIGN.md) records the exact source and numerical contracts.

The public ABI remains 33 scratch tensors, a 408-byte body descriptor, and a 112-byte shared-input descriptor. Five body joins, all norm and bridge joins, and terminal publication remain. Additional GM and UB storage are zero. Quantization uses 92,192 bytes of UB; the routed body uses 116,480 bytes. The complete FFN peak remains the norm phase's 195,584 bytes.

The main library and direct quantization probe compiled with CANN 9.2 and co-loaded under Torch/TorchNPU 2.12 without initializing NPU. Five source/ABI contracts and three probe-fixture contracts passed on x86 and ARM. The 103 main source files match the reviewed source snapshot.

On a3-194 die 4, the direct probe passed M1/2/12/32 with 16/24 mixed groups: 192 cases, 16 captures, 384 changing graph replays, and 3,456 raw-byte checks. This includes all four immediate consumers and the nonfinite/signed-zero fixtures. [The measured report](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-dual-quant-probe-v1-a3-194/summary.json) has wide-report SHA `50436e46100c8f41926ff303783576a6271229ac6c8f1c616e36f5878517d813`.

The original DecoderLayer norm/MLP comparison also passed for two weight sets, M1/2/12/32, 24 groups, and eight changing inputs: 64 cases and 704 raw-byte checks, including 256 immediate-consumer checks. All 72 prepared/replayed Down-tiling checks matched the full 200-byte N512 oracle; the four original NZ allocations and their bytes stayed unchanged. [This separate full-FFN report](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-ffn-dual-quant-wide-v1-a3-194/summary.json) has SHA `1e9fe64cca0a581045b3961dd24b71f03f0fabebbf505861fe879228a9c94300`. TP reductions are outside this gate.

At M12, 24 groups and 65 active experts, paired timing used all 33 common tensor addresses, 16 weight banks, 16 calls per graph, and 30 alternating event samples per arm in each of two owner arenas. Candidate/parent medians were 203.306/199.665 us and 202.739/200.594 us. Median paired savings were -3.652/-2.182 us, with 0/60 wins. All 19,200 boundary checks, 360 event-tail checks and 696 complete active-table checks passed. [The timing report](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-dual-quant-timing-v1-a3-194/summary.json), SHA `92ae4d12bc1500367c693ac3ed5531808f9369dfb877619c9bb74d39d0fc5971`, rejects this scheduling change for the measured workload; it does not identify the cause of the regression.

| Artifact | SHA-256 |
| --- | --- |
| `build-v1/ffn_dual_quant.so` | `715060c0c50a275855eb757cd85128a6c5586deeaa7757dc2945b92bee64d69b` |
| Main manifest | `2a5009cb99dbb366919cca1713645400adbef59e302e774a5160394b482bb222` |
| `probe-build-v1/quant_probe.so` | `41bc2af880217a0520e6dfc8764de6068e1e570bc08ea14e2349333549126c65` |
| Probe manifest | `ca037e2dbd81642310f50fb2aac32ec973c765e84a9ce17306caec8aaa90bc24` |
| Frozen probe checker | `de72807207fa564cb8d754cf6c0190814c967c0506794419951b0048f75bdde9` |

Both builds are frozen under `/home/shanshan/tilert-megakernel-20261021/tests/glm52/whole_layer/ffn_dual_quant/` on a3-78 and a3-194. [The build evidence](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-dual-quant-build-v1/summary.json) retains sources, commands, compilation logs, ARM contract output, and remote file verification without object files or libraries.

The direct probe compares the original two quantizers against the shared-prefix candidate without norm, gate, or TopK. It checks shared codes/scales (`bridge[6:8]`), routed codes/scales (`body[10:12]`), and four immediate same-graph consumers. Twelve raw BF16 fixtures cover finite values, positive/negative zero, infinities, NaN payloads, and the `0x3c2f` reciprocal-versus-division witness. Two different poison patterns per fixture reject missing writes. Fixed legal route metadata remains separate from these numerical inputs.

Inside the task's Torch 2.12/CANN 9.2 container, with the task directory mounted at `/workspace` and a writable evidence directory at `/evidence`:

```bash
source /opt/ascend/cann-9.2.0/set_env.sh
python3 /workspace/tests/glm52/whole_layer/ffn_dual_quant/probe-build-v1/probe/check.py \
  --candidate-build /workspace/tests/glm52/whole_layer/ffn_dual_quant/build-v1 \
  --probe-build /workspace/tests/glm52/whole_layer/ffn_dual_quant/probe-build-v1 \
  --rows 1,2,12,32 --blocks 16,24 \
  --output /evidence/dual-quant-direct.json
```

The full direct probe uses 16 captures and 384 graph replays. The matched timing bundle compares this candidate with c90 at all 33 common tensor addresses, with two independently allocated owner arenas, 16 weight banks, and 30 paired event samples per arm and arena. [Its CPU evidence](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-dual-quant-timing-build-v1/handoff.json) records six x86 and six ARM ABI/contract tests plus actual five-library co-loading without NPU initialization. These checks do not establish a latency gain.
