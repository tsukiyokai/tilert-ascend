# AscendC stage validation on A3

The [subsequent a3-84 runs](a3-84/README.md) add raw-byte checks, NZ expert
weights, composed attention and norm validation, and paired component timings.
The [a3-162 runs](a3-162/README.md) add routed/shared interleaving, the TileRT
body adapter, a signed-zero combine check, and composed indexed attention.
The results below describe the earlier a3-78 runs.

The two native stages matched their CANN references numerically at zero
tolerance on synthetic inputs. These reports did not compare raw bytes,
so they do not distinguish positive and negative zero. They establish
stage correctness for the cases below, without a latency measurement or a
whole-model result.

| Stage | Verification widths | Changing-input graph replays | BF16 outputs compared |
| --- | --- | --- | --- |
| MLA projection tail | 1, 2, 3, 4, 6, 8, 12, 16, 24, 31, 32 | 88 | 6,832,128 |
| Routed W4A8 experts | 1, 2, 4, 8, 12, 16, 31, 32 | 60 | 5,185,536 |

Environment: a3-78, physical die 4 only, Ascend910_9392, 24 AIC / 48 AIV,
driver 25.5.5, CANN/runtime 9.2.0, torch/torch_npu 2.12.0. The isolated
container exposes this die as `npu:0`. Every run used eight mixed core groups
and zero absolute/relative tolerance. Raw reports contain source/library
SHA-256 hashes. The DRIVER version query raises a UnicodeDecodeError in this
Python package; the driver version above comes from host `npu-smi`.

MLA compares value-up, the BF16 boundary, dynamic A8 quantization and local
output projection. It excludes SFA and TP reduction. The INT32-to-FP16 cast
requires `SetDeqScale(half(1.0))`; adding this setting resolved the M=2 code
differences while value-up and activation scales had already matched.

- [M=1,2](mla-m1-m2-v2.json)
- [Remaining nine widths](mla-widths-v2.json)

Routed experts compare all 256 TP-sharded experts, intermediate width 128,
top-8 combine, hot experts, dispersed routes and changing expert IDs. The
reference is the stock `torch_npu` CANN chain with ND weights. The deployed
vLLM-Ascend custom NZ-weight path remains a separate required comparison.
The [isolated vendor check](routed-vendor-isolation.log) records valid ND
results and invalid NZ results for the tested preparation sequence.

- [M=1](routed-m1-queue-nd.json)
- [Remaining seven widths](routed-widths-queue-nd.json)
- [Control with `stream(false)`](routed-m1-sv-nd.json)

The control produced zero outputs. Changing only the binding's stream
accessor to `stream()` made the full M=1 graph comparison pass: queued Torch
dispatch/gather producers must be submitted before a direct AscendC launch.
The kernel binary was otherwise unchanged. This submits the host task queue;
it adds no device synchronization. The later MLA/attention bindings contain
the same correction and require their new library hashes to be tested.

Reproduction commands, after sourcing the matching CANN environment:

```bash
python3 tests/glm52/mla_tail/check.py --device npu:0 \
  --widths 1 2 3 4 6 8 12 16 24 31 32 --replays 8 --blocks 8 \
  --no-timing --output logs/mla.json
python3 tests/glm52/routed_int4/check.py --device npu:0 \
  --library tests/glm52/routed_int4/build-queue/routed_int4_torch.so \
  --widths 1 2 4 8 12 16 31 32 --replays 8 --blocks 8 \
  --vendor-backend torch-npu --vendor-layout nd --no-timing \
  --output logs/routed.json
```

The recorded M=1 routed run used four replays; the reproduction above expands
it to eight. The MLA reports identify the pre-stream-accessor-change library.
No latency run was made. A subsequent occupancy check found VLLM processes
on all 16 dies, so further device work stopped. Real checkpoint weights,
queued-producer MLA replay, sanitizer runs, shared attention, two-rank
communication and paired whole-engine TPOT remain to be verified.

## Additional compiled stages

The single-launch Prolog/SFA/projection composition and the residual/norm/A8
stage compiled and linked against the same CANN 9.2.0 installation. Their
Torch schemas loaded without starting an NPU task. The build manifests
record sources, vendor headers and artifacts:

- [Full shared-indexer attention](attention-full-build-provenance.json)
- [Residual/norm/A8](norm-build-provenance.json)

These are compiler and loader results. Neither composed stage has a device
comparison or latency result in this evidence set.
