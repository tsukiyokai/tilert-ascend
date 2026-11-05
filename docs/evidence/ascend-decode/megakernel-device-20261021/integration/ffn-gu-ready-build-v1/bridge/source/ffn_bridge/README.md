# Post-attention FFN bridge

This isolated MIX entry consumes a BF16 attention TP result and the BF16
residual. It produces AddRMSNorm outputs, FP32 gate logits, ordered top-8
indices and probabilities, and independent shared-expert A8 input. It does
not contain a TP reduction or an expert body. Its separate FFN epilogue
retains `BF16(routed*2.5)` before `BF16(scaled+shared)`.

The fixed geometry is `1 <= M <= 32`, H=6144, 256 experts, top-8, and 16 or
24 mixed groups. The gate weight is contiguous physical ND BF16
`[6144,256]`; the correction bias is FP32 `[256]`. Converting a production
gate weight to this prototype layout is an explicit preparation step.

## Interfaces and lifetime

`tilert_ffn_bridge.prepare(sample, blocks)` returns twelve tensors and a
sync base. Preparation allocates and initializes the gate tiling before
capture. Execution performs no allocation or route read on the CPU. The
caller retains all tensors until every consuming graph has finished.
Scratch tensors must be disjoint, and the tiling and system-workspace
TensorImpl identities must belong to the same prepare call. Calls use the
current Torch-NPU stream with host task-queue submission enabled.

| Slot | Tensor | Dtype and shape |
| --- | --- | --- |
| 0 | Normalized input | BF16 `[M,6144]` |
| 1 | Retained residual | BF16 `[M,6144]` |
| 2 | Reciprocal RMS | FP32 `[M,1]` |
| 3 | Gate logits | FP32 `[M,256]` |
| 4 | Probabilities | FP32 `[M,8]` |
| 5 | Ordered expert IDs | INT32 `[M,8]` |
| 6 | Shared input codes | INT8 `[M,6144]` |
| 7 | Shared input scales | FP32 `[M,1]` |
| 8 | Immutable gate tiling | UINT8 `[512]` |
| 9 | Matmul system workspace | UINT8 `[16777216]` |
| 10 | Combined FFN partial | BF16 `[M,6144]` |
| 11 | Scaled routed partial | BF16 `[M,6144]` |

M32 scratch occupies 18,582,272 bytes. Gate/quant and top-k start only after
the preceding phase's full pipe drain, DDR publication and mixed SyncAll.
The gate and independent shared DynamicQuant execute on AIC and AIV in the
same phase. The norm's 191 KiB UB allocation is retired before quantization;
top-k uses 16,416 bytes after another reset. All resident cores participate
in each mixed barrier, including cores without a row or gate tile.

## Router arithmetic

`prepare_vendor.py` freezes the pinned vLLM-Ascend without-group implementation
and changes only two UB capacities. Both expert-ID buffers cover the full
256-INT32 copy already issued by that source. Sigmoid, Sort, Gather,
ReduceSum, epsilon addition, Div and scaling retain their source order.
For the standalone-to-MIX adaptation, both per-core row counts are one and
only flat AIV IDs below M call the body. This covers M32 at 16 or 24 groups
without interpreting inactive cores as valid rows.

The production oracle is `BaseDeviceAdaptor.moe_gating_top_k`, which invokes
`torch.ops._C_ascend.moe_gating_top_k`. Parameters are k=8, k_group=1,
group_count=1, group_select_mode=1, renorm=1, norm_type=1, out_flag=false,
routed_scaling_factor=1 and epsilon=1e-20. Selection sorts sigmoid plus bias
in descending order. Probabilities gather the original sigmoid scores.
The device check compares the ordered ID tensor and every probability byte,
including ties; CPU sigmoid is not its acceptance oracle. The subsequent
routed body still owns conversion of these probabilities to BF16.

## Build and checks

Use the pinned vLLM-Ascend source and CANN 9.2. A build directory is immutable:
the script rejects an existing path. It preserves all compiled local and
vendor sources, command arguments, compiler logs and artifact hashes.

```bash
VLLM_ASCEND_ROOT=/path/to/pinned/vllm-ascend \
  bash tests/glm52/whole_layer/ffn_bridge/build.sh /path/to/new-build
```

CPU checks cover row/column ownership, GM range and scratch size, adversarial
fixed-logit fixture properties, the two capacity changes and the observable
BF16 epilogue rounding boundary. They do not validate device arithmetic.

Run fixed-logit tests first in a matching Torch/Torch-NPU environment with
the production custom operators installed:

```bash
python tests/glm52/whole_layer/ffn_bridge/check.py \
  --library /path/to/new-build/ffn_bridge.so --topk-only \
  --rows 1 2 12 32 --blocks 16 24 --replays 8 --output /path/to/topk-result
```

After that gate passes, omit `--topk-only` to compare norm, retained
residual, reciprocal RMS, gate logits, top-k, shared A8/scales and FFN
epilogue. Every graph replay changes its inputs; the native graph includes
queued clone producers. A failure retains actual tensors and oracle
outputs. No timing or TPOT claim is made by this numerical check.
