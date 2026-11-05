# W4A8 SwiGLU quantization dependency

The frozen 8K continuation exposes a read-after-write dependency in
`grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h`, from vLLM-Ascend
`f558b58c177916424ae9dcf9bb8e66c0b9031236`. SwiGLU copies its result into
the gate buffer; quantization immediately reads that destination with
`Abs`. The [patch](../../../tests/glm52/patches/gmm1-swiglu-barrier.patch)
adds `PipeBarrier<PIPE_V>()` between those operations. CANN 9.2 emits
`copy_ubuf_to_ubuf` for this copy; the official
[pipeline table](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/appendix/api_pipeline_type_summary.md)
assigns UB-to-UB `DataCopy` to `PIPE_V`.

## Numerical evidence

In the [original 8K capture](cann-routed-8k-20261017.json), all 11 incorrect
scales equal `max(abs(gate_before_swiglu))/127`, bitwise. Correct SwiGLU
values quantized with those incorrect scales reproduce all 1,408 captured
INT8 codes. Their maximum absolute code is only 3–13; the actual nonzero
activation with scale `amax/127` must reach 127. GMM2 reconstructs all
48 expert rows exactly from its actual inputs.

The [single-die replay](../../../tests/glm52/replay_cann_gmm1.py) uses the
real layer-4 dispatch and W4 weight slices, production NZ postprocessing
and the original `_C_ascend` wrapper. Independent CPU arithmetic checks
every INT8 code and FP32 scale, repeated calls and the amax invariant.

| Vendor | Rank shard | Calls | Exact expert rows |
| --- | --- | --- | --- |
| Original | 0 | 8 | 62/64 |
| Rebuilt, barrier | 0 | 8 | 64/64 |
| Rebuilt, barrier | 12 | 32 | 256/256 |
| Rebuilt, barrier | 15 | 32 | 256/256 |
| Complete overlay, original host library | 0 | 32 | 256/256 |

The four patched cases pass 832/832 expert rows. They use shared die 15,
a 1,536 MiB allocator limit and 704 MiB peak reserved memory.
These are precision checks; full TP16 reset and HF parity remain pending.
The [report](cann-swiglu-barrier-20261017.json) binds checkpoint, payload,
source, binary and raw tensor hashes.

## Build and replay

The [builder](../../../tests/glm52/build_gmm1_barrier.sh) recompiles the
selected A8W4 NZ kernel through the installed CANN 9.2 `opc` wrapper in a
new directory. It includes this numerical fix and the
[single-chunk synchronization patch](gmm1-tiling-20261018/README.md#single-chunk-synchronization).
The complete vendor overlay preserves the installed host/API/tiling and
other operators, replacing the selected ELF, matching metadata and two
source headers. The source and installed headers must match before patching;
the output directory must be outside the original source.

```bash
source /workspace/environment/activate.sh
cd /workspace/src/TileRT-Ascend
bash tests/glm52/build_gmm1_barrier.sh \
  /workspace/src/vllm-ascend-unified /workspace/offline/gmm1-barrier-rebuild
original_vendor=/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
overlay_vendor=/workspace/offline/gmm1-barrier-rebuild/overlay/vendors/custom_transformer
ASCEND_CUSTOM_OPP_PATH="$overlay_vendor:$original_vendor" \
  PYTHONPATH="$PWD:${PYTHONPATH:-}" TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python tests/glm52/replay_cann_gmm1.py \
    --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
    --trace /workspace/offline/cann-routed-stage-20261017/continuation8k/rank-0/step-08192.pt \
    --rank 0 --device npu:15 --repeats 32 --output /tmp/gmm1-replay.json
```

Both vendor paths are required: `enable_custom_op()` prepends the original
path if absent. An overlay-only launch therefore loses the intended
selection order. The report's initial environment precedes that bootstrap.
With the ordered paths, `strace openat` records successful reads of the
overlay's original host library and patched `fa3d` kernel. The barrier-only
overlay in the numerical report changes three of 1,373 vendor entries:
that kernel, its JSON metadata and the source header. The current builder
also changes the pipeline header for single-chunk synchronization.
