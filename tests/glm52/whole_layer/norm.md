# Residual, RMS normalization and dynamic A8

`norm_native.h` provides a composable mixed-kernel stage for BF16
`[M,6144]`, with `1 <= M <= 32`. It adds the residual, applies the deployed
vLLM-Ascend A2/A3 single-N `AddRmsNormBias` body, then dynamically quantizes
the BF16 result. CANN 9.2.0 compiles and links the prototype for `dav-2201`;
its three Torch schemas load without starting an NPU task. Device numerical
and timing checks remain outstanding.

The external source is supplied through `VLLM_ASCEND_ROOT`. Build provenance
records the exact single-N, reduction and data-copy headers together with
the native source and library hashes. The vendor body is included unchanged;
only its source-local argument type is renamed. It consumes four fields,
with the same H=6144 BF16 `ub_factor` choices as the vendor host tiler.

The arithmetic retains the vendor boundaries: BF16 residual sum; FP32
square, reduction and reciprocal RMS; BF16 normalized value; FP32 gamma
multiplication and optional beta addition; final BF16 output. DynamicQuant
then uses vector division, FP32-to-INT32 nearest-even conversion and an
explicit unit DEQSCALE before INT32-to-FP16-to-INT8 conversion. Scales are
contiguous FP32 `[M,1]`, matching the attention prolog input. This preserves
the separate norm/quant path selected by `CannModel`'s
`fuse_norm_quant=False` compilation setting.

Each AIV owns its rows through both stages. It drains writes and releases
the norm's 195,584-byte UB allocation before quantization uses 92,192 bytes.
All resident AIC/AIV groups join the final barrier. The normalized tensor,
retained residual, reciprocal RMS, INT8 output and scales are caller-owned
GM buffers; no allocation or host tensor read occurs in captured execution.
The no-residual first-layer norm is a separate contract and is not provided.

```bash
source /opt/ascend/cann-9.2.0/set_env.sh
VLLM_ASCEND_ROOT=/path/to/vllm-ascend \
    bash tests/glm52/whole_layer/norm_build.sh
python tests/glm52/whole_layer/norm_check.py --widths 1 --replays 1 \
    --no-timing --output /tmp/norm-eager.json
python tests/glm52/whole_layer/norm_check.py --widths 1 2 12 31 32 \
    --blocks 1 8 24 --capture --replays 8 --no-timing \
    --output /tmp/norm-replay.json
```

The default baseline is the installed
`torch.ops._C_ascend.npu_add_rms_norm_bias` followed by CANN DynamicQuant.
The harness compares all five outputs exactly, exercises present/absent
beta, cancellation, signed values, changing inputs and a queued producer.
`--baseline source-replay` instead runs the same norm source in a separate
launch followed by CANN DynamicQuant. That mode checks stage composition
and the new A8 implementation; it does not independently validate the
included norm source against the installed vendor operator.

After correctness passes, omit `--no-timing` with `--capture` to record
alternating event-timed graph pairs. Both arms include the same queued
input clone. These timings describe the norm/A8 segment, not a full layer
or end-to-end TPOT.
