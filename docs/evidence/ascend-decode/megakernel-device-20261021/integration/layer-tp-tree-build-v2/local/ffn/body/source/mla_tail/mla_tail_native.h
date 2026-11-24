#ifndef TESTS_GLM52_MLA_TAIL_NATIVE_H
#define TESTS_GLM52_MLA_TAIL_NATIVE_H

// Define this before any matmul header in a composed kernel as well.
#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mla_tail.h"

namespace mla_tail::device {
using namespace AscendC;

template <HardEvent event>
__aicore__ inline void Fence()
{
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}

__aicore__ inline void Publish()
{
    DataSyncBarrier<MemDsbT::DDR>();
}

__aicore__ inline void LoadTiling(TCubeTiling& cube, __gm__ uint8_t* source)
{
    auto* destination = reinterpret_cast<uint32_t*>(&cube);
    auto* words = reinterpret_cast<__gm__ uint32_t*>(source);
    for (uint32_t index = 0; index < sizeof(TCubeTiling) / 4; ++index) {
        destination[index] = words[index];
    }
}

struct VectorWorkspace {
    TBuf<TPosition::VECCALC> bf16_buffer;
    TBuf<TPosition::VECCALC> half_buffer;
    TBuf<TPosition::VECCALC> code_buffer;
    TBuf<TPosition::VECCALC> value_buffer;
    TBuf<TPosition::VECCALC> other_buffer;
    TBuf<TPosition::VECCALC> work_buffer;
    TBuf<TPosition::VECCALC> accumulator_buffer;
    TBuf<TPosition::VECCALC> bias_buffer;
    TBuf<TPosition::VECCALC> scalar_buffer;
    LocalTensor<bfloat16_t> bf16;
    LocalTensor<half> fp16;
    LocalTensor<int8_t> codes;
    LocalTensor<float> value;
    LocalTensor<float> other;
    LocalTensor<float> work;
    LocalTensor<int32_t> accumulator;
    LocalTensor<int32_t> bias;
    LocalTensor<float> scalar;

    __aicore__ inline void Init(TPipe& pipe)
    {
        pipe.InitBuffer(bf16_buffer, kProjectionInput * 2);
        pipe.InitBuffer(half_buffer, kProjectionInput * 2);
        pipe.InitBuffer(code_buffer, kProjectionInput);
        pipe.InitBuffer(value_buffer, kProjectionInput * 4);
        pipe.InitBuffer(other_buffer, kProjectionInput * 4);
        pipe.InitBuffer(work_buffer, kProjectionInput * 4);
        pipe.InitBuffer(accumulator_buffer, kOutputTile * 4);
        pipe.InitBuffer(bias_buffer, kOutputTile * 4);
        pipe.InitBuffer(scalar_buffer, kScaleStride * 4);
        bf16 = bf16_buffer.Get<bfloat16_t>();
        fp16 = half_buffer.Get<half>();
        codes = code_buffer.Get<int8_t>();
        value = value_buffer.Get<float>();
        other = other_buffer.Get<float>();
        work = work_buffer.Get<float>();
        accumulator = accumulator_buffer.Get<int32_t>();
        bias = bias_buffer.Get<int32_t>();
        scalar = scalar_buffer.Get<float>();
    }
};

__aicore__ inline void StageHeads(VectorWorkspace& held,
                                 __gm__ bfloat16_t* latent,
                                 __gm__ bfloat16_t* heads,
                                 int64_t rows, int64_t capacity,
                                 int64_t vector_index, int64_t vectors)
{
    GlobalTensor<bfloat16_t> source;
    GlobalTensor<bfloat16_t> destination;
    source.SetGlobalBuffer(latent, rows * kHeads * kLatent);
    destination.SetGlobalBuffer(heads, capacity * kHeads * kLatent);
    for (int64_t task = vector_index; task < capacity * kHeads;
         task += vectors) {
        const int64_t head = task / capacity;
        const int64_t row = task % capacity;
        if (row < rows) {
            DataCopy(held.bf16, source[(row * kHeads + head) * kLatent],
                     kLatent);
            Fence<HardEvent::MTE2_MTE3>();
        } else {
            Duplicate(held.bf16, static_cast<bfloat16_t>(0), kLatent);
            Fence<HardEvent::V_MTE3>();
        }
        DataCopy(destination[task * kLatent], held.bf16, kLatent);
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::V_MTE2>();
    }
}

using ValueA = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using ValueB = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using ValueC = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using ValueBias = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using OutputA = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
using OutputB = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
using OutputC = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using OutputBias = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

__aicore__ inline void ProjectValues(TPipe& pipe,
                                    __gm__ bfloat16_t* heads,
                                    __gm__ bfloat16_t* weights,
                                    __gm__ float* values,
                                    __gm__ uint8_t* tiling,
                                    int64_t capacity, int64_t blocks)
{
    TCubeTiling cube;
    LoadTiling(cube, tiling);
    Matmul<ValueA, ValueB, ValueC, ValueBias> mm;
    mm.Init(&cube, &pipe);
    GlobalTensor<bfloat16_t> a;
    GlobalTensor<bfloat16_t> b;
    GlobalTensor<float> c;
    for (int64_t tile = GetBlockIdx(); tile < kValueTiles; tile += blocks) {
        const int64_t head = tile / (kValue / kValueTile);
        a.SetGlobalBuffer(heads + head * capacity * kLatent,
                          capacity * kLatent);
        b.SetGlobalBuffer(weights + tile * kLatent * kValueTile,
                          kLatent * kValueTile);
        c.SetGlobalBuffer(values + tile * capacity * kValueTile,
                          capacity * kValueTile);
        mm.SetOrgShape(capacity, kValueTile, kLatent);
        mm.SetTensorA(a);
        mm.SetTensorB(b, false);
        mm.IterateAll(c);
        mm.End();
    }
}

__aicore__ inline void QuantizeValues(VectorWorkspace& held,
                                     __gm__ float* values,
                                     __gm__ int8_t* quantized,
                                     __gm__ float* token_scales,
                                     int64_t capacity, int64_t vector_index,
                                     int64_t vectors)
{
    GlobalTensor<float> source;
    GlobalTensor<int8_t> destination;
    GlobalTensor<float> scales;
    source.SetGlobalBuffer(values, capacity * kProjectionInput);
    destination.SetGlobalBuffer(quantized, capacity * kProjectionInput);
    scales.SetGlobalBuffer(token_scales, capacity * kScaleStride);
    for (int64_t row = vector_index; row < capacity; row += vectors) {
        for (int64_t tile = 0; tile < kValueTiles; ++tile) {
            DataCopy(held.value[tile * kValueTile],
                     source[(tile * capacity + row) * kValueTile],
                     kValueTile);
        }
        Fence<HardEvent::MTE2_V>();
        // The original value-up BMM returns BF16 before npu_dynamic_quant.
        Cast(held.bf16, held.value, RoundMode::CAST_RINT, kProjectionInput);
        PipeBarrier<PIPE_V>();
        Cast(held.value, held.bf16, RoundMode::CAST_NONE, kProjectionInput);
        PipeBarrier<PIPE_V>();
        Abs(held.other, held.value, kProjectionInput);
        Duplicate(held.scalar, 0.0F, kScaleStride);
        PipeBarrier<PIPE_V>();
        ReduceMax(held.scalar, held.other, held.work, kProjectionInput, false);
        Fence<HardEvent::V_S>();
        const float maximum = held.scalar.GetValue(0);
        // npu_dynamic_quant uses vector division. Scalar IEEE division can
        // choose different codes at half-integer thresholds on this device.
        Duplicate(held.work, maximum == 0.0F ? 1.0F : maximum,
                  kProjectionInput);
        Duplicate(held.other, 127.0F, kProjectionInput);
        PipeBarrier<PIPE_V>();
        Div(held.other, held.other, held.work, kProjectionInput);
        PipeBarrier<PIPE_V>();
        Mul(held.value, held.value, held.other, kProjectionInput);
        Muls(held.scalar, held.scalar, 1.0F / 127.0F, kScaleStride);
        PipeBarrier<PIPE_V>();
        Mins(held.value, held.value, 127.0F, kProjectionInput);
        PipeBarrier<PIPE_V>();
        Maxs(held.value, held.value, -127.0F, kProjectionInput);
        PipeBarrier<PIPE_V>();
        // Match dynamic_quant's FP32 -> INT32 -> FP16 -> INT8 cast chain.
        // Rounding to an integer first avoids an FP16 double-rounding edge.
        auto integers = held.work.ReinterpretCast<int32_t>();
        Cast(integers, held.value, RoundMode::CAST_RINT, kProjectionInput);
        PipeBarrier<PIPE_V>();
        // INT32 -> FP16 CAST_ROUND consumes the AIV DEQSCALE register.
        // Initialize it as dynamic_quant does; a previous kernel may change it.
        SetDeqScale(static_cast<half>(1.0F));
        PipeBarrier<PIPE_V>();
        Cast(held.fp16, integers, RoundMode::CAST_ROUND, kProjectionInput);
        PipeBarrier<PIPE_V>();
        Cast(held.codes, held.fp16, RoundMode::CAST_TRUNC, kProjectionInput);
        Fence<HardEvent::V_MTE3>();
        DataCopy(destination[row * kProjectionInput], held.codes,
                 kProjectionInput);
        DataCopy(scales[row * kScaleStride], held.scalar, kScaleStride);
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::V_MTE2>();
    }
}

__aicore__ inline void ProjectOutput(TPipe& pipe,
                                    __gm__ int8_t* quantized,
                                    __gm__ int8_t* weights,
                                    __gm__ int32_t* accumulators,
                                    __gm__ uint8_t* tiling,
                                    int64_t capacity, int64_t blocks)
{
    TCubeTiling cube;
    LoadTiling(cube, tiling);
    Matmul<OutputA, OutputB, OutputC, OutputBias> mm;
    mm.Init(&cube, &pipe);
    GlobalTensor<int8_t> a;
    GlobalTensor<int8_t> b;
    GlobalTensor<int32_t> c;
    a.SetGlobalBuffer(quantized, capacity * kProjectionInput);
    for (int64_t tile = GetBlockIdx(); tile < kOutputTiles; tile += blocks) {
        b.SetGlobalBuffer(weights + tile * kProjectionInput * kOutputTile,
                          kProjectionInput * kOutputTile);
        c.SetGlobalBuffer(accumulators + tile * capacity * kOutputTile,
                          capacity * kOutputTile);
        mm.SetOrgShape(capacity, kOutputTile, kProjectionInput);
        mm.SetTensorA(a);
        mm.SetTensorB(b, false);
        mm.IterateAll(c);
        mm.End();
    }
}

__aicore__ inline void DequantizeOutput(
    VectorWorkspace& held, __gm__ int32_t* accumulators,
    __gm__ float* weight_scales, __gm__ float* token_scales,
    __gm__ uint8_t* bias, int bias_kind, __gm__ bfloat16_t* output,
    int64_t rows, int64_t capacity, int64_t vector_index, int64_t vectors)
{
    GlobalTensor<int32_t> source;
    GlobalTensor<float> weight_scale;
    GlobalTensor<float> token_scale;
    GlobalTensor<int32_t> integer_bias;
    GlobalTensor<float> floating_bias;
    GlobalTensor<bfloat16_t> destination;
    source.SetGlobalBuffer(accumulators, capacity * kHidden);
    weight_scale.SetGlobalBuffer(weight_scales, kHidden);
    token_scale.SetGlobalBuffer(token_scales, capacity * kScaleStride);
    integer_bias.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bias));
    floating_bias.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    destination.SetGlobalBuffer(output, rows * kHidden);
    for (int64_t task = vector_index; task < rows * kOutputTiles;
         task += vectors) {
        const int64_t row = task / kOutputTiles;
        const int64_t tile = task % kOutputTiles;
        DataCopy(held.accumulator,
                 source[(tile * capacity + row) * kOutputTile], kOutputTile);
        DataCopy(held.other, weight_scale[tile * kOutputTile], kOutputTile);
        DataCopy(held.scalar, token_scale[row * kScaleStride], kScaleStride);
        if (bias_kind == 1) {
            DataCopy(held.bias, integer_bias[tile * kOutputTile], kOutputTile);
        }
        Fence<HardEvent::MTE2_V>();
        Fence<HardEvent::MTE2_S>();
        const float row_scale = held.scalar.GetValue(0);
        if (bias_kind == 1) {
            Add(held.accumulator, held.accumulator, held.bias, kOutputTile);
            PipeBarrier<PIPE_V>();
        }
        Cast(held.value, held.accumulator, RoundMode::CAST_RINT, kOutputTile);
        PipeBarrier<PIPE_V>();
        // Preserve the channel-then-token order of this projection's epilogue.
        Mul(held.value, held.value, held.other, kOutputTile);
        PipeBarrier<PIPE_V>();
        Muls(held.value, held.value, row_scale, kOutputTile);
        PipeBarrier<PIPE_V>();
        if (bias_kind == 2) {
            Fence<HardEvent::V_MTE2>();
            DataCopy(held.other, floating_bias[tile * kOutputTile], kOutputTile);
            Fence<HardEvent::MTE2_V>();
            Add(held.value, held.value, held.other, kOutputTile);
            PipeBarrier<PIPE_V>();
        }
        Cast(held.bf16, held.value, RoundMode::CAST_RINT, kOutputTile);
        Fence<HardEvent::V_MTE3>();
        DataCopy(destination[row * kHidden + tile * kOutputTile], held.bf16,
                 kOutputTile);
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::V_MTE2>();
    }
}

constexpr int64_t kDequantRows = 8;

struct OutputWorkspace {
    TBuf<TPosition::VECCALC> accumulator_buffer, value_buffer, output_buffer;
    TBuf<TPosition::VECCALC> weight_buffer, bias_buffer, scale_buffer;
    LocalTensor<int32_t> accumulators, bias;
    LocalTensor<float> values, weights, scales;
    LocalTensor<bfloat16_t> output;

    __aicore__ inline void Init(TPipe& pipe)
    {
        pipe.InitBuffer(accumulator_buffer, kDequantRows * kOutputTile * 4);
        pipe.InitBuffer(value_buffer, kDequantRows * kOutputTile * 4);
        pipe.InitBuffer(output_buffer, kDequantRows * kOutputTile * 2);
        pipe.InitBuffer(weight_buffer, kOutputTile * 4);
        pipe.InitBuffer(bias_buffer, kOutputTile * 4);
        pipe.InitBuffer(scale_buffer, kDequantRows * kScaleStride * 4);
        accumulators = accumulator_buffer.Get<int32_t>();
        values = value_buffer.Get<float>();
        output = output_buffer.Get<bfloat16_t>();
        weights = weight_buffer.Get<float>();
        bias = bias_buffer.Get<int32_t>();
        scales = scale_buffer.Get<float>();
    }
};

enum class ScaleOrder { ChannelThenToken, CombinedScales };

template <ScaleOrder order = ScaleOrder::ChannelThenToken>
__aicore__ inline void DequantizeOutputRows(
    TPipe& pipe, __gm__ int32_t* accumulators, __gm__ float* weight_scales,
    __gm__ float* token_scales, __gm__ uint8_t* bias, int bias_kind,
    __gm__ bfloat16_t* output, int64_t rows, int64_t capacity,
    int64_t vector_index, int64_t vectors)
{
    const int64_t batches = (rows + kDequantRows - 1) / kDequantRows;
    if (vector_index >= batches * kOutputTiles) {
        return;
    }
    OutputWorkspace held;
    held.Init(pipe);
    TBuf<TPosition::VECCALC> combined_buffer;
    LocalTensor<float> combined;
    if constexpr (order == ScaleOrder::CombinedScales) {
        pipe.InitBuffer(combined_buffer, kOutputTile * sizeof(float));
        combined = combined_buffer.Get<float>();
    }
    GlobalTensor<int32_t> source, integer_bias;
    GlobalTensor<float> weight_scale, token_scale, floating_bias;
    GlobalTensor<bfloat16_t> destination;
    source.SetGlobalBuffer(accumulators, capacity * kHidden);
    weight_scale.SetGlobalBuffer(weight_scales, kHidden);
    token_scale.SetGlobalBuffer(token_scales, capacity * kScaleStride);
    integer_bias.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bias));
    floating_bias.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    destination.SetGlobalBuffer(output, rows * kHidden);

    for (int64_t task = vector_index; task < batches * kOutputTiles;
         task += vectors) {
        const int64_t tile = task % kOutputTiles;
        const int64_t begin = (task / kOutputTiles) * kDequantRows;
        const int64_t count = rows - begin < kDequantRows
            ? rows - begin : kDequantRows;
        DataCopy(held.accumulators, source[(tile * capacity + begin) * kOutputTile],
                  count * kOutputTile);
        DataCopy(held.weights, weight_scale[tile * kOutputTile], kOutputTile);
        DataCopy(held.scales, token_scale[begin * kScaleStride],
                  count * kScaleStride);
        if (bias_kind == 1) {
            DataCopy(held.bias, integer_bias[tile * kOutputTile], kOutputTile);
        } else if (bias_kind == 2) {
            DataCopy(held.bias.ReinterpretCast<float>(),
                      floating_bias[tile * kOutputTile], kOutputTile);
        }
        Fence<HardEvent::MTE2_V>();
        Fence<HardEvent::MTE2_S>();

        for (int64_t row = 0; row < count; ++row) {
            const int64_t offset = row * kOutputTile;
            if (bias_kind == 1) {
                Add(held.accumulators[offset], held.accumulators[offset],
                    held.bias, kOutputTile);
                PipeBarrier<PIPE_V>();
            }
            Cast(held.values[offset], held.accumulators[offset],
                   RoundMode::CAST_RINT, kOutputTile);
            PipeBarrier<PIPE_V>();
            const float scale = held.scales.GetValue(row * kScaleStride);
            if constexpr (order == ScaleOrder::CombinedScales) {
                Muls(combined, held.weights, scale, kOutputTile);
                PipeBarrier<PIPE_V>();
                Mul(held.values[offset], held.values[offset], combined,
                    kOutputTile);
            } else {
                Mul(held.values[offset], held.values[offset], held.weights,
                    kOutputTile);
                PipeBarrier<PIPE_V>();
                Muls(held.values[offset], held.values[offset], scale,
                     kOutputTile);
            }
            PipeBarrier<PIPE_V>();
            if (bias_kind == 2) {
                Add(held.values[offset], held.values[offset],
                    held.bias.ReinterpretCast<float>(), kOutputTile);
                PipeBarrier<PIPE_V>();
            }
            Cast(held.output[offset], held.values[offset],
                   RoundMode::CAST_RINT, kOutputTile);
        }
        Fence<HardEvent::V_MTE3>();
        DataCopyParams copy;
        copy.blockCount = static_cast<uint16_t>(count);
        copy.blockLen = kOutputTile * sizeof(bfloat16_t) / 32;
        copy.srcStride = 0;
        copy.dstStride = (kHidden - kOutputTile) * sizeof(bfloat16_t) / 32;
        DataCopy(destination[begin * kHidden + tile * kOutputTile],
                  held.output, copy);
        // Protect the UB source of the previous output DMA before another
        // task's input DMA and vector work reuse this arena.
        Fence<HardEvent::MTE3_MTE2>();
    }
}


__aicore__ inline void Forward(
    TPipe& pipe, __gm__ bfloat16_t* latent,
    __gm__ bfloat16_t* value_weight, __gm__ int8_t* output_weight,
    __gm__ float* output_scale, __gm__ uint8_t* bias, int bias_kind,
    __gm__ uint8_t* scratch, __gm__ uint8_t* tiling,
    __gm__ bfloat16_t* output, int64_t rows, int64_t capacity,
    int64_t blocks, const WorkspaceLayout& layout)
{
    SetSysWorkspace(scratch + layout.system);
    const uint64_t values_offset = layout.values;
    const uint64_t quantized_offset = layout.quantized;
    const uint64_t scales_offset = layout.token_scales;
    const uint64_t accumulators_offset = layout.accumulators;
    auto* heads = reinterpret_cast<__gm__ bfloat16_t*>(scratch);
    auto* values = reinterpret_cast<__gm__ float*>(scratch + values_offset);
    auto* quantized = reinterpret_cast<__gm__ int8_t*>(scratch + quantized_offset);
    auto* scales = reinterpret_cast<__gm__ float*>(scratch + scales_offset);
    auto* accumulators =
        reinterpret_cast<__gm__ int32_t*>(scratch + accumulators_offset);
    VectorWorkspace held;
    if ASCEND_IS_AIV {
        held.Init(pipe);
        StageHeads(held, latent, heads, rows, capacity, GetBlockIdx(), 2 * blocks);
        Publish();
    }
    // Every AIC and AIV participates, including cores with no tile in a phase.
    // Batch scheduling and the physical-core launch limit ensure residency.
    SyncAll<false>();
    if ASCEND_IS_AIC {
        ProjectValues(pipe, heads, value_weight, values, tiling, capacity, blocks);
        Publish();
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        QuantizeValues(held, values, quantized, scales, capacity,
                       GetBlockIdx(), 2 * blocks);
        Publish();
    }
    SyncAll<false>();
    if ASCEND_IS_AIC {
        // Value matmul and its object have ended before reclaiming L1/events.
        pipe.Reset();
        ProjectOutput(pipe, quantized, output_weight, accumulators,
                      tiling + kCubeTilingBytes, capacity, blocks);
        Publish();
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        DequantizeOutputRows(pipe, accumulators, output_scale, scales, bias,
                            bias_kind, output, rows, capacity, GetBlockIdx(),
                            2 * blocks);
        Publish();
    }
}

}  // namespace mla_tail::device

#endif
