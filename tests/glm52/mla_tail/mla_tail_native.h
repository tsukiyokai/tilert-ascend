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
    PipeBarrier<PIPE_ALL>();
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

template <int64_t kHeads = mla_tail::kHeads>
struct VectorWorkspace {
    static constexpr int64_t kProjectionInput = kHeads * kValue;
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

constexpr int64_t kStageRows = 8;

template <int64_t kHeads = mla_tail::kHeads>
__aicore__ inline void StageHeads(VectorWorkspace<kHeads>& held,
                                 __gm__ bfloat16_t* latent,
                                 __gm__ bfloat16_t* heads,
                                 int64_t rows, int64_t capacity,
                                 int64_t vector_index, int64_t vectors)
{
    GlobalTensor<bfloat16_t> source;
    GlobalTensor<bfloat16_t> destination;
    source.SetGlobalBuffer(latent, rows * kHeads * kLatent);
    destination.SetGlobalBuffer(heads, capacity * kHeads * kLatent);
    const int64_t stage_rows = rows <= 6 ? 1 : ((rows + 5) / 6);
    const int64_t batches = (rows + stage_rows - 1) / stage_rows;
    auto stage = held.value.template ReinterpretCast<bfloat16_t>();
    for (int64_t task = vector_index; task < kHeads * batches;
         task += vectors) {
        const int64_t head = task / batches;
        const int64_t begin = (task % batches) * stage_rows;
        const int64_t valid = rows <= begin
            ? 0 : (rows - begin < stage_rows ? rows - begin : stage_rows);
        if (valid <= 0) {
            continue;
        }
        DataCopyParams copy{
            static_cast<uint16_t>(valid),
            static_cast<uint16_t>(kLatent * sizeof(bfloat16_t) / 32),
            static_cast<uint16_t>((kHeads - 1) * kLatent * sizeof(bfloat16_t) / 32),
            0,
        };
        DataCopy(stage, source[(begin * kHeads + head) * kLatent], copy);
        Fence<HardEvent::MTE2_MTE3>();
        DataCopy(destination[(head * capacity + begin) * kLatent], stage,
                 valid * kLatent);
        if (task + vectors < kHeads * batches) {
            Fence<HardEvent::MTE3_MTE2>();
        }
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

template <int64_t kHeads = mla_tail::kHeads>
__aicore__ inline void ProjectValues(TPipe& pipe,
                                    __gm__ bfloat16_t* latent,
                                    __gm__ bfloat16_t* weights,
                                    __gm__ float* values,
                                    __gm__ uint8_t* tiling,
                                    int64_t capacity, int64_t blocks,
                                    int64_t rows = 0)
{
    if (GetBlockIdx() >= kValueTiles) {
        return;
    }
    constexpr int64_t kProjectionInput = kHeads * kValue;
    constexpr int64_t kValueTile = kProjectionInput / kValueTiles;
    TCubeTiling cube;
    LoadTiling(cube, tiling);
    Matmul<ValueA, ValueB, ValueC, ValueBias> mm;
    mm.Init(&cube, &pipe);
    GlobalTensor<bfloat16_t> a;
    GlobalTensor<bfloat16_t> b;
    GlobalTensor<float> c;
    for (int64_t tile = GetBlockIdx(); tile < kValueTiles; tile += blocks) {
        const int64_t head = tile / (kValue / kValueTile);
        a.SetGlobalBuffer(latent + head * kLatent,
                          capacity * kHeads * kLatent);
        b.SetGlobalBuffer(weights + tile * kLatent * kValueTile,
                          kLatent * kValueTile);
        c.SetGlobalBuffer(values + tile * kValueTile,
                          capacity * kProjectionInput);
        mm.SetOrgShape(capacity, kValueTile, kHeads * kLatent, kLatent,
                       kProjectionInput);
        mm.SetSingleShape(capacity, kValueTile, kLatent);
        mm.SetTensorA(a);
        mm.SetTensorB(b, false);
        mm.IterateAll(c);
        mm.End();
    }
}

template <int64_t kHeads = mla_tail::kHeads>
__aicore__ inline void QuantizeValues(VectorWorkspace<kHeads>& held,
                                     __gm__ float* values,
                                     __gm__ int8_t* quantized,
                                     __gm__ float* token_scales,
                                     int64_t capacity, int64_t vector_index,
                                     int64_t vectors, int64_t rows = 0)
{
    const int64_t active_rows = rows > 0 ? rows : capacity;
    if (vector_index >= active_rows) {
        return;
    }
    constexpr int64_t kProjectionInput = kHeads * kValue;
    GlobalTensor<float> source;
    GlobalTensor<int8_t> destination;
    GlobalTensor<float> scales;
    source.SetGlobalBuffer(values, capacity * kProjectionInput);
    destination.SetGlobalBuffer(quantized, capacity * kProjectionInput);
    scales.SetGlobalBuffer(token_scales, capacity * kScaleStride);
    SetDeqScale(static_cast<half>(1.0F));
    for (int64_t row = vector_index; row < active_rows; row += vectors) {
        DataCopy(held.value, source[row * kProjectionInput], kProjectionInput);
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
        Duplicate(held.work, maximum == 0.0F ? 1.0F : maximum, 64);
        Duplicate(held.other, 127.0F, 64);
        PipeBarrier<PIPE_V>();
        Div(held.other, held.other, held.work, 64);
        PipeBarrier<PIPE_V>();
        Mul(held.value, held.value, held.other, 64, kProjectionInput / 64,
            BinaryRepeatParams{1, 1, 0, 8, 8, 0});
        Muls(held.scalar, held.scalar, 1.0F / 127.0F, kScaleStride);
        PipeBarrier<PIPE_V>();
        Mins(held.value, held.value, 127.0F, kProjectionInput);
        PipeBarrier<PIPE_V>();
        Maxs(held.value, held.value, -127.0F, kProjectionInput);
        PipeBarrier<PIPE_V>();
        // Match dynamic_quant's FP32 -> INT32 -> FP16 -> INT8 cast chain.
        // Rounding to an integer first avoids an FP16 double-rounding edge.
        auto integers = held.work.template ReinterpretCast<int32_t>();
        Cast(integers, held.value, RoundMode::CAST_RINT, kProjectionInput);
        PipeBarrier<PIPE_V>();
        Cast(held.fp16, integers, RoundMode::CAST_ROUND, kProjectionInput);
        PipeBarrier<PIPE_V>();
        Cast(held.codes, held.fp16, RoundMode::CAST_TRUNC, kProjectionInput);
        Fence<HardEvent::V_MTE3>();
        DataCopy(destination[row * kProjectionInput], held.codes,
                 kProjectionInput);
        DataCopy(scales[row * kScaleStride], held.scalar, kScaleStride);
        if (row + vectors < active_rows) {
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::V_MTE2>();
        }
    }
}

template <int64_t kHeads = mla_tail::kHeads>
__aicore__ inline void ProjectOutput(TPipe& pipe,
                                    __gm__ int8_t* quantized,
                                    __gm__ int8_t* weights,
                                    __gm__ int32_t* accumulators,
                                    __gm__ uint8_t* tiling,
                                    int64_t capacity, int64_t blocks,
                                    int64_t rows = 0)
{
    const int64_t active_rows = rows > 0 ? rows : capacity;
    constexpr int64_t kProjectionInput = kHeads * kValue;
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
        mm.SetSingleShape(capacity, kOutputTile, kProjectionInput);
        mm.SetTensorA(a);
        mm.SetTensorB(b, false);
        mm.IterateAll(c);
        mm.End();
    }
}

template <int64_t kHeads = mla_tail::kHeads>
__aicore__ inline void DequantizeOutput(
    VectorWorkspace<kHeads>& held, __gm__ int32_t* accumulators,
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

constexpr int64_t kDequantRows = 16;

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

template <ScaleOrder order = ScaleOrder::CombinedScales>
__aicore__ inline void FinishOutputRows(
    OutputWorkspace& held, LocalTensor<float>& combined,
    __gm__ int32_t* accumulators, __gm__ float* weight_scales,
    __gm__ float* token_scales, __gm__ uint8_t* bias, int bias_kind,
    __gm__ bfloat16_t* output, int64_t rows, int64_t capacity,
    int64_t vector_index, int64_t vectors)
{
    const int64_t batch_rows = rows <= 1 ? 1
        : (rows <= 2 * kDequantRows ? ((rows + 1) >> 1) : kDequantRows);
    const int64_t batches = (rows + batch_rows - 1) / batch_rows;
    if (vector_index >= batches * kOutputTiles) {
        return;
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
        const int64_t begin = (task / kOutputTiles) * batch_rows;
        const int64_t count = rows <= begin ? 0
            : (rows - begin < batch_rows ? rows - begin : batch_rows);
        if (count <= 0) {
            continue;
        }
        const int64_t total = count * kOutputTile;
        DataCopy(held.accumulators, source[(tile * capacity + begin) * kOutputTile],
                  total);
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

        if (bias_kind == 1) {
            for (int64_t row = 0; row < count; ++row) {
                Add(held.accumulators[row * kOutputTile],
                    held.accumulators[row * kOutputTile], held.bias, kOutputTile);
            }
            PipeBarrier<PIPE_V>();
        }
        Cast(held.values, held.accumulators, RoundMode::CAST_RINT, total);
        PipeBarrier<PIPE_V>();
        if constexpr (order == ScaleOrder::CombinedScales) {
            for (int64_t row = 0; row < count; ++row) {
                Muls(combined[row * kOutputTile], held.weights,
                     held.scales.GetValue(row * kScaleStride), kOutputTile);
            }
            PipeBarrier<PIPE_V>();
            Mul(held.values, held.values, combined, total);
            PipeBarrier<PIPE_V>();
        } else {
            for (int64_t row = 0; row < count; ++row) {
                Mul(held.values[row * kOutputTile],
                    held.values[row * kOutputTile], held.weights, kOutputTile);
            }
            PipeBarrier<PIPE_V>();
            for (int64_t row = 0; row < count; ++row) {
                Muls(held.values[row * kOutputTile],
                     held.values[row * kOutputTile],
                     held.scales.GetValue(row * kScaleStride), kOutputTile);
            }
            PipeBarrier<PIPE_V>();
        }
        if (bias_kind == 2) {
            for (int64_t row = 0; row < count; ++row) {
                Add(held.values[row * kOutputTile],
                    held.values[row * kOutputTile],
                    held.bias.ReinterpretCast<float>(), kOutputTile);
            }
            PipeBarrier<PIPE_V>();
        }
        Cast(held.output, held.values, RoundMode::CAST_RINT, total);
        Fence<HardEvent::V_MTE3>();
        DataCopyParams copy;
        copy.blockCount = static_cast<uint16_t>(count);
        copy.blockLen = kOutputTile * sizeof(bfloat16_t) / 32;
        copy.srcStride = 0;
        copy.dstStride = (kHidden - kOutputTile) * sizeof(bfloat16_t) / 32;
        DataCopy(destination[begin * kHidden + tile * kOutputTile],
                  held.output, copy);
        if (task + vectors < batches * kOutputTiles) {
            Fence<HardEvent::MTE3_MTE2>();
        }
    }
}

template <ScaleOrder order = ScaleOrder::CombinedScales>
__aicore__ inline void DequantizeOutputRows(
    TPipe& pipe, __gm__ int32_t* accumulators, __gm__ float* weight_scales,
    __gm__ float* token_scales, __gm__ uint8_t* bias, int bias_kind,
    __gm__ bfloat16_t* output, int64_t rows, int64_t capacity,
    int64_t vector_index, int64_t vectors)
{
    OutputWorkspace held;
    TBuf<TPosition::VECCALC> combined_buffer;
    LocalTensor<float> combined;
    held.Init(pipe);
    if constexpr (order == ScaleOrder::CombinedScales) {
        pipe.InitBuffer(combined_buffer,
                        kDequantRows * kOutputTile * sizeof(float));
        combined = combined_buffer.Get<float>();
    }
    FinishOutputRows<order>(held, combined, accumulators, weight_scales,
                            token_scales, bias, bias_kind, output, rows,
                            capacity, vector_index, vectors);
}


template <int64_t kHeads = mla_tail::kHeads>
__aicore__ inline void Forward(
    TPipe& pipe, __gm__ bfloat16_t* latent,
    __gm__ bfloat16_t* value_weight, __gm__ int8_t* output_weight,
    __gm__ float* output_scale, __gm__ uint8_t* bias, int bias_kind,
    __gm__ uint8_t* scratch, __gm__ uint8_t* tiling,
    __gm__ bfloat16_t* output, int64_t rows, int64_t capacity,
    int64_t blocks, const WorkspaceLayout& layout, bool publish_end = true)
{
    SetSysWorkspace(scratch + layout.system);
    const uint64_t values_offset = layout.values;
    const uint64_t quantized_offset = layout.quantized;
    const uint64_t scales_offset = layout.token_scales;
    const uint64_t accumulators_offset = layout.accumulators;
    auto* values = reinterpret_cast<__gm__ float*>(scratch + values_offset);
    auto* quantized = reinterpret_cast<__gm__ int8_t*>(scratch + quantized_offset);
    auto* scales = reinterpret_cast<__gm__ float*>(scratch + scales_offset);
    auto* accumulators =
        reinterpret_cast<__gm__ int32_t*>(scratch + accumulators_offset);
    VectorWorkspace<kHeads> held;
    if ASCEND_IS_AIC {
        ProjectValues<kHeads>(pipe, latent, value_weight, values, tiling, capacity, blocks, rows);
        Publish();
    }
    if ASCEND_IS_AIV {
        held.Init(pipe);
    }
    // Every AIC and AIV participates, including cores with no tile in a phase.
    // Batch scheduling and the physical-core launch limit ensure residency.
    SyncAll<false>();
    if ASCEND_IS_AIC {
        pipe.Reset();
    }
    if ASCEND_IS_AIV {
        QuantizeValues<kHeads>(held, values, quantized, scales, capacity,
                               GetBlockIdx(), 2 * blocks, rows);
        Publish();
    }
    SyncAll<false>();
    if ASCEND_IS_AIC {
        ProjectOutput<kHeads>(pipe, quantized, output_weight, accumulators,
                              tiling + kCubeTilingBytes, capacity, blocks, rows);
        Publish();
    }
    if ASCEND_IS_AIV {
        pipe.Reset();
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        DequantizeOutputRows<ScaleOrder::CombinedScales>(
            pipe, accumulators, output_scale, scales, bias, bias_kind, output,
            rows, capacity, GetBlockIdx(), 2 * blocks);
        if (publish_end) {
            Publish();
        }
    }
}

}  // namespace mla_tail::device

#endif
