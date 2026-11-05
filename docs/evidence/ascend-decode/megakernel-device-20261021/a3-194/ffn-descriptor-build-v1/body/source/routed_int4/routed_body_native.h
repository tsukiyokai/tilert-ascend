#ifndef TILERT_ROUTED_BODY_NATIVE_H
#define TILERT_ROUTED_BODY_NATIVE_H

/** TP16 W4A8 routed experts with explicit scratch and deterministic combine.
 *
 * Numerical order: FP16(s_trunc * high_dot), FP16(s_trunc * low_dot),
 * FP32(16 * high + low + scaled_correction), activation scale. Gate/up stay
 * FP32 through SwiGLU and row quantization. Down results become BF16 before
 * top-k probabilities are multiplied and accumulated in FP32 slot order.
 */
#define ASCENDC_CUBE_ONLY

#include "routed_int4_kernel.h"
#include "../../../source/descriptor.h"
#include "routed_int4_dispatch.h"
#include "routed_int4_trace.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#ifdef ROUTED_INPUT_FRONT
#include "routed_int4_front.h"
#endif

#ifdef ROUTED_SHARED_PIPELINE
#include "routed_int4_pipeline_native.h"
#ifdef ROUTED_INPUT_FRONT
#define routed_int4_fused ffn_descriptor_unused_body_kernel
#else
#define routed_int4_fused routed_shared_fused
#endif
#elif defined(ROUTED_CONTROL)
#define routed_int4_fused routed_int4_control_fused
#elif defined(ROUTED_INPUT_FRONT)
#define routed_int4_fused routed_int4_front_fused
#elif defined(ROUTED_GU_MDL)
#define routed_int4_fused routed_int4_mdl_fused
#elif defined(ROUTED_GU_PAD16)
#define routed_int4_fused routed_int4_pad16_fused
#elif defined(ROUTED_BALANCED_NAMESPACE)
#define routed_int4_fused routed_int4_balanced_fused
#endif

namespace routed_body {

using namespace AscendC;
using namespace routed_int4;

namespace {

constexpr int64_t kSlack = 64;
constexpr int64_t kBlockFloats = 8;
using APacked = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
#ifdef ROUTED_FIXPIPE_HALF
using RoutedProduct = half;
#else
using RoutedProduct = int32_t;
#endif
using CProduct = MatmulType<TPosition::GM, CubeFormat::ND, RoutedProduct>;
using BiasNone = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
template <CubeFormat format>
using Projection = Matmul<APacked,
    MatmulType<TPosition::GM, format, int4b_t, false>, CProduct, BiasNone>;
#ifdef ROUTED_GU_MDL
template <CubeFormat format>
using GateProjection = Matmul<APacked,
    MatmulType<TPosition::GM, format, int4b_t, false>, CProduct, BiasNone,
    CFG_MDL>;
#else
template <CubeFormat format>
using GateProjection = Projection<format>;
#endif

// int4b_t is a storage wrapper; physical pointer arithmetic uses bytes.
__aicore__ inline __gm__ int4b_t* Nibbles(__gm__ int4b_t* base,
                                         int64_t elements)
{
    return (__gm__ int4b_t*)((__gm__ int8_t*)base + elements / 2);
}

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

struct Workspace {
    TBuf<TPosition::VECCALC> input_buf, value_buf, other_buf, work_buf;
    TBuf<TPosition::VECCALC> half_buf, packed_buf, staging_buf, scale_buf;
    TBuf<TPosition::VECCALC> mask_buf, token_buf, input_scale_buf;
    TBuf<TPosition::VECCALC> activation_buf, maximum_buf;
    TBuf<TPosition::VECCALC> count_buf, start_buf, source_buf, slot_buf;
    TBuf<TPosition::VECCALC> weight_buf, original_weight_buf;
    LocalTensor<int8_t> input;
    LocalTensor<float> value, other, work, scale, token, activation, maximum;
    LocalTensor<float> weight, original_weight, input_scale;
    LocalTensor<half> half_value;
    LocalTensor<int4b_t> packed;
    LocalTensor<int32_t> staging, mask, count, start, source, slot;

    __aicore__ inline void Init(TPipe& pipe)
    {
        constexpr int64_t span = kVectorTile + kSlack;
        pipe.InitBuffer(input_buf, span);
        pipe.InitBuffer(value_buf, span * sizeof(float));
        pipe.InitBuffer(other_buf, span * sizeof(float));
        pipe.InitBuffer(work_buf, span * sizeof(float));
        pipe.InitBuffer(half_buf, span * sizeof(half));
        pipe.InitBuffer(packed_buf, span / 2 + kSlack);
        pipe.InitBuffer(staging_buf, span * sizeof(int32_t));
        pipe.InitBuffer(scale_buf, span * sizeof(float));
        pipe.InitBuffer(mask_buf, span * sizeof(int32_t));
        pipe.InitBuffer(token_buf, kMaxRowsPerExpert * sizeof(float));
        pipe.InitBuffer(input_scale_buf, kMaxTokens * sizeof(float));
        pipe.InitBuffer(activation_buf,
                        kMaxRowsPerExpert * kIntermediate * sizeof(float));
        pipe.InitBuffer(maximum_buf,
                        kMaxRowsPerExpert * kBlockFloats * sizeof(float));
        pipe.InitBuffer(count_buf, kMaxExperts * sizeof(int32_t));
        pipe.InitBuffer(start_buf, kMaxExperts * sizeof(int32_t));
        pipe.InitBuffer(source_buf, kMaxRoutes * sizeof(int32_t));
        pipe.InitBuffer(slot_buf, kMaxRoutes * sizeof(int32_t));
        pipe.InitBuffer(weight_buf, kMaxRoutes * sizeof(float));
        pipe.InitBuffer(original_weight_buf, kMaxRoutes * sizeof(float));
        input = input_buf.Get<int8_t>();
        value = value_buf.Get<float>();
        other = other_buf.Get<float>();
        work = work_buf.Get<float>();
        half_value = half_buf.Get<half>();
        packed = packed_buf.Get<int4b_t>();
        staging = staging_buf.Get<int32_t>();
        scale = scale_buf.Get<float>();
        mask = mask_buf.Get<int32_t>();
        token = token_buf.Get<float>();
        input_scale = input_scale_buf.Get<float>();
        activation = activation_buf.Get<float>();
        maximum = maximum_buf.Get<float>();
        count = count_buf.Get<int32_t>();
        start = start_buf.Get<int32_t>();
        source = source_buf.Get<int32_t>();
        slot = slot_buf.Get<int32_t>();
        weight = weight_buf.Get<float>();
        original_weight = original_weight_buf.Get<float>();
        Duplicate(mask, int32_t(-8192), kVectorTile);
        PipeBarrier<PIPE_V>();
    }
};

__aicore__ inline void LoadTiling(TCubeTiling& cube, __gm__ uint8_t* bytes)
{
    auto* destination = (uint32_t*)&cube;
    auto* source = (__gm__ uint32_t*)bytes;
    for (uint32_t word = 0; word < sizeof(TCubeTiling) / sizeof(uint32_t);
         ++word) {
        destination[word] = source[word];
    }
}

template <bool pad_rows = false, class MatmulType>
__aicore__ inline void Project(MatmulType& mm, __gm__ int4b_t* rows,
                               __gm__ int4b_t* weight,
                               __gm__ RoutedProduct* product,
                               __gm__ ScaleStorage* weight_scale, int64_t columns,
                               int64_t inner, int64_t count)
{
    const int64_t matrix_rows = pad_rows ? GateMatrixRows(count) : 2 * count;
    GlobalTensor<int4b_t> a, b;
    GlobalTensor<RoutedProduct> c;
    a.SetGlobalBuffer(rows, matrix_rows * inner);
    b.SetGlobalBuffer(weight, inner * columns);
    c.SetGlobalBuffer(product, matrix_rows * columns);
    mm.SetOrgShape(matrix_rows, columns, inner);
    mm.SetSingleShape(matrix_rows, columns, inner);
    mm.SetTensorA(a);
    mm.SetTensorB(b, false);
#ifdef ROUTED_FIXPIPE_HALF
    GlobalTensor<uint64_t> scale;
    scale.SetGlobalBuffer(weight_scale, columns);
    mm.SetQuantVector(scale);
#endif
    mm.IterateAll(c);
    mm.End();
}

__aicore__ inline void SplitHalves(Workspace& held, int64_t span)
{
    Muls(held.other, held.value, 1.0F / 16.0F, span);
    PipeBarrier<PIPE_V>();
    Floor(held.other, held.other, span);
    PipeBarrier<PIPE_V>();
    Muls(held.work, held.other, -16.0F, span);
    PipeBarrier<PIPE_V>();
    Add(held.work, held.work, held.value, span);
    PipeBarrier<PIPE_V>();
    Adds(held.work, held.work, -8.0F, span);
    PipeBarrier<PIPE_V>();
    DataCopy(held.value, held.other, span);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void Emit(Workspace& held, LocalTensor<float> values,
                            GlobalTensor<int4b_t>& destination,
                            int64_t offset, int64_t count)
{
    Cast(held.half_value, values, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    Cast(held.packed, held.half_value, RoundMode::CAST_RINT, count);
    Fence<HardEvent::V_MTE3>();
    DataCopyPad(destination[offset], held.packed,
                DataCopyExtParams{1, uint32_t(count / 2), 0, 0, 0});
    Fence<HardEvent::MTE3_V>();
}

template <bool balanced = false>
__aicore__ inline void Split(Workspace& held, __gm__ int8_t* rows,
                             __gm__ int4b_t* stacked, int64_t count,
                             int64_t start, int64_t tokens,
                             int64_t first_row = 0, int64_t row_step = 1)
{
    GlobalTensor<int8_t> input;
    GlobalTensor<int4b_t> high, low;
    input.SetGlobalBuffer(rows, tokens * kHidden);
    high.SetGlobalBuffer(stacked, count * kHidden);
    low.SetGlobalBuffer(Nibbles(stacked, count * kHidden), count * kHidden);
    for (int64_t row = balanced ? first_row : 0; row < count;
         row += balanced ? row_step : 1) {
        const int64_t source = held.source.GetValue(start + row);
        for (int64_t column = 0; column < kHidden; column += kVectorTile) {
            Fence<HardEvent::V_MTE2>();
            DataCopy(held.input, input[source * kHidden + column],
                     kVectorTile);
            Fence<HardEvent::MTE2_V>();
            Cast(held.half_value, held.input, RoundMode::CAST_NONE, kVectorTile);
            PipeBarrier<PIPE_V>();
            Cast(held.value, held.half_value, RoundMode::CAST_NONE, kVectorTile);
            PipeBarrier<PIPE_V>();
            SplitHalves(held, kVectorTile);
            Emit(held, held.value, high, row * kHidden + column, kVectorTile);
            Emit(held, held.work, low, row * kHidden + column, kVectorTile);
        }
    }
}

__aicore__ inline void LoadScale(Workspace& held,
                                 GlobalTensor<float>& scales,
                                 int64_t offset, int64_t count)
{
    Fence<HardEvent::V_MTE2>();
    DataCopy(held.scale, scales[offset], count);
    Fence<HardEvent::MTE2_V>();
    // A3 Fixpipe retains 10 mantissa bits in the FP32 scale carrier.
    auto bits = held.scale.ReinterpretCast<uint16_t>();
    auto mask = held.mask.ReinterpretCast<uint16_t>();
    And(bits, bits, mask, 2 * count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ScaledPartial(Workspace& held,
                                     LocalTensor<float> destination,
                                     GlobalTensor<RoutedProduct>& product,
                                     int64_t offset, int64_t count)
{
    Fence<HardEvent::V_MTE2>();
#ifdef ROUTED_FIXPIPE_HALF
    DataCopy(held.half_value, product[offset], count);
    Fence<HardEvent::MTE2_V>();
    Cast(destination, held.half_value, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
#else
    DataCopy(held.staging, product[offset], count);
    Fence<HardEvent::MTE2_V>();
    Cast(destination, held.staging, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    Mul(destination, destination, held.scale, count);
    PipeBarrier<PIPE_V>();
    Cast(held.half_value, destination, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    Cast(destination, held.half_value, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
#endif
}

__aicore__ inline void Recombine(Workspace& held,
                                 GlobalTensor<RoutedProduct>& product,
                                 GlobalTensor<float>& correction,
                                 int64_t row, int64_t rows, int64_t columns,
                                 int64_t offset, int64_t count)
{
    ScaledPartial(held, held.value, product, row * columns + offset, count);
    ScaledPartial(held, held.work, product,
                  (rows + row) * columns + offset, count);
    Muls(held.value, held.value, 16.0F, count);
    PipeBarrier<PIPE_V>();
    Add(held.value, held.value, held.work, count);
    PipeBarrier<PIPE_V>();
    Fence<HardEvent::V_MTE2>();
    DataCopy(held.work, correction[offset], count);
    Fence<HardEvent::MTE2_V>();
    Add(held.value, held.value, held.work, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void SwiGlu(Workspace& held)
{
    DataCopy(held.other, held.value[kIntermediate], kIntermediate);
    PipeBarrier<PIPE_V>();
    Muls(held.work, held.value, -1.0F, kIntermediate);
    PipeBarrier<PIPE_V>();
    Exp(held.work, held.work, kIntermediate);
    PipeBarrier<PIPE_V>();
    Adds(held.work, held.work, 1.0F, kIntermediate);
    PipeBarrier<PIPE_V>();
    Div(held.value, held.value, held.work, kIntermediate);
    PipeBarrier<PIPE_V>();
    Mul(held.value, held.value, held.other, kIntermediate);
    PipeBarrier<PIPE_V>();
}

template <bool balanced = false>
__aicore__ inline void Activate(
    Workspace& held, int64_t count, int64_t start, __gm__ RoutedProduct* product,
    __gm__ float* correction, __gm__ ScaleStorage* scales,
    __gm__ int4b_t* middle, __gm__ float* middle_scales,
    int64_t first_row = 0, int64_t row_step = 1)
{
    GlobalTensor<RoutedProduct> dots;
    GlobalTensor<float> bias, weight_scales, output_scales;
    GlobalTensor<int4b_t> high, low;
    dots.SetGlobalBuffer(product, 2 * count * kGateUp);
    bias.SetGlobalBuffer(correction, kGateUp);
#ifndef ROUTED_FIXPIPE_HALF
    weight_scales.SetGlobalBuffer(scales, kGateUp);
#endif
    output_scales.SetGlobalBuffer(middle_scales, count);
    high.SetGlobalBuffer(middle, count * kIntermediate);
    low.SetGlobalBuffer(Nibbles(middle, count * kIntermediate),
                        count * kIntermediate);
#ifndef ROUTED_FIXPIPE_HALF
    LoadScale(held, weight_scales, 0, kGateUp);
#endif
    for (int64_t row = balanced ? first_row : 0; row < count;
         row += balanced ? row_step : 1) {
        Recombine(held, dots, bias, row, count, kGateUp, 0, kGateUp);
        const int64_t source = held.source.GetValue(start + row);
        const float activation_scale = held.input_scale.GetValue(source);
        Fence<HardEvent::S_V>();
        Muls(held.value, held.value, activation_scale, kGateUp);
        PipeBarrier<PIPE_V>();
        SwiGlu(held);
        DataCopy(held.activation[row * kIntermediate], held.value,
                 kIntermediate);
        Abs(held.work, held.value, kIntermediate);
        PipeBarrier<PIPE_V>();
        auto reduce_work = held.staging.ReinterpretCast<float>();
        ReduceMax(held.maximum[row * kBlockFloats], held.work, reduce_work,
                  kIntermediate, false);
        PipeBarrier<PIPE_V>();
    }
    Fence<HardEvent::V_S>();
    for (int64_t row = balanced ? first_row : 0; row < count;
         row += balanced ? row_step : 1) {
        const float maximum = held.maximum.GetValue(row * kBlockFloats);
        const float quantum = maximum / 127.0F;
        const float factor = maximum == 0.0F ? 0.0F : 1.0F / quantum;
        Muls(held.value, held.activation[row * kIntermediate], factor,
             kIntermediate);
        PipeBarrier<PIPE_V>();
        Mins(held.value, held.value, 127.0F, kIntermediate);
        PipeBarrier<PIPE_V>();
        Maxs(held.value, held.value, -127.0F, kIntermediate);
        PipeBarrier<PIPE_V>();
        // Fused GMM1 casts FP32 -> FP16 -> INT8. Keep that first rounding
        // before obtaining the integer to split into the two signed nibbles.
        Cast(held.half_value, held.value, RoundMode::CAST_RINT, kIntermediate);
        PipeBarrier<PIPE_V>();
        Cast(held.value, held.half_value, RoundMode::CAST_NONE, kIntermediate);
        PipeBarrier<PIPE_V>();
        Cast(held.value, held.value, RoundMode::CAST_RINT, kIntermediate);
        PipeBarrier<PIPE_V>();
        SplitHalves(held, kIntermediate);
        Emit(held, held.value, high, row * kIntermediate, kIntermediate);
        Emit(held, held.work, low, row * kIntermediate, kIntermediate);
        if constexpr (balanced) {
            // Other AIVs own adjacent scales. Write exactly four bytes from
            // an aligned UB source, then wait before reusing that source.
            held.token.SetValue(0, quantum);
            Fence<HardEvent::S_MTE3>();
            DataCopyPad(output_scales[row], held.token,
                DataCopyExtParams{1, sizeof(float), 0, 0, 0});
            Fence<HardEvent::MTE3_S>();
        } else {
            held.token.SetValue(row, quantum);
        }
    }
    if constexpr (!balanced) {
        Fence<HardEvent::S_MTE3>();
        DataCopyPad(output_scales, held.token,
                    DataCopyExtParams{1, uint32_t(count * sizeof(float)), 0, 0, 0});
        Fence<HardEvent::MTE3_S>();
    }
}

template <bool balanced = false>
__aicore__ inline void StoreRoutes(
    Workspace& held, int64_t count, int64_t start, __gm__ RoutedProduct* product,
    __gm__ float* correction, __gm__ ScaleStorage* scales,
    __gm__ float* middle_scales, __gm__ bfloat16_t* routes,
    int64_t begin, int64_t end, int64_t tokens,
    int64_t first_row = 0, int64_t row_step = 1)
{
    GlobalTensor<RoutedProduct> dots;
    GlobalTensor<float> bias, weight_scales, activation_scales;
    GlobalTensor<bfloat16_t> output;
    dots.SetGlobalBuffer(product, 2 * count * kHidden);
    bias.SetGlobalBuffer(correction, kHidden);
#ifndef ROUTED_FIXPIPE_HALF
    weight_scales.SetGlobalBuffer(scales, kHidden);
#endif
    activation_scales.SetGlobalBuffer(middle_scales, count);
    output.SetGlobalBuffer(routes, tokens * kTopK * kHidden);
    Fence<HardEvent::S_MTE2>();
    DataCopyPad(held.token, activation_scales,
                DataCopyExtParams{1, uint32_t(count * sizeof(float)), 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
    Fence<HardEvent::MTE2_S>();
    auto narrowed = held.half_buf.Get<bfloat16_t>();
    for (int64_t column = begin; column < end; column += kVectorTile) {
        const int64_t span = end - column < kVectorTile
            ? end - column : kVectorTile;
#ifndef ROUTED_FIXPIPE_HALF
        LoadScale(held, weight_scales, column, span);
#endif
        for (int64_t row = balanced ? first_row : 0; row < count;
             row += balanced ? row_step : 1) {
            Recombine(held, dots, bias, row, count, kHidden, column, span);
            Muls(held.value, held.value, held.token.GetValue(row), span);
            PipeBarrier<PIPE_V>();
            Cast(narrowed, held.value, RoundMode::CAST_RINT, span);
            const int64_t source = held.source.GetValue(start + row);
            const int64_t slot = held.slot.GetValue(start + row);
            const int64_t offset = (source * kTopK + slot) * kHidden + column;
            Fence<HardEvent::V_MTE3>();
            DataCopy(output[offset], narrowed, span);
            Fence<HardEvent::MTE3_V>();
        }
    }
}

__aicore__ inline void CombineTokens(Workspace& held,
                                      __gm__ bfloat16_t* route_output,
                                      __gm__ bfloat16_t* out,
                                      int64_t tokens, int64_t blocks)
{
    GlobalTensor<bfloat16_t> routes, output;
    routes.SetGlobalBuffer(route_output, tokens * kTopK * kHidden);
    output.SetGlobalBuffer(out, tokens * kHidden);
    auto values = held.half_buf.Get<bfloat16_t>();
    const int64_t tiles_per_token = kHidden / kVectorTile;
    const int64_t vector_cores = 2 * blocks;
    for (int64_t tile = GetBlockIdx(); tile < tokens * tiles_per_token;
         tile += vector_cores) {
        const int64_t token = tile / tiles_per_token;
        const int64_t column = (tile % tiles_per_token) * kVectorTile;
        Duplicate(held.value, 0.0F, kVectorTile);
        PipeBarrier<PIPE_V>();
        for (int64_t slot = 0; slot < kTopK; ++slot) {
            const int64_t original = token * kTopK + slot;
            const float probability = held.original_weight.GetValue(original);
            // Match unpermute's zero-probability behavior, including NaNs.
            if (probability == 0.0F) {
                continue;
            }
            Fence<HardEvent::V_MTE2>();
            DataCopy(values, routes[original * kHidden + column], kVectorTile);
            Fence<HardEvent::MTE2_V>();
            // CANN seeds from original slot 0, preserving a negative zero.
            // A skipped slot 0 leaves +0; later slots are always additions.
            auto term = slot == 0 ? held.value : held.other;
            Cast(term, values, RoundMode::CAST_NONE, kVectorTile);
            PipeBarrier<PIPE_V>();
            Muls(term, term, probability, kVectorTile);
            PipeBarrier<PIPE_V>();
            if (slot != 0) {
                Add(held.value, held.value, term, kVectorTile);
                PipeBarrier<PIPE_V>();
            }
        }
        Cast(values, held.value, RoundMode::CAST_RINT, kVectorTile);
        Fence<HardEvent::V_MTE3>();
        DataCopy(output[token * kHidden + column], values, kVectorTile);
        Fence<HardEvent::MTE3_V>();
    }
}

}  // namespace

template <CubeFormat weight_format, bool from_dispatch>
__aicore__ inline void Forward(
    TPipe& pipe, const ffn_descriptor::BodyArguments& arguments)
{
    Timing timing;
    timing.Begin(0);
#ifdef ROUTED_SHARED_PIPELINE
    static_assert(from_dispatch, "the shared pipeline requires input preparation");
    pipeline::Stages shared(arguments.shared, arguments.rows, arguments.blocks);
    shared.StageInput(pipe);
#endif
    if constexpr (from_dispatch) {
#ifdef ROUTED_INPUT_FRONT
        if (arguments.dispatch.bf16_rows != 0) {
            front::Prepare(pipe, arguments.dispatch, ((__gm__ int8_t*)arguments.scratch[kRestoredRows]), ((__gm__ float*)arguments.scratch[kRestoredScale]), ((__gm__ int32_t*)arguments.scratch[kGroupCount]),
                           ((__gm__ int32_t*)arguments.scratch[kGroupStart]), ((__gm__ int32_t*)arguments.scratch[kSourceRow]), ((__gm__ int32_t*)arguments.scratch[kSourceSlot]), ((__gm__ float*)arguments.scratch[kRowWeight]),
                           arguments.rows, arguments.blocks, timing);
        } else
#endif
        {
            dispatch::Restore(pipe, arguments.dispatch, ((__gm__ int8_t*)arguments.scratch[kRestoredRows]), ((__gm__ float*)arguments.scratch[kRestoredScale]), ((__gm__ int32_t*)arguments.scratch[kGroupCount]),
                              ((__gm__ int32_t*)arguments.scratch[kGroupStart]), ((__gm__ int32_t*)arguments.scratch[kSourceRow]), ((__gm__ int32_t*)arguments.scratch[kSourceSlot]), ((__gm__ float*)arguments.scratch[kRowWeight]),
                              arguments.rows, arguments.blocks, timing);
        }
    }
    timing.Begin(1);
    Workspace held;
#ifdef ROUTED_SHARED_PIPELINE
    shared.InitResident(pipe);
    shared.Gate(pipe);
#endif
    GlobalTensor<int32_t> counts, starts;
    if ASCEND_IS_AIC {
        counts.SetGlobalBuffer(((__gm__ int32_t*)arguments.scratch[kGroupCount]), kMaxExperts);
        starts.SetGlobalBuffer(((__gm__ int32_t*)arguments.scratch[kGroupStart]), kMaxExperts);
        // Replayed graphs update counts at the same GM address. Scalar
        // loads must discard the preceding replay's data-cache lines.
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(counts);
    }
    if ASCEND_IS_AIV {
        held.Init(pipe);
        GlobalTensor<int32_t> sources, slots;
        GlobalTensor<float> weights, input_scales;
        counts.SetGlobalBuffer(((__gm__ int32_t*)arguments.scratch[kGroupCount]), kMaxExperts);
        starts.SetGlobalBuffer(((__gm__ int32_t*)arguments.scratch[kGroupStart]), kMaxExperts);
        sources.SetGlobalBuffer(((__gm__ int32_t*)arguments.scratch[kSourceRow]), kMaxRoutes);
        slots.SetGlobalBuffer(((__gm__ int32_t*)arguments.scratch[kSourceSlot]), kMaxRoutes);
        weights.SetGlobalBuffer(((__gm__ float*)arguments.scratch[kRowWeight]), kMaxRoutes);
        input_scales.SetGlobalBuffer(((__gm__ float*)arguments.scratch[kRestoredScale]), arguments.rows);
        DataCopy(held.count, counts, kMaxExperts);
        DataCopy(held.start, starts, kMaxExperts);
        DataCopy(held.source, sources, kMaxRoutes);
        DataCopy(held.slot, slots, kMaxRoutes);
        DataCopy(held.weight, weights, kMaxRoutes);
        DataCopyPad(held.input_scale, input_scales,
                    DataCopyExtParams{1, uint32_t(arguments.rows * sizeof(float)),
                                      0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_S>();
        for (int64_t route = 0; route < arguments.rows * kTopK; ++route) {
            const int64_t original = held.source.GetValue(route) * kTopK
                                     + held.slot.GetValue(route);
            held.original_weight.SetValue(original, held.weight.GetValue(route));
        }
    }

    // Every core reaches every barrier, including empty expert owners.
    if ASCEND_IS_AIV {
#ifdef ROUTED_ROW_BALANCED
        const int64_t workers = 2 * arguments.blocks;
        int64_t next = GetBlockIdx();
        for (int64_t expert = 0; expert < ffn_bridge::kExperts; ++expert) {
            const int64_t count = held.count.GetValue(expert);
            const int64_t begin = held.start.GetValue(expert);
            if (next >= begin + count) {
                continue;
            }
            Split<true>(held, ((__gm__ int8_t*)arguments.scratch[kRestoredRows]), Nibbles(((__gm__ int4b_t*)arguments.scratch[kStacked]), GateStorageRow(begin) * kHidden),
                        count, begin, arguments.rows, next - begin, workers);
            do {
                next += workers;
            } while (next < begin + count);
        }
        Publish();
#else
        if (GetSubBlockIdx() == 0) {
            int64_t active = 0;
            int64_t next = GetBlockIdx() / 2;
            for (int64_t expert = 0; expert < ffn_bridge::kExperts; ++expert) {
                if (held.count.GetValue(expert) == 0) {
                    continue;
                }
                if (active++ != next) {
                    continue;
                }
                next += arguments.blocks;
                const int64_t begin = held.start.GetValue(expert);
                Split(held, ((__gm__ int8_t*)arguments.scratch[kRestoredRows]), Nibbles(((__gm__ int4b_t*)arguments.scratch[kStacked]), GateStorageRow(begin) * kHidden),
                      held.count.GetValue(expert), held.start.GetValue(expert),
                      arguments.rows);
            }
            Publish();
        }
#endif
    }
    timing.Barrier(1);
    timing.Begin(2);
#ifdef ROUTED_SHARED_PIPELINE
    shared.Activate();
#endif
    if ASCEND_IS_AIC {
#ifdef ROUTED_SHARED_PIPELINE
        // The shared gate projection ended before barrier 1.
        pipe.Reset();
#endif
        TCubeTiling gate_cube;
        LoadTiling(gate_cube, ((__gm__ uint8_t*)arguments.scratch[kGateTiling]));
        GateProjection<weight_format> gate_mm;
        gate_mm.Init(&gate_cube, &pipe);
        int64_t active = 0;
        int64_t next = GetBlockIdx();
        for (int64_t expert = 0; expert < ffn_bridge::kExperts; ++expert) {
            if (counts.GetValue(expert) == 0) {
                continue;
            }
            if (active++ != next) {
                continue;
            }
            next += arguments.blocks;
            const int64_t begin = starts.GetValue(expert);
            Project<kPadGateRows>(gate_mm,
                    Nibbles(((__gm__ int4b_t*)arguments.scratch[kStacked]), GateStorageRow(begin) * kHidden),
                    Nibbles(((__gm__ int4b_t*)arguments.weights[0]), expert * kGateUp * kHidden),
                    ((__gm__ RoutedProduct*)arguments.scratch[kGateProduct]) + GateStorageRow(begin) * kGateUp,
                    ((__gm__ ScaleStorage*)arguments.weights[2]) + expert * kGateUp, kGateUp, kHidden,
                    counts.GetValue(expert));
        }
        Publish();
    }
    timing.Barrier(2);
    timing.Begin(3);
#ifdef ROUTED_SHARED_PIPELINE
    shared.Down(pipe);
#endif
    if ASCEND_IS_AIV {
#ifdef ROUTED_ROW_BALANCED
        const int64_t workers = 2 * arguments.blocks;
        int64_t next = GetBlockIdx();
        for (int64_t expert = 0; expert < ffn_bridge::kExperts; ++expert) {
            const int64_t count = held.count.GetValue(expert);
            const int64_t begin = held.start.GetValue(expert);
            if (next >= begin + count) {
                continue;
            }
            Activate<true>(held, count, begin,
                ((__gm__ RoutedProduct*)arguments.scratch[kGateProduct]) + GateStorageRow(begin) * kGateUp,
                ((__gm__ float*)arguments.weights[1]) + expert * kGateUp, ((__gm__ ScaleStorage*)arguments.weights[2]) + expert * kGateUp,
                Nibbles(((__gm__ int4b_t*)arguments.scratch[kMiddle]), 2 * begin * kIntermediate),
                ((__gm__ float*)arguments.scratch[kMiddleScale]) + begin, next - begin, workers);
            do {
                next += workers;
            } while (next < begin + count);
        }
        Publish();
#else
        if (GetSubBlockIdx() == 0) {
            int64_t active = 0;
            int64_t next = GetBlockIdx() / 2;
            for (int64_t expert = 0; expert < ffn_bridge::kExperts; ++expert) {
                if (held.count.GetValue(expert) == 0) {
                    continue;
                }
                if (active++ != next) {
                    continue;
                }
                next += arguments.blocks;
                const int64_t begin = held.start.GetValue(expert);
                Activate(held, held.count.GetValue(expert),
                         held.start.GetValue(expert),
                         ((__gm__ RoutedProduct*)arguments.scratch[kGateProduct]) + GateStorageRow(begin) * kGateUp,
                         ((__gm__ float*)arguments.weights[1]) + expert * kGateUp, ((__gm__ ScaleStorage*)arguments.weights[2]) + expert * kGateUp,
                         Nibbles(((__gm__ int4b_t*)arguments.scratch[kMiddle]), 2 * begin * kIntermediate),
                         ((__gm__ float*)arguments.scratch[kMiddleScale]) + begin);
            }
            Publish();
        }
#endif
    }
    timing.Barrier(3);
    timing.Begin(4);
#ifdef ROUTED_SHARED_PIPELINE
    shared.Store(pipe);
#endif
    if ASCEND_IS_AIC {
        // Gate matmul has ended on every AIC before reusing L1 and events.
        pipe.Reset();
        TCubeTiling down_cube;
        LoadTiling(down_cube, ((__gm__ uint8_t*)arguments.scratch[kDownTiling]));
        Projection<weight_format> down_mm;
        down_mm.Init(&down_cube, &pipe);
        int64_t active = 0;
        int64_t next = GetBlockIdx();
        for (int64_t expert = 0; expert < ffn_bridge::kExperts; ++expert) {
            if (counts.GetValue(expert) == 0) {
                continue;
            }
            if (active++ != next) {
                continue;
            }
            next += arguments.blocks;
            const int64_t begin = starts.GetValue(expert);
            Project(down_mm, Nibbles(((__gm__ int4b_t*)arguments.scratch[kMiddle]), 2 * begin * kIntermediate),
                    Nibbles(((__gm__ int4b_t*)arguments.weights[3]), expert * kHidden * kIntermediate),
                    ((__gm__ RoutedProduct*)arguments.scratch[kDownProduct]) + 2 * begin * kHidden,
                    ((__gm__ ScaleStorage*)arguments.weights[5]) + expert * kHidden, kHidden, kIntermediate,
                    counts.GetValue(expert));
        }
        Publish();
    }
    timing.Barrier(4);
    timing.Begin(5);
    if ASCEND_IS_AIV {
        const int64_t begin = GetSubBlockIdx() * (kHidden / 2);
#ifdef ROUTED_ROW_BALANCED
        // Both AIVs in a group retain separate H/2 columns for each row.
        int64_t next = GetBlockIdx() / 2;
        for (int64_t expert = 0; expert < ffn_bridge::kExperts; ++expert) {
            const int64_t count = held.count.GetValue(expert);
            const int64_t route_begin = held.start.GetValue(expert);
            if (next >= route_begin + count) {
                continue;
            }
            StoreRoutes<true>(held, count, route_begin,
                ((__gm__ RoutedProduct*)arguments.scratch[kDownProduct]) + 2 * route_begin * kHidden,
                ((__gm__ float*)arguments.weights[4]) + expert * kHidden, ((__gm__ ScaleStorage*)arguments.weights[5]) + expert * kHidden,
                ((__gm__ float*)arguments.scratch[kMiddleScale]) + route_begin, ((__gm__ bfloat16_t*)arguments.scratch[kRouteOutput]),
                begin, begin + kHidden / 2, arguments.rows, next - route_begin, arguments.blocks);
            do {
                next += arguments.blocks;
            } while (next < route_begin + count);
        }
#else
        int64_t active = 0;
        int64_t next = GetBlockIdx() / 2;
        for (int64_t expert = 0; expert < ffn_bridge::kExperts; ++expert) {
            if (held.count.GetValue(expert) == 0) {
                continue;
            }
            if (active++ != next) {
                continue;
            }
            next += arguments.blocks;
            const int64_t route_begin = held.start.GetValue(expert);
            StoreRoutes(held, held.count.GetValue(expert),
                        held.start.GetValue(expert),
                        ((__gm__ RoutedProduct*)arguments.scratch[kDownProduct]) + 2 * route_begin * kHidden,
                        ((__gm__ float*)arguments.weights[4]) + expert * kHidden,
                        ((__gm__ ScaleStorage*)arguments.weights[5]) + expert * kHidden,
                        ((__gm__ float*)arguments.scratch[kMiddleScale]) + route_begin, ((__gm__ bfloat16_t*)arguments.scratch[kRouteOutput]),
                        begin, begin + kHidden / 2, arguments.rows);
        }
#endif
        Publish();
    }
    timing.Barrier(5);
    timing.Begin(6);
    if ASCEND_IS_AIV {
        CombineTokens(held, ((__gm__ bfloat16_t*)arguments.scratch[kRouteOutput]), ((__gm__ bfloat16_t*)arguments.scratch[kOutput]), arguments.rows, arguments.blocks);
    }
    timing.FinishFinal(6);
    timing.Store(((__gm__ int64_t*)nullptr), arguments.rows, arguments.blocks);
}

}  // namespace routed_body

#endif
