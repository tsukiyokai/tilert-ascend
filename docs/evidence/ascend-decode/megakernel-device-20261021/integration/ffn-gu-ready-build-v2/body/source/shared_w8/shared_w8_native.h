#ifndef TESTS_GLM52_SHARED_W8_NATIVE_H
#define TESTS_GLM52_SHARED_W8_NATIVE_H

#include "shared_w8.h"
#include "../mla_tail/mla_tail_dequant.h"

namespace shared_w8::device {
using namespace AscendC;
using mla_tail::device::Fence;
using mla_tail::device::LoadTiling;
using mla_tail::device::Publish;

struct Buffers {
    TBuf<TPosition::VECCALC> input_buf, sum_buf, term_buf, scale_buf;
    TBuf<TPosition::VECCALC> value_buf, work_buf, maximum_buf, half_buf, code_buf;
    LocalTensor<int8_t> input, codes;
    LocalTensor<int32_t> sum, term;
    LocalTensor<float> scales, values, work, maximum;
    LocalTensor<half> halves;

    __aicore__ inline void Init(TPipe& pipe)
    {
        pipe.InitBuffer(input_buf, kInnerTile);
        pipe.InitBuffer(sum_buf, kGateColumns * 4);
        pipe.InitBuffer(term_buf, kGateColumns * 4);
        pipe.InitBuffer(scale_buf, (kGateColumns + kScaleStride) * 4);
        pipe.InitBuffer(value_buf, kGateColumns * 4);
        pipe.InitBuffer(work_buf, kGateColumns * 4);
        pipe.InitBuffer(maximum_buf, kScaleStride * 4);
        pipe.InitBuffer(half_buf, kIntermediate * 2);
        pipe.InitBuffer(code_buf, kIntermediate);
        input = input_buf.Get<int8_t>();
        sum = sum_buf.Get<int32_t>();
        term = term_buf.Get<int32_t>();
        scales = scale_buf.Get<float>();
        values = value_buf.Get<float>();
        work = work_buf.Get<float>();
        maximum = maximum_buf.Get<float>();
        halves = half_buf.Get<half>();
        codes = code_buf.Get<int8_t>();
    }
};

__aicore__ inline void StageInput(Buffers& held, __gm__ int8_t* input,
    __gm__ int8_t* staged, int64_t rows, int64_t capacity, int64_t vectors)
{
    GlobalTensor<int8_t> source, destination;
    source.SetGlobalBuffer(input, rows * kHidden);
    destination.SetGlobalBuffer(staged, capacity * kHidden);
    for (int64_t task = GetBlockIdx(); task < kInnerParts * capacity;
         task += vectors) {
        const int64_t part = task / capacity;
        const int64_t row = task % capacity;
        if (row < rows) {
            DataCopy(held.input, source[row * kHidden + part * kInnerTile],
                     kInnerTile);
            Fence<HardEvent::MTE2_MTE3>();
        } else {
            Duplicate(held.input.ReinterpretCast<int16_t>(), int16_t(0),
                      kInnerTile / 2);
            Fence<HardEvent::V_MTE3>();
        }
        DataCopy(destination[task * kInnerTile], held.input, kInnerTile);
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::V_MTE2>();
    }
}

using Operand = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
using Product = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

__aicore__ inline void Project(TPipe& pipe, __gm__ int8_t* input,
    __gm__ int8_t* weights, __gm__ int32_t* output, __gm__ uint8_t* tiling,
    int64_t capacity, int64_t blocks, bool gate)
{
    TCubeTiling cube;
    LoadTiling(cube, tiling);
    Matmul<Operand, Operand, Product, Product> mm;
    mm.Init(&cube, &pipe);
    const int64_t tasks = gate ? kInnerParts * kGateParts : kDownParts;
    const int64_t columns = gate ? kGateTile : kDownTile;
    const int64_t inner = gate ? kInnerTile : kIntermediate;
    GlobalTensor<int8_t> a, b;
    GlobalTensor<int32_t> c;
    for (int64_t task = GetBlockIdx(); task < tasks; task += blocks) {
        const int64_t input_part = gate ? task / kGateParts : 0;
        a.SetGlobalBuffer(input + input_part * capacity * inner,
                          capacity * inner);
        b.SetGlobalBuffer(weights + task * inner * columns, inner * columns);
        c.SetGlobalBuffer(output + task * capacity * columns, capacity * columns);
        mm.SetOrgShape(capacity, columns, inner);
        mm.SetTensorA(a);
        mm.SetTensorB(b, false);
        mm.IterateAll(c);
        mm.End();
    }
}

__aicore__ inline void Activate(Buffers& held, __gm__ int32_t* partial,
    __gm__ float* channel_scale, __gm__ float* input_scale,
    __gm__ int8_t* middle, __gm__ float* middle_scale,
    int64_t rows, int64_t capacity, int64_t vectors)
{
    GlobalTensor<int32_t> source;
    GlobalTensor<float> channels, tokens, out_scale;
    GlobalTensor<int8_t> output;
    source.SetGlobalBuffer(partial, kInnerParts * kGateColumns * capacity);
    channels.SetGlobalBuffer(channel_scale, kGateColumns);
    tokens.SetGlobalBuffer(input_scale, rows);
    output.SetGlobalBuffer(middle, capacity * kIntermediate);
    out_scale.SetGlobalBuffer(middle_scale, capacity * kScaleStride);
    DataCopyParams load_parts;
    load_parts.blockCount = kGateParts;
    load_parts.blockLen = kGateTile * sizeof(int32_t) / 32;
    load_parts.srcStride = (capacity - 1) * kGateTile * sizeof(int32_t) / 32;
    load_parts.dstStride = 0;
    for (int64_t row = GetBlockIdx(); row < capacity; row += vectors) {
        if (row >= rows) {
            Duplicate(held.codes.ReinterpretCast<int16_t>(), int16_t(0),
                      kIntermediate / 2);
            Duplicate(held.maximum, 0.0F, kScaleStride);
        } else {
            DataCopy(held.sum, source[row * kGateTile], load_parts);
            Fence<HardEvent::MTE2_V>();
            for (int64_t part = 1; part < kInnerParts; ++part) {
                DataCopy(held.term,
                    source[(part * kGateParts * capacity + row) * kGateTile],
                    load_parts);
                Fence<HardEvent::MTE2_V>();
                Add(held.sum, held.sum, held.term, kGateColumns);
                Fence<HardEvent::V_MTE2>();
            }
            DataCopy(held.scales, channels, kGateColumns);
            DataCopyPad(held.scales[kGateColumns], tokens[row],
                DataCopyExtParams{1, sizeof(float), 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
            Fence<HardEvent::MTE2_V>();
            Fence<HardEvent::MTE2_S>();
            // INT8 products and their six-way sum fit exactly in INT32:
            // 6144 * 128 * 128 = 100663296 < 2^31.
            Cast(held.values, held.sum, RoundMode::CAST_NONE, kGateColumns);
            PipeBarrier<PIPE_V>();
            Mul(held.values, held.values, held.scales, kGateColumns);
            PipeBarrier<PIPE_V>();
            Muls(held.values, held.values,
                 held.scales.GetValue(kGateColumns), kGateColumns);
            PipeBarrier<PIPE_V>();
            Muls(held.work, held.values, -1.0F, kIntermediate);
            PipeBarrier<PIPE_V>();
            Exp(held.work, held.work, kIntermediate);
            PipeBarrier<PIPE_V>();
            Adds(held.work, held.work, 1.0F, kIntermediate);
            PipeBarrier<PIPE_V>();
            Div(held.values, held.values, held.work, kIntermediate);
            PipeBarrier<PIPE_V>();
            Mul(held.values, held.values, held.values[kIntermediate],
                kIntermediate);
            PipeBarrier<PIPE_V>();
            Abs(held.work, held.values, kIntermediate);
            Duplicate(held.maximum, 0.0F, kScaleStride);
            PipeBarrier<PIPE_V>();
            ReduceMax(held.maximum, held.work, held.work[kIntermediate],
                      kIntermediate, false);
            PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_S>();
            // CANN 9.2 key 30013: scalar scale and reciprocal multiplication.
            const float scale = held.maximum.GetValue(0) / 127.0F;
            held.maximum.SetValue(0, scale);
            const float factor = scale == 0.0F ? 1.0F : 1.0F / scale;
            Fence<HardEvent::S_V>();
            Muls(held.values, held.values, factor, kIntermediate);
            PipeBarrier<PIPE_V>();
            auto integers = held.sum.ReinterpretCast<int16_t>();
            Cast(integers, held.values, RoundMode::CAST_RINT, kIntermediate);
            PipeBarrier<PIPE_V>();
            Cast(held.halves, integers, RoundMode::CAST_NONE, kIntermediate);
            PipeBarrier<PIPE_V>();
            Cast(held.codes, held.halves, RoundMode::CAST_NONE, kIntermediate);
        }
        Fence<HardEvent::V_MTE3>();
        DataCopy(output[row * kIntermediate], held.codes, kIntermediate);
        DataCopy(out_scale[row * kScaleStride], held.maximum, kScaleStride);
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::V_MTE2>();
    }
}
}  // namespace shared_w8::device

#endif
