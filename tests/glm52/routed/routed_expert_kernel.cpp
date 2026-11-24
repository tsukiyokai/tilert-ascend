/**
 * The routed experts as one MIX kernel, split on the expert axis.
 *
 * The shape of this follows from three measurements rather than from taste.
 * The cube's cost does not depend on how many rows it gets -- 10.742 us at two
 * rows against 11.416 at thirty-two -- so a per-expert matmul at about 10 us,
 * times 44 activated experts, would be 440 us a layer against the vendor's
 * 165 for both grouped matmuls. What the cube is short of is cores: the same
 * 1.57 MB of weights takes 20.610 us on one block, 13.120 on two and 10.795
 * on four, which is 76, 120 and 145 GB/s.
 *
 * So the experts are spread over the cores and each core takes its own at full
 * width. The consequence worth the most is a simplification: a permuted row
 * belongs to exactly one expert and therefore to exactly one core, so the
 * per-token maximum for the intermediate's dynamic quantisation is core-local
 * and this kernel needs no grid barrier at all -- neither of the two
 * `SyncAll<false>()` calls the shared expert's chain needs, and none of its
 * cross-block reduction.
 *
 * Rows a expert are fixed here rather than read from a group list. The cube
 * does not care, by the table above, and the host side of a variable group
 * list is a separate change; 2 is the measured average of 1.45 rounded up.
 *
 * [routed-split-20261020](../../../docs/evidence/ascend-decode/routed-split-20261020/README.md)
 */
#define ASCENDC_CUBE_ONLY

#include "routed_expert_kernel.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

constexpr int64_t HIDDEN = 6144;
constexpr int64_t HALF = 128;
constexpr int64_t GATE_UP = 2 * HALF;
#ifndef ROWS_PER_EXPERT
#define ROWS_PER_EXPERT 2
#endif
constexpr int64_t ROWS = ROWS_PER_EXPERT;
// One vector repeat of FP32, in elements; buffers carry it as slack because
// the operators round their count up to a whole repeat.
constexpr int64_t SLACK = 64;
// A reduction writes a whole 32-byte block, so each row gets its own.
constexpr int64_t BLOCK_FLOATS = 8;
// The scales of a expert travel as whole blocks.
constexpr int64_t SCALE_SPAN = ((ROWS + BLOCK_FLOATS - 1) / BLOCK_FLOATS)
                               * BLOCK_FLOATS;
// One tile of the 6144-wide output a vector core walks at a time.
constexpr int64_t TILE = 2048;

// Flag ids: A3 has sixteen a core, `SyncAll` takes 11 to 13, and the Matmul
// library uses some of the low ones for its own control.
constexpr uint16_t GATE_READY = 4;
constexpr uint16_t MIDDLE_READY = 5;
constexpr uint16_t DOWN_READY = 6;

using AQuant = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
using BQuant = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, true>;
using CProduct = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using BiasNone = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

namespace {

/** Make this core's global writes visible to the others.
 *
 * On architecture 2201 the L2 cache is shared and only the Scalar unit's
 * DCache is per-core, so a cube-to-vector handoff through global memory has
 * no coherence problem. What it needs is completion ordering, and
 * `PipeBarrier` is an issue-order barrier only.
 */
__aicore__ inline void Publish()
{
    DataSyncBarrier<MemDsbT::DDR>();
}

__aicore__ inline void LoadTiling(TCubeTiling& cube, __gm__ uint8_t* tiling)
{
    auto* destination = (uint32_t*)&cube;
    auto* source = (__gm__ uint32_t*)tiling;
    for (uint32_t word = 0; word < sizeof(TCubeTiling) / sizeof(uint32_t);
         ++word) {
        destination[word] = source[word];
    }
}

using Projection = Matmul<AQuant, BQuant, CProduct, BiasNone>;

/** One expert's gate-and-up projection, at full width. */
__aicore__ inline void ProjectGateUp(Projection& mm, __gm__ int8_t* rows,
                                     __gm__ int8_t* weight,
                                     __gm__ int32_t* product)
{
    GlobalTensor<int8_t> a;
    GlobalTensor<int8_t> b;
    GlobalTensor<int32_t> c;
    a.SetGlobalBuffer(rows, ROWS * HIDDEN);
    b.SetGlobalBuffer(weight, GATE_UP * HIDDEN);
    c.SetGlobalBuffer(product, ROWS * GATE_UP);
    mm.SetOrgShape(ROWS, GATE_UP, HIDDEN);
    mm.SetTensorA(a);
    mm.SetTensorB(b, true);
    mm.IterateAll(c);
}

/** One expert's down projection, at full width. */
__aicore__ inline void ProjectDown(Projection& mm, __gm__ int8_t* codes,
                                   __gm__ int8_t* weight,
                                   __gm__ int32_t* product)
{
    GlobalTensor<int8_t> a;
    GlobalTensor<int8_t> b;
    GlobalTensor<int32_t> c;
    a.SetGlobalBuffer(codes, ROWS * HALF);
    b.SetGlobalBuffer(weight, HIDDEN * HALF);
    c.SetGlobalBuffer(product, ROWS * HIDDEN);
    mm.SetOrgShape(ROWS, HIDDEN, HALF);
    mm.SetTensorA(a);
    mm.SetTensorB(b, true);
    mm.IterateAll(c);
}

/** The vector state one expert's activation and quantisation needs. */
struct Epilogue {
    TBuf<TPosition::VECCALC> product_buffer;
    TBuf<TPosition::VECCALC> gate_buffer;
    TBuf<TPosition::VECCALC> up_buffer;
    TBuf<TPosition::VECCALC> work_buffer;
    TBuf<TPosition::VECCALC> scale_buffer;
    TBuf<TPosition::VECCALC> token_buffer;
    TBuf<TPosition::VECCALC> rowmax_buffer;
    TBuf<TPosition::VECCALC> activation_buffer;
    TBuf<TPosition::VECCALC> half_buffer;
    TBuf<TPosition::VECCALC> code_buffer;
    LocalTensor<int32_t> product;
    LocalTensor<float> gate;
    LocalTensor<float> up;
    LocalTensor<float> work;
    LocalTensor<float> scale;
    LocalTensor<float> token;
    LocalTensor<float> rowmax;
    LocalTensor<float> activation;
    LocalTensor<half> half_value;
    LocalTensor<int8_t> code;

    __aicore__ inline void Init(TPipe& pipe)
    {
        pipe.InitBuffer(product_buffer,
                        (ROWS * GATE_UP + SLACK) * sizeof(int32_t));
        pipe.InitBuffer(gate_buffer, (HALF + SLACK) * sizeof(float));
        pipe.InitBuffer(up_buffer, (HALF + SLACK) * sizeof(float));
        pipe.InitBuffer(work_buffer, (HALF + SLACK) * sizeof(float));
        pipe.InitBuffer(scale_buffer, (GATE_UP + SLACK) * sizeof(float));
        pipe.InitBuffer(token_buffer, (SCALE_SPAN + SLACK) * sizeof(float));
        pipe.InitBuffer(rowmax_buffer,
                        ROWS * BLOCK_FLOATS * sizeof(float));
        pipe.InitBuffer(activation_buffer,
                        (ROWS * HALF + SLACK) * sizeof(float));
        pipe.InitBuffer(half_buffer,
                        (ROWS * HALF + SLACK) * sizeof(half));
        pipe.InitBuffer(code_buffer,
                        (ROWS * HALF + SLACK) * sizeof(int8_t));
        product = product_buffer.Get<int32_t>();
        gate = gate_buffer.Get<float>();
        up = up_buffer.Get<float>();
        work = work_buffer.Get<float>();
        scale = scale_buffer.Get<float>();
        token = token_buffer.Get<float>();
        rowmax = rowmax_buffer.Get<float>();
        activation = activation_buffer.Get<float>();
        half_value = half_buffer.Get<half>();
        code = code_buffer.Get<int8_t>();
    }
};

/** `silu(gate) * up`, in place in `gate`. */
__aicore__ inline void SwiGlu(Epilogue& held)
{
    Muls(held.work, held.gate, -1.0F, HALF);
    PipeBarrier<PIPE_V>();
    Exp(held.work, held.work, HALF);
    PipeBarrier<PIPE_V>();
    Adds(held.work, held.work, 1.0F, HALF);
    PipeBarrier<PIPE_V>();
    Div(held.gate, held.gate, held.work, HALF);
    PipeBarrier<PIPE_V>();
    Mul(held.gate, held.gate, held.up, HALF);
}

/** Dequantise, activate and requantise one expert's rows.
 *
 * The row maximum is over this expert's own 128 columns and this core owns
 * every one of them, so there is nothing to reduce across blocks.
 */
__aicore__ inline void Activate(TPipe& pipe, __gm__ int32_t* product,
                                __gm__ float* gu_scale,
                                __gm__ float* token_scale,
                                __gm__ int8_t* codes, __gm__ float* out_scale)
{
    Epilogue held;
    held.Init(pipe);
    GlobalTensor<int32_t> product_global;
    GlobalTensor<float> scale_global;
    GlobalTensor<float> token_global;
    GlobalTensor<int8_t> code_global;
    GlobalTensor<float> out_scale_global;
    product_global.SetGlobalBuffer(product, ROWS * GATE_UP);
    scale_global.SetGlobalBuffer(gu_scale, GATE_UP);
    token_global.SetGlobalBuffer(token_scale, SCALE_SPAN);
    code_global.SetGlobalBuffer(codes, ROWS * HALF);
    out_scale_global.SetGlobalBuffer(out_scale, SCALE_SPAN);

    DataCopy(held.scale, scale_global, GATE_UP);
    DataCopy(held.token, token_global, SCALE_SPAN);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::UB>();

    for (int64_t row = 0; row < ROWS; ++row) {
        DataCopy(held.product, product_global[row * GATE_UP], GATE_UP);
        PipeBarrier<PIPE_ALL>();
        Cast(held.gate, held.product, RoundMode::CAST_NONE, HALF);
        Cast(held.up, held.product[HALF], RoundMode::CAST_NONE, HALF);
        PipeBarrier<PIPE_V>();
        Mul(held.gate, held.gate, held.scale, HALF);
        Mul(held.up, held.up, held.scale[HALF], HALF);
        PipeBarrier<PIPE_V>();
        const float of_row = held.token.GetValue(row);
        Muls(held.gate, held.gate, of_row, HALF);
        Muls(held.up, held.up, of_row, HALF);
        PipeBarrier<PIPE_V>();
        SwiGlu(held);
        PipeBarrier<PIPE_V>();
        // Keep the activation for the second pass over this row, and its
        // largest magnitude for the scale.
        DataCopy(held.activation[row * HALF], held.gate, HALF);
        Abs(held.work, held.gate, HALF);
        PipeBarrier<PIPE_V>();
        ReduceMax(held.rowmax[row * BLOCK_FLOATS], held.work, held.work, HALF,
                  false);
    }
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::UB>();

    for (int64_t row = 0; row < ROWS; ++row) {
        const float largest = held.rowmax.GetValue(row * BLOCK_FLOATS);
        const float quantum = largest / 127.0F;
        held.token.SetValue(row, quantum);
        Muls(held.gate, held.activation[row * HALF], 1.0F / quantum, HALF);
        PipeBarrier<PIPE_V>();
        Cast(held.half_value, held.gate, RoundMode::CAST_RINT, HALF);
        PipeBarrier<PIPE_V>();
        Cast(held.code[row * HALF], held.half_value, RoundMode::CAST_RINT,
             HALF);
    }
    PipeBarrier<PIPE_ALL>();
    DataCopy(code_global, held.code, ROWS * HALF);
    // The scale leaves by MTE. A scalar store to global memory from a vector
    // core does not land -- it goes through the one incoherent cache here.
    DataCopy(out_scale_global, held.token, SCALE_SPAN);
}

/** Scale one expert's down product into BF16. */
__aicore__ inline void Narrow(TPipe& pipe, __gm__ int32_t* product,
                              __gm__ float* down_scale,
                              __gm__ float* middle_scale,
                              __gm__ bfloat16_t* out, int64_t from,
                              int64_t upto)
{
    TBuf<TPosition::VECCALC> product_buffer;
    TBuf<TPosition::VECCALC> value_buffer;
    TBuf<TPosition::VECCALC> scale_buffer;
    TBuf<TPosition::VECCALC> narrow_buffer;
    TBuf<TPosition::VECCALC> row_buffer;
    pipe.InitBuffer(product_buffer, (TILE + SLACK) * sizeof(int32_t));
    pipe.InitBuffer(value_buffer, (TILE + SLACK) * sizeof(float));
    pipe.InitBuffer(scale_buffer, (TILE + SLACK) * sizeof(float));
    pipe.InitBuffer(narrow_buffer, (TILE + SLACK) * sizeof(bfloat16_t));
    pipe.InitBuffer(row_buffer, (SCALE_SPAN + SLACK) * sizeof(float));
    LocalTensor<int32_t> wide = product_buffer.Get<int32_t>();
    LocalTensor<float> value = value_buffer.Get<float>();
    LocalTensor<float> scale = scale_buffer.Get<float>();
    LocalTensor<bfloat16_t> narrow = narrow_buffer.Get<bfloat16_t>();
    LocalTensor<float> row_scale = row_buffer.Get<float>();

    GlobalTensor<int32_t> product_global;
    GlobalTensor<float> scale_global;
    GlobalTensor<float> middle_global;
    GlobalTensor<bfloat16_t> out_global;
    product_global.SetGlobalBuffer(product, ROWS * HIDDEN);
    scale_global.SetGlobalBuffer(down_scale, HIDDEN);
    middle_global.SetGlobalBuffer(middle_scale, SCALE_SPAN);
    out_global.SetGlobalBuffer(out, ROWS * HIDDEN);

    DataCopy(row_scale, middle_global, SCALE_SPAN);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::UB>();

    for (int64_t row = 0; row < ROWS; ++row) {
        const float of_row = row_scale.GetValue(row);
        for (int64_t done = from; done < upto; done += TILE) {
            const int64_t count = (upto - done < TILE) ? upto - done : TILE;
            const int64_t at = row * HIDDEN + done;
            DataCopy(wide, product_global[at], count);
            DataCopy(scale, scale_global[done], count);
            PipeBarrier<PIPE_ALL>();
            Cast(value, wide, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Mul(value, value, scale, count);
            PipeBarrier<PIPE_V>();
            Muls(value, value, of_row, count);
            PipeBarrier<PIPE_V>();
            Cast(narrow, value, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_ALL>();
            DataCopy(out_global[at], narrow, count);
        }
    }
}

}  // namespace

extern "C" __global__ __mix__(1, 2) __schedmode__(1)
__aicore__ void routed_expert_fused(
    uint64_t sync_base, __gm__ int8_t* rows, __gm__ int8_t* gate_up,
    __gm__ float* gu_scale, __gm__ int8_t* down, __gm__ float* down_scale,
    __gm__ float* token_scale, __gm__ int32_t* partial,
    __gm__ int8_t* middle, __gm__ float* middle_scale, __gm__ int32_t* wide,
    __gm__ bfloat16_t* out, __gm__ uint8_t* gu_tiling,
    __gm__ uint8_t* down_tiling, int64_t experts, int64_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);

    // The two halves number the grid differently: the cube's `GetBlockIdx`
    // runs to `blocks`, the vector side's to twice that, and `GetBlockNum`
    // reports the cube's count on both. So the vector side divides.
    const int64_t per_block = (experts + blocks - 1) / blocks;
    TPipe pipe;

    if ASCEND_IS_AIC {
        TCubeTiling gate_up_shape;
        TCubeTiling down_shape;
        LoadTiling(gate_up_shape, gu_tiling);
        LoadTiling(down_shape, down_tiling);
        Projection gate_up_mm;
        Projection down_mm;
        gate_up_mm.Init(&gate_up_shape, &pipe);
        down_mm.Init(&down_shape, &pipe);
        const int64_t first = GetBlockIdx() * per_block;
        for (int64_t step = 0; step < per_block; ++step) {
            const int64_t expert = first + step;
            if (expert >= experts) {
                break;
            }
            ProjectGateUp(gate_up_mm, rows + expert * ROWS * HIDDEN,
                          gate_up + expert * GATE_UP * HIDDEN,
                          partial + expert * ROWS * GATE_UP);
            Publish();
            CrossCoreSetFlag<0x02, PIPE_FIX>(GATE_READY);
            CrossCoreWaitFlag<0x03, PIPE_MTE2>(MIDDLE_READY);
            ProjectDown(down_mm, middle + expert * ROWS * HALF,
                        down + expert * HIDDEN * HALF,
                        wide + expert * ROWS * HIDDEN);
            Publish();
            CrossCoreSetFlag<0x02, PIPE_FIX>(DOWN_READY);
        }
        gate_up_mm.End();
        down_mm.End();
    }
    if ASCEND_IS_AIV {
        const int64_t first = (GetBlockIdx() / 2) * per_block;
        const int64_t sub = GetSubBlockIdx();
        const int64_t share = HIDDEN / 2;
        const int64_t from = sub * share;
        for (int64_t step = 0; step < per_block; ++step) {
            const int64_t expert = first + step;
            if (expert >= experts) {
                break;
            }
            CrossCoreWaitFlag<0x02, PIPE_MTE2>(GATE_READY);
            if (sub == 0) {
                Activate(pipe, partial + expert * ROWS * GATE_UP,
                         gu_scale + expert * GATE_UP,
                         token_scale + expert * SCALE_SPAN,
                         middle + expert * ROWS * HALF,
                         middle_scale + expert * SCALE_SPAN);
                Publish();
                CrossCoreSetFlag<0x03, PIPE_MTE3>(MIDDLE_READY);
            }
            CrossCoreWaitFlag<0x02, PIPE_MTE2>(DOWN_READY);
            Narrow(pipe, wide + expert * ROWS * HIDDEN,
                   down_scale + expert * HIDDEN,
                   middle_scale + expert * SCALE_SPAN,
                   out + expert * ROWS * HIDDEN, from,
                   from + share);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

void routed_expert_launch(uint32_t blocks, void* stream, uint64_t sync_base,
                          void* rows, void* gate_up, float* gu_scale,
                          void* down, float* down_scale, float* token_scale,
                          void* partial, void* middle, float* middle_scale,
                          void* wide, void* out, void* gu_tiling,
                          void* down_tiling, int64_t experts)
{
    routed_expert_fused<<<blocks, nullptr, stream>>>(
        sync_base, (__gm__ int8_t*)rows, (__gm__ int8_t*)gate_up, gu_scale,
        (__gm__ int8_t*)down, down_scale, token_scale,
        (__gm__ int32_t*)partial, (__gm__ int8_t*)middle, middle_scale,
        (__gm__ int32_t*)wide, (__gm__ bfloat16_t*)out,
        (__gm__ uint8_t*)gu_tiling, (__gm__ uint8_t*)down_tiling, experts,
        blocks);
}
