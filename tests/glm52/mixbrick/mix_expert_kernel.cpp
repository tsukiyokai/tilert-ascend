/**
 * The shared expert's first half in one MIX kernel instead of two.
 *
 * A layer runs `QuantBatchMatmulV3` for the shared expert's gate and up
 * projections, 11.661 us on AI_CORE, then `DequantSwigluQuant`, 3.140 on the
 * vector core. Two launches, and the INT32 product goes to global memory and
 * comes back. Fusing them keeps the product where the cube left it and pays
 * the cross-core structure once.
 *
 * The cube-to-vector flag direction is the only one the headers settle, so
 * that is the only one used: the cube finishes the whole projection, raises
 * `GATE_READY`, and the vector side dequantises and applies SwiGLU. Both arms
 * run at the same block count so the comparison is of the boundary, not of a
 * parallel decomposition -- the multi-core N split needs the gate and up rows
 * interleaved so a core owns whole pairs, which is a separate change.
 */
// MIX-mode Matmul is driven from the AIV: `Iterate` on the vector side only
// notifies the cube side, which answers when the matrix work is done. That is
// the KFC channel, and it is why a vendor MIX entry carries a
// `__kfc_workspace__`. Calling the API from inside `if ASCEND_IS_AIC` is the
// server half running the client's code with no client, and it deadlocks.
// This macro selects the other route -- Matmul on the cube side only -- and
// `matmul_intf.h` reads it, so it has to precede the include.
#define ASCENDC_CUBE_ONLY

#include "mix_expert_kernel.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

constexpr int64_t HIDDEN = 6144;
constexpr int64_t HALF = 128;
// One token scale a row, staged on chip; the widest arm verifies 32 rows.
constexpr int64_t MAX_ROWS = 32;
// One vector repeat of FP32, in elements.
constexpr int64_t SLACK = 64;
// One tile of the 6144-wide output a vector core walks at a time. Wide
// enough that a block's whole column span is one tile at four blocks: at 512
// the epilogue made three passes a row and spent 5.393 us moving 0.26 us of
// bytes.
constexpr int64_t TILE = 2048;
// A reduction's destination is written a whole 32-byte block at a time.
constexpr int64_t REDUCE_ALIGN = 8;
// Partial maxima are one float a block a row, in a fixed-size workspace.
constexpr int64_t MAX_BLOCKS = 32;
constexpr int64_t GATE_UP = 2 * HALF;
// CANN's own Matmul-to-vector example uses mode 2 and flag 8. A3 allows
// flag ids 0 through 10 only, and the Matmul library consumes some itself, so
// a hand-written flag beside it is a collision risk -- 8 is what the vendor
// example picks for exactly this handoff. A flag's counter also saturates at
// 15 sets, which a per-layer persistent kernel will have to reckon with.
// Flag 8 is `AIV_INTER_BLOCK_BARRIER` in catlass, and the vendor warns that
// the Matmul high-level API uses this interface for its own cross-core
// control, so sharing an id with it risks a collision. Two Matmul objects run
// in this kernel; stay off 8, 9 and 10.
constexpr uint16_t GATE_READY = 4;
constexpr uint16_t VECTOR_ALIVE = 5;
constexpr uint16_t DOWN_READY = 6;

using AQuant = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
using BQuant = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, true>;
// The vendor's cube-only sample declares C at `VECIN` on the kernel side.
// Copying that made the four-block arm measurably worse here -- 448 448 448
// 224 896 0 against 0 0 448 0 0 0 -- so GM it stays.
using CProduct = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using BiasNone = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

namespace {

/** Push this core's view of a global buffer out of its data cache.
 *
 * A cross-core flag orders the pipes; it does not make one core's cached
 * global writes visible to another's reads. The framework issues a DCCI ALL
 * around every operator -- SuperKernel's capability bit 4 exists to turn that
 * off -- so a hand-written kernel that spans the boundary has to issue its
 * own. Without it the vector half read zeros from a `partial` that verified
 * correct every time when downloaded, and the count of wrong rows differed on
 * every run.
 */
/** Report what each half sees of the grid, into the first slots of a buffer.
 *
 * Every channel range in this kernel is derived from `GetBlockIdx` and
 * `GetBlockNum`, and under mode 2 a cube can only raise a flag for its own
 * two vector subblocks -- so if the two halves number the grid differently, a
 * vector core reads a slice some other cube wrote and no barrier can help.
 * The derivation was a guess; this measures it.
 */
__aicore__ inline void ReportGrid(__gm__ int32_t* buffer, int64_t slot)
{
    GlobalTensor<int32_t> probe;
    probe.SetGlobalBuffer(buffer, 64);
    probe.SetValue(slot, (int32_t)GetBlockNum());
    probe.SetValue(slot + 1 + GetBlockIdx(),
                   (int32_t)(100 * (1 + GetBlockIdx()) + GetSubBlockIdx()));
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE,
                             DcciDst::CACHELINE_OUT>(probe);
    DataSyncBarrier<MemDsbT::DDR>();
}

/** Publish this core's global writes to the others.
 *
 * On NPU architecture 2201 the L2 cache is shared across cores, the DCache is
 * per-core but used only by the Scalar unit, and the ICache is read-only. A
 * cube-to-vector handoff goes L0C, Fixpipe, GM, MTE2, UB -- entirely through
 * the shared L2 -- so it has no coherence problem and needs no cache
 * maintenance. Earlier attempts here flushed the DCache, which is the wrong
 * cache for this path.
 *
 * What it does need is completion ordering, and `PipeBarrier` is not that:
 * it blocks a pipe's issue order, while this blocks until the write is
 * visible to other cores. `MemDsbT::ALL` would not do -- it covers GM and UB
 * but explicitly not L0C or L1 -- and `DDR` is the one that covers the
 * Fixpipe write this hands over.
 */
__aicore__ inline void Publish()
{
    DataSyncBarrier<MemDsbT::DDR>();
}

/** One contiguous run of output channels, from global memory to global.
 *
 * `channel` indexes the gate-and-up axis, so a block asks for its own slice
 * twice: once inside the gate half and once inside the up half. The tiling
 * describes a slice; the full 256-wide shape stays in `SetOrgShape` so the
 * strides of B and C are the real ones.
 */
__aicore__ inline void Project(TPipe& pipe, __gm__ int8_t* x,
                               __gm__ int8_t* gate_up,
                               __gm__ int32_t* partial,
                               __gm__ uint8_t* tiling, int64_t rows,
                               int64_t channel, int64_t channels)
{
    TCubeTiling cube;
    auto* destination = (uint32_t*)&cube;
    auto* source = (__gm__ uint32_t*)tiling;
    for (uint32_t word = 0; word < sizeof(TCubeTiling) / sizeof(uint32_t);
         ++word) {
        destination[word] = source[word];
    }
    Matmul<AQuant, BQuant, CProduct, BiasNone> mm;
    mm.Init(&cube, &pipe);
    GlobalTensor<int8_t> a;
    GlobalTensor<int8_t> b;
    GlobalTensor<int32_t> c;
    a.SetGlobalBuffer((__gm__ int8_t*)x);
    b.SetGlobalBuffer((__gm__ int8_t*)gate_up + channel * HIDDEN);
    c.SetGlobalBuffer((__gm__ int32_t*)partial + channel);
    mm.SetOrgShape(rows, GATE_UP, HIDDEN);
    mm.SetTensorA(a);
    mm.SetTensorB(b, true);
    mm.IterateAll(c);
    mm.End();
}

/** One tile of vector state: a row's product, its two scaled halves. */
struct Epilogue {
    TBuf<TPosition::VECCALC> product_buffer;
    TBuf<TPosition::VECCALC> gate_buffer;
    TBuf<TPosition::VECCALC> up_buffer;
    TBuf<TPosition::VECCALC> work_buffer;
    TBuf<TPosition::VECCALC> scale_buffer;
    TBuf<TPosition::VECCALC> narrow_buffer;
    TBuf<TPosition::VECCALC> token_buffer;
    TBuf<TPosition::VECCALC> half_buffer;
    TBuf<TPosition::VECCALC> code_buffer;
    TBuf<TPosition::VECCALC> reduced_buffer;
    TBuf<TPosition::VECCALC> rowmax_buffer;
    LocalTensor<int32_t> product;
    LocalTensor<float> gate;
    LocalTensor<float> up;
    LocalTensor<float> work;
    LocalTensor<float> scale;
    LocalTensor<bfloat16_t> narrow;
    LocalTensor<float> token;
    LocalTensor<half> half_value;
    LocalTensor<int8_t> code;
    LocalTensor<float> reduced;
    LocalTensor<float> rowmax;

    __aicore__ inline void Init(TPipe& pipe)
    {
        // A repeat of a vector operator is 256 bytes, and the operators
        // round their element count up to one. A buffer sized exactly to the
        // count has no room for that: at one block the counts reach the
        // buffer sizes exactly and the result alternated between right and
        // wrong on every other run, while at four blocks, a quarter of the
        // way in, it did not.
        pipe.InitBuffer(product_buffer, (GATE_UP + SLACK) * sizeof(int32_t));
        pipe.InitBuffer(gate_buffer, (HALF + SLACK) * sizeof(float));
        pipe.InitBuffer(up_buffer, (HALF + SLACK) * sizeof(float));
        pipe.InitBuffer(work_buffer, (HALF + SLACK) * sizeof(float));
        pipe.InitBuffer(scale_buffer, (GATE_UP + SLACK) * sizeof(float));
        pipe.InitBuffer(narrow_buffer, (HALF + SLACK) * sizeof(bfloat16_t));
        pipe.InitBuffer(token_buffer, MAX_ROWS * sizeof(float));
        pipe.InitBuffer(half_buffer, (HALF + SLACK) * sizeof(half));
        pipe.InitBuffer(code_buffer, (HALF + SLACK) * sizeof(int8_t));
        pipe.InitBuffer(reduced_buffer, REDUCE_ALIGN * sizeof(float));
        pipe.InitBuffer(rowmax_buffer,
                        MAX_ROWS * REDUCE_ALIGN * sizeof(float));
        product = product_buffer.Get<int32_t>();
        gate = gate_buffer.Get<float>();
        up = up_buffer.Get<float>();
        work = work_buffer.Get<float>();
        scale = scale_buffer.Get<float>();
        narrow = narrow_buffer.Get<bfloat16_t>();
        token = token_buffer.Get<float>();
        half_value = half_buffer.Get<half>();
        code = code_buffer.Get<int8_t>();
        reduced = reduced_buffer.Get<float>();
        rowmax = rowmax_buffer.Get<float>();
    }
};

/** `silu(gate) * up`, rounded once to BF16 on the way out. */
__aicore__ inline void SwiGlu(Epilogue& held, int64_t count)
{
    Muls(held.work, held.gate, -1.0F, count);
    PipeBarrier<PIPE_V>();
    Exp(held.work, held.work, count);
    PipeBarrier<PIPE_V>();
    Adds(held.work, held.work, 1.0F, count);
    PipeBarrier<PIPE_V>();
    Div(held.gate, held.gate, held.work, count);
    PipeBarrier<PIPE_V>();
    Mul(held.gate, held.gate, held.up, count);
    PipeBarrier<PIPE_V>();
    Cast(held.narrow, held.gate, RoundMode::CAST_RINT, count);
}

/** The vector half: dequantise each row's product and apply SwiGLU. */
__aicore__ inline void Activate(TPipe& pipe, __gm__ int32_t* partial,
                                __gm__ float* gu_scale,
                                __gm__ float* token_scale,
                                __gm__ bfloat16_t* out, int64_t rows,
                                int64_t first, int64_t stride,
                                int64_t channel, int64_t channels,
                                __gm__ float* hidden, __gm__ float* maxima,
                                int64_t block)
{
    Epilogue held;
    held.Init(pipe);
    GlobalTensor<int32_t> product_global;
    GlobalTensor<float> scale_global;
    GlobalTensor<float> token_global;
    GlobalTensor<bfloat16_t> out_global;
    GlobalTensor<float> hidden_global;
    GlobalTensor<float> maxima_global;
    hidden_global.SetGlobalBuffer(hidden, rows * HALF);
    maxima_global.SetGlobalBuffer(maxima,
                                  MAX_BLOCKS * MAX_ROWS * REDUCE_ALIGN);
    // With a length. `SetGlobalBuffer(ptr)` alone leaves the tensor's extent
    // unset, and then only the offset-zero row reads back correctly: row 0
    // was right and rows 1 through 7 wrong, every run, from a `partial` that
    // verified against the host every run.
    product_global.SetGlobalBuffer(partial, rows * GATE_UP);
    scale_global.SetGlobalBuffer(gu_scale, GATE_UP);
    token_global.SetGlobalBuffer(token_scale, MAX_ROWS);
    out_global.SetGlobalBuffer(out, rows * HALF);

    // This core's own slice of both halves' scales, so the loop below reads
    // only what it owns.
    DataCopy(held.scale, scale_global[channel], channels);
    DataCopy(held.scale[channels], scale_global[HALF + channel], channels);
    // The per-row scale is read as a scalar, and a scalar read of global
    // memory from a vector core is not covered by the mixed-core barrier --
    // it returned zero for every row but the first, which showed up as a
    // zeroed output with the INT32 product provably correct. Stage it.
    DataCopy(held.token, token_global, MAX_ROWS);
    PipeBarrier<PIPE_ALL>();
    // The scale is then read by the scalar unit. A pipe barrier orders issue,
    // not retirement, and `HardEvent::MTE2_S` did not help either: the read
    // returned zero for every row but the first on exactly every other run of
    // the same binary, which is unified buffer still holding what the
    // previous kernel left. Wait for the copy to retire.
    DataSyncBarrier<MemDsbT::UB>();

    // Two vector subblocks, alternate rows each. At eight rows the work is
    // 1024 elements, so the split is about keeping both subblocks fed rather
    // than about throughput.
    // Which rows this core owns. A MIX kernel's two vector subblocks take
    // alternate rows; a vector-only kernel is one core and takes them all.
    // Reading `GetSubBlockIdx` here instead would silently write every other
    // row in the vector-only case, which is half the output.
    for (int64_t row = first; row < rows; row += stride) {
        const int64_t gate_at = row * GATE_UP + channel;
        DataCopy(held.product, product_global[gate_at], channels);
        DataCopy(held.product[channels], product_global[gate_at + HALF],
                 channels);
        PipeBarrier<PIPE_ALL>();
        Cast(held.gate, held.product, RoundMode::CAST_NONE, channels);
        Cast(held.up, held.product[channels], RoundMode::CAST_NONE, channels);
        PipeBarrier<PIPE_V>();
        Mul(held.gate, held.gate, held.scale, channels);
        Mul(held.up, held.up, held.scale[channels], channels);
        PipeBarrier<PIPE_V>();
        const float scale = held.token.GetValue(row);
        Muls(held.gate, held.gate, scale, channels);
        Muls(held.up, held.up, scale, channels);
        PipeBarrier<PIPE_V>();
        SwiGlu(held, channels);
        PipeBarrier<PIPE_ALL>();
        DataCopy(out_global[row * HALF + channel], held.narrow, channels);
        // The activation itself, and this block's largest magnitude in it.
        // Per-token quantisation needs the maximum over all `HALF` columns
        // and this split owns only `channels` of them, so each block leaves a
        // partial for the pass after the barrier to reduce.
        DataCopy(hidden_global[row * HALF + channel], held.gate, channels);
        Abs(held.work, held.gate, channels);
        PipeBarrier<PIPE_V>();
        // A reduction's destination is written a whole 32-byte block at a
        // time, so each row gets its own aligned block rather than its own
        // float. Reading the result back as a scalar instead cost a
        // retirement barrier a row and doubled the kernel.
        ReduceMax(held.rowmax[row * REDUCE_ALIGN], held.work, held.work,
                  channels, false);
    }
    PipeBarrier<PIPE_ALL>();
    DataCopy(maxima_global[block * MAX_ROWS * REDUCE_ALIGN], held.rowmax,
             MAX_ROWS * REDUCE_ALIGN);
}

/** The second cube stage: the down projection, split on its own N axis.
 *
 * Its K axis is all `HALF` intermediate columns, which the first stage spread
 * across blocks -- so a grid barrier, not a per-pair flag, is what has to
 * stand between them.
 */
__aicore__ inline void ProjectDown(TPipe& pipe, __gm__ int8_t* middle,
                                   __gm__ int8_t* down, __gm__ int32_t* wide,
                                   __gm__ uint8_t* tiling, int64_t rows,
                                   int64_t column, int64_t columns)
{
    TCubeTiling cube;
    auto* destination = (uint32_t*)&cube;
    auto* source = (__gm__ uint32_t*)tiling;
    for (uint32_t word = 0; word < sizeof(TCubeTiling) / sizeof(uint32_t);
         ++word) {
        destination[word] = source[word];
    }
    Matmul<AQuant, BQuant, CProduct, BiasNone> mm;
    mm.Init(&cube, &pipe);
    GlobalTensor<int8_t> a;
    GlobalTensor<int8_t> b;
    GlobalTensor<int32_t> c;
    a.SetGlobalBuffer((__gm__ int8_t*)middle, rows * HALF);
    b.SetGlobalBuffer((__gm__ int8_t*)down + column * HALF,
                      columns * HALF);
    c.SetGlobalBuffer((__gm__ int32_t*)wide + column, rows * HIDDEN);
    mm.SetOrgShape(rows, HIDDEN, HALF);
    mm.SetTensorA(a);
    mm.SetTensorB(b, true);
    mm.IterateAll(c);
    mm.End();
}

/** Scale the down projection's INT32 product into BF16. */
__aicore__ inline void Narrow(TPipe& pipe, __gm__ int32_t* wide,
                              __gm__ float* down_scale,
                              __gm__ float* middle_scale,
                              __gm__ bfloat16_t* out, int64_t rows,
                              int64_t column, int64_t columns)
{
    TBuf<TPosition::VECCALC> product_buffer;
    TBuf<TPosition::VECCALC> value_buffer;
    TBuf<TPosition::VECCALC> scale_buffer;
    TBuf<TPosition::VECCALC> narrow_buffer;
    pipe.InitBuffer(product_buffer, (TILE + SLACK) * sizeof(int32_t));
    pipe.InitBuffer(value_buffer, (TILE + SLACK) * sizeof(float));
    pipe.InitBuffer(scale_buffer, (TILE + SLACK) * sizeof(float));
    pipe.InitBuffer(narrow_buffer, (TILE + SLACK) * sizeof(bfloat16_t));
    LocalTensor<int32_t> product = product_buffer.Get<int32_t>();
    LocalTensor<float> value = value_buffer.Get<float>();
    LocalTensor<float> scale = scale_buffer.Get<float>();
    LocalTensor<bfloat16_t> narrow = narrow_buffer.Get<bfloat16_t>();
    TBuf<TPosition::VECCALC> row_buffer;
    pipe.InitBuffer(row_buffer, MAX_ROWS * sizeof(float));
    LocalTensor<float> row_scale = row_buffer.Get<float>();

    GlobalTensor<int32_t> wide_global;
    GlobalTensor<float> scale_global;
    GlobalTensor<bfloat16_t> out_global;
    wide_global.SetGlobalBuffer(wide, rows * HIDDEN);
    scale_global.SetGlobalBuffer(down_scale, HIDDEN);
    out_global.SetGlobalBuffer(out, rows * HIDDEN);
    GlobalTensor<float> middle_global;
    middle_global.SetGlobalBuffer(middle_scale, MAX_ROWS);
    DataCopy(row_scale, middle_global, MAX_ROWS);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::UB>();

    for (int64_t row = 0; row < rows; ++row) {
        const float scale_of_row = row_scale.GetValue(row);
        for (int64_t done = 0; done < columns; done += TILE) {
            const int64_t count = (columns - done < TILE) ? columns - done
                                                          : TILE;
            const int64_t at = row * HIDDEN + column + done;
            DataCopy(product, wide_global[at], count);
            DataCopy(scale, scale_global[column + done], count);
            PipeBarrier<PIPE_ALL>();
            Cast(value, product, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Mul(value, value, scale, count);
            PipeBarrier<PIPE_V>();
            Muls(value, value, scale_of_row, count);
            PipeBarrier<PIPE_V>();
            Cast(narrow, value, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_ALL>();
            DataCopy(out_global[at], narrow, count);
        }
    }
}

/** Reduce the partial maxima, quantise the activation, publish the scale.
 *
 * Runs after the barrier, so every block's partial is readable. Each block
 * reduces the whole row -- the work is one float a block, cheaper than
 * arranging for one core to do it and broadcast -- and then quantises only
 * its own slice of the activation.
 */
__aicore__ inline void Quantise(TPipe& pipe, __gm__ float* hidden,
                                __gm__ float* maxima, __gm__ int8_t* middle,
                                __gm__ float* scale_out, int64_t rows,
                                int64_t channel, int64_t channels,
                                int64_t blocks)
{
    TBuf<TPosition::VECCALC> value_buffer;
    TBuf<TPosition::VECCALC> half_buffer;
    TBuf<TPosition::VECCALC> code_buffer;
    TBuf<TPosition::VECCALC> partial_buffer;
    pipe.InitBuffer(value_buffer, (MAX_ROWS * HALF + SLACK) * sizeof(float));
    pipe.InitBuffer(half_buffer, (MAX_ROWS * HALF + SLACK) * sizeof(half));
    pipe.InitBuffer(code_buffer, (MAX_ROWS * HALF + SLACK) * sizeof(int8_t));
    pipe.InitBuffer(partial_buffer,
                    MAX_BLOCKS * MAX_ROWS * REDUCE_ALIGN * sizeof(float));
    LocalTensor<float> value = value_buffer.Get<float>();
    LocalTensor<half> half_value = half_buffer.Get<half>();
    LocalTensor<int8_t> code = code_buffer.Get<int8_t>();
    LocalTensor<float> partial = partial_buffer.Get<float>();
    TBuf<TPosition::VECCALC> scale_buffer;
    pipe.InitBuffer(scale_buffer, MAX_ROWS * sizeof(float));
    LocalTensor<float> scale_row = scale_buffer.Get<float>();

    GlobalTensor<float> hidden_global;
    GlobalTensor<float> maxima_global;
    GlobalTensor<int8_t> middle_global;
    GlobalTensor<float> scale_global;
    hidden_global.SetGlobalBuffer(hidden, rows * HALF);
    maxima_global.SetGlobalBuffer(maxima,
                                  MAX_BLOCKS * MAX_ROWS * REDUCE_ALIGN);
    middle_global.SetGlobalBuffer(middle, rows * HALF);
    scale_global.SetGlobalBuffer(scale_out, MAX_ROWS);

    // Only the blocks that ran. Copying the whole fixed-size workspace was
    // 32 KiB a core and cost most of the pass.
    DataCopy(partial, maxima_global, blocks * MAX_ROWS * REDUCE_ALIGN);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::UB>();

    // Every row's scale first, from the partials. One scalar read a block a
    // row, all on chip.
    for (int64_t row = 0; row < rows; ++row) {
        float largest = 0.0F;
        for (int64_t block = 0; block < blocks; ++block) {
            const float one = partial.GetValue(
                (block * MAX_ROWS + row) * REDUCE_ALIGN);
            largest = (one > largest) ? one : largest;
        }
        scale_row.SetValue(row, largest / 127.0F);
    }

    // Then the whole slice in one strided copy, cast whole, and out again.
    // Walking it row by row was forty operations on 32 elements each, where
    // one vector repeat is 64 floats -- about 11 us of pure fixed cost.
    //
    // The strides are in 32-byte data blocks, which is also why the channel
    // count has to leave a whole number of them: at eight blocks the INT8
    // slice is 16 bytes and there is no such copy.
    DataCopyParams gather;
    gather.blockCount = static_cast<uint16_t>(rows);
    gather.blockLen = static_cast<uint16_t>(channels * sizeof(float) / 32);
    gather.srcStride = static_cast<uint16_t>((HALF - channels)
                                             * sizeof(float) / 32);
    gather.dstStride = 0;
    DataCopy(value, hidden_global[channel], gather);
    PipeBarrier<PIPE_ALL>();
    for (int64_t row = 0; row < rows; ++row) {
        Muls(value[row * channels], value[row * channels],
             1.0F / scale_row.GetValue(row), channels);
    }
    PipeBarrier<PIPE_V>();
    Cast(half_value, value, RoundMode::CAST_RINT, rows * channels);
    PipeBarrier<PIPE_V>();
    Cast(code, half_value, RoundMode::CAST_RINT, rows * channels);
    PipeBarrier<PIPE_ALL>();
    DataCopyParams scatter;
    scatter.blockCount = static_cast<uint16_t>(rows);
    scatter.blockLen = static_cast<uint16_t>(channels / 32);
    scatter.srcStride = 0;
    scatter.dstStride = static_cast<uint16_t>((HALF - channels) / 32);
    DataCopy(middle_global[channel], code, scatter);
    PipeBarrier<PIPE_ALL>();
    DataCopy(scale_global, scale_row, MAX_ROWS);
    DataSyncBarrier<MemDsbT::DDR>();
}

}  // namespace

// `__mix__(cube, vec)` is the vendor spelling and it is not interchangeable
// with `[[bisheng::core_ratio(cube, vec)]]`: the macro has two definitions,
// one per compilation pass -- `__attribute__((aic))` for the cube half and
// `core_ratio` for the mix -- while the C++11 attribute applies
// unconditionally to both, which hung every launch.
extern "C" __global__ __mix__(1, 2) __schedmode__(1)
__aicore__ void mix_expert_fused(
    uint64_t sync_base, __gm__ int8_t* x, __gm__ int8_t* gate_up,
    __gm__ int32_t* partial, __gm__ float* gu_scale,
    __gm__ float* token_scale, __gm__ bfloat16_t* out,
    __gm__ uint8_t* tiling, int64_t rows, int64_t stages,
    int64_t blocks, __gm__ int8_t* middle, __gm__ int8_t* down,
    __gm__ float* down_scale, __gm__ int32_t* wide,
    __gm__ bfloat16_t* narrow_out, __gm__ uint8_t* down_tiling,
    __gm__ float* middle_scale, __gm__ float* hidden, __gm__ float* maxima)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    SetSyncBaseAddr(sync_base);
    // `stages` localises a hang: bit 0 the cube half, bit 1 the cube's set,
    // bit 2 the vector half, bit 3 the vector's wait. Set and wait are
    // separate bits so a hang can be attributed to one side: a set with no
    // waiter that completes proves the set is fine, and a wait with no setter
    // that hangs proves the vector half is alive at all.
    // Each block owns a contiguous run of SwiGLU outputs, and therefore the
    // same run inside each of the two halves. Splitting the 256-wide axis
    // straight across blocks would put gate `i` and up `i` on different
    // cores, and SwiGLU needs them together; two projections of a slice each
    // keeps them on one core and needs no weight permutation.
    //
    // The two halves number the grid differently, and it is measured, not
    // assumed: at four blocks the cube reports `GetBlockNum` 4 with
    // `GetBlockIdx` 0 to 3, while the vector side reports `GetBlockNum` 4
    // with `GetBlockIdx` 0 to 7 and `GetSubBlockIdx` 0 or 1. So the count
    // cannot be used to convert between them. The vector subblocks 2i and
    // 2i+1 belong to cube i -- and under mode 2 a cube can raise a flag only
    // for those two -- so a vector core must take its range from
    // `GetBlockIdx() / 2`. Taking it from `GetBlockIdx()` sent each vector
    // core to a slice some other cube wrote, which no barrier can order.
    const int64_t channels = HALF / blocks;
    if (stages & 64) {
        if ASCEND_IS_AIC {
            ReportGrid(partial, 0);
        }
        if ASCEND_IS_AIV {
            ReportGrid(partial, 32);
        }
        return;
    }
    // One `TPipe` a half, declared once and shared by every stage on it.
    // A second one constructed mid-kernel, after the first has gone out of
    // scope, silently did not work: the stage that owned it never ran.
    TPipe pipe;
    if ASCEND_IS_AIC {
        const int64_t channel = GetBlockIdx() * channels;
        if (stages & 1) {
            // Bit 5 projects the whole axis in one call instead of two. Two
            // `Matmul` objects sharing one `TPipe` is the one variable the
            // regression arrived with, and the original single-call kernel
            // was repeatably correct.
            if (stages & 32) {
                Project(pipe, x, gate_up, partial, tiling, rows, 0,
                        GATE_UP);
            } else {
                Project(pipe, x, gate_up, partial, tiling, rows, channel,
                        channels);
                Project(pipe, x, gate_up, partial, tiling, rows,
                        HALF + channel, channels);
            }
            // Drain the fixpipe, then publish out of the data cache, then
            // raise the flag. All three, in that order.
            Publish();
        }
        if (stages & 2) {
            // Mode 2 is the only one that crosses from cube to vector. Mode 0
            // synchronises cores of the same kind, so a mode-0 set from the
            // cube raises a flag only another cube would consume and the
            // waiting vector core never sees it -- which is what hung every
            // run that included this flag.
            // Once, not once per waiter. Under mode 2 a single set wakes
            // both of this cube's vector subblocks, consecutive sets of one
            // flagId are not allowed, and a second set leaves the counter a
            // surplus that survives the process -- which showed as a
            // perfectly alternating 896, 0, 896, 0 across runs of the same
            // binary.
            CrossCoreSetFlag<0x02, PIPE_FIX>(GATE_READY);
        }
        if (stages & 16) {
            CrossCoreWaitFlag(VECTOR_ALIVE);
        }
    }
    if ASCEND_IS_AIV {
        const int64_t channel = (GetBlockIdx() / 2) * channels;
        if (stages & 8) {
            // Mode and pipe both spelled out. A bare `CrossCoreWaitFlag`
            // defaults to mode 0 on `PIPE_S`, and `PIPE_S` is listed as
            // unsupported for every cross-core mode this part has -- so the
            // default argument is an unsupported value. `PIPE_MTE2` is what
            // the consumer actually blocks on.
            //
            // One set raises the counter of both of a cube's vector
            // subblocks, and each must wait exactly once or the leftover
            // count survives into the next launch. So the handshake stays
            // out of any branch that the two subblocks could take
            // differently: split the work, never the handshake.
            CrossCoreWaitFlag<0x02, PIPE_MTE2>(GATE_READY);
        }
        if (stages & 16) {
            CrossCoreSetFlag<0x02, PIPE_MTE3>(VECTOR_ALIVE);
        }
        if (stages & 4) {
            // Every row, on whichever vector subblocks run. Splitting rows
            // by `GetSubBlockIdx` left half the output unwritten, and at
            // eight rows the vector work is too small to be worth a split.
            Activate(pipe, partial, gu_scale, token_scale, out, rows,
                     0, 1, channel, channels, hidden, maxima,
                     GetBlockIdx() / 2);
        }
    }
    if (stages & 128) {
        // The down projection reads every block's slice of the intermediate,
        // not just its own pair's, and the per-token scale is a maximum over
        // columns this split spreads across blocks. So two grid barriers,
        // both at top level: every core must arrive even where its own stage
        // work is skipped.
        // Bit 8 runs the two barriers and nothing between or after them,
        // which is the only way to price them apart from the work.
        const bool barriers_only = (stages & 256) != 0;
        Publish();
        SyncAll<false>();
        if ASCEND_IS_AIV {
            if (!barriers_only) {
                Quantise(pipe, hidden, maxima, middle, middle_scale, rows,
                         (GetBlockIdx() / 2) * channels, channels, blocks);
            }
        }
        Publish();
        SyncAll<false>();
        if (barriers_only) {
            return;
        }
        const int64_t columns = HIDDEN / blocks;
        if ASCEND_IS_AIC {
            ProjectDown(pipe, middle, down, wide, down_tiling, rows,
                        GetBlockIdx() * columns, columns);
            Publish();
            CrossCoreSetFlag<0x02, PIPE_FIX>(DOWN_READY);
        }
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag<0x02, PIPE_MTE2>(DOWN_READY);
            // Bit 9 skips the epilogue, which is what prices it: inferring
            // it from the difference between two code versions is how the
            // last attribution went wrong.
            if ((stages & 512) == 0) {
                Narrow(pipe, wide, down_scale, middle_scale, narrow_out,
                       rows, (GetBlockIdx() / 2) * columns, columns);
            }
        }
    }
}

// The AIC:AIV ratio is declared once, by KERNEL_TASK_TYPE_DEFAULT in the
// body. `[[bisheng::core_ratio]]` says the same thing again and the sanitizer
// build rejects it outright in this position.
extern "C" __global__ __mix__(1, 2) __schedmode__(1)
__aicore__ void mix_expert_project(
    uint64_t sync_base, __gm__ int8_t* x, __gm__ int8_t* gate_up,
    __gm__ int32_t* partial, __gm__ uint8_t* tiling, int64_t rows,
    int64_t blocks)
{
    const int64_t block = GetBlockIdx() * blocks / GetBlockNum();
    const int64_t channels = HALF / blocks;
    const int64_t channel = block * channels;
    // Also a MIX kernel, so that the staged arm's matmul is the same kernel
    // kind as the fused one's and the comparison is of the boundary alone.
    // An ND input makes the Matmul library convert to NZ, which is not purely
    // a cube activity.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    SetSyncBaseAddr(sync_base);
    TPipe pipe;
    if ASCEND_IS_AIC {
        Project(pipe, x, gate_up, partial, tiling, rows, channel, channels);
        Project(pipe, x, gate_up, partial, tiling, rows, HALF + channel,
                channels);
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void mix_expert_activate(__gm__ int32_t* partial, __gm__ float* gu_scale,
                         __gm__ float* token_scale, __gm__ bfloat16_t* out,
                         int64_t rows, int64_t blocks, __gm__ float* hidden,
                         __gm__ float* maxima)
{
    TPipe pipe;
    const int64_t block = GetBlockIdx() * blocks / GetBlockNum();
    const int64_t channels = HALF / blocks;
    Activate(pipe, partial, gu_scale, token_scale, out, rows, 0, 1,
             block * channels, channels, hidden, maxima, block);
}

void mix_expert_fused_launch(uint32_t blocks, void* stream, uint64_t sync_base,
                             void* x, void* gate_up, void* partial,
                             float* gu_scale, float* token_scale, void* out,
                             void* tiling, int64_t rows, int64_t stages,
                             void* middle, void* down, float* down_scale,
                             void* wide, void* narrow_out, void* down_tiling,
                             float* middle_scale, float* hidden,
                             float* maxima)
{
    mix_expert_fused<<<blocks, nullptr, stream>>>(
        sync_base, (__gm__ int8_t*)x, (__gm__ int8_t*)gate_up,
        (__gm__ int32_t*)partial, gu_scale, token_scale,
        (__gm__ bfloat16_t*)out, (__gm__ uint8_t*)tiling, rows, stages,
        blocks, (__gm__ int8_t*)middle, (__gm__ int8_t*)down, down_scale,
        (__gm__ int32_t*)wide, (__gm__ bfloat16_t*)narrow_out,
        (__gm__ uint8_t*)down_tiling, middle_scale, hidden, maxima);
}

void mix_expert_staged_launch(uint32_t blocks, void* stream,
                              uint64_t sync_base, void* x, void* gate_up,
                              void* partial, float* gu_scale,
                              float* token_scale, void* out, void* tiling,
                              int64_t rows, float* hidden, float* maxima)
{
    mix_expert_project<<<blocks, nullptr, stream>>>(
        sync_base, (__gm__ int8_t*)x, (__gm__ int8_t*)gate_up,
        (__gm__ int32_t*)partial, (__gm__ uint8_t*)tiling, rows, blocks);
    mix_expert_activate<<<blocks, nullptr, stream>>>(
        (__gm__ int32_t*)partial, gu_scale, token_scale,
        (__gm__ bfloat16_t*)out, rows, blocks, hidden, maxima);
}
