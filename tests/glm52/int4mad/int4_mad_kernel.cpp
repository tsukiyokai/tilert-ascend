/**
 * Does the A3 cube multiply two INT4 operands, driven by hand?
 *
 * The high-level `Matmul` will not pass `int4b_t` in any configuration,
 * because it plans for global memory to hold INT4 while the L1 tile holds
 * INT8 and A3's MTE2 does not widen on the way in. The chip's own path is
 * open: INT4 copies into A1, B1 and A2, and the library's `load_to_l0_utils`
 * already answers `LOAD2DTRANSPOSE` for `int4b_t`. So this drives L1 and L0
 * directly -- `DataCopy`, `LoadData`, `Mmad` -- instead of `IterateAll`.
 *
 * Both operands must be INT4: `Mmad` lists
 * "dst: int32_t, fm: int4b_t, filter: int4b_t" and dispatches to `mad_s4`,
 * and this part has no mixed-width mad. An INT8 activation reaches an INT4
 * weight by splitting into two INT4 halves and recombining 16 × high + low,
 * which is what the vendor's `a8w4_msd` stages do -- so this probe settles
 * the primitive those two mads rest on.
 *
 * The first question is arithmetic, not layout, so both operands are filled
 * with ones and every element of the product should be K whatever order the
 * fractals are in. A second filling, bytes of 0x21, makes the nibbles of a
 * byte differ: the sum over K becomes 96 if both halves are read, 64 or 128
 * if only one is. A prefix filling then reads out the fractal's own order.
 *
 * One mad covers at most 128 of K -- a K of 256 answered 128 -- so a real K
 * accumulates over several, the first initialising L0C and the rest adding.
 * Each step also loads its own 128 of K into the same L0 slot, which keeps
 * L0A and L0B at a kilobyte each and avoids a repeat count large enough to
 * misbehave: eight fractals in one `LoadData` lost one of four mads' worth
 * and corrupted a cell.
 *
 * Reusing one L0 slot needs the two queues ordered both ways, and
 * `PipeBarrier<PIPE_ALL>` does not do it -- with it the same code answered
 * 128 exactly, 256 as 192, 512 as noise, 1024 as half and 6144 exactly,
 * which is the signature of a hazard rather than of arithmetic. `MTE1_M`
 * makes the mad wait for its operands; `M_MTE1` stops the next load
 * overwriting operands the mad is still reading.
 */
#include "int4_mad_kernel.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr int64_t ROWS = 16;
#ifndef INNER_K
#define INNER_K 64
#endif
constexpr int64_t INNER = INNER_K;
constexpr int64_t COLUMNS = 16;
// How much of K one repeat of `LoadData` covers. An INT4 fractal holds 64,
// but the parameter may count in INT8 fractals of 32, which would load half
// the tile and leave the rest of L0 at zero -- and a sum of 64 real terms and
// 64 zeros is indistinguishable from a K of 64.
#ifndef REPEAT_K
#define REPEAT_K 64
#endif
constexpr int64_t INT4_FRACTAL_K = REPEAT_K;
// The most of K one mad covers.
constexpr int64_t MAD_K = 128;

}  // namespace

extern "C" __global__ __mix__(1, 2) __aicore__ void int4_mad(
    __gm__ int4b_t* left,
    __gm__ int4b_t* right, __gm__ int32_t* out)
{
    // A cube-only task type was rejected at submission -- the queue reported
    // `sqe_type=3(place holder), errType=0x20(sq sw status error)`. The MIX
    // shape this project already runs cube code through is what works.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    if ASCEND_IS_AIV {
        return;
    }
    TPipe pipe;
    TBuf<TPosition::A1> left_stage;
    TBuf<TPosition::B1> right_stage;
    TBuf<TPosition::A2> left_tile;
    TBuf<TPosition::B2> right_tile;
    TBuf<TPosition::CO1> product_tile;
    pipe.InitBuffer(left_stage, ROWS * INNER / 2);
    pipe.InitBuffer(right_stage, COLUMNS * INNER / 2);
    // L0 holds one step of K, not the whole of it.
    pipe.InitBuffer(left_tile, ROWS * MAD_K / 2);
    pipe.InitBuffer(right_tile, COLUMNS * MAD_K / 2);
    pipe.InitBuffer(product_tile, ROWS * COLUMNS * sizeof(int32_t));

    LocalTensor<int4b_t> left_l1 = left_stage.Get<int4b_t>();
    LocalTensor<int4b_t> right_l1 = right_stage.Get<int4b_t>();
    LocalTensor<int4b_t> left_l0 = left_tile.Get<int4b_t>();
    LocalTensor<int4b_t> right_l0 = right_tile.Get<int4b_t>();
    LocalTensor<int32_t> product = product_tile.Get<int32_t>();

    GlobalTensor<int4b_t> left_global;
    GlobalTensor<int4b_t> right_global;
    GlobalTensor<int32_t> out_global;
    left_global.SetGlobalBuffer(left, ROWS * INNER);
    right_global.SetGlobalBuffer(right, COLUMNS * INNER);
    out_global.SetGlobalBuffer(out, ROWS * COLUMNS);

    DataCopy(left_l1, left_global, ROWS * INNER);
    DataCopy(right_l1, right_global, COLUMNS * INNER);
    PipeBarrier<PIPE_ALL>();

    const uint8_t fractals = (uint8_t)(MAD_K / INT4_FRACTAL_K);
    for (int64_t done = 0; done < INNER; done += MAD_K) {
        const int64_t span = (INNER - done < MAD_K) ? INNER - done : MAD_K;
        // The fractals along K come first in the layout, so a step of K
        // starts whole fractals in.
        LoadData(left_l0, left_l1[done * ROWS],
                 LoadData2DParams(0, fractals, 1, 0, 0, false, 0));
        LoadData(right_l0, right_l1[done * COLUMNS],
                 LoadData2DParams(0, fractals, 1, 0, 0, false, 0));
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        MmadParams step((uint16_t)ROWS, (uint16_t)COLUMNS, (uint16_t)span,
                        false, 0, false, false, false);
        step.cmatrixInitVal = (done == 0);
        step.cmatrixSource = false;
        Mmad(product, left_l0, right_l0, step);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    }

    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    Fixpipe(out_global, product,
            FixpipeParamsV220((uint16_t)COLUMNS, (uint16_t)ROWS,
                              (uint16_t)0, (uint32_t)COLUMNS, false));
}

void int4_mad_launch(void* stream, void* left, void* right, void* out)
{
    int4_mad<<<1, nullptr, stream>>>((__gm__ int4b_t*)left,
                                      (__gm__ int4b_t*)right,
                                      (__gm__ int32_t*)out);
}
