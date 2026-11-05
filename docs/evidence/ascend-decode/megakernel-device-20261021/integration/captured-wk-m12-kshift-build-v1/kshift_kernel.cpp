#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr int32_t kRows = 12;
constexpr int32_t kColumns = 160;
constexpr int32_t kInner = 6144;
constexpr int32_t kTile = 16;
constexpr int32_t kMmadInner = 128;
constexpr int32_t kOuterCount = 12;

// CANN 9.2 MatMulV2 key 229585, M12/N160/K6144: ten N16 tiles.
// Adjacent tiles reverse twelve K512 segments; pairs rotate by two.
__aicore__ inline int32_t OuterK(int32_t iteration, int32_t column_tile,
                               bool shifted)
{
    if (!shifted) {
        return iteration;
    }
    const int32_t direction = (column_tile & 1) ? kOuterCount - 1 - iteration : iteration;
    return (direction + (column_tile / 2) * 2) % kOuterCount;
}

}  // namespace

extern "C" __global__ __mix__(1, 2) __aicore__
void wk_m12_kshift_kernel(GM_ADDR hidden, GM_ADDR weight, GM_ADDR accumulators,
                      GM_ADDR projection, int32_t shifted)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    if ASCEND_IS_AIV {
        return;
    }
    const int32_t block = GetBlockIdx();
    if (block >= 10) {
        return;
    }
    const int32_t column_tile = block;
    TPipe pipe;
    TBuf<TPosition::A1> a_stage;
    TBuf<TPosition::B1> b_stage;
    TBuf<TPosition::A2> a_operand;
    TBuf<TPosition::B2> b_operand;
    TBuf<TPosition::CO1> c_accumulator;
    pipe.InitBuffer(a_stage, kTile * kMmadInner * sizeof(bfloat16_t));
    pipe.InitBuffer(b_stage, kTile * kMmadInner * sizeof(bfloat16_t));
    pipe.InitBuffer(a_operand, kTile * kMmadInner * sizeof(bfloat16_t));
    pipe.InitBuffer(b_operand, kTile * kMmadInner * sizeof(bfloat16_t));
    pipe.InitBuffer(c_accumulator, kTile * kTile * sizeof(float));
    auto a_l1 = a_stage.Get<bfloat16_t>();
    auto b_l1 = b_stage.Get<bfloat16_t>();
    auto a_l0 = a_operand.Get<bfloat16_t>();
    auto b_l0 = b_operand.Get<bfloat16_t>();
    auto c_l0 = c_accumulator.Get<float>();
    GlobalTensor<bfloat16_t> a_gm;
    GlobalTensor<bfloat16_t> b_gm;
    GlobalTensor<float> c_gm;
    GlobalTensor<bfloat16_t> output_gm;
    a_gm.SetGlobalBuffer((__gm__ bfloat16_t*)hidden, kRows * kInner);
    b_gm.SetGlobalBuffer((__gm__ bfloat16_t*)weight, kColumns * kInner);
    c_gm.SetGlobalBuffer((__gm__ float*)accumulators, kRows * kColumns);
    output_gm.SetGlobalBuffer((__gm__ bfloat16_t*)projection, kRows * kColumns);

    Nd2NzParams copy{};
    copy.ndNum = 1;
    copy.nValue = kRows;
    copy.dValue = kMmadInner;
    copy.srcDValue = kInner;
    copy.dstNzC0Stride = kTile;
    copy.dstNzNStride = 1;
    Nd2NzParams copy_b = copy;
    copy_b.nValue = kTile;
    LoadData2DParams load{};
    load.repeatTimes = kMmadInner / kTile;
    load.srcStride = 1;
    load.ifTranspose = false;
    for (int32_t outer = 0; outer < kOuterCount; ++outer) {
        const int32_t k_segment = OuterK(outer, column_tile, shifted != 0);
        for (int32_t step = 0; step < 4; ++step) {
            const int32_t k_start = k_segment * 512 + step * kMmadInner;
            DataCopy(a_l1, a_gm[k_start], copy);
            DataCopy(b_l1, b_gm[column_tile * kTile * kInner + k_start], copy_b);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            LoadData(a_l0, a_l1, load);
            LoadData(b_l0, b_l1, load);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
            MmadParams mad{};
            mad.m = kRows;
            mad.n = kTile;
            mad.k = kMmadInner;
            mad.cmatrixInitVal = outer == 0 && step == 0;
            mad.cmatrixSource = false;
            Mmad(c_l0, a_l0, b_l0, mad);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        }
    }
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    FixpipeParamsV220 store{};
    store.nSize = kTile;
    store.mSize = kRows;
    store.srcStride = kTile;
    store.dstStride = kColumns;
    store.ndNum = 1;
    const int64_t offset = column_tile * kTile;
    store.quantPre = QuantMode_t::NoQuant;
    Fixpipe(c_gm[offset], c_l0, store);
    store.quantPre = QuantMode_t::F322BF16;
    Fixpipe(output_gm[offset], c_l0, store);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
}

extern "C" void wk_m12_kshift_launch(void* stream, void* hidden, void* weight,
    void* accumulators, void* projection, int32_t shifted)
{
    wk_m12_kshift_kernel<<<10, nullptr, stream>>>(
        (GM_ADDR)hidden, (GM_ADDR)weight, (GM_ADDR)accumulators,
        (GM_ADDR)projection, shifted);
}
