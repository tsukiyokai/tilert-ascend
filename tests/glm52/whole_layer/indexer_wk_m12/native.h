#ifndef TESTS_GLM52_INDEXER_WK_M12_NATIVE_H
#define TESTS_GLM52_INDEXER_WK_M12_NATIVE_H

#include "kernel_operator.h"
#include "indexer_front_contract.h"

namespace indexer_wk_m12::device {
using namespace AscendC;

constexpr int32_t kRows = 12;
constexpr int32_t kColumns = 160;
constexpr int32_t kInner = 6144;
constexpr int32_t kTile = 16;
constexpr int32_t kPackedColumns = 32;
constexpr int32_t kMmadInner = 128;
constexpr int32_t kSegments = 12;
constexpr int32_t kTasks = 10;

static_assert(indexer_front::kHidden == kInner);
static_assert(indexer_front::kKeyColumns == kColumns);
static_assert(indexer_front::kKeyTile == kPackedColumns);

// CANN 9.2 MatMulV2 key 229585 at M12/N160/K6144 uses ten N16
// tiles. Adjacent tiles reverse K512 traversal; each pair rotates it by two.
__aicore__ inline int32_t Segment(int32_t iteration, int32_t column_tile)
{
    const int32_t direction = (column_tile & 1)
        ? kSegments - 1 - iteration : iteration;
    return (direction + (column_tile / 2) * 2) % kSegments;
}

// Caller supplies a drained, reset AIC pipe. Weight and result retain the
// front-end's [5,K,32] and [5,capacity,32] layouts. No cross-core barrier here.
__aicore__ inline void Project(TPipe& pipe, const indexer_front::Config& c,
                              GM_ADDR hidden, GM_ADDR weight, GM_ADDR result)
{
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
    GlobalTensor<bfloat16_t> a_gm, b_gm;
    GlobalTensor<float> c_gm;
    a_gm.SetGlobalBuffer((__gm__ bfloat16_t*)hidden, kRows * kInner);
    b_gm.SetGlobalBuffer((__gm__ bfloat16_t*)weight, kColumns * kInner);
    c_gm.SetGlobalBuffer((__gm__ float*)result, c.capacity * kColumns);

    Nd2NzParams copy_a{};
    copy_a.ndNum = 1;
    copy_a.nValue = kRows;
    copy_a.dValue = kMmadInner;
    copy_a.srcDValue = kInner;
    copy_a.dstNzC0Stride = kTile;
    copy_a.dstNzNStride = 1;
    Nd2NzParams copy_b{};
    copy_b.ndNum = 1;
    copy_b.nValue = kMmadInner;
    copy_b.dValue = kTile;
    copy_b.srcDValue = kPackedColumns;
    copy_b.dstNzC0Stride = kMmadInner;
    copy_b.dstNzNStride = 1;
    LoadData2DParams load_a{};
    load_a.repeatTimes = kMmadInner / kTile;
    load_a.srcStride = 1;
    load_a.ifTranspose = false;
    LoadData2DParams load_b = load_a;
    // KN32 input becomes [K1,K0,N0] in L1, then [K1,N0,K0] in L0B.
    load_b.ifTranspose = true;

    for (int32_t task = GetBlockIdx(); task < kTasks; task += c.blocks) {
        const int32_t column_tile = task;
        const int32_t weight_tile = column_tile / 2;
        const int32_t column_offset = (column_tile % 2) * kTile;
        for (int32_t outer = 0; outer < kSegments; ++outer) {
            const int32_t segment = Segment(outer, column_tile);
            for (int32_t step = 0; step < 4; ++step) {
                const int32_t k_start = segment * 512 + step * kMmadInner;
                const int64_t a_offset = k_start;
                const int64_t b_offset = weight_tile * kInner * kPackedColumns
                    + k_start * kPackedColumns + column_offset;
                DataCopy(a_l1, a_gm[a_offset], copy_a);
                DataCopy(b_l1, b_gm[b_offset], copy_b);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
                LoadData(a_l0, a_l1, load_a);
                LoadData(b_l0, b_l1, load_b);
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
        store.dstStride = kPackedColumns;
        store.ndNum = 1;
        store.quantPre = QuantMode_t::NoQuant;
        const int64_t offset = weight_tile * c.capacity * kPackedColumns
            + column_offset;
        Fixpipe(c_gm[offset], c_l0, store);
        // A physical core may own another tile when fewer than 10 groups run.
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    }
}

}  // namespace indexer_wk_m12::device

#endif
