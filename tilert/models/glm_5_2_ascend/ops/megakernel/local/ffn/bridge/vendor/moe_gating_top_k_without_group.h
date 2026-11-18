#ifndef MOE_GATING_TOP_K_E_K_WITHOUT_GROUP_H
#define MOE_GATING_TOP_K_E_K_WITHOUT_GROUP_H
#include "kernel_operator.h"
#include "common.h"
#include "kernel_utils.h"
namespace MoeGatingTopK {
using namespace AscendC;

template <typename T>
class MoeGatingTopKWithoutGroup {
public:
    __aicore__ inline MoeGatingTopKWithoutGroup() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
                                const MoeGatingTopKTilingData *tilingData, TPipe *tPipe, int64_t overrideBlockIdx = -1)
    {
        tilingData_ = tilingData;
        pipe_ = tPipe;
        blockIdx_ = overrideBlockIdx >= 0 ? overrideBlockIdx : GetBlockIdx();
        xGm_.SetGlobalBuffer((__gm__ T *)x + 256 * blockIdx_, 256);
        biasGm_.SetGlobalBuffer((__gm__ T *)bias, 256);
        yGm_.SetGlobalBuffer((__gm__ T *)y + 8 * blockIdx_, 8);
        expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + 8 * blockIdx_, 8);
        pipe_->InitBuffer(storage_, 2560 * sizeof(float));
    }

    __aicore__ inline void PreloadBias()
    {
        LocalTensor<float> base = storage_.Get<float>();
        LocalTensor<float> biasTensor = base[256];
        LocalTensor<int32_t> expertIdTensor = base[512].ReinterpretCast<int32_t>();
        DataCopy(biasTensor, biasGm_, 256);
        event_t evMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evMte2V);
        WaitFlag<HardEvent::MTE2_V>(evMte2V);
        ArithProgression(expertIdTensor, static_cast<int32_t>(0), static_cast<int32_t>(1), 256);
    }

    __aicore__ inline void ProcessPreloaded()
    {
        LocalTensor<float> base = storage_.Get<float>();
        LocalTensor<float> xInLocalTensor = base;
        LocalTensor<float> biasTensor = base[256];
        LocalTensor<uint32_t> expertIdTensor = base[512].ReinterpretCast<uint32_t>();
        LocalTensor<float> xNormTensor = base[768];
        LocalTensor<float> xNormWithBiasTensor = base[1024];
        LocalTensor<int32_t> topKExpertId = base[1280].ReinterpretCast<int32_t>();
        LocalTensor<float> yOutTensor = base[1344];
        LocalTensor<float> calcTmpTensor = base[1408];

        DataCopy(xInLocalTensor, xGm_, 256);
        event_t evMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evMte2V);
        WaitFlag<HardEvent::MTE2_V>(evMte2V);

        LocalTensor<uint8_t> calcNormTmpTensor = calcTmpTensor.ReinterpretCast<uint8_t>();
        Sigmoid(xNormTensor, xInLocalTensor, calcNormTmpTensor, 256);
        PipeBarrier<PIPE_V>();
        Add(xNormWithBiasTensor, xNormTensor, biasTensor, 256);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> sortedScore = calcTmpTensor;
        LocalTensor<float> sortTmp = calcTmpTensor[512];
        Sort32(sortTmp, xNormWithBiasTensor, expertIdTensor, 8);
        PipeBarrier<PIPE_V>();
        const uint16_t len32[4] = {32, 32, 32, 32};
        uint32_t sortedNum[4];
        MrgSortCal<float>(sortedScore, MrgSortSrcList(sortTmp[0], sortTmp[64], sortTmp[128], sortTmp[192]),
                          len32, sortedNum, 0b1111, 2);
        PipeBarrier<PIPE_V>();
        const uint16_t len8[4] = {8, 8, 8, 8};
        MrgSortCal<float>(sortTmp, MrgSortSrcList(sortedScore[0], sortedScore[256], sortedScore[0], sortedScore[0]),
                          len8, sortedNum, 0b0011, 1);
        PipeBarrier<PIPE_V>();

        GatherMaskParams gatherMaskParams;
        gatherMaskParams.repeatTimes = 1;
        gatherMaskParams.src0BlockStride = 1;
        gatherMaskParams.src0RepeatStride = 8;
        gatherMaskParams.src1RepeatStride = 0;
        uint64_t rsvdCnt = 0;
        GatherMask(topKExpertId, sortTmp.ReinterpretCast<int32_t>(), static_cast<uint8_t>(2), false,
                   static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);
        PipeBarrier<PIPE_V>();

        LocalTensor<int32_t> topKExpertIdWithByte = calcTmpTensor.ReinterpretCast<int32_t>();
        Muls(topKExpertIdWithByte, topKExpertId, static_cast<int32_t>(sizeof(float)), 8);
        PipeBarrier<PIPE_V>();
        Gather(yOutTensor, xNormTensor, topKExpertIdWithByte.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), 8);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> maxValueTensor = calcTmpTensor;
        LocalTensor<float> tmpTensor = calcTmpTensor[32];
        const int32_t effK = static_cast<int32_t>(tilingData_->k);
        ReduceSum(maxValueTensor, yOutTensor, tmpTensor, effK);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        float sumValue = maxValueTensor.GetValue(0) + tilingData_->eps;
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Duplicate(tmpTensor, sumValue, 8);
        PipeBarrier<PIPE_V>();
        Div(yOutTensor, yOutTensor, tmpTensor, 8);

        event_t evVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evVMte3);
        WaitFlag<HardEvent::V_MTE3>(evVMte3);
        if (effK < 8) {
            event_t evVS2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(evVS2);
            WaitFlag<HardEvent::V_S>(evVS2);
            for (int32_t ki = effK; ki < 8; ++ki) {
                yOutTensor.SetValue(ki, 0.0F);
                topKExpertId.SetValue(ki, -1);
            }
            event_t evSMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(evSMte3);
            WaitFlag<HardEvent::S_MTE3>(evSMte3);
        }
        DataCopy(yGm_, yOutTensor, 8);
        DataCopy(expertIdxGm_, topKExpertId, 8);
    }

    __aicore__ inline void Process()
    {
        PreloadBias();
        PipeBarrier<PIPE_V>();
        ProcessPreloaded();
    }

private:
    TPipe *pipe_;
    TBuf<TPosition::VECCALC> storage_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    int64_t blockIdx_ = 0;
    const MoeGatingTopKTilingData *tilingData_;
};
}
#endif
