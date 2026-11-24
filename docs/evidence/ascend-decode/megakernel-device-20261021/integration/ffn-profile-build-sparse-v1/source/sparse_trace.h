#ifndef TILERT_FFN_PROFILE_TRACE_H
#define TILERT_FFN_PROFILE_TRACE_H

#include "kernel_operator.h"

namespace ffn_trace {
constexpr int kMarks = 13;
constexpr int kWords = 32;
constexpr int kStampOffset = 8;

struct Clock {
    int64_t start;
    int64_t finish;

    __aicore__ inline void Mark(int index)
    {
        if (index == 3) {
            start = AscendC::GetSystemCycle();
        } else if (index == 4) {
            finish = AscendC::GetSystemCycle();
        }
    }

    __aicore__ inline void Store(__gm__ int64_t* trace, int64_t rows,
                                 int64_t blocks)
    {
        int64_t kind = 0;
        if ASCEND_IS_AIV {
            kind = 1;
        }
        const int64_t index = AscendC::GetBlockIdx();
        const int64_t slot = kind == 0 ? index : blocks + index;
        AscendC::GlobalTensor<int64_t> output;
        output.SetGlobalBuffer(trace + slot * kWords, kWords);
        for (int word = 0; word < kWords; ++word) {
            output.SetValue(word, 0);
        }
        output.SetValue(0, 1);
        output.SetValue(1, kind);
        output.SetValue(2, index);
        output.SetValue(3, rows);
        output.SetValue(4, blocks);
        // Preserve the 13-field store footprint; only fields 3->4 measure time.
        for (int mark = 0; mark < kMarks; ++mark) {
            output.SetValue(kStampOffset + mark, mark < 4 ? start : finish);
        }
        for (int word = 0; word < kWords; word += 8) {
            AscendC::DataCacheCleanAndInvalid<int64_t,
                AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(output[word]);
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
};

struct BodyTrace {
    Clock& clock;

    __aicore__ inline explicit BodyTrace(Clock& value) : clock(value) {}
    __aicore__ inline void Begin(int stage) { clock.Mark(4 + stage); }
    __aicore__ inline void Barrier(int) { AscendC::SyncAll<false>(); }
    __aicore__ inline void FinishFinal(int) {}
    __aicore__ inline void Store(__gm__ int64_t*, int64_t, int64_t) {}
};
}  // namespace ffn_trace

#endif
