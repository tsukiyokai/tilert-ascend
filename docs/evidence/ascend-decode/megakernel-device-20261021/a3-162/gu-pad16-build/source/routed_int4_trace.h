#ifndef TESTS_GLM52_ROUTED_INT4_TRACE_H
#define TESTS_GLM52_ROUTED_INT4_TRACE_H

#include "kernel_operator.h"
#include "routed_int4_profile.h"

namespace routed_int4 {

struct NoTrace {
    __aicore__ inline void Begin(int) {}
    __aicore__ inline void Barrier(int) { AscendC::SyncAll<false>(); }
    __aicore__ inline void FinishFinal(int) {}
    __aicore__ inline void Store(__gm__ int64_t*, int64_t, int64_t) {}
};

#ifdef ROUTED_ENABLE_PROFILE
struct StageTrace {
    int64_t ticks[profile::kStages * profile::kMarksPerStage];

    __aicore__ inline void Begin(int stage)
    {
        ticks[stage * profile::kMarksPerStage] = AscendC::GetSystemCycle();
    }

    __aicore__ inline void Barrier(int stage)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        ticks[stage * profile::kMarksPerStage + 1] = AscendC::GetSystemCycle();
        AscendC::SyncAll<false>();
        ticks[stage * profile::kMarksPerStage + 2] = AscendC::GetSystemCycle();
    }

    __aicore__ inline void FinishFinal(int stage)
    {
        // Keep trace stores from idle cores out of the final combine stage.
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        Barrier(stage);
    }

    __aicore__ inline void Store(__gm__ int64_t* trace,
                                 int64_t tokens, int64_t blocks)
    {
        int64_t kind = 0;
        if ASCEND_IS_AIV {
            kind = 1;
        }
        const int64_t index = AscendC::GetBlockIdx();
        const int64_t slot = kind == 0 ? index : blocks + index;
        AscendC::GlobalTensor<int64_t> record;
        record.SetGlobalBuffer(trace + slot * profile::kRecordWords,
                                profile::kRecordWords);
        for (int word = 0; word < profile::kRecordWords; ++word) {
            record.SetValue(word, 0);
        }
        record.SetValue(0, profile::kVersion);
        record.SetValue(1, kind);
        record.SetValue(2, index);
        record.SetValue(3, tokens);
        record.SetValue(4, blocks);
        for (int mark = 0; mark < profile::kStages * profile::kMarksPerStage;
             ++mark) {
            record.SetValue(profile::kTimestampOffset + mark, ticks[mark]);
        }
        for (int word = 0; word < profile::kRecordWords; word += 8) {
            AscendC::DataCacheCleanAndInvalid<int64_t,
                AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(record[word]);
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
};
using Timing = StageTrace;
#else
using Timing = NoTrace;
#endif

}  // namespace routed_int4

#endif
