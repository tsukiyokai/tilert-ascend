#ifndef GMM_A8W4_ACTIVE_GROUPS_H
#define GMM_A8W4_ACTIVE_GROUPS_H

#include "kernel_operator.h"

namespace GroupedMatmulDequantSwigluQuant {
using namespace AscendC;

// Single-chunk decode reserves its unused second activation bank for this table.
// Word 0 is the count; each following word packs (exclusive row end, expert id).
class ActiveGroups {
public:
    static constexpr uint32_t EXPERTS = 256;
    static constexpr uint32_t WORDS = 260;
    bool enabled = false;

    __aicore__ inline void Init(GM_ADDR address)
    {
        enabled = address != nullptr;
        if (enabled) {
            records.SetGlobalBuffer((__gm__ uint64_t *)address);
        }
    }

    __aicore__ inline void Publish(GM_ADDR countsAddress, TPipe *pipe)
    {
        if ASCEND_IS_AIV {
            if (!enabled || GetBlockIdx() != 0) {
                return;
            }
            TBuf<TPosition::VECCALC> buffer;
            pipe->InitBuffer(buffer, (EXPERTS + WORDS) * sizeof(uint64_t));
            auto counts = buffer.Get<int64_t>();
            auto output = buffer.GetWithOffset<uint64_t>(WORDS, EXPERTS * sizeof(uint64_t));
            GlobalTensor<int64_t> input;
            input.SetGlobalBuffer((__gm__ int64_t *)countsAddress);
            DataCopy(counts, input, EXPERTS);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            uint32_t active = 0;
            uint32_t end = 0;
            for (uint32_t expert = 0; expert < EXPERTS; ++expert) {
                const auto rows = counts.GetValue(expert);
                if (rows > 0) {
                    end += rows;
                    output.SetValue(++active, (static_cast<uint64_t>(end) << 32) | expert);
                }
            }
            output.SetValue(0, active);
            const uint32_t length = ((active + 4) / 4) * 4;
            for (uint32_t i = active + 1; i < length; ++i) {
                output.SetValue(i, 0);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(records, output, length);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            pipe->Reset();
        }
    }

    __aicore__ inline void Invalidate()
    {
        // SyncAll publishes MTE3 writes, but scalar GM reads use per-core DCache.
        DataCacheCleanAndInvalid<uint64_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(records);
    }

    __aicore__ inline uint32_t Size() { return records.GetValue(0); }
    __aicore__ inline uint64_t Get(uint32_t slot) { return records.GetValue(slot + 1); }
    __aicore__ inline uint32_t End(uint32_t slot) { return Get(slot) >> 32; }

    __aicore__ inline uint32_t FindRow(uint32_t row)
    {
        uint32_t left = 0;
        uint32_t right = Size();
        while (left < right) {
            const uint32_t mid = (left + right) / 2;
            if (End(mid) <= row) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

private:
    GlobalTensor<uint64_t> records;
};
} // namespace GroupedMatmulDequantSwigluQuant
#endif
