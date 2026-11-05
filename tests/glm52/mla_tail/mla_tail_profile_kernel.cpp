#include "mla_tail_native.h"
#include "mla_tail_profile.h"

using namespace AscendC;
using namespace mla_tail;

namespace {

struct StageTrace {
    int64_t ticks[profile::kStageCount * profile::kMarksPerStage];

    __aicore__ inline void Begin(int stage)
    {
        ticks[stage * profile::kMarksPerStage] = GetSystemCycle();
    }

    __aicore__ inline void Finish(int stage)
    {
        // Delimit completed work rather than asynchronous instruction issue.
        PipeBarrier<PIPE_ALL>();
        ticks[stage * profile::kMarksPerStage + 1] = GetSystemCycle();
        SyncAll<false>();
        ticks[stage * profile::kMarksPerStage + 2] = GetSystemCycle();
    }

    __aicore__ inline void Store(__gm__ int64_t* trace, int64_t rows,
                                 int64_t capacity, int64_t blocks)
    {
        int64_t kind = 0;
        if ASCEND_IS_AIV {
            kind = 1;
        }
        const int64_t index = GetBlockIdx();
        const int64_t slot = kind == 0 ? index : blocks + index;
        GlobalTensor<int64_t> record;
        record.SetGlobalBuffer(trace + slot * profile::kRecordWords,
                                profile::kRecordWords);
        for (int word = 0; word < profile::kRecordWords; ++word) {
            record.SetValue(word, 0);
        }
        record.SetValue(0, profile::kVersion);
        record.SetValue(1, kind);
        record.SetValue(2, index);
        record.SetValue(3, rows);
        record.SetValue(4, capacity);
        record.SetValue(5, blocks);
        for (int mark = 0; mark < profile::kStageCount * profile::kMarksPerStage;
             ++mark) {
            record.SetValue(profile::kTimestampOffset + mark, ticks[mark]);
        }
        // A core owns the complete 512-byte record. Publish scalar-cache
        // writes before kernel completion; no peer reads these records.
        for (int word = 0; word < profile::kRecordWords; word += 8) {
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(record[word]);
        }
        DataSyncBarrier<MemDsbT::DDR>();
    }
};

}  // namespace

extern "C" __global__ __mix__(1, 2) __schedmode__(1)
__aicore__ void mla_tail_profile_fused(
    uint64_t sync_base, __gm__ bfloat16_t* latent,
    __gm__ bfloat16_t* value_weight, __gm__ int8_t* output_weight,
    __gm__ float* output_scale, __gm__ uint8_t* bias, int bias_kind,
    __gm__ uint8_t* scratch, __gm__ uint8_t* tiling,
    __gm__ bfloat16_t* output, __gm__ int64_t* trace,
    int64_t rows, int64_t capacity, int64_t blocks,
    uint64_t values_offset, uint64_t quantized_offset,
    uint64_t scales_offset, uint64_t accumulators_offset,
    uint64_t system_offset)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    SetSysWorkspace(scratch + system_offset);
    TPipe pipe;
    auto* heads = reinterpret_cast<__gm__ bfloat16_t*>(scratch);
    auto* values = reinterpret_cast<__gm__ float*>(scratch + values_offset);
    auto* quantized = reinterpret_cast<__gm__ int8_t*>(scratch + quantized_offset);
    auto* scales = reinterpret_cast<__gm__ float*>(scratch + scales_offset);
    auto* accumulators =
        reinterpret_cast<__gm__ int32_t*>(scratch + accumulators_offset);
    device::VectorWorkspace held;
    StageTrace timing;

    timing.Begin(0);
    if ASCEND_IS_AIV {
        held.Init(pipe);
        device::StageHeads(held, latent, heads, rows, capacity,
                            GetBlockIdx(), 2 * blocks);
        device::Publish();
    }
    timing.Finish(0);

    timing.Begin(1);
    if ASCEND_IS_AIC {
        device::ProjectValues(pipe, heads, value_weight, values, tiling,
                               capacity, blocks);
        device::Publish();
    }
    timing.Finish(1);

    timing.Begin(2);
    if ASCEND_IS_AIV {
        device::QuantizeValues(held, values, quantized, scales, capacity,
                                GetBlockIdx(), 2 * blocks);
        device::Publish();
    }
    timing.Finish(2);

    timing.Begin(3);
    if ASCEND_IS_AIC {
        pipe.Reset();
        device::ProjectOutput(pipe, quantized, output_weight, accumulators,
                               tiling + kCubeTilingBytes, capacity, blocks);
        device::Publish();
    }
    timing.Finish(3);

    timing.Begin(4);
    if ASCEND_IS_AIV {
        device::DequantizeOutput(held, accumulators, output_scale, scales,
            bias, bias_kind, output, rows, capacity, GetBlockIdx(), 2 * blocks);
        device::Publish();
    }
    // This fifth grid barrier is instrumentation-only. It keeps trace stores
    // from idle AICs out of the final AIV stage being measured.
    timing.Finish(4);
    timing.Store(trace, rows, capacity, blocks);
}

extern "C" void mla_tail_profile_launch(
    uint32_t blocks, void* stream, uint64_t sync_base,
    void* latent, void* value_weight, void* output_weight,
    void* output_scale, void* bias, int bias_kind, void* scratch,
    void* tiling, void* output, void* trace, int64_t rows, int64_t capacity)
{
    const WorkspaceLayout layout = MakeWorkspaceLayout(capacity);
    mla_tail_profile_fused<<<blocks, nullptr, stream>>>(
        sync_base, (__gm__ bfloat16_t*)latent,
        (__gm__ bfloat16_t*)value_weight, (__gm__ int8_t*)output_weight,
        (__gm__ float*)output_scale, (__gm__ uint8_t*)bias, bias_kind,
        (__gm__ uint8_t*)scratch, (__gm__ uint8_t*)tiling,
        (__gm__ bfloat16_t*)output, (__gm__ int64_t*)trace,
        rows, capacity, blocks, layout.values, layout.quantized,
        layout.token_scales, layout.accumulators, layout.system);
}
