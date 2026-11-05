#define ASCENDC_CUBE_ONLY
#define C_TEMPLATE 0
#define V_TEMPLATE 1
#define EVENT_ID4 4U
#define EVENT_ID5 5U
#define EVENT_ID6 6U
#define EVENT_ID7 7U

#include "attention_sfa_tiling.h"
#include "attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_kernel_mla.h"

using namespace AscendC;

// Compilation instantiates the complete mixed SFA path. This file has no host
// launcher and is not a numerical or performance harness.
extern "C" __global__ __mix__(1, 2) __schedmode__(1) __aicore__
void sfa_packed_shadow_compile_probe(
    uint64_t sync_base, SparseFlashAttentionTilingDataMla tiling,
    GM_ADDR query, GM_ADDR shadow, GM_ADDR original_kv,
    GM_ADDR query_rope, GM_ADDR original_rope, GM_ADDR topk,
    GM_ADDR query_lengths, GM_ADDR kv_lengths, GM_ADDR page_table,
    GM_ADDR workspace, GM_ADDR output)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    TPipe pipe;
    {
        using Type = SFAType<bfloat16_t, bfloat16_t, bfloat16_t,
            false, SFA_LAYOUT::TND, SFA_LAYOUT::PA_BSND, V_TEMPLATE, true>;
        SparseFlashAttentionMla<Type> attention;
        attention.Init(query, shadow, original_kv, topk,
            query_lengths, kv_lengths, page_table, query_rope, original_rope,
            output, nullptr, nullptr, workspace, &tiling, nullptr, &pipe);
        attention.Process();
    }
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
    SyncAll<false>();
    pipe.Reset();
}
