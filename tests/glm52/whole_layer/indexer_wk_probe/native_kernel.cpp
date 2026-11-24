#include "indexer_front_native.h"

using namespace AscendC;

extern "C" __global__ __mix__(1, 2) __schedmode__(1)
__aicore__ void wk_geometry_kernel(GM_ADDR hidden, GM_ADDR weight,
    GM_ADDR output, GM_ADDR tiling, GM_ADDR workspace, int32_t columns)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSysWorkspace(workspace);
    if ASCEND_IS_AIC {
        TPipe pipe;
        indexer_front::Config config{};
        config.rows = 32;
        config.capacity = 32;
        config.blocks = GetBlockNum();
        indexer_front::device::Project<bfloat16_t, float>(
            pipe, config, hidden, weight, output, tiling, columns, 6144,
            160 / columns);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
    }
}

extern "C" void wk_geometry_launch(void* stream, uint32_t blocks,
    void* hidden, void* weight, void* output, void* tiling, void* workspace,
    int32_t columns)
{
    wk_geometry_kernel<<<blocks, nullptr, stream>>>(
        (GM_ADDR)hidden, (GM_ADDR)weight, (GM_ADDR)output, (GM_ADDR)tiling,
        (GM_ADDR)workspace, columns);
}
