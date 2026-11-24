#define ORIG_DTYPE_X 3
#define DT_BF16 27
#include "dequant_swiglu_quant.h"

extern "C" __global__ __aicore__ void shared_swiglu_source_reference(
    GM_ADDR product, GM_ADDR channel_scale, GM_ADDR token_scale,
    GM_ADDR codes, GM_ADDR scales, int64_t rows, int64_t vectors)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    DequantSwigluQuantBaseTilingData tiling{};
    tiling.inDimx = rows;
    tiling.inDimy = 256;
    tiling.outDimy = 128;
    tiling.UbFactorDimx = 8;
    tiling.UbFactorDimy = 128;
    tiling.usedCoreNum = vectors;
    tiling.maxCoreNum = vectors;
    tiling.inGroupNum = 1;
    tiling.quantMode = 1;
    tiling.swigluMode = 1;
    tiling.gluAlpha = 1.0F;
    AscendC::TPipe pipe;
    DequantSwigluQuantOps::DequantSwigluQuantBase<float, float, float, int32_t>
        op(&pipe);
    op.Init(product, channel_scale, token_scale, nullptr, nullptr, nullptr,
            nullptr, codes, scales, &tiling);
    op.Process();
}

extern "C" void shared_swiglu_reference_launch(uint32_t vectors, void* stream,
    void* product, void* channel_scale, void* token_scale, void* codes,
    void* scales, int64_t rows)
{
    shared_swiglu_source_reference<<<vectors, nullptr, stream>>>(
        (GM_ADDR)product, (GM_ADDR)channel_scale, (GM_ADDR)token_scale,
        (GM_ADDR)codes, (GM_ADDR)scales, rows, vectors);
}
