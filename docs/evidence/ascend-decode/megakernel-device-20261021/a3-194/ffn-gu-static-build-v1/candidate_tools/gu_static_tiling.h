#ifndef TILERT_FFN_GU_STATIC_TILING_H
#define TILERT_FFN_GU_STATIC_TILING_H

#include "lib/matmul_intf.h"

namespace ffn_gu_static {

// M and singleCoreM retain the host tiling's capacity. SetSingleShape supplies
// each expert's effective 2*count rows through MatmulShapeInfo at runtime.
__aicore__ constexpr AscendC::MatmulApiStaticTiling GateTiling()
{
    AscendC::MatmulApiStaticTiling value{};
    value.N = 256;
    value.Ka = 6144;
    value.Kb = 6144;
    value.singleCoreN = 256;
    value.singleCoreK = 6144;
    value.baseM = 64;
    value.baseN = 256;
    value.baseK = 256;
    value.depthA1 = 24;
    value.depthB1 = 8;
    value.stepM = 1;
    value.stepN = 1;
    value.stepKa = 24;
    value.stepKb = 4;
    value.iterateOrder = 1;
    value.shareL1Size = 458752;
    value.shareL0CSize = 65536;
    value.shareUbSize = 0;
    value.dbL0A = 2;
    value.dbL0B = 2;
    value.dbL0C = 1;
    value.cfg = AscendC::CFG_NORM;
    return value;
}

inline constexpr auto kGateTiling = GateTiling();
static_assert(kGateTiling.M == -1 && kGateTiling.singleCoreM == -1);
static_assert(!kGateTiling.cfg.doBasicBlock);
static_assert(!kGateTiling.cfg.doSpecialBasicBlock);
static_assert(!kGateTiling.cfg.doMultiDataLoad);
static_assert(!kGateTiling.cfg.enableStaticPadZeros);
static_assert(kGateTiling.cfg.singleCoreM == 0);

}  // namespace ffn_gu_static

#endif
