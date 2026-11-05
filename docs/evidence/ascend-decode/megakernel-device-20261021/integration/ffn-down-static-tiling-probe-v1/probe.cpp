#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using A = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
using B = MatmulType<TPosition::GM, CubeFormat::NZ, int4b_t, false>;
using C = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using Bias = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

constexpr MatmulConfig DownConfig()
{
    MatmulConfig config = CFG_NORM;
    config.basicM = 32;
    config.basicN = 1024;
    config.basicK = 128;
    config.singleCoreM = 64;
    config.singleCoreN = 6144;
    config.singleCoreK = 128;
    config.enableSetBias = false;
    return config;
}

constexpr auto kTiling = GetMatmulApiTiling<A, B, C, Bias>(DownConfig());
static_assert(kTiling.baseM == 32 && kTiling.baseN == 1024 && kTiling.baseK == 128);
static_assert(kTiling.dbL0B == 1);
static_assert(kTiling.shareL1Size <= 524288);
static_assert(kTiling.shareL0CSize <= 131072);

extern "C" const int32_t down_tiling_fields[] __attribute__((used)) = {
    kTiling.M, kTiling.N, kTiling.Ka, kTiling.Kb,
    kTiling.singleCoreM, kTiling.singleCoreN, kTiling.singleCoreK,
    kTiling.baseM, kTiling.baseN, kTiling.baseK,
    kTiling.depthA1, kTiling.depthB1,
    kTiling.stepM, kTiling.stepN, kTiling.stepKa, kTiling.stepKb,
    kTiling.iterateOrder, kTiling.shareL1Size, kTiling.shareL0CSize,
    kTiling.shareUbSize, kTiling.dbL0A, kTiling.dbL0B, kTiling.dbL0C,
};
