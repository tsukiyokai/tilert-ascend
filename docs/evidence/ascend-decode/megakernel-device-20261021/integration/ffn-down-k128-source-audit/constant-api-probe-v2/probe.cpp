#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using A = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
using C = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using Bias = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

constexpr MatmulConfig DownConfig(int base_m, int base_n, int base_k)
{
    MatmulConfig config = CFG_NORM;
    config.basicM = base_m;
    config.basicN = base_n;
    config.basicK = base_k;
    config.singleCoreM = 64;
    config.singleCoreN = 6144;
    config.singleCoreK = 128;
    config.enableSetBias = false;
    config.enableQuantVector = false;
    return config;
}

template <CubeFormat format>
using B = MatmulType<TPosition::GM, format, int4b_t, false>;

#define DEFINE_TILING(name, format, base_m, base_n, base_k) \
    constexpr auto name = GetMatmulApiTiling<A, B<format>, C, Bias>( \
        DownConfig(base_m, base_n, base_k)); \
    extern "C" const int32_t name##_fields[] __attribute__((used)) = { \
        name.M, name.N, name.Ka, name.Kb, \
        name.singleCoreM, name.singleCoreN, name.singleCoreK, \
        name.baseM, name.baseN, name.baseK, name.depthA1, name.depthB1, \
        name.stepM, name.stepN, name.stepKa, name.stepKb, name.iterateOrder, \
        name.shareL1Size, name.shareL0CSize, name.shareUbSize, \
        name.dbL0A, name.dbL0B, name.dbL0C, \
        name.transLength, name.shareMode, name.isBias, \
    }

DEFINE_TILING(n512_nd, CubeFormat::ND, 64, 512, 128);
DEFINE_TILING(n512_nz, CubeFormat::NZ, 64, 512, 128);
DEFINE_TILING(n1024_nd, CubeFormat::ND, 32, 1024, 64);
DEFINE_TILING(n1024_nz, CubeFormat::NZ, 32, 1024, 64);
DEFINE_TILING(k128_nd, CubeFormat::ND, 32, 1024, 128);
DEFINE_TILING(k128_nz, CubeFormat::NZ, 32, 1024, 128);

static_assert(k128_nz.baseM == 32 && k128_nz.baseN == 1024 && k128_nz.baseK == 128);
static_assert(k128_nz.dbL0B == 1 && k128_nd.dbL0B == 1);
static_assert(k128_nz.shareL1Size == 397312 && k128_nd.shareL1Size == 397312);
static_assert(k128_nz.shareL0CSize == 131072);
