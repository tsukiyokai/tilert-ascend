#include "down_plan.h"
#define g_tilingKey ffn_down_k128_planner_tiling_key
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#undef g_tilingKey

namespace {
using namespace AscendC;
using A = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
using C = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using Bias = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
template <CubeFormat format>
using B = MatmulType<TPosition::GM, format, int4b_t, false>;

constexpr MatmulConfig PlannerConfig(int base_k)
{
    MatmulConfig config = CFG_NORM;
    config.basicM = 32;
    config.basicN = 1024;
    config.basicK = base_k;
    config.singleCoreM = 64;
    config.singleCoreN = 6144;
    config.singleCoreK = 128;
    config.enableSetBias = false;
    config.enableQuantVector = false;
    return config;
}

constexpr auto parentNd = GetMatmulApiTiling<A, B<CubeFormat::ND>, C, Bias>(PlannerConfig(64));
constexpr auto parentNz = GetMatmulApiTiling<A, B<CubeFormat::NZ>, C, Bias>(PlannerConfig(64));
constexpr auto downNd = GetMatmulApiTiling<A, B<CubeFormat::ND>, C, Bias>(PlannerConfig(128));
constexpr auto downNz = GetMatmulApiTiling<A, B<CubeFormat::NZ>, C, Bias>(PlannerConfig(128));

template <class T>
constexpr bool SameOtherFields(const T& parent, const T& candidate)
{
    return parent.M == candidate.M && parent.N == candidate.N &&
        parent.Ka == candidate.Ka && parent.Kb == candidate.Kb &&
        parent.singleCoreM == candidate.singleCoreM &&
        parent.singleCoreN == candidate.singleCoreN &&
        parent.singleCoreK == candidate.singleCoreK &&
        parent.baseM == candidate.baseM && parent.baseN == candidate.baseN &&
        parent.stepM == candidate.stepM && parent.stepN == candidate.stepN &&
        parent.iterateOrder == candidate.iterateOrder &&
        parent.shareL1Size == candidate.shareL1Size &&
        parent.shareL0CSize == candidate.shareL0CSize &&
        parent.shareUbSize == candidate.shareUbSize &&
        parent.dbL0A == candidate.dbL0A && parent.dbL0C == candidate.dbL0C &&
        parent.transLength == candidate.transLength &&
        parent.shareMode == candidate.shareMode && parent.isBias == candidate.isBias;
}

template <class T>
constexpr ffn_down_k128::DownPlan Extract(const T& value)
{
    return {value.baseK, value.depthA1, value.depthB1,
            value.stepKa, value.stepKb, value.dbL0B};
}

template <class T>
constexpr bool ExpectedCandidate(const T& value)
{
    return value.M == 64 && value.N == 6144 && value.Ka == 128 && value.Kb == 128 &&
        value.singleCoreM == 64 && value.singleCoreN == 6144 && value.singleCoreK == 128 &&
        value.baseM == 32 && value.baseN == 1024 && value.baseK == 128 &&
        value.depthA1 == 2 && value.depthB1 == 6 && value.stepM == 2 && value.stepN == 6 &&
        value.stepKa == 1 && value.stepKb == 1 && value.iterateOrder == 0 &&
        value.shareL1Size == 397312 && value.shareL0CSize == 131072 &&
        value.shareUbSize == 0 && value.dbL0A == 2 && value.dbL0B == 1 && value.dbL0C == 1;
}

static_assert(SameOtherFields(parentNd, downNd) && SameOtherFields(parentNz, downNz));
static_assert(ExpectedCandidate(downNd) && ExpectedCandidate(downNz));
// Both accepted control plans also use this compile-time-only value. The
// runtime record retains its validated transLength=0 and other 176 bytes.
static_assert(parentNd.transLength == 131072 && parentNz.transLength == 131072);
static_assert(downNd.transLength == 131072 && downNz.transLength == 131072);
}

extern "C" const ffn_down_k128::DownPlan ffn_down_k128_plan_nd = Extract(downNd);
extern "C" const ffn_down_k128::DownPlan ffn_down_k128_plan_nz = Extract(downNz);
