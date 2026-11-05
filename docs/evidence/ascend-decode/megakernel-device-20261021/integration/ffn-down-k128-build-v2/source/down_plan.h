#ifndef TILERT_FFN_DOWN_K128_STATIC_PLAN_H
#define TILERT_FFN_DOWN_K128_STATIC_PLAN_H

#include <cstdint>
#include <type_traits>

namespace ffn_down_k128 {
struct DownPlan {
    int32_t baseK;
    int32_t depthA1;
    int32_t depthB1;
    int32_t stepKa;
    int32_t stepKb;
    int32_t dbL0B;
};
static_assert(sizeof(DownPlan) == 24);
static_assert(std::is_standard_layout_v<DownPlan>);

template <class Cube>
void ApplyDownPlan(Cube& cube, const DownPlan& plan)
{
    cube.baseK = plan.baseK;
    cube.depthA1 = plan.depthA1;
    cube.depthB1 = plan.depthB1;
    cube.stepKa = plan.stepKa;
    cube.stepKb = plan.stepKb;
    cube.dbL0B = plan.dbL0B;
}
}

extern "C" const ffn_down_k128::DownPlan ffn_down_k128_plan_nd;
extern "C" const ffn_down_k128::DownPlan ffn_down_k128_plan_nz;

#endif
