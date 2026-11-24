#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(AddRmsNormBias)
    .INPUT(x1, ge::TensorType::ALL())
    .INPUT(x2, ge::TensorType::ALL())
    .INPUT(gamma, ge::TensorType::ALL())
    .OPTIONAL_INPUT(beta, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(rstd, ge::TensorType::ALL())
    .OUTPUT(x, ge::TensorType::ALL())
    .ATTR(epsilon, Float, 9.99999997e-07)
    .OP_END_FACTORY_REG(AddRmsNormBias);

REG_OP(MoeGatingTopK)
    .INPUT(x, ge::TensorType::ALL())
    .OPTIONAL_INPUT(bias, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(expert_idx, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .REQUIRED_ATTR(k, Int)
    .ATTR(k_group, Int, 1)
    .ATTR(group_count, Int, 1)
    .ATTR(group_select_mode, Int, 0)
    .ATTR(renorm, Int, 0)
    .ATTR(norm_type, Int, 0)
    .ATTR(out_flag, Bool, false)
    .ATTR(routed_scaling_factor, Float, 1)
    .ATTR(eps, Float, 9.99999968e-21)
    .OP_END_FACTORY_REG(MoeGatingTopK);

}

#endif
