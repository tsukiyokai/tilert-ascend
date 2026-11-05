#include <dlfcn.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "aclnn_torch_adapter/op_api_common.h"
#include "moe/moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h"
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
    ops.def(
        "moe_gating_top_k(Tensor x, "
                            "int k, "
                            "int k_group, "
                            "int group_count, "
                            "int group_select_mode, "
                            "int renorm, "
                            "int norm_type, "
                            "bool out_flag, "
                            "float routed_scaling_factor, "
                            "float eps,"
                            "Tensor? bias_opt=None)"

        "-> (Tensor y ,Tensor expert_idx, Tensor out)"
        );
    ops.impl("moe_gating_top_k", torch::kPrivateUse1,&vllm_ascend::moe_gating_top_k);
    ops.def(
        "npu_add_rms_norm_bias(Tensor x1, "
                            "Tensor x2, "
                            "Tensor gamma, "
                            "Tensor? beta=None, "
                            "float epsilon=1e-6)"
        "-> (Tensor y ,Tensor rstd, Tensor x)"
        );
    ops.impl("npu_add_rms_norm_bias", torch::kPrivateUse1, &vllm_ascend::npu_add_rms_norm_bias);
}
extern "C" const char* ffn_bridge_oracle_api_library(const char* name) {
    void* address = GetOpApiFuncAddr(name);
    Dl_info information{};
    if (address == nullptr || dladdr(address, &information) == 0) return nullptr;
    return information.dli_fname;
}
