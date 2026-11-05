#include <torch/extension.h>
#include <torch/library.h>
namespace vllm_ascend { namespace meta {
std::tuple<at::Tensor,at::Tensor, at::Tensor> npu_add_rms_norm_bias_meta(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    const c10::optional<at::Tensor> &beta,
    double epsilon)
{
    int64_t dim_x = x1.dim();
    int64_t dim_gamma = gamma.dim();
    int64_t diff = dim_x - dim_gamma;
    c10::SymDimVector new_shape;
    at::Tensor rstd;

    if (diff > 0) {
        new_shape.reserve(dim_x);
        auto x1_sizes = x1.sym_sizes();
        for (int64_t i = 0; i < diff; ++i) {
            new_shape.push_back(x1_sizes[i]);
        }
        for (int64_t i = 0; i < dim_gamma; ++i) {
            new_shape.push_back(c10::SymInt(1));
        }
    } else {
        new_shape.assign(dim_x, c10::SymInt(1));
    }
    rstd = at::empty_symint(new_shape, x1.options().dtype(at::kFloat));
    at::Tensor y = at::empty_symint(x1.sym_sizes(), x1.options());
    at::Tensor x = at::empty_symint(x1.sym_sizes(), x1.options());
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, rstd, x);
}
std::tuple<at::Tensor,at::Tensor, at::Tensor> moe_gating_top_k_meta(
    const at::Tensor& x,
    int64_t k,
    int64_t k_group,
    int64_t group_count,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag,
    double routed_scaling_factor,
    double eps,
    const c10::optional<at::Tensor>& bias_opt

    )
{
    TORCH_CHECK(x.dim() == 2, "The x should be 2D");
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
        "float16、float32 or bfloat16 tensor expected but got a tensor with dtype: ",
        x.scalar_type());

    auto rows = x.sym_size(0);
    auto expert_num = x.sym_size(1);
    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    if (bias.defined()) {
        TORCH_CHECK(x.scalar_type() == bias.scalar_type(), "The dtype of x and bias should be same");
        TORCH_CHECK(bias.dim() == 1, "The bias should be 1D");
    }
    at::Tensor y = at::empty_symint(c10::SymDimVector{rows, c10::SymInt(k)}, x.options());
    at::Tensor expert_idx = at::empty_symint(c10::SymDimVector{rows, c10::SymInt(k)}, x.options().dtype(at::kInt));
    at::Tensor out = at::empty_symint(c10::SymDimVector{rows, expert_num}, x.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y,expert_idx,out);
}
std::tuple<at::Tensor, at::Tensor> grouped_matmul_swiglu_quant_v2_meta(
    const at::Tensor & x,
    const at::TensorList &weight,
    const at::TensorList &weight_scale,
    const at::Tensor & x_scale,
    const at::Tensor & group_list,
    const c10::optional<at::Tensor> & smooth_scale,
    const c10::optional<at::TensorList> weight_assist_matrix,
    const c10::optional<at::Tensor> & bias,
    c10::optional<int64_t> dequant_mode,
    c10::optional<int64_t> dequant_dtype,
    c10::optional<int64_t> quant_mode,
    c10::optional<int64_t> quant_dtype,
    bool transpose_weight,
    int64_t group_list_type,
    at::IntArrayRef tuning_config,
    double swiglu_limit)
{

    auto m = x.sym_size(0);
    auto n = weight_scale[0].sym_size(weight_scale[0].dim() - 1);

    c10::SymDimVector output_shape = {m, n / c10::SymInt(2)};
    c10::SymDimVector scale_shape = {m};
    at::Tensor output =  at::empty_symint(output_shape, x.options().dtype(at::kChar));
    at::Tensor output_scale =  at::empty_symint(scale_shape, x.options().dtype(at::kFloat));



    return std::tuple<at::Tensor, at::Tensor>(output, output_scale);
}
}}
TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) {
    ops.impl("npu_add_rms_norm_bias", &vllm_ascend::meta::npu_add_rms_norm_bias_meta);
    ops.impl("moe_gating_top_k", &vllm_ascend::meta::moe_gating_top_k_meta);
    ops.impl("grouped_matmul_swiglu_quant_v2", &vllm_ascend::meta::grouped_matmul_swiglu_quant_v2_meta);
}
