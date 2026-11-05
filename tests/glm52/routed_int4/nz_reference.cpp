/** Stock CANN W4A8 NZ reference with an explicit physical descriptor. */
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <array>
#include <memory>
#include <tuple>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"
#include "aclnnop/aclnn_grouped_matmul_weight_nz.h"
#include "routed_int4_contract.h"

namespace {
using namespace routed_int4;
using TensorOwner = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
using ListOwner = std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)>;

void CheckStatus(aclnnStatus status, const char* operation)
{
    TORCH_CHECK(status == ACL_SUCCESS, operation, " failed: ", status,
                "; ", aclGetRecentErrMsg());
}

void CheckTensor(const at::Tensor& value, at::ScalarType dtype,
                   at::IntArrayRef shape, const at::Device& device,
                   const char* name)
{
    TORCH_CHECK(value.device() == device && value.scalar_type() == dtype,
                name, " has the wrong device or dtype");
    TORCH_CHECK(value.is_contiguous() && value.sizes().equals(shape),
                name, " must be contiguous with shape ", shape);
    TORCH_CHECK(at_npu::native::get_npu_format(value) == ACL_FORMAT_ND,
                name, " must use physical ND storage");
}

void CheckInputs(const at::Tensor& rows, const at::Tensor& weight,
                   const at::Tensor& weight_scale, const at::Tensor& input_scale,
                   const at::Tensor& counts, const at::Tensor& correction,
                   int64_t inner, int64_t columns, bool down)
{
    TORCH_CHECK(rows.device().type() == c10::DeviceType::PrivateUse1,
                "rows must be on an Ascend device");
    TORCH_CHECK(rows.dim() == 2 && rows.size(0) >= 1 && rows.size(0) <= kMaxRoutes,
                "rows must have shape [R,K], 1 <= R <= 256");
    const auto device = rows.device();
    CheckTensor(rows, at::kChar, {rows.size(0), inner}, device, "rows");
    CheckTensor(input_scale, at::kFloat, {rows.size(0)}, device, "input_scale");
    CheckTensor(counts, at::kLong, {kMaxExperts}, device, "counts");
    CheckTensor(correction, at::kFloat, {kMaxExperts, columns}, device, "correction");
    if (down) {
        CheckTensor(weight_scale, at::kLong, {kMaxExperts, 1, columns},
                     device, "weight_scale");
    } else {
        CheckTensor(weight_scale, at::kLong, {kMaxExperts, columns},
                     device, "weight_scale");
    }
    TORCH_CHECK(weight.device() == device && weight.scalar_type() == at::kInt
                && weight.is_contiguous() && weight.storage_offset() == 0,
                "weight must be a complete contiguous NZ INT32 carrier");
    TORCH_CHECK(weight.sizes().equals({kMaxExperts, inner, columns / 8})
                && at_npu::native::get_npu_format(weight) == ACL_FORMAT_FRACTAL_NZ,
                "weight must retain logical INT32 [256,K,N/8] and physical NZ");
    TORCH_CHECK(weight.storage().nbytes()
                >= static_cast<size_t>(kMaxExperts * inner * columns / 2),
                "weight storage is too small for its NZ INT4 bank");
}

TensorOwner ND(const at::Tensor& value, aclDataType dtype)
{
    const int64_t elements = value.numel();
    auto* descriptor = aclCreateTensor(
        value.sizes().data(), value.dim(), dtype, value.strides().data(),
        0, ACL_FORMAT_ND, &elements, 1, value.data_ptr());
    TORCH_CHECK(descriptor != nullptr, "aclCreateTensor failed for an ND tensor");
    return TensorOwner(descriptor, &aclDestroyTensor);
}

TensorOwner NZ(const at::Tensor& weight, int64_t inner, int64_t columns)
{
    // Units are INT32 carriers, not the INT8 units retained by dtype view.
    // CANN's INT32-to-INT4 conversion multiplies the final storage axis by 8.
    const std::array<int64_t, 5> storage{
        kMaxExperts, columns / 64, inner / 16, 16, 8};
    auto* descriptor = aclCreateTensor(
        weight.sizes().data(), weight.dim(), ACL_INT32, weight.strides().data(),
        0, ACL_FORMAT_FRACTAL_NZ, storage.data(), storage.size(), weight.data_ptr());
    TORCH_CHECK(descriptor != nullptr, "aclCreateTensor failed for the NZ bank");
    return TensorOwner(descriptor, &aclDestroyTensor);
}

ListOwner One(TensorOwner tensor)
{
    const aclTensor* member = tensor.get();
    auto* descriptor = aclCreateTensorList(&member, 1);
    TORCH_CHECK(descriptor != nullptr, "aclCreateTensorList failed");
    // aclDestroyTensorList owns and destroys the contained descriptor.
    tensor.release();
    return ListOwner(descriptor, &aclDestroyTensorList);
}

std::tuple<at::Tensor, at::Tensor> GateUp(
    const at::Tensor& rows, const at::Tensor& weight,
    const at::Tensor& weight_scale, const at::Tensor& input_scale,
    const at::Tensor& counts, const at::Tensor& correction)
{
    CheckInputs(rows, weight, weight_scale, input_scale, counts, correction,
                 kHidden, kGateUp, false);
    const c10::OptionalDeviceGuard guard(rows.device());
    auto output = at::empty({rows.size(0), kIntermediate}, rows.options());
    auto scale = at::empty({rows.size(0)}, rows.options().dtype(at::kFloat));
    auto x = ND(rows, ACL_INT8);
    auto b = One(NZ(weight, kHidden, kGateUp));
    auto bs = One(ND(weight_scale, ACL_UINT64));
    auto assist = One(ND(correction, ACL_FLOAT));
    auto xs = ND(input_scale, ACL_FLOAT);
    auto groups = ND(counts, ACL_INT64);
    auto y = ND(output, ACL_INT8);
    auto ys = ND(scale, ACL_FLOAT);
    uint64_t workspace_bytes = 0;
    aclOpExecutor* executor = nullptr;
    // Flush Torch's queued producers before calling the direct runtime API.
    const auto stream = c10_npu::getCurrentNPUStream().stream();
    CheckStatus(aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(
        x.get(), b.get(), bs.get(), assist.get(), nullptr, xs.get(), nullptr,
        groups.get(), 0, 0, 0, 1, nullptr, y.get(), ys.get(),
        &workspace_bytes, &executor), "NZ GMM1 GetWorkspaceSize");
    auto workspace = at::empty({static_cast<int64_t>(workspace_bytes)},
                                rows.options().dtype(at::kByte));
    CheckStatus(aclnnGroupedMatmulSwigluQuantWeightNzV2(
        workspace.data_ptr(), workspace_bytes, executor, stream), "NZ GMM1");
    return {output, scale};
}

at::Tensor Down(
    const at::Tensor& rows, const at::Tensor& weight,
    const at::Tensor& weight_scale, const at::Tensor& input_scale,
    const at::Tensor& counts, const at::Tensor& correction)
{
    CheckInputs(rows, weight, weight_scale, input_scale, counts, correction,
                 kIntermediate, kHidden, true);
    const c10::OptionalDeviceGuard guard(rows.device());
    auto output = at::empty({rows.size(0), kHidden}, rows.options().dtype(at::kBFloat16));
    auto x = One(ND(rows, ACL_INT8));
    auto b = One(NZ(weight, kIntermediate, kHidden));
    auto bs = One(ND(weight_scale, ACL_UINT64));
    auto bias = One(ND(correction, ACL_FLOAT));
    auto xs = One(ND(input_scale, ACL_FLOAT));
    auto groups = ND(counts, ACL_INT64);
    auto y = One(ND(output, ACL_BF16));
    uint64_t workspace_bytes = 0;
    aclOpExecutor* executor = nullptr;
    const auto stream = c10_npu::getCurrentNPUStream().stream();
    CheckStatus(aclnnGroupedMatmulWeightNzGetWorkspaceSize(
        x.get(), b.get(), bias.get(), bs.get(), nullptr, nullptr, nullptr,
        xs.get(), groups.get(), nullptr, nullptr, nullptr, 2, 0, 1, 0,
        nullptr, 0, y.get(), nullptr, nullptr, &workspace_bytes, &executor),
        "NZ GMM2 GetWorkspaceSize");
    auto workspace = at::empty({static_cast<int64_t>(workspace_bytes)},
                                rows.options().dtype(at::kByte));
    CheckStatus(aclnnGroupedMatmulWeightNz(
        workspace.data_ptr(), workspace_bytes, executor, stream), "NZ GMM2");
    return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(tilert_reference, library)
{
    library.def("nz_gmm1(Tensor rows, Tensor weight, Tensor weight_scale, "
                "Tensor input_scale, Tensor counts, Tensor correction) -> (Tensor, Tensor)");
    library.def("nz_gmm2(Tensor(a!) rows, Tensor weight, Tensor weight_scale, "
                "Tensor input_scale, Tensor counts, Tensor correction) -> Tensor");
}

TORCH_LIBRARY_IMPL(tilert_reference, PrivateUse1, library)
{
    library.impl("nz_gmm1", &GateUp);
    library.impl("nz_gmm2", &Down);
}
