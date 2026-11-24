#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "norm_contract.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* address, uint32_t* length);

namespace {
using namespace whole_layer_norm;

struct DeviceLaunch {
    int64_t maximum_blocks;
    uint64_t sync_base;
};
std::mutex launch_mutex;
std::unordered_map<int, DeviceLaunch> launches;

void CheckTensor(const at::Tensor& tensor, const at::Tensor& reference,
    at::ScalarType dtype, at::IntArrayRef shape, const char* name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name, " device differs");
    TORCH_CHECK(tensor.scalar_type() == dtype && tensor.sizes().equals(shape),
                name, " has the wrong dtype or shape");
    TORCH_CHECK(tensor.is_contiguous() &&
                at_npu::native::get_npu_format(tensor) == ACL_FORMAT_ND,
                name, " must have contiguous physical ND storage");
}

std::tuple<int64_t, int64_t> Prepare(const at::Tensor& input, int64_t blocks)
{
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
                "norm_prepare requires an NPU tensor");
    const c10::OptionalDeviceGuard device_guard(input.device());
    std::lock_guard<std::mutex> guard(launch_mutex);
    int64_t cubes = 0;
    int64_t vectors = 0;
    TORCH_CHECK(aclrtGetDeviceInfo(input.get_device(), ACL_DEV_ATTR_AICORE_CORE_NUM,
                                  &cubes) == ACL_SUCCESS, "cannot query AIC count");
    TORCH_CHECK(aclrtGetDeviceInfo(input.get_device(), ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                  &vectors) == ACL_SUCCESS, "cannot query AIV count");
    DeviceLaunch launch{std::min(cubes, vectors / 2), 0};
    uint32_t length = 0;
    TORCH_CHECK(launch.maximum_blocks > 0 &&
                rtGetC2cCtrlAddr(&launch.sync_base, &length) == 0 &&
                launch.sync_base != 0, "cannot prepare mixed groups and C2C sync");
    if (blocks == 0) {
        blocks = std::min<int64_t>(8, launch.maximum_blocks);
    }
    TORCH_CHECK(blocks > 0 && blocks <= launch.maximum_blocks,
                "blocks exceed the physical mixed-group residency limit");
    launches.insert_or_assign(input.get_device(), launch);
    return {blocks, static_cast<int64_t>(launch.sync_base)};
}

void CheckInputs(const at::Tensor& input, const at::Tensor& residual,
    const at::Tensor& gamma, const c10::optional<at::Tensor>& beta,
    const at::Tensor& normalized, const at::Tensor& retained,
    const at::Tensor& rstd, double epsilon, int64_t blocks, int64_t sync_base)
{
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                input.dim() == 2, "input must be an NPU matrix");
    const int64_t rows = input.size(0);
    TORCH_CHECK(rows >= 1 && rows <= kMaximumRows, "rows must be in [1,32]");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0 &&
                epsilon <= std::numeric_limits<float>::max() &&
                static_cast<float>(epsilon) > 0,
                "epsilon must be finite and positive in FP32");
    CheckTensor(input, input, at::kBFloat16, {rows, kHidden}, "input");
    CheckTensor(residual, input, at::kBFloat16, {rows, kHidden}, "residual");
    CheckTensor(gamma, input, at::kBFloat16, {kHidden}, "gamma");
    CheckTensor(normalized, input, at::kBFloat16, {rows, kHidden}, "normalized");
    CheckTensor(retained, input, at::kBFloat16, {rows, kHidden}, "retained");
    CheckTensor(rstd, input, at::kFloat, {rows, 1}, "rstd");
    std::vector<at::Tensor> read{input, residual, gamma};
    if (beta.has_value()) {
        CheckTensor(*beta, input, at::kBFloat16, {kHidden}, "beta");
        read.push_back(*beta);
    }
    const std::vector<at::Tensor> written{normalized, retained, rstd};
    for (size_t i = 0; i < written.size(); ++i) {
        for (const auto& source : read) {
            at::assert_no_overlap(written[i], source);
        }
        for (size_t j = 0; j < i; ++j) {
            at::assert_no_overlap(written[i], written[j]);
        }
    }
    std::lock_guard<std::mutex> guard(launch_mutex);
    const auto found = launches.find(input.get_device());
    TORCH_CHECK(found != launches.end() && blocks > 0 &&
                blocks <= found->second.maximum_blocks &&
                static_cast<uint64_t>(sync_base) == found->second.sync_base,
                "call norm_prepare for this context before capture");
}

at::Tensor BiasOut(const at::Tensor& input, const at::Tensor& residual,
    const at::Tensor& gamma, const c10::optional<at::Tensor>& beta,
    at::Tensor& normalized, at::Tensor& retained, at::Tensor& rstd,
    double epsilon, int64_t blocks, int64_t sync_base)
{
    const c10::OptionalDeviceGuard device_guard(input.device());
    CheckInputs(input, residual, gamma, beta, normalized, retained, rstd,
                epsilon, blocks, sync_base);
    whole_layer_norm_launch(blocks, c10_npu::getCurrentNPUStream().stream(),
        sync_base, input.data_ptr(), residual.data_ptr(), gamma.data_ptr(),
        beta.has_value() ? beta->data_ptr() : nullptr, normalized.data_ptr(),
        retained.data_ptr(), rstd.data_ptr<float>(), nullptr, nullptr,
        input.size(0), static_cast<float>(epsilon));
    return normalized;
}

at::Tensor QuantOut(const at::Tensor& input, const at::Tensor& residual,
    const at::Tensor& gamma, const c10::optional<at::Tensor>& beta,
    at::Tensor& normalized, at::Tensor& retained, at::Tensor& rstd,
    at::Tensor& quantized, at::Tensor& scales,
    double epsilon, int64_t blocks, int64_t sync_base)
{
    const c10::OptionalDeviceGuard device_guard(input.device());
    CheckInputs(input, residual, gamma, beta, normalized, retained, rstd,
                epsilon, blocks, sync_base);
    CheckTensor(quantized, input, at::kChar, {input.size(0), kHidden}, "quantized");
    CheckTensor(scales, input, at::kFloat, {input.size(0), 1}, "scales");
    for (const auto& tensor : {input, residual, gamma, normalized, retained, rstd}) {
        at::assert_no_overlap(quantized, tensor);
        at::assert_no_overlap(scales, tensor);
    }
    if (beta.has_value()) {
        at::assert_no_overlap(quantized, *beta);
        at::assert_no_overlap(scales, *beta);
    }
    at::assert_no_overlap(quantized, scales);
    whole_layer_norm_launch(blocks, c10_npu::getCurrentNPUStream().stream(),
        sync_base, input.data_ptr(), residual.data_ptr(), gamma.data_ptr(),
        beta.has_value() ? beta->data_ptr() : nullptr, normalized.data_ptr(),
        retained.data_ptr(), rstd.data_ptr<float>(), quantized.data_ptr(),
        scales.data_ptr<float>(), input.size(0), static_cast<float>(epsilon));
    return quantized;
}
}

TORCH_LIBRARY_FRAGMENT(tilert, library)
{
    library.def("norm_prepare(Tensor input, int blocks=0) -> (int, int)");
    library.def("norm_bias_out(Tensor input, Tensor residual, Tensor gamma, "
        "Tensor? beta, Tensor(a!) normalized, Tensor(b!) retained, Tensor(c!) rstd, "
        "float epsilon, int blocks, int sync_base) -> Tensor(a!)");
    library.def("norm_quant_out(Tensor input, Tensor residual, Tensor gamma, "
        "Tensor? beta, Tensor(a!) normalized, Tensor(b!) retained, Tensor(c!) rstd, "
        "Tensor(d!) quantized, Tensor(e!) scales, "
        "float epsilon, int blocks, int sync_base) -> Tensor(d!)");
}

TORCH_LIBRARY_IMPL(tilert, PrivateUse1, library)
{
    library.impl("norm_prepare", &Prepare);
    library.impl("norm_bias_out", &BiasOut);
    library.impl("norm_quant_out", &QuantOut);
}
