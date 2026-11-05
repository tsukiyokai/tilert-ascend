#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "contract.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* address, uint32_t* length);

namespace {
using namespace ffn_bridge;
using Tensors = std::vector<at::Tensor>;

struct Prepared {
    c10::weak_intrusive_ptr<c10::TensorImpl> tiling;
    c10::weak_intrusive_ptr<c10::TensorImpl> system;
    int device;
    int64_t rows;
    int64_t blocks;
    uint64_t sync;
};
std::mutex prepare_mutex;
std::unordered_map<const c10::TensorImpl*, Prepared> prepared;

void CheckTensor(const at::Tensor& tensor, const at::Tensor& reference,
    at::ScalarType dtype, at::IntArrayRef shape, const char* name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name, " device differs");
    TORCH_CHECK(tensor.scalar_type() == dtype && tensor.sizes().equals(shape),
                name, " dtype or shape differs");
    TORCH_CHECK(tensor.is_contiguous() &&
                at_npu::native::get_npu_format(tensor) == ACL_FORMAT_ND,
                name, " requires contiguous physical ND storage");
}

Tensors Allocate(const at::Tensor& input, int64_t rows)
{
    auto make = [&](at::IntArrayRef shape, at::ScalarType dtype) {
        return at::empty(shape, input.options().dtype(dtype));
    };
    return {
        make({rows, kHidden}, at::kBFloat16),
        make({rows, kHidden}, at::kBFloat16),
        make({rows, 1}, at::kFloat),
        make({rows, kExperts}, at::kFloat),
        make({rows, kTopK}, at::kFloat),
        make({rows, kTopK}, at::kInt),
        make({rows, kHidden}, at::kChar),
        make({rows, 1}, at::kFloat),
        make({kTilingBytes}, at::kByte),
        make({kSystemBytes}, at::kByte),
        make({rows, kHidden}, at::kBFloat16),
        make({rows, kHidden}, at::kBFloat16),
    };
}

std::tuple<Tensors, int64_t> Prepare(const at::Tensor& input, int64_t blocks)
{
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                input.dim() == 2, "prepare requires an NPU matrix");
    const c10::OptionalDeviceGuard device_guard(input.device());
    const int64_t rows = input.size(0);
    TORCH_CHECK(ValidGeometry(rows, blocks), "require M in [1,32], blocks 16 or 24");
    CheckTensor(input, input, at::kBFloat16, {rows, kHidden}, "input");
    int64_t cubes = 0, vectors = 0;
    TORCH_CHECK(aclrtGetDeviceInfo(input.get_device(), ACL_DEV_ATTR_AICORE_CORE_NUM,
                                  &cubes) == ACL_SUCCESS &&
                aclrtGetDeviceInfo(input.get_device(), ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                  &vectors) == ACL_SUCCESS,
                "cannot query physical mixed-group capacity");
    TORCH_CHECK(blocks <= std::min(cubes, vectors / 2),
                "blocks exceed physical mixed-group residency");
    uint64_t sync = 0;
    uint32_t sync_length = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&sync, &sync_length) == 0 && sync != 0,
                "cannot obtain C2C synchronization base");
    std::array<uint8_t, kTilingBytes> data{};
    TORCH_CHECK(ffn_profile_bridge_build_tiling(Capacity(rows), data.data()) == 0,
                "gate tiling failed");
    auto scratch = Allocate(input, rows);
    TORCH_CHECK(aclrtMemcpy(scratch[kTiling].data_ptr(), kTilingBytes,
                data.data(), kTilingBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
                "cannot initialize immutable gate tiling");
    std::lock_guard<std::mutex> guard(prepare_mutex);
    for (auto item = prepared.begin(); item != prepared.end();) {
        if (item->second.tiling.expired() || item->second.system.expired()) {
            item = prepared.erase(item);
        } else {
            ++item;
        }
    }
    prepared.insert_or_assign(scratch[kTiling].unsafeGetTensorImpl(), Prepared{
        c10::weak_intrusive_ptr<c10::TensorImpl>(scratch[kTiling].getIntrusivePtr()),
        c10::weak_intrusive_ptr<c10::TensorImpl>(scratch[kSystem].getIntrusivePtr()),
        input.get_device(), rows, blocks, sync});
    return {scratch, static_cast<int64_t>(sync)};
}

int64_t CheckScratch(const Tensors& scratch, int64_t blocks, int64_t sync)
{
    TORCH_CHECK(scratch.size() == kBuffers, "bridge scratch requires 12 tensors");
    const auto& ref = scratch[kNormalized];
    TORCH_CHECK(ref.device().type() == c10::DeviceType::PrivateUse1 && ref.dim() == 2,
                "scratch requires NPU matrices");
    const int64_t rows = ref.size(0);
    TORCH_CHECK(ValidGeometry(rows, blocks), "invalid bridge geometry");
    for (const auto slot : {kNormalized, kRetained, kCombined, kScaledRouted}) {
        CheckTensor(scratch[slot], ref, at::kBFloat16, {rows, kHidden}, "BF16 output");
    }
    for (const auto slot : {kRstd, kSharedScales}) {
        CheckTensor(scratch[slot], ref, at::kFloat, {rows, 1}, "row scalar");
    }
    CheckTensor(scratch[kLogits], ref, at::kFloat, {rows, kExperts}, "logits");
    CheckTensor(scratch[kProbabilities], ref, at::kFloat, {rows, kTopK}, "probabilities");
    CheckTensor(scratch[kIds], ref, at::kInt, {rows, kTopK}, "expert IDs");
    CheckTensor(scratch[kSharedCodes], ref, at::kChar, {rows, kHidden}, "shared A8");
    CheckTensor(scratch[kTiling], ref, at::kByte, {kTilingBytes}, "tiling");
    CheckTensor(scratch[kSystem], ref, at::kByte, {kSystemBytes}, "system workspace");
    for (size_t i = 0; i < scratch.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            at::assert_no_overlap(scratch[i], scratch[j]);
        }
    }
    std::lock_guard<std::mutex> guard(prepare_mutex);
    const auto found = prepared.find(scratch[kTiling].unsafeGetTensorImpl());
    TORCH_CHECK(found != prepared.end(), "call prepare before capture");
    const auto& metadata = found->second;
    TORCH_CHECK(metadata.tiling.lock().get() == scratch[kTiling].unsafeGetTensorImpl() &&
                metadata.system.lock().get() == scratch[kSystem].unsafeGetTensorImpl() &&
                metadata.device == ref.get_device() && metadata.rows == rows &&
                metadata.blocks == blocks && metadata.sync == static_cast<uint64_t>(sync),
                "scratch, geometry and synchronization must belong to one prepare call");
    return rows;
}

std::array<void*, kBuffers> Addresses(const Tensors& scratch, const Tensors& input)
{
    std::array<void*, kBuffers> result;
    for (int64_t i = 0; i < kBuffers; ++i) {
        for (const auto& tensor : input) {
            at::assert_no_overlap(scratch[i], tensor);
        }
        result[i] = scratch[i].data_ptr();
    }
    return result;
}

void Forward(const at::Tensor& input, const at::Tensor& residual,
    const at::Tensor& gamma, const at::Tensor& weight, const at::Tensor& bias,
    Tensors scratch, double epsilon, int64_t blocks, int64_t sync)
{
    const c10::OptionalDeviceGuard device_guard(input.device());
    const int64_t rows = CheckScratch(scratch, blocks, sync);
    const auto& ref = scratch[kNormalized];
    CheckTensor(input, ref, at::kBFloat16, {rows, kHidden}, "input");
    CheckTensor(residual, ref, at::kBFloat16, {rows, kHidden}, "residual");
    CheckTensor(gamma, ref, at::kBFloat16, {kHidden}, "gamma");
    CheckTensor(weight, ref, at::kBFloat16, {kHidden, kExperts}, "gate weight");
    CheckTensor(bias, ref, at::kFloat, {kExperts}, "correction bias");
    TORCH_CHECK(std::isfinite(epsilon) && static_cast<float>(epsilon) > 0 &&
                std::isfinite(static_cast<float>(epsilon)), "epsilon must be positive FP32");
    auto addresses = Addresses(scratch, {input, residual, gamma, weight, bias});
    ffn_bridge_launch(c10_npu::getCurrentNPUStream().stream(), sync,
        input.data_ptr(), residual.data_ptr(), gamma.data_ptr(), weight.data_ptr(),
        bias.data_ptr(), addresses.data(), rows, blocks, static_cast<float>(epsilon));
}

void TopK(const at::Tensor& logits, const at::Tensor& bias, Tensors scratch,
          int64_t blocks, int64_t sync)
{
    const c10::OptionalDeviceGuard device_guard(logits.device());
    const int64_t rows = CheckScratch(scratch, blocks, sync);
    CheckTensor(logits, scratch[0], at::kFloat, {rows, kExperts}, "fixed logits");
    CheckTensor(bias, scratch[0], at::kFloat, {kExperts}, "correction bias");
    auto addresses = Addresses(scratch, {logits, bias});
    ffn_bridge_topk_launch(c10_npu::getCurrentNPUStream().stream(), sync,
        logits.data_ptr(), bias.data_ptr(), addresses.data(), rows, blocks);
}

void Finalize(const at::Tensor& routed, const at::Tensor& shared,
              Tensors scratch, int64_t blocks, int64_t sync)
{
    const c10::OptionalDeviceGuard device_guard(routed.device());
    const int64_t rows = CheckScratch(scratch, blocks, sync);
    CheckTensor(routed, scratch[0], at::kBFloat16, {rows, kHidden}, "routed output");
    CheckTensor(shared, scratch[0], at::kBFloat16, {rows, kHidden}, "shared output");
    auto addresses = Addresses(scratch, {routed, shared});
    ffn_bridge_epilogue_launch(c10_npu::getCurrentNPUStream().stream(), sync,
        routed.data_ptr(), shared.data_ptr(), addresses.data(), rows, blocks);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(tilert_ffn_bridge, library)
{
    library.def("prepare(Tensor input, int blocks=24) -> (Tensor[], int)");
    library.def("forward_out(Tensor input, Tensor residual, Tensor gamma, "
        "Tensor gate_weight, Tensor correction_bias, Tensor(a!)[] scratch, "
        "float epsilon, int blocks, int sync_base) -> ()");
    library.def("topk_out(Tensor logits, Tensor correction_bias, "
        "Tensor(a!)[] scratch, int blocks, int sync_base) -> ()");
    library.def("finalize_out(Tensor routed, Tensor shared, "
        "Tensor(a!)[] scratch, int blocks, int sync_base) -> ()");
}

TORCH_LIBRARY_IMPL(tilert_ffn_bridge, PrivateUse1, library)
{
    library.impl("prepare", &Prepare);
    library.impl("forward_out", &Forward);
    library.impl("topk_out", &TopK);
    library.impl("finalize_out", &Finalize);
}
