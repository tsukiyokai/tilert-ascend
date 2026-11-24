#include <array>
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "source/contract.h"

extern "C" void ffn_dual_quant_probe_launch(void* stream, uint64_t sync,
    void** bridge, void** body, int64_t rows, int64_t blocks, bool dual);

namespace ffn_dual_quant_probe_host {
void CheckTensor(const at::Tensor& value, const at::Tensor& input,
    at::ScalarType dtype, at::IntArrayRef shape)
{
    TORCH_CHECK(value.device() == input.device() && value.scalar_type() == dtype &&
                value.sizes() == shape && value.is_contiguous(),
                "quant probe tensor shape, dtype or device mismatch");
    TORCH_CHECK(at_npu::native::get_npu_format(value) == 2,
                "quant probe requires ND storage");
}

void Forward(at::TensorList bridge, at::TensorList body,
    int64_t blocks, int64_t sync, bool dual)
{
    TORCH_CHECK(bridge.size() == 12 && body.size() == 21,
                "use the complete FFN prepare result");
    const auto& input = bridge[ffn_bridge::kNormalized];
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                input.dim() == 2, "quant probe requires an NPU matrix");
    const c10::OptionalDeviceGuard guard(input.device());
    const auto rows = input.size(0);
    TORCH_CHECK(ffn_bridge::ValidGeometry(rows, blocks) && sync != 0,
                "use the geometry and sync returned by FFN prepare");
    CheckTensor(input, input, at::kBFloat16, {rows, 6144});
    CheckTensor(bridge[4], input, at::kFloat, {rows, 8});
    CheckTensor(bridge[5], input, at::kInt, {rows, 8});
    CheckTensor(bridge[6], input, at::kChar, {rows, 6144});
    CheckTensor(bridge[7], input, at::kFloat, {rows, 1});
    CheckTensor(bridge[9], input, at::kByte, {ffn_bridge::kSystemBytes});
    CheckTensor(body[10], input, at::kChar, {rows, 6144});
    CheckTensor(body[11], input, at::kFloat, {rows});
    for (int64_t i = 12; i < 16; ++i) {
        CheckTensor(body[i], input, at::kInt, {256});
    }
    CheckTensor(body[16], input, at::kFloat, {256});
    CheckTensor(body[20], input, at::kInt, {264});
    std::array<void*, 12> bridge_addresses;
    std::array<void*, 21> body_addresses;
    for (size_t i = 0; i < bridge.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            at::assert_no_overlap(bridge[i], bridge[j]);
        }
        for (const auto& tensor : body) {
            at::assert_no_overlap(bridge[i], tensor);
        }
        bridge_addresses[i] = bridge[i].data_ptr();
    }
    for (size_t i = 0; i < body.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            at::assert_no_overlap(body[i], body[j]);
        }
        body_addresses[i] = body[i].data_ptr();
    }
    ffn_dual_quant_probe_launch(c10_npu::getCurrentNPUStream().stream(), sync,
        bridge_addresses.data(), body_addresses.data(), rows, blocks, dual);
}
}

TORCH_LIBRARY_FRAGMENT(tilert_ffn_dual_quant_probe, library)
{
    library.def("forward_out(Tensor(a!)[] bridge, Tensor(b!)[] body, "
                "int blocks, int sync, bool dual) -> ()");
}

TORCH_LIBRARY_IMPL(tilert_ffn_dual_quant_probe, PrivateUse1, library)
{
    library.impl("forward_out", &ffn_dual_quant_probe_host::Forward);
}
