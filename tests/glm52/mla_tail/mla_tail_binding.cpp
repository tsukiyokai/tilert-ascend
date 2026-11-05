#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "mla_tail.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* address, uint32_t* length);

namespace {

using namespace mla_tail;

struct DeviceLaunch {
    int64_t maximum_blocks;
    uint64_t sync_base;
};

std::mutex launch_mutex;
std::unordered_map<int, DeviceLaunch> launch_devices;

void CheckCapacity(int64_t capacity)
{
    TORCH_CHECK(capacity == 16 || capacity == 32,
                "mla_tail capacity must be 16 or 32");
}

void CheckTensor(const at::Tensor& value, at::ScalarType type,
                 at::IntArrayRef shape, const char* name)
{
    TORCH_CHECK(value.scalar_type() == type, name, " has the wrong dtype");
    TORCH_CHECK(value.sizes().equals(shape), name, " has the wrong shape: ",
                value.sizes(), "; expected ", shape);
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous ND storage");
    if (value.device().type() == c10::DeviceType::PrivateUse1) {
        TORCH_CHECK(at_npu::native::get_npu_format(value) == ACL_FORMAT_ND,
                    name, " must use physical ND storage");
    }
}

DeviceLaunch PrepareDevice(int device)
{
    std::lock_guard<std::mutex> guard(launch_mutex);
    int current = -1;
    TORCH_CHECK(aclrtGetDevice(&current) == ACL_SUCCESS && current == device,
                "mla_tail_prepare requires the reference device to be current");
    int64_t cubes = 0;
    int64_t vectors = 0;
    TORCH_CHECK(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_AICORE_CORE_NUM,
                                  &cubes) == ACL_SUCCESS,
                "cannot query the physical AIC count");
    TORCH_CHECK(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                  &vectors) == ACL_SUCCESS,
                "cannot query the physical AIV count");
    DeviceLaunch launch{std::min(cubes, vectors / 2), 0};
    TORCH_CHECK(launch.maximum_blocks > 0, "no mixed 1:2 core group available");
    uint32_t length = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&launch.sync_base, &length) == 0,
                "rtGetC2cCtrlAddr failed");
    TORCH_CHECK(launch.sync_base != 0, "rtGetC2cCtrlAddr returned a null address");
    // Runtime contexts can be recreated at the same ordinal. Refresh the
    // context-owned synchronization address on every explicit preparation.
    launch_devices.insert_or_assign(device, launch);
    return launch;
}

std::tuple<at::Tensor, at::Tensor, int64_t, int64_t> Prepare(
    const at::Tensor& reference, int64_t capacity, int64_t blocks)
{
    CheckCapacity(capacity);
    TORCH_CHECK(reference.device().type() == c10::DeviceType::PrivateUse1,
                "mla_tail_prepare needs an NPU reference tensor");
    const c10::OptionalDeviceGuard device_guard(reference.device());
    const DeviceLaunch launch = PrepareDevice(reference.get_device());
    if (blocks == 0) {
        blocks = std::min<int64_t>(8, launch.maximum_blocks);
    }
    TORCH_CHECK(blocks > 0 && blocks <= launch.maximum_blocks,
                "blocks must be within the physical 1:2 residency limit: ",
                launch.maximum_blocks);
    std::vector<uint8_t> host_tiling(kTilingBytes);
    TORCH_CHECK(mla_tail_build_tiling(capacity, host_tiling.data()) == 0,
                "mla_tail matmul tiling failed");
    const WorkspaceLayout layout = MakeWorkspaceLayout(capacity);
    auto scratch = at::empty({static_cast<int64_t>(layout.bytes)},
                             reference.options().dtype(at::kByte));
    auto tiling = at::empty({capacity, static_cast<int64_t>(kTilingBytes)},
                            reference.options().dtype(at::kByte));
    // The leading extent carries capacity in tensor metadata, so a captured
    // call can reject a mismatched tiling without reading a device value.
    TORCH_CHECK(aclrtMemcpy(tiling.data_ptr(), tiling.nbytes(),
                            host_tiling.data(), kTilingBytes,
                            ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
                "mla_tail tiling upload failed");
    return {scratch, tiling, blocks, static_cast<int64_t>(launch.sync_base)};
}

int CheckInputs(const at::Tensor& latent, const at::Tensor& value_weight,
                 const at::Tensor& output_weight, const at::Tensor& scale,
                 const c10::optional<at::Tensor>& bias,
                 const at::Tensor& scratch, const at::Tensor& tiling,
                 const at::Tensor& output, int64_t capacity)
{
    CheckCapacity(capacity);
    TORCH_CHECK(latent.dim() == 3, "latent must be [M,4,512]");
    const int64_t rows = latent.size(0);
    TORCH_CHECK(rows > 0 && rows <= capacity, "rows must be within capacity");
    CheckTensor(latent, at::kBFloat16, {rows, kHeads, kLatent}, "latent");
    CheckTensor(value_weight, at::kBFloat16,
                 {kValueTiles, kLatent, kValueTile}, "value_weight");
    CheckTensor(output_weight, at::kChar,
                 {kOutputTiles, kProjectionInput, kOutputTile}, "output_weight");
    CheckTensor(scale, at::kFloat, {kHidden}, "output_scale");
    CheckTensor(output, at::kBFloat16, {rows, kHidden}, "output");
    CheckTensor(tiling, at::kByte,
                 {capacity, static_cast<int64_t>(kTilingBytes)}, "tiling");
    const WorkspaceLayout layout = MakeWorkspaceLayout(capacity);
    CheckTensor(scratch, at::kByte,
                 {static_cast<int64_t>(layout.bytes)}, "scratch");
    for (const auto& tensor : {value_weight, output_weight, scale, scratch,
                               tiling, output}) {
        TORCH_CHECK(tensor.device() == latent.device(),
                    "all mla_tail tensors must be on one device");
    }
    if (!bias.has_value()) {
        return 0;
    }
    TORCH_CHECK(bias->device() == latent.device(), "bias device mismatch");
    TORCH_CHECK(bias->scalar_type() == at::kInt ||
                bias->scalar_type() == at::kFloat,
                "bias must be INT32 accumulation bias or FP32 output bias");
    CheckTensor(*bias, bias->scalar_type(), {kHidden}, "bias");
    return bias->scalar_type() == at::kInt ? 1 : 2;
}

at::Tensor TailOut(
    const at::Tensor& latent, const at::Tensor& value_weight,
    const at::Tensor& output_weight, const at::Tensor& output_scale,
    const c10::optional<at::Tensor>& bias, at::Tensor& scratch,
    const at::Tensor& tiling, at::Tensor& output, int64_t capacity,
    int64_t blocks, int64_t sync_base)
{
    const int bias_kind = CheckInputs(latent, value_weight, output_weight,
                                      output_scale, bias, scratch, tiling,
                                      output, capacity);
    TORCH_CHECK(latent.device().type() == c10::DeviceType::PrivateUse1,
                "mla_tail_out needs NPU tensors");
    const c10::OptionalDeviceGuard device_guard(latent.device());
    {
        std::lock_guard<std::mutex> guard(launch_mutex);
        const auto found = launch_devices.find(latent.get_device());
        TORCH_CHECK(found != launch_devices.end(),
                    "call mla_tail_prepare before capture");
        TORCH_CHECK(blocks > 0 && blocks <= found->second.maximum_blocks,
                    "block count exceeds the prepared residency limit");
        TORCH_CHECK(static_cast<uint64_t>(sync_base) == found->second.sync_base,
                    "sync base belongs to another device or preparation");
    }
    for (const auto& tensor : {latent, value_weight, output_weight,
                               output_scale, tiling}) {
        at::assert_no_overlap(output, tensor);
        at::assert_no_overlap(scratch, tensor);
    }
    at::assert_no_overlap(output, scratch);
    if (bias.has_value()) {
        at::assert_no_overlap(output, *bias);
        at::assert_no_overlap(scratch, *bias);
    }
    // Flush Torch's host task queue before a direct AscendC launch. Otherwise
    // queued producers on this stream can be submitted after their consumer.
    void* stream = c10_npu::getCurrentNPUStream().stream();
    mla_tail_launch(
        static_cast<uint32_t>(blocks), stream,
        static_cast<uint64_t>(sync_base), latent.data_ptr(),
        value_weight.data_ptr(), output_weight.data_ptr(),
        output_scale.data_ptr<float>(), bias.has_value() ? bias->data_ptr() : nullptr,
        bias_kind, scratch.data_ptr(), tiling.data_ptr(), output.data_ptr(),
        latent.size(0), capacity);
    return output;
}

at::Tensor TailMeta(
    const at::Tensor& latent, const at::Tensor& value_weight,
    const at::Tensor& output_weight, const at::Tensor& output_scale,
    const c10::optional<at::Tensor>& bias, at::Tensor& scratch,
    const at::Tensor& tiling, at::Tensor& output, int64_t capacity,
    int64_t blocks, int64_t sync_base)
{
    CheckInputs(latent, value_weight, output_weight, output_scale, bias,
                 scratch, tiling, output, capacity);
    return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(tilert, library)
{
    library.def("mla_tail_prepare(Tensor reference, int capacity, int blocks=0) "
                "-> (Tensor, Tensor, int, int)");
    library.def("mla_tail_out(Tensor latent, Tensor value_weight, "
                "Tensor output_weight, Tensor output_scale, Tensor? bias, "
                "Tensor(a!) scratch, Tensor tiling, Tensor(b!) output, "
                "int capacity, int blocks, int sync_base) -> Tensor(b!)");
}

TORCH_LIBRARY_IMPL(tilert, PrivateUse1, library)
{
    library.impl("mla_tail_prepare", &Prepare);
    library.impl("mla_tail_out", &TailOut);
}

TORCH_LIBRARY_IMPL(tilert, Meta, library)
{
    library.impl("mla_tail_out", &TailMeta);
}
