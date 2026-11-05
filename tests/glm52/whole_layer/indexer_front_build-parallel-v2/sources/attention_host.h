#ifndef TESTS_GLM52_WHOLE_LAYER_ATTENTION_HOST_H
#define TESTS_GLM52_WHOLE_LAYER_ATTENTION_HOST_H

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
#include "attention_contract.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* address, uint32_t* length);

namespace whole_layer_attention::host {

using namespace whole_layer_attention;

struct DeviceLaunch {
    int64_t maximum_blocks;
    uint64_t sync_base;
};

inline std::mutex launch_mutex;
inline std::unordered_map<int, DeviceLaunch> launches;

inline void CheckShape(const at::Tensor& tensor, at::ScalarType dtype,
                at::IntArrayRef shape, const char* name, int format = ACL_FORMAT_ND)
{
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.sizes().equals(shape), name, " shape ", tensor.sizes(),
                " differs from ", shape);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    if (tensor.device().type() == c10::DeviceType::PrivateUse1) {
        TORCH_CHECK(at_npu::native::get_npu_format(tensor) == format,
                    name, " has the wrong physical format; expected ", format);
    }
}

inline void CheckGeometry(int64_t rows, int64_t pages, int64_t table_pages)
{
    TORCH_CHECK(rows >= 1 && rows <= kMaximumRows, "rows must be in [1,32]");
    constexpr int64_t page_limit = std::numeric_limits<int32_t>::max() / kPage;
    TORCH_CHECK(pages > 0 && pages <= page_limit, "invalid physical page count");
    TORCH_CHECK(table_pages > 0 && table_pages <= page_limit,
                "invalid page-table capacity");
}

inline void CheckScalar(double value, const char* name)
{
    TORCH_CHECK(std::isfinite(value) && value > 0 &&
                value <= std::numeric_limits<float>::max(),
                name, " must be finite, positive and representable in FP32");
    TORCH_CHECK(static_cast<float>(value) > 0, name, " underflows FP32");
}

inline DeviceLaunch PrepareDevice(int device)
{
    std::lock_guard<std::mutex> guard(launch_mutex);
    int64_t cubes = 0;
    int64_t vectors = 0;
    TORCH_CHECK(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_AICORE_CORE_NUM,
                                  &cubes) == ACL_SUCCESS, "cannot query AIC count");
    TORCH_CHECK(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                  &vectors) == ACL_SUCCESS, "cannot query AIV count");
    DeviceLaunch launch{std::min(cubes, vectors / 2), 0};
    uint32_t length = 0;
    TORCH_CHECK(launch.maximum_blocks > 0, "no resident mixed 1:2 groups");
    TORCH_CHECK(rtGetC2cCtrlAddr(&launch.sync_base, &length) == 0 &&
                launch.sync_base != 0, "cannot obtain the current C2C sync base");
    launches.insert_or_assign(device, launch);
    return launch;
}

inline std::tuple<at::Tensor, int64_t, int64_t> Prepare(
    const at::Tensor& token_x, int64_t pages, int64_t table_pages, int64_t blocks)
{
    TORCH_CHECK(token_x.device().type() == c10::DeviceType::PrivateUse1,
                "shared_attention_prepare requires an NPU token tensor");
    const c10::OptionalDeviceGuard device_guard(token_x.device());
    TORCH_CHECK(token_x.dim() == 2, "token_x must be [M,6144]");
    CheckGeometry(token_x.size(0), pages, table_pages);
    CheckShape(token_x, at::kChar, {token_x.size(0), kHidden}, "token_x");
    const auto launch = PrepareDevice(token_x.get_device());
    if (blocks == 0) {
        blocks = std::min<int64_t>(8, launch.maximum_blocks);
    }
    TORCH_CHECK(blocks > 0 && blocks <= launch.maximum_blocks,
                "blocks exceed physical mixed 1:2 residency: ", launch.maximum_blocks);
    const auto config = MakeConfig(token_x.size(0), pages, table_pages, blocks,
                                    1.0F, 1e-5F, 1e-5F);
    auto scratch = at::empty({static_cast<int64_t>(config.bytes)},
                             token_x.options().dtype(at::kByte));
    return {scratch, blocks, static_cast<int64_t>(launch.sync_base)};
}

inline Config CheckInputs(
    const at::Tensor& token_x, const at::Tensor& token_scale,
    const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
    const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr,
    const at::Tensor& scale_dq, const at::Tensor& scale_uq_qr,
    const at::Tensor& scale_dkv_kr, const at::Tensor& gamma_cq,
    const at::Tensor& gamma_ckv, const at::Tensor& sin, const at::Tensor& cos,
    const at::Tensor& slots, const at::Tensor& kv_cache,
    const at::Tensor& rope_cache, const at::Tensor& topk,
    const at::Tensor& query_lengths, const at::Tensor& kv_lengths,
    const at::Tensor& page_table, const at::Tensor& scratch,
    const at::Tensor& output, double scale, double epsilon_cq,
    double epsilon_ckv, int64_t blocks, bool check_phase_buffers = true)
{
    TORCH_CHECK(token_x.dim() == 2 && kv_cache.dim() == 4 && page_table.dim() == 2,
                "token_x, kv_cache and page_table ranks must be 2,4,2");
    const int64_t rows = token_x.size(0);
    const int64_t pages = kv_cache.size(0);
    const int64_t table_pages = page_table.size(1);
    CheckGeometry(rows, pages, table_pages);
    CheckScalar(scale, "scale");
    CheckScalar(epsilon_cq, "epsilon_cq");
    CheckScalar(epsilon_ckv, "epsilon_ckv");
    TORCH_CHECK(blocks > 0 && blocks <= 64, "invalid mixed group count");
    CheckShape(token_x, at::kChar, {rows, kHidden}, "token_x");
    CheckShape(token_scale, at::kFloat, {rows, 1}, "token_scale");
    CheckShape(weight_dq, at::kChar, {kHidden, kQueryRank}, "weight_dq", ACL_FORMAT_FRACTAL_NZ);
    CheckShape(weight_uq_qr, at::kChar, {kQueryRank, kHeads * (kNope + kRope)},
                "weight_uq_qr", ACL_FORMAT_FRACTAL_NZ);
    CheckShape(weight_dkv_kr, at::kChar, {kHidden, kLatent + kRope},
                "weight_dkv_kr", ACL_FORMAT_FRACTAL_NZ);
    CheckShape(weight_uk, at::kBFloat16, {kHeads, kNope, kLatent}, "weight_uk");
    CheckShape(scale_dq, at::kFloat, {1, kQueryRank}, "scale_dq");
    CheckShape(scale_uq_qr, at::kFloat, {1, kHeads * (kNope + kRope)}, "scale_uq_qr");
    CheckShape(scale_dkv_kr, at::kFloat, {1, kLatent + kRope}, "scale_dkv_kr");
    CheckShape(gamma_cq, at::kBFloat16, {kQueryRank}, "gamma_cq");
    CheckShape(gamma_ckv, at::kBFloat16, {kLatent}, "gamma_ckv");
    CheckShape(sin, at::kBFloat16, {rows, kRope}, "sin");
    CheckShape(cos, at::kBFloat16, {rows, kRope}, "cos");
    CheckShape(slots, at::kLong, {rows}, "slots");
    CheckShape(kv_cache, at::kBFloat16, {pages, kPage, 1, kLatent}, "kv_cache");
    CheckShape(rope_cache, at::kBFloat16, {pages, kPage, 1, kRope}, "rope_cache");
    CheckShape(topk, at::kInt, {rows, 1, kTopK}, "topk");
    CheckShape(query_lengths, at::kInt, {1}, "query_lengths");
    CheckShape(kv_lengths, at::kInt, {1}, "kv_lengths");
    CheckShape(page_table, at::kInt, {1, table_pages}, "page_table");
    if (check_phase_buffers) {
        CheckShape(output, at::kBFloat16, {rows, kHeads, kLatent}, "output");
    }
    const Config config = MakeConfig(rows, pages, table_pages, blocks,
        static_cast<float>(scale), static_cast<float>(epsilon_cq),
        static_cast<float>(epsilon_ckv));
    if (check_phase_buffers) {
        CheckShape(scratch, at::kByte, {static_cast<int64_t>(config.bytes)}, "scratch");
    }
    for (const auto& tensor : {token_scale, weight_dq, weight_uq_qr, weight_uk,
            weight_dkv_kr, scale_dq, scale_uq_qr, scale_dkv_kr, gamma_cq,
            gamma_ckv, sin, cos, slots, kv_cache, rope_cache, topk,
            query_lengths, kv_lengths, page_table, scratch, output}) {
        TORCH_CHECK(tensor.device() == token_x.device(), "attention devices differ");
    }
    return config;
}

}  // namespace whole_layer_attention::host

#endif
