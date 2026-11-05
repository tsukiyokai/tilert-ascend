/** Explicit, capture-stable storage for the routed INT4 stage. */
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "routed_int4_kernel.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace ffn_descriptor_body_host {
using namespace routed_int4;
#ifdef ROUTED_FIXPIPE_HALF
constexpr at::ScalarType kScaleDtype = at::kLong;
constexpr at::ScalarType kProductDtype = at::kHalf;
#else
constexpr at::ScalarType kScaleDtype = at::kFloat;
constexpr at::ScalarType kProductDtype = at::kInt;
#endif

struct DeviceLaunch {
    int64_t maximum_blocks;
    uint64_t sync_base;
};

using WeakTensor = c10::weak_intrusive_ptr<c10::TensorImpl,
                                         c10::UndefinedTensorImpl>;
struct PreparedTiling {
    WeakTensor gate;
    WeakTensor down;
    int64_t tokens;
    int64_t experts;
    int64_t blocks;
    int64_t weight_layout;
    bool from_dispatch;
#ifdef ROUTED_SHARED_PIPELINE
    WeakTensor shared;
#endif
};

std::mutex launch_mutex;
std::unordered_map<int, DeviceLaunch> launch_devices;
std::unordered_map<const c10::TensorImpl*, PreparedTiling> prepared_tilings;

constexpr size_t ScratchCount(bool from_dispatch)
{
#ifdef ROUTED_SHARED_PIPELINE
    return pipeline::kScratchCount;
#else
    return from_dispatch ? kDispatchScratchCount : kScratchCount;
#endif
}

void CheckND(const at::Tensor& value, const char* name)
{
    TORCH_CHECK(at_npu::native::get_npu_format(value) == ACL_FORMAT_ND,
                name, " must have physical ND storage; convert NZ before calling");
}

void CheckShape(const at::Tensor& value, at::ScalarType dtype,
                 at::IntArrayRef shape, const at::Device& device,
                 const char* name)
{
    TORCH_CHECK(value.device() == device, name, " has the wrong device");
    TORCH_CHECK(value.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(value.sizes().equals(shape), name, " has shape ",
                value.sizes(), ", expected ", shape);
    CheckND(value, name);
}

void CheckWeight(const at::Tensor& value, int64_t experts, int64_t inner,
                  int64_t columns, const at::Device& device,
                  int64_t layout, const char* name)
{
    if (layout == kWeightND) {
        CheckShape(value, at::kChar, {experts, inner, columns / 2},
                   device, name);
        return;
    }
    TORCH_CHECK(value.device() == device && value.scalar_type() == at::kInt,
                name, " must be an INT32 NZ carrier on ", device);
    TORCH_CHECK(value.is_contiguous() && value.storage_offset() == 0,
                name, " must be a contiguous view of the complete NZ bank");
    TORCH_CHECK(value.sizes().equals({experts, inner, columns / 8}),
                name, " must have logical INT32 shape [E,K,N/8]");
    TORCH_CHECK(at_npu::native::get_npu_format(value) == ACL_FORMAT_FRACTAL_NZ,
                name, " must retain physical INT8 NZ storage [E,N/64,K/16,16,32]");
    TORCH_CHECK(value.storage().nbytes()
                >= static_cast<size_t>(experts * inner * columns / 2),
                name, " has insufficient physical NZ storage");
}

void CheckGeometry(const at::Tensor& rows, int64_t experts, int64_t blocks)
{
    TORCH_CHECK(rows.device().type() == c10::DeviceType::PrivateUse1,
                "rows must be on an Ascend device");
    TORCH_CHECK(rows.scalar_type() == at::kChar && rows.is_contiguous(),
                "rows must be contiguous INT8");
    TORCH_CHECK(rows.dim() == 2 && rows.size(1) == kHidden,
                "rows must have shape [M,6144]");
    TORCH_CHECK(rows.size(0) >= 1 && rows.size(0) <= kMaxTokens,
                "M must be in [1,32]");
    TORCH_CHECK(experts >= kTopK && experts <= kMaxExperts,
                "experts must be in [8,256]");
    TORCH_CHECK(blocks >= 1, "blocks must be positive");
    CheckND(rows, "rows");
}

DeviceLaunch PrepareDevice(const at::Tensor& rows, int64_t blocks)
{
    int64_t cubes = 0;
    int64_t vectors = 0;
    const int device = rows.get_device();
    TORCH_CHECK(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_AICORE_CORE_NUM,
                                  &cubes) == ACL_SUCCESS,
                "cannot query the physical AIC count");
    TORCH_CHECK(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                  &vectors) == ACL_SUCCESS,
                "cannot query the physical AIV count");
    DeviceLaunch launch{std::min(cubes, vectors / 2), 0};
    TORCH_CHECK(blocks <= launch.maximum_blocks,
                "blocks exceed the mixed-core residency limit: ",
                launch.maximum_blocks);
    uint32_t length = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&launch.sync_base, &length) == 0
                && launch.sync_base != 0, "rtGetC2cCtrlAddr failed");
    // Refresh context metadata on every prepare. Scratch is never cached here.
    const std::lock_guard<std::mutex> lock(launch_mutex);
    launch_devices[device] = launch;
    return launch;
}

void UploadTiling(const at::Tensor& destination, int64_t columns,
                  int64_t inner, int64_t weight_layout)
{
    std::array<uint8_t, kTilingBytes> host{};
    TORCH_CHECK(routed_int4_build_tiling(kMaxStackedRows, columns, inner,
                                        columns, host.data(), weight_layout) == 0,
                "routed_int4_build_tiling failed");
    TORCH_CHECK(aclrtMemcpy(destination.data_ptr(), kTilingBytes, host.data(),
                            kTilingBytes, ACL_MEMCPY_HOST_TO_DEVICE)
                == ACL_SUCCESS, "tiling upload failed");
}

// Call outside capture. No process-wide singleton owns scratch. The returned
// tensors may serve sequential layers on the same device and stream, and
// must remain alive until all launches and graph replays using them finish.
std::tuple<std::vector<at::Tensor>, int64_t> Prepare(
    const at::Tensor& rows, int64_t experts, int64_t blocks,
    int64_t weight_layout, bool from_dispatch)
{
#ifdef ROUTED_SHARED_PIPELINE
    TORCH_CHECK(from_dispatch, "the shared pipeline requires input preparation");
#endif
    CheckGeometry(rows, experts, blocks);
    TORCH_CHECK(IsWeightLayout(weight_layout), "weight_layout must be 0 (ND) or 1 (NZ)");
    const c10::OptionalDeviceGuard guard(rows.device());
    const auto launch = PrepareDevice(rows, blocks);
    const auto options = rows.options();
    const int64_t tokens = rows.size(0);
    const int64_t routes = tokens * kTopK;
    std::vector<at::Tensor> scratch(ScratchCount(from_dispatch));
#ifdef ROUTED_GU_PAD16
    // Dynamic routes can turn former live rows into padding. Defined integer
    // padding may persist: only the first 2*count products are consumed.
    scratch[kStacked] = at::zeros({GateStorageRow(routes), kHidden / 2}, options);
#else
    scratch[kStacked] = at::empty({GateStorageRow(routes), kHidden / 2}, options);
#endif
    scratch[kGateProduct] = at::empty(
        {GateStorageRow(routes), kGateUp}, options.dtype(kProductDtype));
    scratch[kMiddle] = at::empty(
        {2 * routes, kIntermediate / 2}, options);
    scratch[kMiddleScale] = at::empty(
        {routes}, options.dtype(at::kFloat));
    scratch[kDownProduct] = at::empty(
        {2 * routes, kHidden}, options.dtype(kProductDtype));
    scratch[kRouteOutput] = at::empty(
        {tokens, kTopK, kHidden}, options.dtype(at::kBFloat16));
    scratch[kGateTiling] = at::empty(
        {int64_t(kTilingBytes)}, options.dtype(at::kByte));
    scratch[kDownTiling] = at::empty(
        {int64_t(kTilingBytes)}, options.dtype(at::kByte));
    scratch[kMatmulWorkspace] = at::empty(
        {int64_t(kMatmulWorkspaceBytes)}, options.dtype(at::kByte));
    scratch[kOutput] = at::empty(
        {tokens, kHidden}, options.dtype(at::kBFloat16));
    if (from_dispatch) {
        scratch[kRestoredRows] = at::empty({tokens, kHidden}, options);
        scratch[kRestoredScale] = at::empty({tokens}, options.dtype(at::kFloat));
        for (size_t index : {kGroupCount, kGroupStart, kSourceRow, kSourceSlot}) {
            scratch[index] = at::empty({kMaxRoutes}, options.dtype(at::kInt));
        }
        scratch[kRowWeight] = at::empty({kMaxRoutes}, options.dtype(at::kFloat));
    }
    UploadTiling(scratch[kGateTiling], kGateUp, kHidden, weight_layout);
    UploadTiling(scratch[kDownTiling], kHidden, kIntermediate, weight_layout);
#ifdef ROUTED_SHARED_PIPELINE
    const int64_t capacity = pipeline::Capacity(tokens);
    const int64_t payload_bytes = shared_w8::MakeLayout(capacity).system;
    scratch[pipeline::kSharedPayload] = at::empty(
        {payload_bytes}, options.dtype(at::kByte));
    scratch[pipeline::kSharedTiling] = at::empty(
        {2 * int64_t(shared_w8::kTilingStride)}, options.dtype(at::kByte));
    scratch[pipeline::kSharedOutput] = at::empty(
        {tokens, kHidden}, options.dtype(at::kBFloat16));
    std::array<uint8_t, 2 * shared_w8::kTilingStride> shared_tiling{};
    TORCH_CHECK(routed_shared_build_tiling(capacity, shared_tiling.data()) == 0,
                "shared projection tiling failed");
    TORCH_CHECK(aclrtMemcpy(scratch[pipeline::kSharedTiling].data_ptr(),
                shared_tiling.size(), shared_tiling.data(), shared_tiling.size(),
                ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
                "shared tiling upload failed");
#endif
    {
        const std::lock_guard<std::mutex> lock(launch_mutex);
        // Weak owners validate prepare/execute pairing without retaining HBM.
        for (auto item = prepared_tilings.begin(); item != prepared_tilings.end();) {
            if (item->second.gate.expired() || item->second.down.expired()) {
                item = prepared_tilings.erase(item);
            } else {
                ++item;
            }
        }
        prepared_tilings.insert_or_assign(
            scratch[kGateTiling].unsafeGetTensorImpl(),
            PreparedTiling{WeakTensor(scratch[kGateTiling].getIntrusivePtr()),
                           WeakTensor(scratch[kDownTiling].getIntrusivePtr()),
                           tokens, experts, blocks, weight_layout, from_dispatch
#ifdef ROUTED_SHARED_PIPELINE
                           , WeakTensor(scratch[pipeline::kSharedTiling].getIntrusivePtr())
#endif
                           });
    }
    // This address is owned by the current runtime device context. The
    // context must remain alive together with the tensors and captured graph.
    return {std::move(scratch), static_cast<int64_t>(launch.sync_base)};
}

void CheckScratch(at::TensorList scratch, const at::Tensor& rows,
                   bool from_dispatch)
{
    TORCH_CHECK(scratch.size() == ScratchCount(from_dispatch),
                "scratch must come from routed_experts_prepare");
    const int64_t tokens = rows.size(0);
    const int64_t routes = tokens * kTopK;
    const auto device = rows.device();
    CheckShape(scratch[kStacked], at::kChar,
                {GateStorageRow(routes), kHidden / 2}, device, "stacked");
    CheckShape(scratch[kGateProduct], kProductDtype,
                {GateStorageRow(routes), kGateUp}, device, "gate_product");
    CheckShape(scratch[kMiddle], at::kChar,
                {2 * routes, kIntermediate / 2}, device, "middle");
    CheckShape(scratch[kMiddleScale], at::kFloat,
                {routes}, device, "middle_scale");
    CheckShape(scratch[kDownProduct], kProductDtype,
                {2 * routes, kHidden}, device, "down_product");
    CheckShape(scratch[kRouteOutput], at::kBFloat16,
                {tokens, kTopK, kHidden}, device, "route_output");
    CheckShape(scratch[kGateTiling], at::kByte,
                {int64_t(kTilingBytes)}, device, "gate_tiling");
    CheckShape(scratch[kDownTiling], at::kByte,
                {int64_t(kTilingBytes)}, device, "down_tiling");
    CheckShape(scratch[kMatmulWorkspace], at::kByte,
                {int64_t(kMatmulWorkspaceBytes)}, device, "matmul_workspace");
    CheckShape(scratch[kOutput], at::kBFloat16,
                {tokens, kHidden}, device, "output");
    if (from_dispatch) {
        CheckShape(scratch[kRestoredRows], at::kChar,
                    {tokens, kHidden}, device, "restored_rows");
        CheckShape(scratch[kRestoredScale], at::kFloat,
                    {tokens}, device, "restored_scale");
        for (size_t index : {kGroupCount, kGroupStart, kSourceRow, kSourceSlot}) {
            CheckShape(scratch[index], at::kInt, {kMaxRoutes}, device,
                        "dispatch routing metadata");
        }
        CheckShape(scratch[kRowWeight], at::kFloat,
                    {kMaxRoutes}, device, "dispatch row_weight");
    }
#ifdef ROUTED_SHARED_PIPELINE
    CheckShape(scratch[pipeline::kSharedPayload], at::kByte,
        {int64_t(shared_w8::MakeLayout(pipeline::Capacity(tokens)).system)},
        device, "shared payload");
    CheckShape(scratch[pipeline::kSharedTiling], at::kByte,
        {2 * int64_t(shared_w8::kTilingStride)}, device, "shared tiling");
    CheckShape(scratch[pipeline::kSharedOutput], at::kBFloat16,
        {tokens, kHidden}, device, "shared output");
#endif
}

#ifdef ROUTED_SHARED_PIPELINE
pipeline::Input CheckSharedInputs(at::TensorList inputs, at::TensorList scratch,
                                  const at::Tensor& rows)
{
    TORCH_CHECK(inputs.size() == pipeline::kInputCount,
                "shared_args must contain input, token scale, GU weight/scale, down weight/scale");
    const auto device = rows.device();
    const int64_t tokens = rows.size(0);
    CheckShape(inputs[pipeline::kInput], at::kChar, {tokens, kHidden},
                device, "shared input");
    CheckShape(inputs[pipeline::kInputScale], at::kFloat, {tokens},
                device, "shared input scale");
    CheckShape(inputs[pipeline::kGateWeight], at::kChar, {24, 1024, 64},
                device, "shared GU weight");
    CheckShape(inputs[pipeline::kGateScale], at::kFloat, {kGateUp},
                device, "shared GU scale");
    CheckShape(inputs[pipeline::kDownWeight], at::kChar, {24, 128, 256},
                device, "shared down weight");
    CheckShape(inputs[pipeline::kDownScale], at::kFloat, {kHidden},
                device, "shared down scale");
    const int64_t capacity = pipeline::Capacity(tokens);
    const auto layout = shared_w8::MakeLayout(capacity);
    return {
        reinterpret_cast<uint64_t>(inputs[pipeline::kInput].data_ptr()),
        reinterpret_cast<uint64_t>(inputs[pipeline::kInputScale].data_ptr()),
        reinterpret_cast<uint64_t>(inputs[pipeline::kGateWeight].data_ptr()),
        reinterpret_cast<uint64_t>(inputs[pipeline::kGateScale].data_ptr()),
        reinterpret_cast<uint64_t>(inputs[pipeline::kDownWeight].data_ptr()),
        reinterpret_cast<uint64_t>(inputs[pipeline::kDownScale].data_ptr()),
        reinterpret_cast<uint64_t>(scratch[pipeline::kSharedPayload].data_ptr()),
        reinterpret_cast<uint64_t>(scratch[pipeline::kSharedTiling].data_ptr()),
        reinterpret_cast<uint64_t>(scratch[pipeline::kSharedOutput].data_ptr()),
        capacity, layout.partial, layout.middle, layout.scales, layout.down};
}
#endif

at::Tensor Execute(
    const at::Tensor& rows, const at::Tensor& gate_up,
    const at::Tensor& gu_bias, const at::Tensor& gu_scale,
    const at::Tensor& down, const at::Tensor& down_bias,
    const at::Tensor& down_scale, const at::Tensor& token_scale,
    const at::Tensor& group_count, const at::Tensor& group_start,
    const at::Tensor& source_row, const at::Tensor& source_slot,
    const at::Tensor& row_weight, at::TensorList scratch,
    int64_t sync_base, int64_t experts, int64_t blocks, int64_t weight_layout,
    DispatchInput dispatch, at::TensorList dispatch_tensors,
    at::TensorList shared_inputs = {})
{
    const bool from_dispatch = dispatch.rows != 0
#ifdef ROUTED_INPUT_FRONT
                               || dispatch.bf16_rows != 0
#endif
                               ;
    CheckGeometry(rows, experts, blocks);
    TORCH_CHECK(IsWeightLayout(weight_layout), "weight_layout must be 0 (ND) or 1 (NZ)");
    const c10::OptionalDeviceGuard guard(rows.device());
    const int64_t tokens = rows.size(0);
    const auto device = rows.device();
    CheckWeight(gate_up, experts, kHidden, kGateUp, device, weight_layout,
                "gate_up");
    CheckWeight(down, experts, kIntermediate, kHidden, device, weight_layout,
                "down");
    CheckShape(gu_bias, at::kFloat, {experts, kGateUp}, device, "gu_bias");
    CheckShape(gu_scale, kScaleDtype, {experts, kGateUp}, device, "gu_scale");
    CheckShape(down_bias, at::kFloat, {experts, kHidden}, device, "down_bias");
    CheckShape(down_scale, kScaleDtype, {experts, kHidden}, device, "down_scale");
    CheckShape(token_scale, at::kFloat, {tokens}, device, "token_scale");
    CheckShape(group_count, at::kInt, {kMaxExperts}, device, "group_count");
    CheckShape(group_start, at::kInt, {kMaxExperts}, device, "group_start");
    CheckShape(source_row, at::kInt, {kMaxRoutes}, device, "source_row");
    CheckShape(source_slot, at::kInt, {kMaxRoutes}, device, "source_slot");
    CheckShape(row_weight, at::kFloat, {kMaxRoutes}, device, "row_weight");
    CheckScratch(scratch, rows, from_dispatch);
#ifdef ROUTED_SHARED_PIPELINE
    const auto shared = CheckSharedInputs(shared_inputs, scratch, rows);
#endif
    {
        const std::lock_guard<std::mutex> lock(launch_mutex);
        const auto found = launch_devices.find(rows.get_device());
        TORCH_CHECK(found != launch_devices.end(), "prepare before execution");
        TORCH_CHECK(blocks <= found->second.maximum_blocks,
                    "blocks exceed the prepared residency limit");
        TORCH_CHECK(static_cast<uint64_t>(sync_base) == found->second.sync_base,
                    "sync_base does not match this device preparation");
        const auto prepared = prepared_tilings.find(
            scratch[kGateTiling].unsafeGetTensorImpl());
        TORCH_CHECK(prepared != prepared_tilings.end(),
                    "tiling tensors must come from routed_experts_prepare");
        const auto& metadata = prepared->second;
#ifdef ROUTED_SHARED_PIPELINE
        TORCH_CHECK(metadata.shared.lock().get()
                    == scratch[pipeline::kSharedTiling].unsafeGetTensorImpl(),
                    "shared and routed tilings must belong to the same preparation");
#endif
        TORCH_CHECK(metadata.gate.lock().get() == scratch[kGateTiling].unsafeGetTensorImpl()
                    && metadata.down.lock().get() == scratch[kDownTiling].unsafeGetTensorImpl(),
                    "gate and down tilings must belong to the same preparation");
        TORCH_CHECK(metadata.tokens == tokens && metadata.experts == experts
                    && metadata.blocks == blocks && metadata.weight_layout == weight_layout
                    && metadata.from_dispatch == from_dispatch,
                    "M, experts, blocks, weight_layout and input mode must match preparation");
    }
    for (size_t index = 0; index < scratch.size(); ++index) {
        for (size_t previous = 0; previous < index; ++previous) {
            at::assert_no_overlap(scratch[index], scratch[previous]);
        }
        for (const auto& input : {gate_up, gu_bias, gu_scale, down,
                                  down_bias, down_scale}) {
            at::assert_no_overlap(scratch[index], input);
        }
        for (const auto& input : shared_inputs) {
            at::assert_no_overlap(scratch[index], input);
        }
        if (from_dispatch) {
            for (const auto& input : dispatch_tensors) {
                at::assert_no_overlap(scratch[index], input);
            }
        } else {
            for (const auto& input : {rows, token_scale, group_count, group_start,
                                      source_row, source_slot, row_weight}) {
                at::assert_no_overlap(scratch[index], input);
            }
        }
    }
    // Values are device routing data, and deliberately never read on the
    // host during capture/replay. The router guarantees counts <= M, prefix
    // starts, and exactly one route per original (token, top-k slot).

    return scratch[kOutput];
}

at::Tensor RoutedExperts(
    const at::Tensor& rows, const at::Tensor& gate_up,
    const at::Tensor& gu_bias, const at::Tensor& gu_scale,
    const at::Tensor& down, const at::Tensor& down_bias,
    const at::Tensor& down_scale, const at::Tensor& token_scale,
    const at::Tensor& group_count, const at::Tensor& group_start,
    const at::Tensor& source_row, const at::Tensor& source_slot,
    const at::Tensor& row_weight, at::TensorList scratch,
    int64_t sync_base, int64_t experts, int64_t blocks, int64_t weight_layout)
{
    return Execute(rows, gate_up, gu_bias, gu_scale, down, down_bias,
                   down_scale, token_scale, group_count, group_start,
                   source_row, source_slot, row_weight, scratch, sync_base,
                   experts, blocks, weight_layout, {}, {});
}

at::Tensor FromDispatch(
    const at::Tensor& sorted_rows, const at::Tensor& gate_up,
    const at::Tensor& gu_bias, const at::Tensor& gu_scale,
    const at::Tensor& down, const at::Tensor& down_bias,
    const at::Tensor& down_scale, const at::Tensor& sorted_scales,
    const at::Tensor& inverse, const at::Tensor& counts,
    const at::Tensor& probabilities, at::TensorList scratch,
    int64_t sync_base, int64_t experts, int64_t blocks, int64_t weight_layout
#ifdef ROUTED_SHARED_PIPELINE
    , at::TensorList shared_inputs
#endif
    )
{
    TORCH_CHECK(scratch.size() == ScratchCount(true),
                "prepare with from_dispatch=True before execution");
    TORCH_CHECK(sorted_rows.dim() == 2 && sorted_rows.size(0) % kTopK == 0,
                "sorted_rows must have shape [M*8,6144]");
    const int64_t routes = sorted_rows.size(0);
    const int64_t tokens = routes / kTopK;
    const auto device = sorted_rows.device();
    CheckGeometry(scratch[kRestoredRows], experts, blocks);
    CheckShape(sorted_rows, at::kChar, {routes, kHidden}, device, "sorted_rows");
    CheckShape(sorted_scales, at::kFloat, {routes}, device, "sorted_scales");
    CheckShape(inverse, at::kInt, {routes}, device, "inverse");
    CheckShape(counts, at::kLong, {kMaxExperts}, device, "counts");
    CheckShape(probabilities, at::kFloat, {tokens, kTopK}, device, "probabilities");
    CheckShape(scratch[kRestoredRows], at::kChar,
                {tokens, kHidden}, device, "restored_rows");
    const DispatchInput input{
        reinterpret_cast<uint64_t>(sorted_rows.data_ptr()),
        reinterpret_cast<uint64_t>(sorted_scales.data_ptr()),
        reinterpret_cast<uint64_t>(inverse.data_ptr()),
        reinterpret_cast<uint64_t>(counts.data_ptr()),
        reinterpret_cast<uint64_t>(probabilities.data_ptr())};
    return Execute(scratch[kRestoredRows], gate_up, gu_bias, gu_scale, down,
                   down_bias, down_scale, scratch[kRestoredScale],
                   scratch[kGroupCount], scratch[kGroupStart],
                   scratch[kSourceRow], scratch[kSourceSlot], scratch[kRowWeight],
                   scratch, sync_base, experts, blocks, weight_layout, input,
                   {sorted_rows, sorted_scales, inverse, counts, probabilities}
#ifdef ROUTED_SHARED_PIPELINE
                   , shared_inputs
#endif
                   );
}

#ifdef ROUTED_INPUT_FRONT
at::Tensor FromBFloat16(
    const at::Tensor& hidden, const at::Tensor& gate_up,
    const at::Tensor& gu_bias, const at::Tensor& gu_scale,
    const at::Tensor& down, const at::Tensor& down_bias,
    const at::Tensor& down_scale, const at::Tensor& choices,
    const at::Tensor& probabilities, at::TensorList scratch,
    int64_t sync_base, int64_t experts, int64_t blocks, int64_t weight_layout
#ifdef ROUTED_SHARED_PIPELINE
    , at::TensorList shared_inputs
#endif
    )
{
    TORCH_CHECK(scratch.size() == ScratchCount(true),
                "prepare with from_dispatch=True before BF16 execution");
    TORCH_CHECK(hidden.dim() == 2, "hidden must have shape [M,6144]");
    const int64_t tokens = hidden.size(0);
    const auto device = hidden.device();
    CheckGeometry(scratch[kRestoredRows], experts, blocks);
    CheckShape(hidden, at::kBFloat16, {tokens, kHidden}, device, "hidden");
    CheckShape(choices, at::kInt, {tokens, kTopK}, device, "choices");
    CheckShape(probabilities, at::kFloat, {tokens, kTopK}, device, "probabilities");
    CheckShape(scratch[kRestoredRows], at::kChar,
                {tokens, kHidden}, device, "restored_rows");
    DispatchInput input;
    input.probabilities = reinterpret_cast<uint64_t>(probabilities.data_ptr());
    input.bf16_rows = reinterpret_cast<uint64_t>(hidden.data_ptr());
    input.expert_ids = reinterpret_cast<uint64_t>(choices.data_ptr());
    return Execute(scratch[kRestoredRows], gate_up, gu_bias, gu_scale, down,
                   down_bias, down_scale, scratch[kRestoredScale],
                   scratch[kGroupCount], scratch[kGroupStart],
                   scratch[kSourceRow], scratch[kSourceSlot], scratch[kRowWeight],
                   scratch, sync_base, experts, blocks, weight_layout, input,
                   {hidden, choices, probabilities}
#ifdef ROUTED_SHARED_PIPELINE
                   , shared_inputs
#endif
                   );
}
#endif

#ifdef ROUTED_SHARED_PIPELINE
std::tuple<std::vector<at::Tensor>, int64_t> PreparePipeline(
    const at::Tensor& rows, int64_t experts, int64_t blocks, int64_t weight_layout)
{
    return Prepare(rows, experts, blocks, weight_layout, true);
}

std::tuple<at::Tensor, at::Tensor> PipelineFromDispatch(
    const at::Tensor& sorted_rows, const at::Tensor& gate_up,
    const at::Tensor& gu_bias, const at::Tensor& gu_scale,
    const at::Tensor& down, const at::Tensor& down_bias,
    const at::Tensor& down_scale, const at::Tensor& sorted_scales,
    const at::Tensor& inverse, const at::Tensor& counts,
    const at::Tensor& probabilities, at::TensorList shared_inputs,
    at::TensorList scratch, int64_t sync_base, int64_t experts,
    int64_t blocks, int64_t weight_layout)
{
    auto routed = FromDispatch(sorted_rows, gate_up, gu_bias, gu_scale, down,
        down_bias, down_scale, sorted_scales, inverse, counts, probabilities,
        scratch, sync_base, experts, blocks, weight_layout, shared_inputs);
    return {routed, scratch[pipeline::kSharedOutput]};
}

#ifdef ROUTED_INPUT_FRONT
std::tuple<at::Tensor, at::Tensor> PipelineFromBFloat16(
    const at::Tensor& hidden, const at::Tensor& gate_up,
    const at::Tensor& gu_bias, const at::Tensor& gu_scale,
    const at::Tensor& down, const at::Tensor& down_bias,
    const at::Tensor& down_scale, const at::Tensor& choices,
    const at::Tensor& probabilities, at::TensorList shared_inputs,
    at::TensorList scratch, int64_t sync_base, int64_t experts,
    int64_t blocks, int64_t weight_layout)
{
    auto routed = FromBFloat16(hidden, gate_up, gu_bias, gu_scale, down,
        down_bias, down_scale, choices, probabilities, scratch, sync_base,
        experts, blocks, weight_layout, shared_inputs);
    return {routed, scratch[pipeline::kSharedOutput]};
}
#endif
#endif

}  // namespace
