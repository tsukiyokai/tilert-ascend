#include "routed_body_native.h"

using namespace AscendC;
using namespace routed_int4;
using namespace routed_body;

template <CubeFormat weight_format, bool from_dispatch>
__global__ __mix__(1, 2) __schedmode__(1)
__aicore__ void routed_int4_fused(
    uint64_t sync_base, __gm__ int8_t* rows, __gm__ int4b_t* gate_up,
    __gm__ float* gu_bias, __gm__ ScaleStorage* gu_scale, __gm__ int4b_t* down,
    __gm__ float* down_bias, __gm__ ScaleStorage* down_scale,
    __gm__ float* token_scale, __gm__ int32_t* group_count,
    __gm__ int32_t* group_start, __gm__ int32_t* source_row,
    __gm__ int32_t* source_slot, __gm__ float* row_weight,
    __gm__ int4b_t* stacked, __gm__ RoutedProduct* partial,
    __gm__ int4b_t* middle, __gm__ float* middle_scale,
    __gm__ RoutedProduct* wide, __gm__ bfloat16_t* route_output,
    __gm__ bfloat16_t* out, __gm__ uint8_t* gu_tiling,
    __gm__ uint8_t* down_tiling, GM_ADDR workspace,
    int64_t experts, int64_t tokens, int64_t blocks, DispatchInput dispatch_input,
    __gm__ int64_t* trace
#ifdef ROUTED_SHARED_PIPELINE
    , pipeline::Input shared_input
#endif
    )
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    SetSysWorkspace(workspace);
    TPipe pipe;
    routed_body::Forward<weight_format, from_dispatch>(
        pipe, rows, gate_up, gu_bias, gu_scale, down, down_bias, down_scale,
        token_scale, group_count, group_start, source_row, source_slot,
        row_weight, stacked, partial, middle, middle_scale, wide, route_output,
        out, gu_tiling, down_tiling, workspace, experts, tokens, blocks,
        dispatch_input, trace
#ifdef ROUTED_SHARED_PIPELINE
        , shared_input
#endif
        );
}

#ifndef ROUTED_ENABLE_PROFILE
namespace {

template <CubeFormat weight_format, bool from_dispatch>
void Launch(
    uint32_t blocks, void* stream, uint64_t sync_base, void* rows,
    void* gate_up, float* gu_bias, ScaleStorage* gu_scale, void* down,
    float* down_bias, ScaleStorage* down_scale, float* token_scale,
    void* group_count, void* group_start, void* source_row,
    void* source_slot, float* row_weight, void* stacked, void* partial,
    void* middle, float* middle_scale, void* wide, void* route_output,
    void* out, void* gu_tiling, void* down_tiling, void* workspace,
    int64_t experts, int64_t tokens, DispatchInput dispatch_input
#ifdef ROUTED_SHARED_PIPELINE
    , pipeline::Input shared_input
#endif
    )
{
    routed_int4_fused<weight_format, from_dispatch><<<blocks, nullptr, stream>>>(
        sync_base, (__gm__ int8_t*)rows, (__gm__ int4b_t*)gate_up,
        gu_bias, gu_scale, (__gm__ int4b_t*)down, down_bias, down_scale,
        token_scale, (__gm__ int32_t*)group_count, (__gm__ int32_t*)group_start,
        (__gm__ int32_t*)source_row, (__gm__ int32_t*)source_slot, row_weight,
        (__gm__ int4b_t*)stacked, (__gm__ RoutedProduct*)partial,
        (__gm__ int4b_t*)middle, middle_scale, (__gm__ RoutedProduct*)wide,
        (__gm__ bfloat16_t*)route_output, (__gm__ bfloat16_t*)out,
        (__gm__ uint8_t*)gu_tiling, (__gm__ uint8_t*)down_tiling,
        (GM_ADDR)workspace, experts, tokens, blocks, dispatch_input, nullptr
#ifdef ROUTED_SHARED_PIPELINE
        , shared_input
#endif
        );
}

}  // namespace

#ifdef ROUTED_SHARED_PIPELINE
extern "C"
#endif
void routed_int4_launch(
    uint32_t blocks, void* stream, uint64_t sync_base, void* rows,
    void* gate_up, float* gu_bias, ScaleStorage* gu_scale, void* down,
    float* down_bias, ScaleStorage* down_scale, float* token_scale,
    void* group_count, void* group_start, void* source_row,
    void* source_slot, float* row_weight, void* stacked, void* partial,
    void* middle, float* middle_scale, void* wide, void* route_output,
    void* out, void* gu_tiling, void* down_tiling, void* workspace,
    int64_t experts, int64_t tokens, int64_t weight_layout,
    DispatchInput dispatch_input
#ifdef ROUTED_SHARED_PIPELINE
    , pipeline::Input shared_input
#endif
    )
{
#ifdef ROUTED_SHARED_PIPELINE
    const auto launch = weight_layout == kWeightNZ ? &Launch<CubeFormat::NZ, true>
                                                  : &Launch<CubeFormat::ND, true>;
#else
    const bool has_input = dispatch_input.rows != 0
#ifdef ROUTED_INPUT_FRONT
                           || dispatch_input.bf16_rows != 0
#endif
                           ;
    const auto launch = has_input
        ? (weight_layout == kWeightNZ ? &Launch<CubeFormat::NZ, true>
                                      : &Launch<CubeFormat::ND, true>)
        : (weight_layout == kWeightNZ ? &Launch<CubeFormat::NZ, false>
                                      : &Launch<CubeFormat::ND, false>);
#endif
    launch(blocks, stream, sync_base, rows, gate_up, gu_bias, gu_scale,
           down, down_bias, down_scale, token_scale, group_count, group_start,
           source_row, source_slot, row_weight, stacked, partial, middle,
           middle_scale, wide, route_output, out, gu_tiling, down_tiling,
           workspace, experts, tokens, dispatch_input
#ifdef ROUTED_SHARED_PIPELINE
           , shared_input
#endif
           );
}
#else
template <CubeFormat weight_format>
void ProfileLaunch(uint32_t blocks, void* stream, uint64_t sync_base,
                    const uint64_t* p, int64_t tokens)
{
    const uint64_t* s = p + profile::kScratchOffset;
    const DispatchInput dispatch_input{p[6], p[7], p[8], p[9], p[10]};
    routed_int4_fused<weight_format, true><<<blocks, nullptr, stream>>>(
        sync_base, (__gm__ int8_t*)s[kRestoredRows], (__gm__ int4b_t*)p[0],
        (__gm__ float*)p[1], (__gm__ ScaleStorage*)p[2], (__gm__ int4b_t*)p[3],
        (__gm__ float*)p[4], (__gm__ ScaleStorage*)p[5],
        (__gm__ float*)s[kRestoredScale], (__gm__ int32_t*)s[kGroupCount],
        (__gm__ int32_t*)s[kGroupStart], (__gm__ int32_t*)s[kSourceRow],
        (__gm__ int32_t*)s[kSourceSlot], (__gm__ float*)s[kRowWeight],
        (__gm__ int4b_t*)s[kStacked], (__gm__ RoutedProduct*)s[kGateProduct],
        (__gm__ int4b_t*)s[kMiddle], (__gm__ float*)s[kMiddleScale],
        (__gm__ RoutedProduct*)s[kDownProduct], (__gm__ bfloat16_t*)s[kRouteOutput],
        (__gm__ bfloat16_t*)s[kOutput], (__gm__ uint8_t*)s[kGateTiling],
        (__gm__ uint8_t*)s[kDownTiling], (GM_ADDR)s[kMatmulWorkspace],
        kMaxExperts, tokens, blocks, dispatch_input,
        (__gm__ int64_t*)p[profile::kTraceAddress]);
}

extern "C" void routed_int4_profile_launch(
    uint32_t blocks, void* stream, uint64_t sync_base,
    const uint64_t* addresses, int64_t tokens, int64_t weight_layout)
{
    if (weight_layout == kWeightNZ) {
        ProfileLaunch<CubeFormat::NZ>(blocks, stream, sync_base, addresses, tokens);
    } else {
        ProfileLaunch<CubeFormat::ND>(blocks, stream, sync_base, addresses, tokens);
    }
}
#endif
