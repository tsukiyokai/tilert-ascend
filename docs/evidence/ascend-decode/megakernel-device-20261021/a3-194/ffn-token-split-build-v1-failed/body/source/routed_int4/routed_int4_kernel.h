#ifndef TESTS_GLM52_ROUTED_INT4_KERNEL_H
#define TESTS_GLM52_ROUTED_INT4_KERNEL_H

#include <cstdint>

#include "routed_int4_contract.h"

#ifdef ROUTED_SHARED_PIPELINE
#include "routed_int4_pipeline.h"
#ifdef ROUTED_INPUT_FRONT
#define routed_int4_launch ffn_token_split_unused_body_launch
#define routed_int4_build_tiling ffn_token_split_body_build_routed_tiling
#else
#define routed_int4_launch routed_shared_launch
#define routed_int4_build_tiling routed_shared_build_routed_tiling
#endif
#elif defined(ROUTED_FIXPIPE_HALF)
#define routed_int4_launch routed_int4_half_launch
#define routed_int4_build_tiling routed_int4_half_build_tiling
#elif defined(ROUTED_CONTROL)
#define routed_int4_launch routed_int4_control_launch
#define routed_int4_build_tiling routed_int4_control_build_tiling
#elif defined(ROUTED_INPUT_FRONT)
#define routed_int4_launch routed_int4_front_launch
#define routed_int4_build_tiling routed_int4_front_build_tiling
#elif defined(ROUTED_GU_MDL)
#define routed_int4_launch routed_int4_mdl_launch
#define routed_int4_build_tiling routed_int4_mdl_build_tiling
#elif defined(ROUTED_GU_PAD16)
#define routed_int4_launch routed_int4_pad16_launch
#define routed_int4_build_tiling routed_int4_pad16_build_tiling
#elif defined(ROUTED_BALANCED_NAMESPACE)
#define routed_int4_launch routed_int4_balanced_launch
#define routed_int4_build_tiling routed_int4_balanced_build_tiling
#endif

// Scratch belongs to the caller and may be shared by sequential layers on
// one stream. Routing arrays describe expert-major rows; source_slot retains
// the original top-k column. Only the first tokens * kTopK routes are used.
#if defined(ROUTED_SHARED_PIPELINE) || defined(ROUTED_GU_MDL) \
    || defined(ROUTED_GU_PAD16) || defined(ROUTED_INPUT_FRONT)
extern "C"
#endif
void routed_int4_launch(
    uint32_t blocks, void* stream, uint64_t sync_base, void* rows,
    void* gate_up, float* gu_bias, routed_int4::ScaleStorage* gu_scale, void* down,
    float* down_bias, routed_int4::ScaleStorage* down_scale, float* token_scale,
    void* group_count, void* group_start, void* source_row,
    void* source_slot, float* row_weight, void* stacked, void* partial,
    void* middle, float* middle_scale, void* wide, void* route_output,
    void* out, void* gu_tiling, void* down_tiling, void* workspace,
    int64_t experts, int64_t tokens,
    int64_t weight_layout = routed_int4::kWeightND,
    routed_int4::DispatchInput dispatch = {}
#ifdef ROUTED_SHARED_PIPELINE
    , routed_int4::pipeline::Input shared_input = {}
#endif
    );

/** Fill a 256-byte buffer with the cube tiling for one INT4 projection. */
extern "C" int routed_int4_build_tiling(int rows, int columns, int inner,
                                        int width, void* buffer,
                                        int64_t weight_layout = routed_int4::kWeightND);

#endif
