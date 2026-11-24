#ifndef TESTS_GLM52_WHOLE_LAYER_ATTENTION_INDEXED_SFA_H
#define TESTS_GLM52_WHOLE_LAYER_ATTENTION_INDEXED_SFA_H

// Load SDK declarations globally before isolating the vendor SFA declarations.
// Its global ConstInfo, RunInfo and Min collide with LI/QLI name lookup when
// both operators are instantiated in the same translation unit.
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "attention_sfa_tiling.h"

#define C_TEMPLATE 0
#define V_TEMPLATE 1
#define EVENT_ID4 4U
#define EVENT_ID5 5U
#define EVENT_ID6 6U
#define EVENT_ID7 7U
namespace whole_layer_indexed_sfa {
#include "attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_kernel_mla.h"
}  // namespace whole_layer_indexed_sfa
#undef EVENT_ID4
#undef EVENT_ID5
#undef EVENT_ID6
#undef EVENT_ID7

// These are the only vendor SFA names used by attention_native.h. The vendor
// include guards keep its internal declarations within the namespace above.
using whole_layer_indexed_sfa::SFA_LAYOUT;
using whole_layer_indexed_sfa::SFAType;
using whole_layer_indexed_sfa::SparseFlashAttentionMla;

#endif
