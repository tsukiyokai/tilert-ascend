#include <dlfcn.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/sparse_flash_attention/sparse_flash_attention_torch_adpt.h"

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
    ops.def(
        "npu_sparse_flash_attention(Tensor query, Tensor key, Tensor value,"
        "                           Tensor sparse_indices, float scale_value, *,"
        "                           Tensor? block_table=None, Tensor? actual_seq_lengths_query=None,"
        "                           Tensor? actual_seq_lengths_kv=None, Tensor? query_rope=None,"
        "                           Tensor? key_rope=None, int sparse_block_size=1,"
        "                           str layout_query='BSND', str layout_kv='BSND',"
        "                           int sparse_mode=3, int pre_tokens=9223372036854775807,"
        "                           int next_tokens=9223372036854775807, int attention_mode=2,"
        "                           bool return_softmax_lse=False) -> (Tensor attention_out, Tensor softmax_max, Tensor softmax_sum)"
    );
    ops.impl("npu_sparse_flash_attention", torch::kPrivateUse1, &vllm_ascend::npu_sparse_flash_attention);
}

extern "C" const char* vendor_hook_sfa_api_library(const char* name) {
    void* address = GetOpApiFuncAddr(name);
    Dl_info information{};
    if (address == nullptr || dladdr(address, &information) == 0) {
        return nullptr;
    }
    return information.dli_fname;
}
