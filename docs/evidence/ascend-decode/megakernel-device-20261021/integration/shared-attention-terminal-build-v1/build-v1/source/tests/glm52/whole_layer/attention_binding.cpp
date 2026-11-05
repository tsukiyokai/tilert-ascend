#include "attention_host.h"

namespace {

using namespace whole_layer_attention_terminal;
using namespace whole_layer_attention_terminal::host;

at::Tensor AttentionOut(
    const at::Tensor& token_x, const at::Tensor& token_scale,
    const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
    const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr,
    const at::Tensor& scale_dq, const at::Tensor& scale_uq_qr,
    const at::Tensor& scale_dkv_kr, const at::Tensor& gamma_cq,
    const at::Tensor& gamma_ckv, const at::Tensor& sin, const at::Tensor& cos,
    const at::Tensor& slots, at::Tensor& kv_cache, at::Tensor& rope_cache,
    const at::Tensor& topk, const at::Tensor& query_lengths,
    const at::Tensor& kv_lengths, const at::Tensor& page_table,
    at::Tensor& scratch, at::Tensor& output, double scale,
    double epsilon_cq, double epsilon_ckv, int64_t blocks, int64_t sync_base)
{
    TORCH_CHECK(token_x.device().type() == c10::DeviceType::PrivateUse1,
                "shared_attention_out requires NPU tensors");
    const c10::OptionalDeviceGuard device_guard(token_x.device());
    const Config config = CheckInputs(token_x, token_scale, weight_dq,
        weight_uq_qr, weight_uk, weight_dkv_kr, scale_dq, scale_uq_qr,
        scale_dkv_kr, gamma_cq, gamma_ckv, sin, cos, slots, kv_cache,
        rope_cache, topk, query_lengths, kv_lengths, page_table,
        scratch, output, scale, epsilon_cq, epsilon_ckv, blocks);
    {
        std::lock_guard<std::mutex> guard(launch_mutex);
        const auto found = launches.find(token_x.get_device());
        TORCH_CHECK(found != launches.end() &&
                    found->second.sync_base == static_cast<uint64_t>(sync_base) &&
                    blocks <= found->second.maximum_blocks,
                    "prepare this device/context before attention capture");
    }
    const std::vector<at::Tensor> written{kv_cache, rope_cache, scratch, output};
    const std::vector<at::Tensor> read{token_x, token_scale, weight_dq, weight_uq_qr,
        weight_uk, weight_dkv_kr, scale_dq, scale_uq_qr, scale_dkv_kr, gamma_cq,
        gamma_ckv, sin, cos, slots, topk, query_lengths, kv_lengths, page_table};
    for (size_t index = 0; index < written.size(); ++index) {
        for (const auto& input : read) {
            at::assert_no_overlap(written[index], input);
        }
        for (size_t previous = 0; previous < index; ++previous) {
            at::assert_no_overlap(written[index], written[previous]);
        }
    }
    // Direct launches must follow Torch producers still in the host task queue.
    whole_layer_attention_terminal_launch(c10_npu::getCurrentNPUStream().stream(),
        static_cast<uint64_t>(sync_base), config, token_x.data_ptr(),
        token_scale.data_ptr(), weight_dq.data_ptr(), weight_uq_qr.data_ptr(),
        weight_uk.data_ptr(), weight_dkv_kr.data_ptr(), scale_dq.data_ptr(),
        scale_uq_qr.data_ptr(), scale_dkv_kr.data_ptr(), gamma_cq.data_ptr(),
        gamma_ckv.data_ptr(), sin.data_ptr(), cos.data_ptr(), slots.data_ptr(),
        kv_cache.data_ptr(), rope_cache.data_ptr(), topk.data_ptr(),
        query_lengths.data_ptr(), kv_lengths.data_ptr(), page_table.data_ptr(),
        scratch.data_ptr(), output.data_ptr());
    return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(tilert_attention_terminal, library)
{
    library.def("shared_attention_prepare(Tensor token_x, int pages, "
                "int table_pages, int blocks=0) -> (Tensor, int, int)");
    library.def("shared_attention_out(Tensor token_x, Tensor token_scale, "
        "Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, "
        "Tensor scale_dq, Tensor scale_uq_qr, Tensor scale_dkv_kr, Tensor gamma_cq, "
        "Tensor gamma_ckv, Tensor sin, Tensor cos, Tensor slots, "
        "Tensor(a!) kv_cache, Tensor(b!) rope_cache, Tensor topk, "
        "Tensor query_lengths, Tensor kv_lengths, Tensor page_table, "
        "Tensor(c!) scratch, Tensor(d!) output, float scale, float epsilon_cq, "
        "float epsilon_ckv, int blocks, int sync_base) -> Tensor(d!)");
}

TORCH_LIBRARY_IMPL(tilert_attention_terminal, PrivateUse1, library)
{
    library.impl("shared_attention_prepare", &Prepare);
    library.impl("shared_attention_out", &AttentionOut);
}
