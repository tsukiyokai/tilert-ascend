#include "attention_host.h"
#include "attention_export_contract.h"

namespace {

using namespace whole_layer_attention;
using namespace whole_layer_attention::host;

std::tuple<at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t>
PrepareExport(const at::Tensor& token_x, int64_t pages, int64_t table_pages,
              int64_t blocks)
{
    TORCH_CHECK(token_x.device().type() == c10::DeviceType::PrivateUse1,
                "export attention requires an NPU token tensor");
    const c10::OptionalDeviceGuard device_guard(token_x.device());
    TORCH_CHECK(token_x.dim() == 2, "token_x must be [M,6144]");
    const int64_t rows = token_x.size(0);
    CheckGeometry(rows, pages, table_pages);
    CheckShape(token_x, at::kChar, {rows, kHidden}, "token_x");
    const auto launch = PrepareDevice(token_x.get_device());
    if (blocks == 0) {
        blocks = launch.maximum_blocks;
    }
    TORCH_CHECK(blocks > 0 && blocks <= launch.maximum_blocks,
                "blocks exceed physical mixed 1:2 residency");
    const auto config = MakeConfig(rows, pages, table_pages, blocks,
                                    1.0F, 1e-5F, 1e-5F);
    auto scratch = at::empty({static_cast<int64_t>(config.bytes)},
                             token_x.options().dtype(at::kByte));
    auto query_norm = at::empty({rows, kQueryRank}, token_x.options());
    auto query_scale = at::empty({rows}, token_x.options().dtype(at::kFloat));
    return {scratch, query_norm, query_scale, blocks,
            static_cast<int64_t>(launch.sync_base)};
}

at::Tensor ExportOut(
    const at::Tensor& token_x, const at::Tensor& token_scale,
    const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
    const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr,
    const at::Tensor& scale_dq, const at::Tensor& scale_uq_qr,
    const at::Tensor& scale_dkv_kr, const at::Tensor& gamma_cq,
    const at::Tensor& gamma_ckv, const at::Tensor& sin, const at::Tensor& cos,
    const at::Tensor& slots, at::Tensor& kv_cache, at::Tensor& rope_cache,
    const at::Tensor& topk, const at::Tensor& query_lengths,
    const at::Tensor& kv_lengths, const at::Tensor& page_table,
    at::Tensor& scratch, at::Tensor& output, at::Tensor& query_norm,
    at::Tensor& query_norm_scale, double scale, double epsilon_cq,
    double epsilon_ckv, int64_t blocks, int64_t sync_base)
{
    TORCH_CHECK(token_x.device().type() == c10::DeviceType::PrivateUse1,
                "export attention requires NPU tensors");
    const c10::OptionalDeviceGuard device_guard(token_x.device());
    const Config config = CheckInputs(token_x, token_scale, weight_dq,
        weight_uq_qr, weight_uk, weight_dkv_kr, scale_dq, scale_uq_qr,
        scale_dkv_kr, gamma_cq, gamma_ckv, sin, cos, slots, kv_cache,
        rope_cache, topk, query_lengths, kv_lengths, page_table,
        scratch, output, scale, epsilon_cq, epsilon_ckv, blocks);
    CheckShape(query_norm, at::kChar, {config.rows, kQueryRank}, "query_norm");
    CheckShape(query_norm_scale, at::kFloat, {config.rows}, "query_norm_scale");
    {
        std::lock_guard<std::mutex> guard(launch_mutex);
        const auto found = launches.find(token_x.get_device());
        TORCH_CHECK(found != launches.end() &&
                    found->second.sync_base == static_cast<uint64_t>(sync_base) &&
                    blocks <= found->second.maximum_blocks,
                    "prepare this device/context before export attention capture");
    }
    const std::vector<at::Tensor> written{kv_cache, rope_cache, scratch, output,
                                         query_norm, query_norm_scale};
    const std::vector<at::Tensor> read{token_x, token_scale, weight_dq,
        weight_uq_qr, weight_uk, weight_dkv_kr, scale_dq, scale_uq_qr,
        scale_dkv_kr, gamma_cq, gamma_ckv, sin, cos, slots, topk, query_lengths,
        kv_lengths, page_table};
    for (size_t index = 0; index < written.size(); ++index) {
        TORCH_CHECK(written[index].device() == token_x.device(),
                    "export attention output devices differ");
        for (const auto& input : read) {
            at::assert_no_overlap(written[index], input);
        }
        for (size_t previous = 0; previous < index; ++previous) {
            at::assert_no_overlap(written[index], written[previous]);
        }
    }
    whole_layer_export_attention_launch(
        c10_npu::getCurrentNPUStream().stream(), static_cast<uint64_t>(sync_base),
        config, token_x.data_ptr(), token_scale.data_ptr(), weight_dq.data_ptr(),
        weight_uq_qr.data_ptr(), weight_uk.data_ptr(), weight_dkv_kr.data_ptr(),
        scale_dq.data_ptr(), scale_uq_qr.data_ptr(), scale_dkv_kr.data_ptr(),
        gamma_cq.data_ptr(), gamma_ckv.data_ptr(), sin.data_ptr(), cos.data_ptr(),
        slots.data_ptr(), kv_cache.data_ptr(), rope_cache.data_ptr(),
        topk.data_ptr(), query_lengths.data_ptr(), kv_lengths.data_ptr(),
        page_table.data_ptr(), scratch.data_ptr(), output.data_ptr(),
        query_norm.data_ptr(), query_norm_scale.data_ptr());
    return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(tilert, library)
{
    library.def("export_attention_prepare(Tensor token_x, int pages, "
                "int table_pages, int blocks=0) -> (Tensor, Tensor, Tensor, int, int)");
    library.def("export_attention_out(Tensor token_x, Tensor token_scale, "
        "Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, "
        "Tensor scale_dq, Tensor scale_uq_qr, Tensor scale_dkv_kr, Tensor gamma_cq, "
        "Tensor gamma_ckv, Tensor sin, Tensor cos, Tensor slots, "
        "Tensor(a!) kv_cache, Tensor(b!) rope_cache, Tensor topk, "
        "Tensor query_lengths, Tensor kv_lengths, Tensor page_table, "
        "Tensor(c!) scratch, Tensor(d!) output, Tensor(e!) query_norm, "
        "Tensor(f!) query_norm_scale, float scale, float epsilon_cq, "
        "float epsilon_ckv, int blocks, int sync_base) -> Tensor(d!)");
}

TORCH_LIBRARY_IMPL(tilert, PrivateUse1, library)
{
    library.impl("export_attention_prepare", &PrepareExport);
    library.impl("export_attention_out", &ExportOut);
}
