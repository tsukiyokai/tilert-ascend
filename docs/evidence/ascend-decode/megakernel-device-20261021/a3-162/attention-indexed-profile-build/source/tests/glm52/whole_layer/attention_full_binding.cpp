#include "attention_host.h"
#include "attention_full_contract.h"

namespace {

using namespace whole_layer_attention;
using namespace whole_layer_attention::host;

std::tuple<at::Tensor, at::Tensor, int64_t, int64_t> PrepareFull(
    const at::Tensor& token_x, int64_t pages, int64_t table_pages, int64_t blocks)
{
    TORCH_CHECK(token_x.device().type() == c10::DeviceType::PrivateUse1,
                "shared_full_attention_prepare requires NPU tensors");
    const c10::OptionalDeviceGuard device_guard(token_x.device());
    TORCH_CHECK(token_x.dim() == 2, "token_x must be [M,6144]");
    CheckGeometry(token_x.size(0), pages, table_pages);
    CheckShape(token_x, at::kChar, {token_x.size(0), kHidden}, "token_x");
    const auto launch = PrepareDevice(token_x.get_device());
    if (blocks == 0) {
        blocks = std::min<int64_t>(8, launch.maximum_blocks);
    }
    TORCH_CHECK(blocks > 0 && blocks <= launch.maximum_blocks,
                "blocks exceed physical mixed 1:2 residency");
    const auto config = MakeFullConfig(MakeConfig(token_x.size(0), pages,
        table_pages, blocks, 1.0F, 1e-5F, 1e-5F));
    std::vector<uint8_t> host_tiling(mla_tail::kTilingBytes);
    TORCH_CHECK(mla_tail_build_tiling(config.capacity, host_tiling.data()) == 0,
                "tail matmul tiling failed");
    auto scratch = at::empty({static_cast<int64_t>(config.bytes)},
                             token_x.options().dtype(at::kByte));
    auto tiling = at::empty({config.capacity, static_cast<int64_t>(mla_tail::kTilingBytes)},
                            token_x.options().dtype(at::kByte));
    TORCH_CHECK(aclrtMemcpy(tiling.data_ptr(), tiling.nbytes(), host_tiling.data(),
                host_tiling.size(), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
                "tail tiling upload failed");
    return {scratch, tiling, blocks, static_cast<int64_t>(launch.sync_base)};
}

at::Tensor FullOut(
    const at::Tensor& token_x, const at::Tensor& token_scale,
    const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
    const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr,
    const at::Tensor& scale_dq, const at::Tensor& scale_uq_qr,
    const at::Tensor& scale_dkv_kr, const at::Tensor& gamma_cq,
    const at::Tensor& gamma_ckv, const at::Tensor& sin, const at::Tensor& cos,
    const at::Tensor& slots, at::Tensor& kv_cache, at::Tensor& rope_cache,
    const at::Tensor& topk, const at::Tensor& query_lengths,
    const at::Tensor& kv_lengths, const at::Tensor& page_table,
    const at::Tensor& value_weight, const at::Tensor& output_weight,
    const at::Tensor& output_scale, const c10::optional<at::Tensor>& bias,
    at::Tensor& scratch, const at::Tensor& tiling, at::Tensor& output,
    double scale, double epsilon_cq, double epsilon_ckv,
    int64_t blocks, int64_t sync_base)
{
    TORCH_CHECK(token_x.device().type() == c10::DeviceType::PrivateUse1,
                "shared_full_attention_out requires NPU tensors");
    const c10::OptionalDeviceGuard device_guard(token_x.device());
    const Config attention = CheckInputs(token_x, token_scale, weight_dq,
        weight_uq_qr, weight_uk, weight_dkv_kr, scale_dq, scale_uq_qr,
        scale_dkv_kr, gamma_cq, gamma_ckv, sin, cos, slots, kv_cache,
        rope_cache, topk, query_lengths, kv_lengths, page_table,
        scratch, output, scale, epsilon_cq, epsilon_ckv, blocks, false);
    const FullConfig config = MakeFullConfig(attention);
    CheckShape(value_weight, at::kBFloat16, {16, 512, 64}, "value_weight");
    CheckShape(output_weight, at::kChar, {24, 1024, 256}, "output_weight");
    CheckShape(output_scale, at::kFloat, {6144}, "output_scale");
    CheckShape(output, at::kBFloat16, {attention.rows, 6144}, "output");
    CheckShape(scratch, at::kByte, {static_cast<int64_t>(config.bytes)}, "scratch");
    CheckShape(tiling, at::kByte,
        {config.capacity, static_cast<int64_t>(mla_tail::kTilingBytes)}, "tail_tiling");
    int bias_kind = 0;
    if (bias.has_value()) {
        TORCH_CHECK(bias->scalar_type() == at::kInt || bias->scalar_type() == at::kFloat,
                    "bias must be INT32 accumulation bias or FP32 output bias");
        CheckShape(*bias, bias->scalar_type(), {6144}, "bias");
        bias_kind = bias->scalar_type() == at::kInt ? 1 : 2;
    }
    {
        std::lock_guard<std::mutex> guard(launch_mutex);
        const auto found = launches.find(token_x.get_device());
        TORCH_CHECK(found != launches.end() &&
                    found->second.sync_base == static_cast<uint64_t>(sync_base) &&
                    blocks <= found->second.maximum_blocks,
                    "prepare the current device/context before full attention capture");
    }
    std::vector<at::Tensor> read{token_x, token_scale, weight_dq, weight_uq_qr,
        weight_uk, weight_dkv_kr, scale_dq, scale_uq_qr, scale_dkv_kr, gamma_cq,
        gamma_ckv, sin, cos, slots, topk, query_lengths, kv_lengths, page_table,
        value_weight, output_weight, output_scale, tiling};
    if (bias.has_value()) {
        read.push_back(*bias);
    }
    const std::vector<at::Tensor> written{kv_cache, rope_cache, scratch, output};
    for (const auto& input : read) {
        TORCH_CHECK(input.device() == token_x.device(), "full attention devices differ");
    }
    for (size_t index = 0; index < written.size(); ++index) {
        for (const auto& input : read) {
            at::assert_no_overlap(written[index], input);
        }
        for (size_t previous = 0; previous < index; ++previous) {
            at::assert_no_overlap(written[index], written[previous]);
        }
    }
    whole_layer_full_attention_launch(c10_npu::getCurrentNPUStream().stream(),
        static_cast<uint64_t>(sync_base), config, token_x.data_ptr(),
        token_scale.data_ptr(), weight_dq.data_ptr(), weight_uq_qr.data_ptr(),
        weight_uk.data_ptr(), weight_dkv_kr.data_ptr(), scale_dq.data_ptr(),
        scale_uq_qr.data_ptr(), scale_dkv_kr.data_ptr(), gamma_cq.data_ptr(),
        gamma_ckv.data_ptr(), sin.data_ptr(), cos.data_ptr(), slots.data_ptr(),
        kv_cache.data_ptr(), rope_cache.data_ptr(), topk.data_ptr(),
        query_lengths.data_ptr(), kv_lengths.data_ptr(), page_table.data_ptr(),
        value_weight.data_ptr(), output_weight.data_ptr(), output_scale.data_ptr(),
        bias.has_value() ? bias->data_ptr() : nullptr, bias_kind, scratch.data_ptr(),
        tiling.data_ptr(), output.data_ptr());
    return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(tilert, library)
{
    library.def("shared_full_attention_prepare(Tensor token_x, int pages, "
                "int table_pages, int blocks=0) -> (Tensor, Tensor, int, int)");
    library.def("shared_full_attention_out(Tensor token_x, Tensor token_scale, "
        "Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, "
        "Tensor scale_dq, Tensor scale_uq_qr, Tensor scale_dkv_kr, Tensor gamma_cq, "
        "Tensor gamma_ckv, Tensor sin, Tensor cos, Tensor slots, "
        "Tensor(a!) kv_cache, Tensor(b!) rope_cache, Tensor topk, "
        "Tensor query_lengths, Tensor kv_lengths, Tensor page_table, "
        "Tensor value_weight, Tensor output_weight, Tensor output_scale, Tensor? bias, "
        "Tensor(c!) scratch, Tensor tiling, Tensor(d!) output, float scale, "
        "float epsilon_cq, float epsilon_ckv, int blocks, int sync_base) -> Tensor(d!)");
}

TORCH_LIBRARY_IMPL(tilert, PrivateUse1, library)
{
    library.impl("shared_full_attention_prepare", &PrepareFull);
    library.impl("shared_full_attention_out", &FullOut);
}
