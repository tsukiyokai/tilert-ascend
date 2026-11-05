#include "attention_host.h"
#include "attention_norm_contract.h"

namespace {

using namespace whole_layer_attention_terminal;
using namespace whole_layer_attention_terminal::host;

std::tuple<at::Tensor, at::Tensor, int64_t, int64_t> PrepareNormAttention(
    const at::Tensor& input, int64_t pages, int64_t table_pages, int64_t blocks)
{
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                input.dim() == 2, "input must be an NPU BF16 matrix");
    const c10::OptionalDeviceGuard device_guard(input.device());
    CheckGeometry(input.size(0), pages, table_pages);
    CheckShape(input, at::kBFloat16, {input.size(0), kHidden}, "input");
    const auto launch = PrepareDevice(input.get_device());
    if (blocks == 0) {
        blocks = std::min<int64_t>(8, launch.maximum_blocks);
    }
    TORCH_CHECK(blocks > 0 && blocks <= launch.maximum_blocks,
                "blocks exceed physical mixed 1:2 residency");
    const auto config = MakeNormConfig(MakeConfig(input.size(0), pages,
        table_pages, blocks, 1.0F, 1e-5F, 1e-5F), 1e-5F);
    std::vector<uint8_t> host_tiling(mla_tail_attention_terminal::kTilingBytes);
    TORCH_CHECK(mla_tail_attention_terminal_build_tiling(config.full.capacity, host_tiling.data()) == 0,
                "tail matmul tiling failed");
    auto scratch = at::empty({static_cast<int64_t>(config.bytes)},
                             input.options().dtype(at::kByte));
    auto tiling = at::empty(
        {config.full.capacity, static_cast<int64_t>(mla_tail_attention_terminal::kTilingBytes)},
        input.options().dtype(at::kByte));
    TORCH_CHECK(aclrtMemcpy(tiling.data_ptr(), tiling.nbytes(), host_tiling.data(),
                host_tiling.size(), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
                "tail tiling upload failed");
    return {scratch, tiling, blocks, static_cast<int64_t>(launch.sync_base)};
}

at::Tensor NormAttentionOut(
    const at::Tensor& input, const at::Tensor& residual,
    const at::Tensor& norm_gamma, const c10::optional<at::Tensor>& norm_beta,
    at::Tensor& retained, const at::Tensor& weight_dq,
    const at::Tensor& weight_uq_qr, const at::Tensor& weight_uk,
    const at::Tensor& weight_dkv_kr, const at::Tensor& scale_dq,
    const at::Tensor& scale_uq_qr, const at::Tensor& scale_dkv_kr,
    const at::Tensor& gamma_cq, const at::Tensor& gamma_ckv,
    const at::Tensor& sin, const at::Tensor& cos, const at::Tensor& slots,
    at::Tensor& kv_cache, at::Tensor& rope_cache, const at::Tensor& topk,
    const at::Tensor& query_lengths, const at::Tensor& kv_lengths,
    const at::Tensor& page_table, const at::Tensor& value_weight,
    const at::Tensor& output_weight, const at::Tensor& output_scale,
    const c10::optional<at::Tensor>& bias, at::Tensor& scratch,
    const at::Tensor& tiling, at::Tensor& output, double norm_epsilon,
    double scale, double epsilon_cq, double epsilon_ckv,
    int64_t blocks, int64_t sync_base)
{
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                input.dim() == 2, "input must be an NPU BF16 matrix");
    const c10::OptionalDeviceGuard device_guard(input.device());
    TORCH_CHECK(kv_cache.dim() == 4 && page_table.dim() == 2,
                "cache and page table ranks must be 4 and 2");
    const int64_t rows = input.size(0);
    CheckGeometry(rows, kv_cache.size(0), page_table.size(1));
    CheckScalar(norm_epsilon, "norm_epsilon");
    TORCH_CHECK(blocks > 0 && blocks <= 64, "invalid mixed group count");
    const auto shape_config = MakeNormConfig(MakeConfig(rows, kv_cache.size(0),
        page_table.size(1), blocks, 1.0F, 1e-5F, 1e-5F), norm_epsilon);
    CheckShape(scratch, at::kByte, {static_cast<int64_t>(shape_config.bytes)},
                "scratch");
    // Metadata-only views validate the quantized producer/consumer contract.
    const auto codes = scratch.narrow(0, 0, rows * kHidden).view(at::kChar)
        .view({rows, kHidden});
    const auto scales = scratch.narrow(0, shape_config.scale_offset, rows * 4)
        .view(at::kFloat).view({rows, 1});
    const auto attention = CheckInputs(codes, scales, weight_dq, weight_uq_qr,
        weight_uk, weight_dkv_kr, scale_dq, scale_uq_qr, scale_dkv_kr,
        gamma_cq, gamma_ckv, sin, cos, slots, kv_cache, rope_cache, topk,
        query_lengths, kv_lengths, page_table, scratch, output,
        scale, epsilon_cq, epsilon_ckv, blocks, false);
    const auto config = MakeNormConfig(attention, static_cast<float>(norm_epsilon));
    CheckShape(input, at::kBFloat16, {rows, kHidden}, "input");
    CheckShape(residual, at::kBFloat16, {rows, kHidden}, "residual");
    CheckShape(norm_gamma, at::kBFloat16, {kHidden}, "norm_gamma");
    CheckShape(retained, at::kBFloat16, {rows, kHidden}, "retained");
    CheckShape(value_weight, at::kBFloat16, {16, 512, 64}, "value_weight");
    CheckShape(output_weight, at::kChar, {24, 1024, 256}, "output_weight");
    CheckShape(output_scale, at::kFloat, {kHidden}, "output_scale");
    CheckShape(output, at::kBFloat16, {rows, kHidden}, "output");
    CheckShape(tiling, at::kByte,
        {config.full.capacity, static_cast<int64_t>(mla_tail_attention_terminal::kTilingBytes)}, "tiling");
    if (norm_beta.has_value()) {
        CheckShape(*norm_beta, at::kBFloat16, {kHidden}, "norm_beta");
    }
    int bias_kind = 0;
    if (bias.has_value()) {
        TORCH_CHECK(bias->scalar_type() == at::kInt || bias->scalar_type() == at::kFloat,
                    "bias must be INT32 accumulation bias or FP32 output bias");
        CheckShape(*bias, bias->scalar_type(), {kHidden}, "bias");
        bias_kind = bias->scalar_type() == at::kInt ? 1 : 2;
    }
    {
        std::lock_guard<std::mutex> guard(launch_mutex);
        const auto found = launches.find(input.get_device());
        TORCH_CHECK(found != launches.end() && blocks > 0 &&
                    found->second.sync_base == static_cast<uint64_t>(sync_base) &&
                    blocks <= found->second.maximum_blocks,
                    "prepare this context before norm-attention capture");
    }
    std::vector<at::Tensor> read{input, residual, norm_gamma, weight_dq, weight_uq_qr,
        weight_uk, weight_dkv_kr, scale_dq, scale_uq_qr, scale_dkv_kr, gamma_cq,
        gamma_ckv, sin, cos, slots, topk, query_lengths, kv_lengths, page_table,
        value_weight, output_weight, output_scale, tiling};
    if (bias.has_value()) {
        read.push_back(*bias);
    }
    if (norm_beta.has_value()) {
        read.push_back(*norm_beta);
    }
    const std::vector<at::Tensor> written{retained, kv_cache, rope_cache, scratch, output};
    for (const auto& tensor : read) {
        TORCH_CHECK(tensor.device() == input.device(), "norm-attention devices differ");
    }
    for (size_t index = 0; index < written.size(); ++index) {
        TORCH_CHECK(written[index].device() == input.device(),
                    "norm-attention devices differ");
        for (const auto& source : read) {
            at::assert_no_overlap(written[index], source);
        }
        for (size_t previous = 0; previous < index; ++previous) {
            at::assert_no_overlap(written[index], written[previous]);
        }
    }
    whole_layer_norm_attention_launch(c10_npu::getCurrentNPUStream().stream(),
        static_cast<uint64_t>(sync_base), config, input.data_ptr(), residual.data_ptr(),
        norm_gamma.data_ptr(), norm_beta.has_value() ? norm_beta->data_ptr() : nullptr,
        retained.data_ptr(), weight_dq.data_ptr(), weight_uq_qr.data_ptr(),
        weight_uk.data_ptr(), weight_dkv_kr.data_ptr(), scale_dq.data_ptr(),
        scale_uq_qr.data_ptr(), scale_dkv_kr.data_ptr(), gamma_cq.data_ptr(),
        gamma_ckv.data_ptr(), sin.data_ptr(), cos.data_ptr(), slots.data_ptr(),
        kv_cache.data_ptr(), rope_cache.data_ptr(), topk.data_ptr(),
        query_lengths.data_ptr(), kv_lengths.data_ptr(), page_table.data_ptr(),
        value_weight.data_ptr(), output_weight.data_ptr(), output_scale.data_ptr(),
        bias.has_value() ? bias->data_ptr() : nullptr, bias_kind,
        scratch.data_ptr(), tiling.data_ptr(), output.data_ptr());
    return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(tilert_attention_terminal, library)
{
    library.def("shared_norm_attention_prepare(Tensor input, int pages, "
                "int table_pages, int blocks=0) -> (Tensor, Tensor, int, int)");
    library.def("shared_norm_attention_out(Tensor input, Tensor residual, "
        "Tensor norm_gamma, Tensor? norm_beta, Tensor(a!) retained, "
        "Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, "
        "Tensor scale_dq, Tensor scale_uq_qr, Tensor scale_dkv_kr, Tensor gamma_cq, "
        "Tensor gamma_ckv, Tensor sin, Tensor cos, Tensor slots, "
        "Tensor(b!) kv_cache, Tensor(c!) rope_cache, Tensor topk, "
        "Tensor query_lengths, Tensor kv_lengths, Tensor page_table, "
        "Tensor value_weight, Tensor output_weight, Tensor output_scale, Tensor? bias, "
        "Tensor(d!) scratch, Tensor tiling, Tensor(e!) output, float norm_epsilon, "
        "float scale, float epsilon_cq, float epsilon_ckv, "
        "int blocks, int sync_base) -> Tensor(e!)");
}

TORCH_LIBRARY_IMPL(tilert_attention_terminal, PrivateUse1, library)
{
    library.impl("shared_norm_attention_prepare", &PrepareNormAttention);
    library.impl("shared_norm_attention_out", &NormAttentionOut);
}
