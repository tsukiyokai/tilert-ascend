#include "attention_host.h"
#include "attention_indexed_contract.h"

namespace {

namespace host = whole_layer_attention::host;
namespace indexed = whole_layer_indexed;
namespace main_attention = whole_layer_attention;

c10::Dict<std::string, int64_t> Describe(const indexed::Config& c)
{
    c10::Dict<std::string, int64_t> result;
    result.insert("query_norm", c.query_norm);
    result.insert("query_norm_scale", c.query_norm_scale);
    result.insert("query", c.query);
    result.insert("key", c.key);
    result.insert("weights", c.weights);
    result.insert("query_scale", c.query_scale);
    result.insert("key_scale", c.key_scale);
    result.insert("topk", c.topk);
    result.insert("latent", c.latent);
    result.insert("attention_query", c.attention_offset + c.attention.query_offset);
    result.insert("attention_rope", c.attention_offset + c.attention.rope_offset);
    result.insert("front_query_projection", c.front_offset + c.front.query_projection);
    result.insert("front_key_projection", c.front_offset + c.front.key_projection);
    result.insert("front_key_normalized", c.front_offset + c.front.key_normalized);
    result.insert("front_query_rope", c.front_offset + c.front.query_rope);
    result.insert("front_key_rope", c.front_offset + c.front.key_rope);
    result.insert("front_norm_stats", c.front_offset + c.front.norm_stats);
    result.insert("front_query_hadamard", c.front_offset + c.front.query_hadamard);
    result.insert("front_key_hadamard", c.front_offset + c.front.key_hadamard);
    result.insert("front_begin", c.front_offset);
    result.insert("front_bytes", c.front.bytes);
    result.insert("tail_values", c.tail_offset + c.tail.values);
    result.insert("tail_quantized", c.tail_offset + c.tail.quantized);
    result.insert("tail_scales", c.tail_offset + c.tail.token_scales);
    result.insert("tail_accumulators", c.tail_offset + c.tail.accumulators);
    result.insert("capacity", c.front.capacity);
#ifdef ATTENTION_INDEXED_PROFILE
    result.insert("trace", c.trace);
    result.insert("trace_words", indexed::kTraceWords);
    result.insert("trace_marks", indexed::kTraceMarks);
    result.insert("trace_stamp_offset", indexed::kTraceStampOffset);
#endif
    result.insert("bytes", c.bytes);
    return result;
}

std::tuple<at::Tensor, at::Tensor, c10::Dict<std::string, int64_t>, int64_t, int64_t>
Prepare(const at::Tensor& token_x, int64_t pages, int64_t table_pages,
         int64_t index_pages, int64_t index_table_pages, bool quantized,
         int64_t blocks)
{
    TORCH_CHECK(token_x.device().type() == c10::DeviceType::PrivateUse1,
                "indexed attention requires an NPU token tensor");
    const c10::OptionalDeviceGuard device_guard(token_x.device());
    TORCH_CHECK(token_x.dim() == 2, "token_x must be [M,6144]");
    const auto rows = token_x.size(0);
    host::CheckShape(token_x, at::kChar, {rows, 6144}, "token_x");
    host::CheckGeometry(rows, pages, table_pages);
    host::CheckGeometry(rows, index_pages, index_table_pages);
    const auto launch = host::PrepareDevice(token_x.get_device());
    if (blocks == 0) {
        blocks = launch.maximum_blocks;
    }
    TORCH_CHECK(blocks > 0 && blocks <= launch.maximum_blocks,
                "blocks exceed physical mixed 1:2 residency");
    const auto c = indexed::MakeConfig(main_attention::MakeConfig(
        rows, pages, table_pages, blocks, 1.0F, 1e-5F, 1e-5F),
        index_pages, index_table_pages, quantized);
    auto scratch = at::empty({static_cast<int64_t>(c.bytes)},
                             token_x.options().dtype(at::kByte));
    auto tiling = at::empty({c.front.capacity, indexed::kTilingBytes},
                            token_x.options().dtype(at::kByte));
    auto cpu = at::zeros(tiling.sizes(), at::TensorOptions().dtype(at::kByte));
    auto* bytes = cpu.data_ptr<uint8_t>();
    TORCH_CHECK(indexer_front_build_tiling(c.front.capacity, bytes) == 0,
                "indexer front tiling failed");
    TORCH_CHECK(mla_tail_build_tiling(c.front.capacity,
                bytes + indexer_front::kTilingBytes) == 0,
                "MLA tail tiling failed");
    tiling.copy_(cpu);
    return {scratch, tiling, Describe(c), blocks,
            static_cast<int64_t>(launch.sync_base)};
}

at::Tensor Execute(
    const at::Tensor& hidden, const at::Tensor& token_x,
    const at::Tensor& token_scale, at::TensorList main_weights,
    at::TensorList front_weights, at::TensorList tail_weights,
    const std::optional<at::Tensor>& bias, at::TensorList main_metadata,
    at::TensorList indexer_metadata, at::TensorList caches,
    at::Tensor& scratch, const at::Tensor& tiling, at::Tensor& output,
    double scale, double epsilon_cq, double epsilon_ckv,
    int64_t blocks, int64_t sync_base)
{
    TORCH_CHECK(token_x.device().type() == c10::DeviceType::PrivateUse1,
                "indexed attention requires NPU tensors");
    const c10::OptionalDeviceGuard device_guard(token_x.device());
    TORCH_CHECK(main_weights.size() == 9 && front_weights.size() == 6 &&
                tail_weights.size() == 3 && main_metadata.size() == 6 &&
                indexer_metadata.size() == 6 && caches.size() == 4,
                "indexed attention requires weight lists 9/6/3, metadata 6/6, caches 4");
    TORCH_CHECK(token_x.dim() == 2 && caches[0].dim() == 4 &&
                caches[2].dim() == 4 && main_metadata[5].dim() == 2 &&
                indexer_metadata[5].dim() == 2, "invalid indexed attention ranks");
    const auto rows = token_x.size(0);
    const auto pages = caches[0].size(0);
    const auto table_pages = main_metadata[5].size(1);
    const auto index_pages = caches[2].size(0);
    const auto index_table_pages = indexer_metadata[5].size(1);
    host::CheckGeometry(rows, pages, table_pages);
    host::CheckGeometry(rows, index_pages, index_table_pages);
    host::CheckScalar(scale, "scale");
    host::CheckScalar(epsilon_cq, "epsilon_cq");
    host::CheckScalar(epsilon_ckv, "epsilon_ckv");
    const bool quantized = caches[2].scalar_type() == at::kChar;
    const auto query_scale_dtype = front_weights[1].scalar_type();
    TORCH_CHECK(query_scale_dtype == at::kFloat || query_scale_dtype == at::kBFloat16,
                "WQ channel scales require their original FP32 or BF16 dtype");
    const auto attention = main_attention::MakeConfig(rows, pages, table_pages,
        blocks, static_cast<float>(scale), static_cast<float>(epsilon_cq),
        static_cast<float>(epsilon_ckv));
    const auto c = indexed::MakeConfig(attention, index_pages, index_table_pages,
                                       quantized, query_scale_dtype == at::kBFloat16);
    host::CheckShape(scratch, at::kByte, {static_cast<int64_t>(c.bytes)}, "scratch");
    // This metadata-only view describes the kernel-owned top-k prefix.
    auto topk = scratch.narrow(0, c.topk, rows * 2048 * 4).view(at::kInt)
                    .view({rows, 1, 2048});
    host::CheckInputs(token_x, token_scale, main_weights[0], main_weights[1],
        main_weights[2], main_weights[3], main_weights[4], main_weights[5],
        main_weights[6], main_weights[7], main_weights[8], main_metadata[0],
        main_metadata[1], main_metadata[2], caches[0], caches[1], topk,
        main_metadata[3], main_metadata[4], main_metadata[5], scratch, output,
        scale, epsilon_cq, epsilon_ckv, blocks, false);
    host::CheckShape(hidden, at::kBFloat16, {rows, 6144}, "hidden");
    host::CheckShape(front_weights[0], at::kChar, {16, 2048, 256}, "indexer_wq");
    host::CheckShape(front_weights[1], query_scale_dtype, {4096}, "indexer_wq_scale");
    host::CheckShape(front_weights[2], at::kBFloat16, {5, 6144, 32}, "indexer_wk");
    host::CheckShape(front_weights[3], at::kFloat, {128}, "indexer_gamma");
    host::CheckShape(front_weights[4], at::kFloat, {128}, "indexer_beta");
    host::CheckShape(front_weights[5], at::kBFloat16, {128, 128}, "hadamard");
    host::CheckShape(tail_weights[0], at::kBFloat16, {16, 512, 64}, "value_weight");
    host::CheckShape(tail_weights[1], at::kChar, {24, 1024, 256}, "output_weight");
    host::CheckShape(tail_weights[2], at::kFloat, {6144}, "output_scale");
    host::CheckShape(indexer_metadata[0], at::kBFloat16, {rows, 64}, "indexer_sin");
    host::CheckShape(indexer_metadata[1], at::kBFloat16, {rows, 64}, "indexer_cos");
    host::CheckShape(indexer_metadata[2], at::kLong, {rows}, "indexer_slots");
    host::CheckShape(indexer_metadata[3], at::kInt, {1}, "indexer_query_lengths");
    host::CheckShape(indexer_metadata[4], at::kInt, {1}, "indexer_key_lengths");
    host::CheckShape(indexer_metadata[5], at::kInt, {1, index_table_pages},
                     "indexer_page_table");
    host::CheckShape(caches[2], quantized ? at::kChar : at::kBFloat16,
                     {index_pages, 128, 1, 128}, "indexer_cache");
    if (quantized) {
        host::CheckShape(caches[3], at::kHalf, {index_pages, 128}, "indexer_scales");
    } else {
        host::CheckShape(caches[3], at::kHalf, {0}, "unused_indexer_scales");
    }
    host::CheckShape(tiling, at::kByte,
                     {c.front.capacity, indexed::kTilingBytes}, "tiling");
    host::CheckShape(output, at::kBFloat16, {rows, 6144}, "output");
    int bias_kind = 0;
    if (bias.has_value()) {
        TORCH_CHECK(bias->scalar_type() == at::kInt || bias->scalar_type() == at::kFloat,
                    "bias must be INT32 accumulation or FP32 output bias");
        host::CheckShape(*bias, bias->scalar_type(), {6144}, "bias");
        bias_kind = bias->scalar_type() == at::kInt ? 1 : 2;
    }
    {
        std::lock_guard<std::mutex> guard(host::launch_mutex);
        const auto found = host::launches.find(token_x.get_device());
        TORCH_CHECK(found != host::launches.end() && blocks > 0 &&
                    blocks <= found->second.maximum_blocks &&
                    found->second.sync_base == static_cast<uint64_t>(sync_base),
                    "prepare this device/context before indexed attention capture");
    }
    std::vector<at::Tensor> read{hidden, token_x, token_scale, tiling};
    for (const auto& list : {main_weights, front_weights, tail_weights,
                             main_metadata, indexer_metadata}) {
        read.insert(read.end(), list.begin(), list.end());
    }
    if (bias.has_value()) {
        read.push_back(*bias);
    }
    std::vector<at::Tensor> written{scratch, output};
    written.insert(written.end(), caches.begin(), caches.end());
    for (const auto& tensor : read) {
        TORCH_CHECK(tensor.device() == token_x.device(), "indexed input devices differ");
    }
    for (size_t index = 0; index < written.size(); ++index) {
        TORCH_CHECK(written[index].device() == token_x.device(),
                    "indexed output devices differ");
        for (const auto& input : read) {
            at::assert_no_overlap(written[index], input);
        }
        for (size_t previous = 0; previous < index; ++previous) {
            at::assert_no_overlap(written[index], written[previous]);
        }
    }
    indexed::Addresses addresses{};
    addresses.hidden = hidden.data_ptr();
    addresses.token_x = token_x.data_ptr();
    addresses.token_scale = token_scale.data_ptr();
    for (int index = 0; index < 9; ++index) {
        addresses.main_weights[index] = main_weights[index].data_ptr();
    }
    for (int index = 0; index < 6; ++index) {
        addresses.front_weights[index] = front_weights[index].data_ptr();
        addresses.main_metadata[index] = main_metadata[index].data_ptr();
        addresses.indexer_metadata[index] = indexer_metadata[index].data_ptr();
    }
    for (int index = 0; index < 3; ++index) {
        addresses.tail_weights[index] = tail_weights[index].data_ptr();
    }
    for (int index = 0; index < 4; ++index) {
        addresses.caches[index] = caches[index].data_ptr();
    }
    addresses.bias = bias.has_value() ? bias->data_ptr() : nullptr;
    addresses.scratch = scratch.data_ptr();
    addresses.tiling = tiling.data_ptr();
    addresses.output = output.data_ptr();
    whole_layer_indexed_attention_launch(
        c10_npu::getCurrentNPUStream().stream(), static_cast<uint64_t>(sync_base),
        c, addresses, bias_kind);
    return output;
}

}  // namespace

#ifndef ATTENTION_OP_NAMESPACE
#define ATTENTION_OP_NAMESPACE tilert
#endif

TORCH_LIBRARY_FRAGMENT(ATTENTION_OP_NAMESPACE, library)
{
    library.def("indexed_attention_prepare(Tensor token_x, int pages, int table_pages, "
        "int index_pages, int index_table_pages, bool quantized, int blocks=0) "
        "-> (Tensor, Tensor, Dict(str, int), int, int)");
    library.def("indexed_attention_out(Tensor hidden, Tensor token_x, Tensor token_scale, "
        "Tensor[] main_weights, Tensor[] front_weights, Tensor[] tail_weights, "
        "Tensor? bias, Tensor[] main_metadata, Tensor[] indexer_metadata, "
        "Tensor(a!)[] caches, Tensor(b!) scratch, Tensor tiling, Tensor(c!) output, "
        "float scale, float epsilon_cq, float epsilon_ckv, int blocks, int sync_base) "
        "-> Tensor(c!)");
}

TORCH_LIBRARY_IMPL(ATTENTION_OP_NAMESPACE, PrivateUse1, library)
{
    library.impl("indexed_attention_prepare", &Prepare);
    library.impl("indexed_attention_out", &Execute);
}
