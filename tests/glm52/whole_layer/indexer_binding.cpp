#include "attention_host.h"
#include "indexer_contract.h"

namespace {

namespace host = whole_layer_attention::host;
namespace indexer = whole_layer_indexer;

bool CheckQuery(const at::Tensor& query)
{
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1,
                "indexer requires NPU tensors");
    TORCH_CHECK(query.dim() == 3, "query must be [M,32,128]");
    const bool quantized = query.scalar_type() == at::kChar;
    host::CheckShape(query, quantized ? at::kChar : at::kBFloat16,
        {query.size(0), indexer::kHeads, indexer::kHeadDim}, "query");
    return quantized;
}

std::tuple<at::Tensor, int64_t, int64_t> Prepare(
    const at::Tensor& query, int64_t pages, int64_t table_pages, int64_t blocks)
{
    const bool quantized = CheckQuery(query);
    const c10::OptionalDeviceGuard device_guard(query.device());
    host::CheckGeometry(query.size(0), pages, table_pages);
    const auto launch = host::PrepareDevice(query.get_device());
    if (blocks == 0) {
        // CANN partitions sorted lists over the full physical grid. Changing
        // that partition can change the stable order of equal-score indices.
        blocks = launch.maximum_blocks;
    }
    TORCH_CHECK(blocks > 0 && blocks <= launch.maximum_blocks,
                "blocks exceed physical mixed 1:2 residency");
    const auto config = indexer::MakeConfig(query.size(0), pages,
        table_pages, blocks, quantized);
    auto scratch = at::empty({static_cast<int64_t>(config.bytes)},
                             query.options().dtype(at::kByte));
    return {scratch, blocks, static_cast<int64_t>(launch.sync_base)};
}

at::Tensor IndexerOut(
    const at::Tensor& query, const at::Tensor& new_keys, const at::Tensor& weights,
    const std::optional<at::Tensor>& query_scale,
    const std::optional<at::Tensor>& new_key_scale,
    const at::Tensor& slots, at::Tensor& key_cache,
    const std::optional<at::Tensor>& scale_cache,
    const at::Tensor& query_lengths, const at::Tensor& key_lengths,
    const at::Tensor& page_table, at::Tensor& scratch, at::Tensor& output,
    int64_t blocks, int64_t sync_base)
{
    const bool quantized = CheckQuery(query);
    const c10::OptionalDeviceGuard device_guard(query.device());
    TORCH_CHECK(key_cache.dim() == 4 && page_table.dim() == 2,
                "key cache and page table must have rank 4 and 2");
    const int64_t rows = query.size(0);
    const int64_t pages = key_cache.size(0);
    const int64_t table_pages = page_table.size(1);
    host::CheckGeometry(rows, pages, table_pages);
    {
        std::lock_guard<std::mutex> guard(host::launch_mutex);
        const auto found = host::launches.find(query.get_device());
        TORCH_CHECK(found != host::launches.end() &&
                    found->second.sync_base == static_cast<uint64_t>(sync_base) &&
                    blocks > 0 && blocks <= found->second.maximum_blocks,
                    "prepare this device/context before indexer capture");
    }
    const auto config = indexer::MakeConfig(rows, pages, table_pages, blocks,
                                           quantized);
    const auto dtype = quantized ? at::kChar : at::kBFloat16;
    host::CheckShape(new_keys, dtype, {rows, indexer::kHeadDim}, "new_keys");
    host::CheckShape(weights, quantized ? at::kHalf : at::kBFloat16,
                     {rows, indexer::kHeads}, "weights");
    host::CheckShape(slots, at::kLong, {rows}, "slots");
    host::CheckShape(key_cache, dtype,
        {pages, indexer::kPage, 1, indexer::kHeadDim}, "key_cache");
    host::CheckShape(query_lengths, at::kInt, {1}, "query_lengths");
    host::CheckShape(key_lengths, at::kInt, {1}, "key_lengths");
    host::CheckShape(page_table, at::kInt, {1, table_pages}, "page_table");
    host::CheckShape(scratch, at::kByte, {static_cast<int64_t>(config.bytes)},
                     "scratch");
    host::CheckShape(output, at::kInt, {rows, 1, indexer::kTopK}, "output");
    TORCH_CHECK(query_scale.has_value() == quantized &&
                new_key_scale.has_value() == quantized &&
                scale_cache.has_value() == quantized,
                "INT8 requires all three scale tensors; BF16 requires none");
    std::vector<at::Tensor> read{query, new_keys, weights, slots,
        query_lengths, key_lengths, page_table};
    std::vector<at::Tensor> written{key_cache, scratch, output};
    if (quantized) {
        host::CheckShape(*query_scale, at::kHalf, {rows, indexer::kHeads},
                         "query_scale");
        host::CheckShape(*new_key_scale, at::kHalf, {rows}, "new_key_scale");
        host::CheckShape(*scale_cache, at::kHalf, {pages, indexer::kPage},
                         "scale_cache");
        read.push_back(*query_scale);
        read.push_back(*new_key_scale);
        written.push_back(*scale_cache);
    }
    for (const auto& tensor : read) {
        TORCH_CHECK(tensor.device() == query.device(), "indexer devices differ");
    }
    for (size_t index = 0; index < written.size(); ++index) {
        TORCH_CHECK(written[index].device() == query.device(),
                    "indexer devices differ");
        for (const auto& input : read) {
            at::assert_no_overlap(written[index], input);
        }
        for (size_t previous = 0; previous < index; ++previous) {
            at::assert_no_overlap(written[index], written[previous]);
        }
    }
    whole_layer_indexer_launch(c10_npu::getCurrentNPUStream().stream(),
        static_cast<uint64_t>(sync_base), config, query.data_ptr(),
        new_keys.data_ptr(), weights.data_ptr(),
        quantized ? query_scale->data_ptr() : nullptr,
        quantized ? new_key_scale->data_ptr() : nullptr,
        slots.data_ptr(), key_cache.data_ptr(),
        quantized ? scale_cache->data_ptr() : nullptr,
        query_lengths.data_ptr(), key_lengths.data_ptr(), page_table.data_ptr(),
        scratch.data_ptr(), output.data_ptr());
    return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(tilert, library)
{
    library.def("indexer_prepare(Tensor query, int pages, int table_pages, "
                "int blocks=0) -> (Tensor, int, int)");
    library.def("indexer_out(Tensor query, Tensor new_keys, Tensor weights, "
        "Tensor? query_scale, Tensor? new_key_scale, Tensor slots, "
        "Tensor(a!) key_cache, Tensor(b!)? scale_cache, Tensor query_lengths, "
        "Tensor key_lengths, Tensor page_table, Tensor(c!) scratch, "
        "Tensor(d!) output, int blocks, int sync_base) -> Tensor(d!)");
}

TORCH_LIBRARY_IMPL(tilert, PrivateUse1, library)
{
    library.impl("indexer_prepare", &Prepare);
    library.impl("indexer_out", &IndexerOut);
}
