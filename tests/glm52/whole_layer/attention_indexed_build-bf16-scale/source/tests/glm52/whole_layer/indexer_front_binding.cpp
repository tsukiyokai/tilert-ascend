#include "attention_host.h"
#include "indexer_front_contract.h"

namespace {
namespace host = whole_layer_attention::host;
namespace front = indexer_front;

int64_t CheckHidden(const at::Tensor& hidden)
{
    TORCH_CHECK(hidden.device().type() == c10::DeviceType::PrivateUse1,
                "indexer front requires NPU tensors");
    TORCH_CHECK(hidden.dim() == 2, "hidden must be [M,6144]");
    const int64_t rows = hidden.size(0);
    TORCH_CHECK(rows >= 1 && rows <= 32, "rows must lie in [1,32]");
    host::CheckShape(hidden, at::kBFloat16, {rows, front::kHidden}, "hidden");
    return rows;
}

std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>, int64_t, int64_t>
Prepare(const at::Tensor& hidden, bool quantized, int64_t blocks)
{
    const int64_t rows = CheckHidden(hidden);
    const c10::OptionalDeviceGuard device_guard(hidden.device());
    const auto launch = host::PrepareDevice(hidden.get_device());
    if (blocks == 0) {
        blocks = launch.maximum_blocks;
    }
    TORCH_CHECK(blocks > 0 && blocks <= launch.maximum_blocks,
                "blocks exceed physical mixed 1:2 residency");
    const auto c = front::MakeConfig(rows, blocks, quantized);
    const auto options = hidden.options();
    auto scratch = at::empty({static_cast<int64_t>(c.bytes)}, options.dtype(at::kByte));
    auto tiling = at::empty({c.capacity, front::kTilingBytes}, options.dtype(at::kByte));
    auto cpu = at::zeros(tiling.sizes(), at::TensorOptions().dtype(at::kByte));
    TORCH_CHECK(indexer_front_build_tiling(c.capacity, cpu.data_ptr()) == 0,
                "indexer front cube tiling failed");
    tiling.copy_(cpu);
    const auto encoded = options.dtype(quantized ? at::kChar : at::kBFloat16);
    const auto weights = options.dtype(quantized ? at::kHalf : at::kBFloat16);
    std::vector<at::Tensor> output{
        at::empty({rows, front::kHeads, front::kDim}, encoded),
        at::empty({rows, front::kDim}, encoded),
        at::empty({rows, front::kHeads}, weights),
        at::empty({quantized ? rows : 0, front::kHeads}, options.dtype(at::kHalf)),
        at::empty({quantized ? rows : 0}, options.dtype(at::kHalf))};
    return {scratch, tiling, output, blocks, static_cast<int64_t>(launch.sync_base)};
}

std::vector<at::Tensor> Execute(
    const at::Tensor& hidden, const at::Tensor& query_norm,
    const at::Tensor& query_norm_scale, const at::Tensor& query_weight,
    const at::Tensor& query_weight_scale, const at::Tensor& key_weight,
    const at::Tensor& gamma, const at::Tensor& beta, const at::Tensor& cosine,
    const at::Tensor& sine, const at::Tensor& hadamard, at::Tensor& scratch,
    const at::Tensor& tiling, at::TensorList outputs, bool quantized,
    int64_t blocks, int64_t sync_base)
{
    const int64_t rows = CheckHidden(hidden);
    const c10::OptionalDeviceGuard device_guard(hidden.device());
    {
        std::lock_guard<std::mutex> guard(host::launch_mutex);
        const auto found = host::launches.find(hidden.get_device());
        TORCH_CHECK(found != host::launches.end() &&
                    found->second.sync_base == static_cast<uint64_t>(sync_base) &&
                    blocks > 0 && blocks <= found->second.maximum_blocks,
                    "prepare this device/context before indexer front capture");
    }
    const auto scale_dtype = query_weight_scale.scalar_type();
    TORCH_CHECK(scale_dtype == at::kFloat || scale_dtype == at::kBFloat16,
                "WQ channel scales require their original FP32 or BF16 dtype");
    const auto c = front::MakeConfig(rows, blocks, quantized,
                                      scale_dtype == at::kBFloat16);
    host::CheckShape(query_norm, at::kChar, {rows, front::kQueryRank}, "query_norm");
    host::CheckShape(query_norm_scale, at::kFloat, {rows}, "query_norm_scale");
    host::CheckShape(query_weight, at::kChar,
        {16, front::kQueryRank, front::kQueryTile}, "query_weight");
    host::CheckShape(query_weight_scale, scale_dtype,
                     {front::kQueryColumns}, "query_weight_scale");
    host::CheckShape(key_weight, at::kBFloat16,
        {5, front::kHidden, front::kKeyTile}, "key_weight");
    host::CheckShape(gamma, at::kFloat, {front::kDim}, "gamma");
    host::CheckShape(beta, at::kFloat, {front::kDim}, "beta");
    host::CheckShape(cosine, at::kBFloat16, {rows, front::kRope}, "cosine");
    host::CheckShape(sine, at::kBFloat16, {rows, front::kRope}, "sine");
    host::CheckShape(hadamard, at::kBFloat16,
                     {front::kDim, front::kDim}, "hadamard");
    host::CheckShape(scratch, at::kByte, {static_cast<int64_t>(c.bytes)}, "scratch");
    host::CheckShape(tiling, at::kByte, {c.capacity, front::kTilingBytes}, "tiling");
    TORCH_CHECK(outputs.size() == 5, "five indexer front outputs required");
    const auto encoded = quantized ? at::kChar : at::kBFloat16;
    host::CheckShape(outputs[0], encoded, {rows, front::kHeads, front::kDim}, "query");
    host::CheckShape(outputs[1], encoded, {rows, front::kDim}, "key");
    host::CheckShape(outputs[2], quantized ? at::kHalf : at::kBFloat16,
                     {rows, front::kHeads}, "weights");
    host::CheckShape(outputs[3], at::kHalf,
                     {quantized ? rows : 0, front::kHeads}, "query_scale");
    host::CheckShape(outputs[4], at::kHalf,
                     {quantized ? rows : 0}, "key_scale");
    const std::vector<at::Tensor> read{hidden, query_norm, query_norm_scale,
        query_weight, query_weight_scale, key_weight, gamma, beta, cosine,
        sine, hadamard, tiling};
    std::vector<at::Tensor> written{scratch};
    written.insert(written.end(), outputs.begin(), outputs.end());
    for (const auto& tensor : read) {
        TORCH_CHECK(tensor.device() == hidden.device(), "front input devices differ");
    }
    for (size_t i = 0; i < written.size(); ++i) {
        TORCH_CHECK(written[i].device() == hidden.device(), "front output devices differ");
        for (const auto& tensor : read) {
            at::assert_no_overlap(written[i], tensor);
        }
        for (size_t j = 0; j < i; ++j) {
            at::assert_no_overlap(written[i], written[j]);
        }
    }
    const front::Addresses a{hidden.data_ptr(), query_norm.data_ptr(),
        query_norm_scale.data_ptr(), query_weight.data_ptr(),
        query_weight_scale.data_ptr(), key_weight.data_ptr(), gamma.data_ptr(),
        beta.data_ptr(), cosine.data_ptr(), sine.data_ptr(), hadamard.data_ptr(),
        scratch.data_ptr(), tiling.data_ptr(), outputs[0].data_ptr(),
        outputs[1].data_ptr(), outputs[2].data_ptr(),
        quantized ? outputs[3].data_ptr() : nullptr,
        quantized ? outputs[4].data_ptr() : nullptr};
    indexer_front_launch(c10_npu::getCurrentNPUStream().stream(),
                          static_cast<uint64_t>(sync_base), c, a);
    return outputs.vec();
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(tilert, library)
{
    library.def("indexer_front_prepare(Tensor hidden, bool quantized, int blocks=0) "
                "-> (Tensor, Tensor, Tensor[], int, int)");
    library.def("indexer_front_out(Tensor hidden, Tensor query_norm, "
        "Tensor query_norm_scale, Tensor query_weight, Tensor query_weight_scale, "
        "Tensor key_weight, Tensor gamma, Tensor beta, Tensor cosine, Tensor sine, "
        "Tensor hadamard, Tensor(a!) scratch, Tensor tiling, Tensor(b!)[] outputs, "
        "bool quantized, int blocks, int sync_base) -> Tensor(b!)[]");
}

TORCH_LIBRARY_IMPL(tilert, PrivateUse1, library)
{
    library.impl("indexer_front_prepare", &Prepare);
    library.impl("indexer_front_out", &Execute);
}
