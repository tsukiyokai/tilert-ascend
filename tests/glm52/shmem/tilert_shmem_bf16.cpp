#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <mutex>

extern "C" {
int32_t rtGetC2cCtrlAddr(uint64_t* address, uint32_t* length);
void* aclshmem_malloc(size_t size);
void aclshmem_free(void* ptr);
}

void tp_push_demo(uint32_t block_dim, void* stream, uint64_t fftsAddr,
                  float* partial, float* out, uint8_t* window, int32_t* errors,
                  int rows, int hidden, int rounds, int round_base, int stage,
                  const int32_t* active_ptr = nullptr);

void launch_indexer_prep_qk(uint32_t block_dim, void* stream, void* kw,
                            void* q_in, void* cos_gm, void* sin_gm,
                            void* ln_weight, void* ln_bias, void* slot_mapping,
                            void* k_cache, void* qk_or_q_out, void* weights_out,
                            int T, int N, int64_t num_slots, float eps,
                            int is_c8, int slot_is_i64,
                            const int32_t* active_ptr = nullptr);

void launch_indexer_split_cast_scatter_c8(
    uint32_t block_dim, void* stream, void* qk_int8, void* qk_scale,
    void* slot_mapping, void* k_cache, void* k_scale_cache, void* q_int8_out,
    void* q_scale_out, int T, int N, int64_t num_slots, int slot_is_i64,
    const int32_t* active_ptr = nullptr);

void launch_indexer_prep_qk_c8(
    uint32_t block_dim, void* stream, void* kw, void* q_in, void* cos_gm,
    void* sin_gm, void* ln_weight, void* ln_bias, void* slot_mapping,
    void* k_cache, void* k_scale_cache, void* q_int8_out, void* q_scale_out,
    void* weights_out, int T, int N, int64_t num_slots, float eps,
    int slot_is_i64, const int32_t* active_ptr = nullptr);

void launch_tree_splice_topk(uint32_t block_dim, void* stream, void* topk_indices,
                             const void* prefix_len_ptr, const void* ancestors,
                             void* cache_buf, int T, int K, int W);

namespace {

constexpr int64_t kDefaultCapacity = 4LL * 1024 * 1024;
constexpr uint32_t kDefaultBlocks = 48;

struct ShmemState {
    std::mutex mutex;
    uint64_t ffts = 0;
    uint8_t* window = nullptr;
    int32_t* errors = nullptr;
    int64_t capacity = 0;
    int32_t round = 0;
    const int32_t* active_shmem_ptr = nullptr;
    const int32_t* active_idx_ptr = nullptr;

    void Ensure(int64_t numel) {
        std::lock_guard<std::mutex> lock(mutex);
        TORCH_CHECK(numel > 0, "onesided_all_reduce requires non-empty input");
        if (window != nullptr && numel <= capacity) {
            return;
        }
        if (window != nullptr) {
            aclshmem_free(window);
            window = nullptr;
        }
        if (errors == nullptr) {
            errors = static_cast<int32_t*>(aclshmem_malloc(64));
            TORCH_CHECK(errors != nullptr, "aclshmem_malloc(errors) failed");
        }
        capacity = std::max(numel, kDefaultCapacity);
        const size_t window_bytes =
            64 + static_cast<size_t>((capacity * 4 + 31) & ~31LL) * 32;
        window = static_cast<uint8_t*>(aclshmem_malloc(window_bytes));
        TORCH_CHECK(window != nullptr, "aclshmem_malloc(window) failed");
        if (ffts == 0) {
            uint32_t length = 0;
            TORCH_CHECK(rtGetC2cCtrlAddr(&ffts, &length) == 0 && ffts != 0,
                        "rtGetC2cCtrlAddr failed");
        }
    }
};

ShmemState& GetState() {
    static ShmemState state;
    return state;
}

at::Tensor OneSidedAllReduceOut(const at::Tensor& partial, at::Tensor& out) {
    TORCH_CHECK(partial.device().type() == c10::DeviceType::PrivateUse1 &&
                    out.device().type() == c10::DeviceType::PrivateUse1,
                "onesided_all_reduce_out requires NPU tensors");
    TORCH_CHECK(partial.is_contiguous() && out.is_contiguous(),
                "onesided_all_reduce_out requires contiguous tensors");
    const auto dtype = partial.scalar_type();
    TORCH_CHECK((dtype == at::kBFloat16 || dtype == at::kFloat) &&
                    out.scalar_type() == dtype &&
                    out.numel() == partial.numel(),
                "onesided_all_reduce_out requires matching BFloat16 or Float32 tensors");
    const c10::OptionalDeviceGuard device_guard(partial.device());
    const int64_t numel = partial.numel();
    auto& state = GetState();
    state.Ensure(numel);
    void* stream = c10_npu::getCurrentNPUStream().stream(true);
    const int stage = (dtype == at::kBFloat16) ? 10 : 0;
    const int round = state.round++;
    const int32_t* act_ptr =
        (numel == 196608) ? state.active_shmem_ptr : nullptr;
    tp_push_demo(kDefaultBlocks, stream, state.ffts,
                 reinterpret_cast<float*>(partial.data_ptr()),
                 reinterpret_cast<float*>(out.data_ptr()), state.window,
                 state.errors, 1, static_cast<int>(numel), 1, round, stage,
                 act_ptr);
    return out;
}

at::Tensor OneSidedAllReduceOutMeta(const at::Tensor& partial,
                                    at::Tensor& out) {
    return out;
}

at::Tensor OneSidedAllReduce(const at::Tensor& partial) {
    auto out = at::empty_like(partial);
    return OneSidedAllReduceOut(partial, out);
}

at::Tensor OneSidedAllReduceMeta(const at::Tensor& partial) {
    return at::empty_like(partial);
}

at::Tensor OneSidedAllReduceScaleAdd(const at::Tensor& partial,
                                     const at::Tensor& routed,
                                     double scale) {
    TORCH_CHECK(partial.device().type() == c10::DeviceType::PrivateUse1,
                "onesided_all_reduce_scale_add requires an NPU tensor");
    TORCH_CHECK(partial.is_contiguous() && routed.is_contiguous(),
                "onesided_all_reduce_scale_add requires contiguous tensors");
    TORCH_CHECK(partial.scalar_type() == at::kBFloat16 &&
                    routed.scalar_type() == at::kBFloat16,
                "onesided_all_reduce_scale_add requires BFloat16 tensors");
    const int64_t numel = partial.numel();
    const int64_t routed_numel = routed.numel();
    TORCH_CHECK(routed_numel == numel ||
                    (routed_numel > 0 && numel % routed_numel == 0),
                "onesided_all_reduce_scale_add requires matching or slice numel");
    const c10::OptionalDeviceGuard device_guard(partial.device());
    auto& state = GetState();
    state.Ensure(numel);
    auto out = at::empty_like(partial);
    void* stream = c10_npu::getCurrentNPUStream().stream(true);
    const int round = state.round++;
    union {
        float f;
        int32_t i;
    } scale_bits = {static_cast<float>(scale)};
    const int stage = (routed_numel == numel) ? 13 : 15;
    const int32_t* act_ptr =
        (numel == 196608) ? state.active_shmem_ptr : nullptr;
    tp_push_demo(kDefaultBlocks, stream, state.ffts,
                 reinterpret_cast<float*>(partial.data_ptr()),
                 reinterpret_cast<float*>(out.data_ptr()), state.window,
                 reinterpret_cast<int32_t*>(routed.data_ptr()), 1,
                 static_cast<int>(numel), scale_bits.i, round, stage, act_ptr);
    return out;
}

at::Tensor OneSidedAllReduceScaleAddMeta(const at::Tensor& partial,
                                         const at::Tensor& routed,
                                         double scale) {
    return at::empty_like(partial);
}

at::Tensor OneSidedTopTokens(const at::Tensor& local_max,
                             const at::Tensor& local_idx,
                             int64_t vocab_offset) {
    TORCH_CHECK(local_max.device().type() == c10::DeviceType::PrivateUse1 &&
                    local_idx.device().type() == c10::DeviceType::PrivateUse1,
                "onesided_top_tokens requires NPU tensors");
    TORCH_CHECK(local_max.is_contiguous() && local_idx.is_contiguous(),
                "onesided_top_tokens requires contiguous tensors");
    TORCH_CHECK(local_max.scalar_type() == at::kFloat &&
                    local_idx.scalar_type() == at::kLong,
                "onesided_top_tokens requires (float32, int64) tensors");
    const int64_t rows = local_max.numel();
    TORCH_CHECK(rows > 0 && rows <= 64 && local_idx.numel() == rows,
                "onesided_top_tokens supports 1..64 tokens");
    const c10::OptionalDeviceGuard device_guard(local_max.device());
    auto& state = GetState();
    state.Ensure(196608);
    auto out = at::empty_like(local_idx);
    void* stream = c10_npu::getCurrentNPUStream().stream(true);
    const int round = state.round++;
    const int32_t* act_ptr = (rows == 32) ? state.active_shmem_ptr : nullptr;
    tp_push_demo(kDefaultBlocks, stream, state.ffts,
                 reinterpret_cast<float*>(local_max.data_ptr()),
                 reinterpret_cast<float*>(out.data_ptr()), state.window,
                 reinterpret_cast<int32_t*>(local_idx.data_ptr()),
                 static_cast<int>(rows), static_cast<int>(vocab_offset), 1,
                 round, 14, act_ptr);
    return out;
}

at::Tensor OneSidedTopTokensMeta(const at::Tensor& local_max,
                                 const at::Tensor& local_idx,
                                 int64_t vocab_offset) {
    return at::empty_like(local_idx);
}

std::tuple<at::Tensor, at::Tensor> IndexerPrepQk(
    const at::Tensor& kw, const at::Tensor& q_in, const at::Tensor& cos_t,
    const at::Tensor& sin_t, const at::Tensor& ln_weight,
    const at::Tensor& ln_bias, const at::Tensor& slot_mapping,
    const at::Tensor& k_cache, double eps, bool is_c8) {
    const c10::OptionalDeviceGuard device_guard(kw.device());
    const int64_t T = kw.size(0);
    const int64_t N = q_in.numel() / (T * 128);
    const int64_t W = kw.size(1) - 128;
    const int64_t num_slots = k_cache.numel() / 128;

    at::Tensor qk_or_q_out = is_c8
        ? at::empty({T * (1 + N), 128}, kw.options())
        : at::empty({T, N, 128}, kw.options());
    at::Tensor weights_out = at::empty(
        {T, W}, is_c8 ? kw.options().dtype(at::kHalf) : kw.options());

    void* stream = c10_npu::getCurrentNPUStream().stream(true);
    const int slot_is_i64 = (slot_mapping.scalar_type() == at::kLong) ? 1 : 0;
    const int32_t* act_ptr = (T == 32) ? GetState().active_idx_ptr : nullptr;
    launch_indexer_prep_qk(
        kDefaultBlocks, stream, kw.data_ptr(), q_in.data_ptr(),
        cos_t.data_ptr(), sin_t.data_ptr(), ln_weight.data_ptr(),
        ln_bias.data_ptr(), slot_mapping.data_ptr(), k_cache.data_ptr(),
        qk_or_q_out.data_ptr(), weights_out.data_ptr(), static_cast<int>(T),
        static_cast<int>(N), num_slots, static_cast<float>(eps), is_c8 ? 1 : 0,
        slot_is_i64, act_ptr);
    return {qk_or_q_out, weights_out};
}

std::tuple<at::Tensor, at::Tensor> IndexerPrepQkMeta(
    const at::Tensor& kw, const at::Tensor& q_in, const at::Tensor& cos_t,
    const at::Tensor& sin_t, const at::Tensor& ln_weight,
    const at::Tensor& ln_bias, const at::Tensor& slot_mapping,
    const at::Tensor& k_cache, double eps, bool is_c8) {
    const int64_t T = kw.size(0);
    const int64_t N = q_in.numel() / (T * 128);
    const int64_t W = kw.size(1) - 128;
    at::Tensor qk_or_q_out = is_c8
        ? at::empty({T * (1 + N), 128}, kw.options())
        : at::empty({T, N, 128}, kw.options());
    at::Tensor weights_out = at::empty(
        {T, W}, is_c8 ? kw.options().dtype(at::kHalf) : kw.options());
    return {qk_or_q_out, weights_out};
}

std::tuple<at::Tensor, at::Tensor> IndexerSplitCastScatterC8(
    const at::Tensor& qk_int8, const at::Tensor& qk_scale,
    const at::Tensor& slot_mapping, const at::Tensor& k_cache,
    const at::Tensor& k_scale_cache, int64_t T, int64_t N) {
    const c10::OptionalDeviceGuard device_guard(qk_int8.device());
    const int64_t num_slots = k_cache.numel() / 128;
    at::Tensor q_int8_out = at::empty({T, N, 128}, k_cache.options());
    at::Tensor q_scale_out = at::empty({T, N}, k_scale_cache.options());

    void* stream = c10_npu::getCurrentNPUStream().stream(true);
    const int slot_is_i64 = (slot_mapping.scalar_type() == at::kLong) ? 1 : 0;
    void* scale_ptr =
        (qk_int8.scalar_type() == at::kBFloat16)
            ? ((qk_int8.data_ptr() == qk_scale.data_ptr())
                   ? nullptr
                   : reinterpret_cast<void*>(1ULL))
            : qk_scale.data_ptr();
    const int32_t* act_ptr = (T == 32) ? GetState().active_idx_ptr : nullptr;
    launch_indexer_split_cast_scatter_c8(
        kDefaultBlocks, stream, qk_int8.data_ptr(), scale_ptr,
        slot_mapping.data_ptr(), k_cache.data_ptr(), k_scale_cache.data_ptr(),
        q_int8_out.data_ptr(), q_scale_out.data_ptr(), static_cast<int>(T),
        static_cast<int>(N), num_slots, slot_is_i64, act_ptr);
    return {q_int8_out, q_scale_out};
}

std::tuple<at::Tensor, at::Tensor> IndexerSplitCastScatterC8Meta(
    const at::Tensor& qk_int8, const at::Tensor& qk_scale,
    const at::Tensor& slot_mapping, const at::Tensor& k_cache,
    const at::Tensor& k_scale_cache, int64_t T, int64_t N) {
    at::Tensor q_int8_out = at::empty({T, N, 128}, k_cache.options());
    at::Tensor q_scale_out = at::empty({T, N}, k_scale_cache.options());
    return {q_int8_out, q_scale_out};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> IndexerPrepQkC8(
    const at::Tensor& kw, const at::Tensor& q_in, const at::Tensor& cos_t,
    const at::Tensor& sin_t, const at::Tensor& ln_weight,
    const at::Tensor& ln_bias, const at::Tensor& slot_mapping,
    const at::Tensor& k_cache, const at::Tensor& k_scale_cache, double eps) {
    const c10::OptionalDeviceGuard device_guard(kw.device());
    const int64_t T = kw.size(0);
    const int64_t N = q_in.numel() / (T * 128);
    const int64_t W = kw.size(1) - 128;
    if (T != 32 || (N != 32 && N != 64) || W != 32) {
        auto [qk_or_q, weights] = IndexerPrepQk(
            kw, q_in, cos_t, sin_t, ln_weight, ln_bias, slot_mapping, k_cache,
            eps, true);
        auto [q_int8_out, q_scale_out] = IndexerSplitCastScatterC8(
            qk_or_q, kw, slot_mapping, k_cache, k_scale_cache, T, N);
        return {q_int8_out, q_scale_out, weights};
    }
    const int64_t num_slots = k_cache.numel() / 128;
    at::Tensor q_int8_out = at::empty({T, N, 128}, k_cache.options());
    at::Tensor q_scale_out = at::empty({T, N}, k_scale_cache.options());
    at::Tensor weights_out = at::empty({T, W}, kw.options().dtype(at::kHalf));

    void* stream = c10_npu::getCurrentNPUStream().stream(true);
    const int slot_is_i64 = (slot_mapping.scalar_type() == at::kLong) ? 1 : 0;
    const int32_t* act_ptr = GetState().active_idx_ptr;
    launch_indexer_prep_qk_c8(
        kDefaultBlocks, stream, kw.data_ptr(), q_in.data_ptr(),
        cos_t.data_ptr(), sin_t.data_ptr(), ln_weight.data_ptr(),
        ln_bias.data_ptr(), slot_mapping.data_ptr(), k_cache.data_ptr(),
        k_scale_cache.data_ptr(), q_int8_out.data_ptr(), q_scale_out.data_ptr(),
        weights_out.data_ptr(), static_cast<int>(T), static_cast<int>(N),
        num_slots, static_cast<float>(eps), slot_is_i64, act_ptr);
    return {q_int8_out, q_scale_out, weights_out};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> IndexerPrepQkC8Meta(
    const at::Tensor& kw, const at::Tensor& q_in, const at::Tensor& cos_t,
    const at::Tensor& sin_t, const at::Tensor& ln_weight,
    const at::Tensor& ln_bias, const at::Tensor& slot_mapping,
    const at::Tensor& k_cache, const at::Tensor& k_scale_cache, double eps) {
    const int64_t T = kw.size(0);
    const int64_t N = q_in.numel() / (T * 128);
    const int64_t W = kw.size(1) - 128;
    at::Tensor q_int8_out = at::empty({T, N, 128}, k_cache.options());
    at::Tensor q_scale_out = at::empty({T, N}, k_scale_cache.options());
    at::Tensor weights_out = at::empty({T, W}, kw.options().dtype(at::kHalf));
    return {q_int8_out, q_scale_out, weights_out};
}

void SetActiveTokensPtr(const at::Tensor& active_tensor, bool enable_shmem,
                        bool enable_idx) {
    auto& state = GetState();
    const int32_t* ptr = active_tensor.defined()
        ? active_tensor.data_ptr<int32_t>()
        : nullptr;
    state.active_shmem_ptr = enable_shmem ? ptr : nullptr;
    state.active_idx_ptr = enable_idx ? ptr : nullptr;
}

void SetActiveTokensPtrMeta(const at::Tensor& active_tensor, bool enable_shmem,
                            bool enable_idx) {}

at::Tensor& TreeSpliceTopk(at::Tensor& topk_indices,
                           const at::Tensor& prefix_len,
                           const at::Tensor& ancestors,
                           const c10::optional<at::Tensor>& cache_buf) {
    const c10::OptionalDeviceGuard device_guard(topk_indices.device());
    const int T = static_cast<int>(topk_indices.size(0));
    const int K = static_cast<int>(topk_indices.size(-1));
    const int W = static_cast<int>(ancestors.size(-1));
    void* stream = c10_npu::getCurrentNPUStream().stream(true);
    void* c_ptr = (cache_buf.has_value() && cache_buf->defined() &&
                   cache_buf->is_contiguous() && cache_buf->size(-1) == K)
                      ? cache_buf->data_ptr()
                      : nullptr;
    launch_tree_splice_topk(
        static_cast<uint32_t>(T), stream, topk_indices.data_ptr(),
        prefix_len.data_ptr(), ancestors.data_ptr(), c_ptr, T, K, W);
    return topk_indices;
}

at::Tensor& TreeSpliceTopkMeta(at::Tensor& topk_indices,
                               const at::Tensor& prefix_len,
                               const at::Tensor& ancestors,
                               const c10::optional<at::Tensor>& cache_buf) {
    return topk_indices;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(tilert, m) {
    m.def("onesided_all_reduce(Tensor partial) -> Tensor");
    m.def("onesided_all_reduce_out(Tensor partial, Tensor(a!) out) -> Tensor(a!)");
    m.def(
        "onesided_all_reduce_scale_add(Tensor partial, Tensor routed, float scale) -> Tensor");
    m.def(
        "onesided_top_tokens(Tensor local_max, Tensor local_idx, int vocab_offset) -> Tensor");
    m.def(
        "indexer_prep_qk(Tensor kw, Tensor q_in, Tensor cos_t, Tensor sin_t, Tensor ln_weight, Tensor ln_bias, Tensor slot_mapping, Tensor k_cache, float eps, bool is_c8) -> (Tensor, Tensor)");
    m.def(
        "indexer_split_cast_scatter_c8(Tensor qk_int8, Tensor qk_scale, Tensor slot_mapping, Tensor k_cache, Tensor k_scale_cache, int T, int N) -> (Tensor, Tensor)");
    m.def(
        "indexer_prep_qk_c8(Tensor kw, Tensor q_in, Tensor cos_t, Tensor sin_t, Tensor ln_weight, Tensor ln_bias, Tensor slot_mapping, Tensor k_cache, Tensor k_scale_cache, float eps) -> (Tensor, Tensor, Tensor)");
    m.def(
        "set_active_tokens_ptr(Tensor active_tensor, bool enable_shmem, bool enable_idx) -> ()");
    m.def(
        "tree_splice_topk(Tensor(a!) topk_indices, Tensor prefix_len, Tensor ancestors, Tensor? cache_buf=None) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(tilert, PrivateUse1, m) {
    m.impl("onesided_all_reduce", &OneSidedAllReduce);
    m.impl("onesided_all_reduce_out", &OneSidedAllReduceOut);
    m.impl("onesided_all_reduce_scale_add", &OneSidedAllReduceScaleAdd);
    m.impl("onesided_top_tokens", &OneSidedTopTokens);
    m.impl("indexer_prep_qk", &IndexerPrepQk);
    m.impl("indexer_split_cast_scatter_c8", &IndexerSplitCastScatterC8);
    m.impl("indexer_prep_qk_c8", &IndexerPrepQkC8);
    m.impl("set_active_tokens_ptr", &SetActiveTokensPtr);
    m.impl("tree_splice_topk", &TreeSpliceTopk);
}

TORCH_LIBRARY_IMPL(tilert, Meta, m) {
    m.impl("onesided_all_reduce", &OneSidedAllReduceMeta);
    m.impl("onesided_all_reduce_out", &OneSidedAllReduceOutMeta);
    m.impl("onesided_all_reduce_scale_add", &OneSidedAllReduceScaleAddMeta);
    m.impl("onesided_top_tokens", &OneSidedTopTokensMeta);
    m.impl("indexer_prep_qk", &IndexerPrepQkMeta);
    m.impl("indexer_split_cast_scatter_c8", &IndexerSplitCastScatterC8Meta);
    m.impl("indexer_prep_qk_c8", &IndexerPrepQkC8Meta);
    m.impl("set_active_tokens_ptr", &SetActiveTokensPtrMeta);
    m.impl("tree_splice_topk", &TreeSpliceTopkMeta);
}
