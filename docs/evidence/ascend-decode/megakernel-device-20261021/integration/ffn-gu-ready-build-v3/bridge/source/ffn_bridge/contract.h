#ifndef TILERT_FFN_BRIDGE_CONTRACT_H
#define TILERT_FFN_BRIDGE_CONTRACT_H

#include <cstdint>

namespace ffn_bridge {

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define BRIDGE_DEVICE __aicore__
#else
#define BRIDGE_DEVICE
#endif

constexpr int64_t kHidden = 6144;
constexpr int64_t kExperts = 256;
constexpr int64_t kTopK = 8;
constexpr int64_t kMaximumRows = 32;
constexpr int64_t kGateTile = 16;
constexpr int64_t kTilingBytes = 512;
constexpr int64_t kSystemBytes = 16 * 1024 * 1024;
constexpr int64_t kTopKUbBytes = 16416;

enum Buffer : int64_t {
    kNormalized,
    kRetained,
    kRstd,
    kLogits,
    kProbabilities,
    kIds,
    kSharedCodes,
    kSharedScales,
    kTiling,
    kSystem,
    kCombined,
    kScaledRouted,
    kBuffers,
};

constexpr bool ValidGeometry(int64_t rows, int64_t blocks)
{
    return rows >= 1 && rows <= kMaximumRows && (blocks == 16 || blocks == 24);
}

BRIDGE_DEVICE constexpr int64_t Capacity(int64_t rows)
{
    return rows <= 16 ? 16 : 32;
}

constexpr int64_t RowOwner(int64_t row, int64_t blocks)
{
    return row % (2 * blocks);
}

constexpr int64_t ScratchBytes(int64_t rows)
{
    // Four BF16 [M,H] outputs, shared A8, logits, two row scalars,
    // FP32 probabilities and INT32 indices, plus tiling and system workspace.
    return rows * (9 * kHidden + 4 * kExperts + 8 + 8 * kTopK)
        + kTilingBytes + kSystemBytes;
}

static_assert(kExperts % kGateTile == 0);
static_assert(kTopK * sizeof(int32_t) == 32);
static_assert(kTopKUbBytes < 192 * 1024);

#undef BRIDGE_DEVICE

}  // namespace ffn_bridge

extern "C" int ffn_gu_ready_bridge_build_tiling(int capacity, void* buffer);
extern "C" void ffn_bridge_launch(void* stream, uint64_t sync_base,
    void* input, void* residual, void* gamma, void* gate, void* bias,
    void** scratch, int64_t rows, int64_t blocks, float epsilon);
extern "C" void ffn_bridge_topk_launch(void* stream, uint64_t sync_base,
    void* logits, void* bias, void** scratch, int64_t rows, int64_t blocks);
extern "C" void ffn_bridge_epilogue_launch(void* stream, uint64_t sync_base,
    void* routed, void* shared, void** scratch, int64_t rows, int64_t blocks);

#endif
