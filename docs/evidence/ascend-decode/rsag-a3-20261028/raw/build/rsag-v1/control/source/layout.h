#ifndef TRT_RSAG_CONTROL_LAYOUT_H
#define TRT_RSAG_CONTROL_LAYOUT_H

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define RSAG_DEVICE __aicore__
#else
#define RSAG_DEVICE
#endif

namespace trt_rsag_control_contract {

constexpr int64_t kHidden = 6144;
constexpr int64_t kLineElements = 256;
constexpr uint64_t kPayloadAlignment = 512;
constexpr uint64_t kFlagAlignment = 64;
constexpr uint32_t kMaximumEpoch = UINT32_MAX - 1;

RSAG_DEVICE constexpr uint64_t Align512(uint64_t bytes)
{
    return (bytes + 511) / 512 * 512;
}

RSAG_DEVICE constexpr bool AlignedPayload(uintptr_t address)
{
    return address != 0 && address % kPayloadAlignment == 0;
}

RSAG_DEVICE constexpr bool ValidGeometry(int64_t rows, int32_t ranks)
{
    return rows >= 1 && rows <= 32 && ranks >= 2 && ranks <= 16;
}

struct Range {
    int64_t begin;
    int64_t count;
};

RSAG_DEVICE constexpr Range OwnerRange(int64_t rows, int32_t ranks, int32_t owner)
{
    if (!ValidGeometry(rows, ranks) || owner < 0 || owner >= ranks) {
        return {-1, 0};
    }
    const int64_t lines = rows * kHidden / kLineElements;
    const int64_t quotient = lines / ranks;
    const int64_t remainder = lines % ranks;
    return {(owner * quotient + (owner < remainder ? owner : remainder))
                * kLineElements,
            (quotient + (owner < remainder)) * kLineElements};
}

struct Layout {
    uint64_t epoch;
    uint64_t scatter_ready;
    uint64_t gather_ready;
    uint64_t scatter;
    uint64_t scatter_slot_bytes;
    uint64_t scatter_plane_bytes;
    uint64_t gather;
    uint64_t gather_plane_bytes;
    uint64_t bytes;
};

RSAG_DEVICE constexpr Layout MakeLayout(int64_t capacity, int32_t ranks)
{
    if (!ValidGeometry(capacity, ranks)) {
        return {};
    }
    Layout layout{};
    layout.scatter_ready = kFlagAlignment;
    layout.gather_ready = layout.scatter_ready + 2 * ranks * kFlagAlignment;
    layout.scatter = Align512(layout.gather_ready + 2 * ranks * kFlagAlignment);
    layout.scatter_slot_bytes = ((24 * capacity + ranks - 1) / ranks) * 512;
    layout.scatter_plane_bytes = ranks * layout.scatter_slot_bytes;
    layout.gather = layout.scatter + 2 * layout.scatter_plane_bytes;
    layout.gather_plane_bytes = capacity * kHidden * sizeof(uint16_t);
    layout.bytes = layout.gather + 2 * layout.gather_plane_bytes;
    return layout;
}

static_assert(OwnerRange(1, 16, 7).count == 512);
static_assert(OwnerRange(1, 16, 8).count == 256);
static_assert(MakeLayout(32, 16).bytes == 1577472);

}  // namespace trt_rsag_control_contract

#undef RSAG_DEVICE
#endif
