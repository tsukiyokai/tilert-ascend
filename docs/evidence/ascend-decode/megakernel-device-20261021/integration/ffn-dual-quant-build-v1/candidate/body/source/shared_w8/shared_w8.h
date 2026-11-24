#ifndef TESTS_GLM52_SHARED_W8_H
#define TESTS_GLM52_SHARED_W8_H

#include <cstdint>

#ifndef SHARED_W8_DYNAMIC_PERFORMANCE_QUANT
#define SHARED_W8_DYNAMIC_PERFORMANCE_QUANT 0
#endif

namespace shared_w8 {
constexpr bool kDynamicPerformanceQuant = SHARED_W8_DYNAMIC_PERFORMANCE_QUANT != 0;
constexpr int64_t kHidden = 6144;
constexpr int64_t kIntermediate = 128;
constexpr int64_t kGateColumns = 2 * kIntermediate;
constexpr int64_t kInnerTile = 1024;
constexpr int64_t kGateTile = 64;
constexpr int64_t kInnerParts = kHidden / kInnerTile;
constexpr int64_t kGateParts = kGateColumns / kGateTile;
constexpr int64_t kDownTile = 256;
constexpr int64_t kDownParts = kHidden / kDownTile;
constexpr int64_t kScaleStride = 8;
constexpr uint64_t kTilingStride = 512;
constexpr uint64_t kSystemBytes = 16ULL << 20;

constexpr uint64_t Align(uint64_t bytes)
{
    return (bytes + 511) / 512 * 512;
}

struct Layout {
    uint64_t staged;
    uint64_t partial;
    uint64_t middle;
    uint64_t scales;
    uint64_t down;
    uint64_t system;
    uint64_t bytes;
};

constexpr Layout MakeLayout(int64_t capacity)
{
    Layout layout{};
    layout.partial = Align(capacity * kHidden);
    layout.middle = Align(layout.partial
        + kInnerParts * kGateColumns * capacity * sizeof(int32_t));
    layout.scales = Align(layout.middle + capacity * kIntermediate);
    layout.down = Align(layout.scales + capacity * kScaleStride * sizeof(float));
    layout.system = Align(layout.down + capacity * kHidden * sizeof(int32_t));
    layout.bytes = layout.system + kSystemBytes;
    return layout;
}
}  // namespace shared_w8

extern "C" int shared_w8_prepare(int capacity, int blocks, void* tiling,
                                  uint64_t* sync_base);
extern "C" uint64_t shared_w8_workspace_field(int capacity, int field);
extern "C" int shared_w8_dynamic_quant();
extern "C" void shared_w8_launch(
    uint32_t blocks, void* stream, uint64_t sync_base,
    void* input, void* input_scale, void* gate_weight, void* gate_scale,
    void* down_weight, void* down_scale, void* scratch, void* tiling,
    void* output, int64_t rows, int64_t capacity);

#endif
