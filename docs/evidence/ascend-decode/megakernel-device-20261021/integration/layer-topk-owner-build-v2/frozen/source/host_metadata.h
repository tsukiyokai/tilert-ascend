#ifndef TILERT_LAYER_TOPK_OWNER_HOST_METADATA_H
#define TILERT_LAYER_TOPK_OWNER_HOST_METADATA_H

#include "contract.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace layer_topk_owner {

// Geometry: M, main pages/table columns, index pages/table columns, groups,
// index-cache INT8 flag, WQ-scale BF16 flag, output-bias kind.
inline bool MakeHostConfig(const int64_t* geometry, const float* scalars,
                           Config& config)
{
    if (geometry == nullptr || scalars == nullptr ||
        !ffn_bridge::ValidGeometry(geometry[0], geometry[5]) ||
        (geometry[6] != 0 && geometry[6] != 1) ||
        (geometry[7] != 0 && geometry[7] != 1) ||
        geometry[8] < 0 || geometry[8] > 2) {
        return false;
    }
    constexpr int64_t page_limit = std::numeric_limits<int32_t>::max() / 128;
    for (int i = 1; i <= 4; ++i) {
        if (geometry[i] < 1 || geometry[i] > page_limit) {
            return false;
        }
    }
    // Scalars: attention scale, CQ/CKV epsilon, input/FFN epsilon.
    for (int i = 0; i < 5; ++i) {
        if (!(std::isfinite(scalars[i]) && scalars[i] > 0)) {
            return false;
        }
    }
    config = {};
    const auto main = whole_layer_attention::MakeConfig(
        geometry[0], geometry[1], geometry[2], geometry[5],
        scalars[0], scalars[1], scalars[2]);
    config.indexed = whole_layer_indexed::MakeConfig(
        main, geometry[3], geometry[4], geometry[6], geometry[7]);
    config.shared = whole_layer_attention::MakeFullConfig(main);
    config.input_epsilon = scalars[3];
    config.ffn_epsilon = scalars[4];
    config.bias_kind = geometry[8];
    return true;
}

inline bool MakeHostAddresses(const Config& config,
    const uint64_t* norm, const uint64_t* attention, uint64_t shared_topk,
    const uint64_t* post, const uint64_t* routed_weights,
    const uint64_t* bridge, const uint64_t* body,
    const uint64_t* shared_weights, Addresses& addresses)
{
    if (!(norm && attention && post && routed_weights && bridge && body &&
          shared_weights)) {
        return false;
    }
    addresses = {};
    static_assert(sizeof(NormAddresses) == 8 * sizeof(uint64_t));
    static_assert(sizeof(AttentionAddresses) == 41 * sizeof(uint64_t));
    std::memcpy(&addresses.norm, norm, sizeof(addresses.norm));
    std::memcpy(&addresses.attention, attention, sizeof(addresses.attention));
    addresses.shared_topk = shared_topk;
    addresses.post_gamma = post[0];
    addresses.gate = post[1];
    addresses.correction_bias = post[2];
    std::copy_n(bridge, ffn_bridge::kBuffers, addresses.bridge);
    void* routed[ffn_original_shared_gu_static::kRoutedWeights];
    void* bridge_pointers[ffn_bridge::kBuffers];
    void* body_pointers[ffn_original_shared_gu_static::kBodyBuffers];
    for (int i = 0; i < ffn_original_shared_gu_static::kRoutedWeights; ++i) {
        routed[i] = reinterpret_cast<void*>(routed_weights[i]);
        if (routed_weights[i] == 0) {
            return false;
        }
    }
    for (int i = 0; i < ffn_bridge::kBuffers; ++i) {
        bridge_pointers[i] = reinterpret_cast<void*>(bridge[i]);
    }
    for (int i = 0; i < ffn_original_shared_gu_static::kBodyBuffers; ++i) {
        body_pointers[i] = reinterpret_cast<void*>(body[i]);
    }
    for (int i = 0; i < 4; ++i) {
        if (shared_weights[i] == 0) {
            return false;
        }
    }
    const auto rows = config.indexed.attention.rows;
    const int64_t capacity = routed_int4::pipeline::Capacity(rows);
    const auto layout = shared_w8::MakeLayout(capacity);
    const routed_int4::pipeline::Input shared{
        bridge[ffn_bridge::kSharedCodes], bridge[ffn_bridge::kSharedScales],
        shared_weights[0], shared_weights[1], shared_weights[2], shared_weights[3],
        body[routed_int4::pipeline::kSharedPayload],
        body[routed_int4::pipeline::kSharedTiling],
        body[routed_int4::pipeline::kSharedOutput],
        capacity, layout.partial, layout.middle, layout.scales, layout.down};
    addresses.body = ffn_original_shared_gu_static::MakeBodyArguments(
        routed, bridge_pointers, body_pointers, shared, rows,
        config.indexed.attention.blocks);
    return Valid(config, addresses);
}

}  // namespace layer_topk_owner

#endif
