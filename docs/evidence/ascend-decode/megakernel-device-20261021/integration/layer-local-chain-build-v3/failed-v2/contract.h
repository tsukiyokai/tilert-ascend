#ifndef TILERT_LAYER_LOCAL_CHAIN_CONTRACT_H
#define TILERT_LAYER_LOCAL_CHAIN_CONTRACT_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../attention/source/tests/glm52/whole_layer/attention_indexed_contract.h"
#include "../attention/source/tests/glm52/whole_layer/attention_full_contract.h"
#include "../ffn/source/descriptor.h"

namespace layer_local_chain {

struct NormAddresses {
    uint64_t input, residual, gamma, normalized, retained, rstd, codes, scales;
};

struct Config {
    whole_layer_indexed::Config indexed;
    whole_layer_attention::FullConfig shared;
    float input_epsilon;
    float ffn_epsilon;
    int32_t bias_kind;
    int32_t reserved;
};

struct Addresses {
    NormAddresses norm;
    whole_layer_indexed::Addresses attention;
    uint64_t shared_topk;
    uint64_t post_gamma, gate, correction_bias;
    uint64_t bridge[ffn_bridge::kBuffers];
    ffn_original_shared_gu_static::BodyArguments body;
};

static_assert(std::is_standard_layout_v<Config>);
static_assert(std::is_trivially_copyable_v<Config>);
static_assert(std::is_standard_layout_v<Addresses>);
static_assert(std::is_trivially_copyable_v<Addresses>);
static_assert(sizeof(NormAddresses) == 64);
static_assert(sizeof(Config) + sizeof(Addresses) + sizeof(uint64_t) < 4096);

// Inputs are validated tensors owned by the caller. Local attention output
// feeds the FFN directly; this interface performs neither TP reduction.
inline bool Valid(const Config& config, const Addresses& addresses)
{
    const auto& main = config.indexed.attention;
    const auto& norm = addresses.norm;
    if (!(norm.input && norm.residual && norm.gamma && norm.normalized &&
          norm.retained && norm.rstd && norm.codes && norm.scales &&
          addresses.post_gamma && addresses.gate && addresses.correction_bias &&
          addresses.attention.scratch && addresses.attention.tiling &&
          addresses.attention.output)) {
        return false;
    }
    for (auto address : addresses.bridge) {
        if (address == 0) {
            return false;
        }
    }
    for (auto address : addresses.body.scratch) {
        if (address == 0) {
            return false;
        }
    }
    return ffn_bridge::ValidGeometry(main.rows, main.blocks)
        && config.shared.attention.rows == main.rows
        && config.shared.attention.blocks == main.blocks
        && addresses.body.rows == main.rows
        && addresses.body.blocks == main.blocks
        && addresses.norm.normalized == reinterpret_cast<uint64_t>(
            addresses.attention.hidden)
        && addresses.norm.codes == reinterpret_cast<uint64_t>(
            addresses.attention.token_x)
        && addresses.norm.scales == reinterpret_cast<uint64_t>(
            addresses.attention.token_scale)
        && addresses.body.dispatch.bf16_rows ==
            addresses.bridge[ffn_bridge::kNormalized]
        && addresses.body.dispatch.expert_ids ==
            addresses.bridge[ffn_bridge::kIds]
        && addresses.body.dispatch.probabilities ==
            addresses.bridge[ffn_bridge::kProbabilities]
        && addresses.body.dispatch.active_list ==
            addresses.body.scratch[routed_int4::pipeline::kActiveList]
        && addresses.body.shared.input ==
            addresses.bridge[ffn_bridge::kSharedCodes]
        && addresses.body.shared.input_scale ==
            addresses.bridge[ffn_bridge::kSharedScales]
        && addresses.body.shared.capacity == ffn_bridge::Capacity(main.rows);
}

}  // namespace layer_local_chain

extern "C" int layer_local_indexed_launch(void* stream, uint64_t sync,
    const layer_local_chain::Config* config,
    const layer_local_chain::Addresses* addresses);
extern "C" int layer_local_shared_launch(void* stream, uint64_t sync,
    const layer_local_chain::Config* config,
    const layer_local_chain::Addresses* addresses);

#endif
