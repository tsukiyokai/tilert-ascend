#ifndef TILERT_LAYER_TP_CHAIN_CONTRACT_H
#define TILERT_LAYER_TP_CHAIN_CONTRACT_H

#include "../local/source/contract.h"
#include "../communication/communication.h"

namespace layer_tp_chain {

using Config = layer_local_chain::Config;
using Addresses = layer_local_chain::Addresses;
using AttentionAddresses = layer_local_chain::AttentionAddresses;

struct Communication {
    uint64_t window;
    uint64_t window_bytes;
    uint64_t attention_sum;
    uint64_t output;
    int64_t capacity;
    int32_t rank;
    int32_t ranks;
};

inline bool ValidCommunication(const Config& config, const Addresses& addresses,
                               const Communication& communication)
{
    const auto rows = config.indexed.attention.rows;
    if (communication.ranks < 2 || communication.ranks > 16 ||
        communication.rank < 0 || communication.rank >= communication.ranks ||
        communication.capacity < rows || communication.capacity > 32 ||
        communication.window == 0 || communication.window % 64 != 0 ||
        communication.attention_sum == 0 || communication.output == 0 ||
        communication.attention_sum == communication.output ||
        communication.attention_sum == addresses.attention.output ||
        communication.output == addresses.bridge[ffn_bridge::kCombined]) {
        return false;
    }
    return communication.window_bytes == whole_layer::MakeCommunicationLayout(
        communication.capacity, communication.ranks).bytes;
}

inline bool Valid(const Config& config, const Addresses& addresses,
                  const Communication& communication)
{
    return layer_local_chain::Valid(config, addresses) &&
           ValidCommunication(config, addresses, communication);
}

static_assert(std::is_standard_layout_v<Communication>);
static_assert(std::is_trivially_copyable_v<Communication>);
static_assert(sizeof(Communication) == 48);
static_assert(sizeof(Config) + sizeof(Addresses) + sizeof(Communication) +
              sizeof(uint64_t) == 1648);

}  // namespace layer_tp_chain

extern "C" int layer_tp_indexed_launch(void* stream, uint64_t sync,
    const layer_tp_chain::Config* config,
    const layer_tp_chain::Addresses* addresses,
    const layer_tp_chain::Communication* communication);
extern "C" int layer_tp_shared_launch(void* stream, uint64_t sync,
    const layer_tp_chain::Config* config,
    const layer_tp_chain::Addresses* addresses,
    const layer_tp_chain::Communication* communication);

#endif
