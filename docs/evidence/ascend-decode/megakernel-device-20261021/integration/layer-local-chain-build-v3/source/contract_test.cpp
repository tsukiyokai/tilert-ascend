#include "contract.h"

#include <cassert>
#include <iostream>

using namespace layer_local_chain;

int main()
{
    Addresses addresses{};
    addresses.norm = {0x1000, 0x2000, 0x3000, 0x4000,
                      0x5000, 0x6000, 0x7000, 0x8000};
    addresses.attention.hidden = addresses.norm.normalized;
    addresses.attention.token_x = addresses.norm.codes;
    addresses.attention.token_scale = addresses.norm.scales;
    addresses.attention.scratch = 0x9000;
    addresses.attention.tiling = 0xa000;
    addresses.attention.output = 0xb000;
    addresses.post_gamma = 0xc000;
    addresses.gate = 0xd000;
    addresses.correction_bias = 0xe000;
    for (int i = 0; i < ffn_bridge::kBuffers; ++i) {
        addresses.bridge[i] = 0x10000 + i * 4096;
    }
    for (int i = 0; i < ffn_original_shared_gu_static::kBodyBuffers; ++i) {
        addresses.body.scratch[i] = 0x20000 + i * 4096;
    }
    addresses.body.dispatch.bf16_rows = addresses.bridge[ffn_bridge::kNormalized];
    addresses.body.dispatch.expert_ids = addresses.bridge[ffn_bridge::kIds];
    addresses.body.dispatch.probabilities = addresses.bridge[ffn_bridge::kProbabilities];
    addresses.body.dispatch.active_list =
        addresses.body.scratch[routed_int4::pipeline::kActiveList];
    addresses.body.shared.input = addresses.bridge[ffn_bridge::kSharedCodes];
    addresses.body.shared.input_scale = addresses.bridge[ffn_bridge::kSharedScales];
    int geometries = 0;
    for (int rows : {1, 2, 12, 32}) {
        for (int blocks : {16, 24}) {
            Config config{};
            config.indexed = whole_layer_indexed::MakeConfig(
                whole_layer_attention::MakeConfig(rows, 66, 64, blocks,
                    0.0441941738F, 1e-5F, 1e-5F), 66, 64, true);
            config.shared = whole_layer_attention::MakeFullConfig(
                config.indexed.attention);
            addresses.body.rows = rows;
            addresses.body.blocks = blocks;
            addresses.body.shared.capacity = ffn_bridge::Capacity(rows);
            assert(Valid(config, addresses));
            auto wrong = addresses;
            wrong.body.dispatch.probabilities += 4096;
            assert(!Valid(config, wrong));
            wrong = addresses;
            wrong.norm.residual = 0;
            assert(!Valid(config, wrong));
            wrong = addresses;
            wrong.norm.normalized += 4096;
            assert(!Valid(config, wrong));
            wrong = addresses;
            wrong.body.rows = rows + 1;
            assert(!Valid(config, wrong));
            ++geometries;
        }
    }
    std::cout << "{\"geometries\":" << geometries
              << ",\"config_bytes\":" << sizeof(Config)
              << ",\"address_bytes\":" << sizeof(Addresses)
              << ",\"kernel_argument_bytes\":"
              << sizeof(Config) + sizeof(Addresses) + sizeof(uint64_t)
              << ",\"pass\":true}\n";
}
