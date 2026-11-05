#include "contract.h"
#include "../communication/tree_plan.h"

#include <cassert>
#include <iostream>

int main()
{
    using namespace layer_tp_tree_chain;
    Addresses addresses{};
    addresses.attention.output = 0x1000;
    addresses.bridge[ffn_bridge::kCombined] = 0x2000;
    int cases = 0;
    int rejected = 0;
    for (int rows : {1, 2, 12, 32}) {
        for (int ranks = 2; ranks <= 16; ++ranks) {
            Config config{};
            config.indexed.attention.rows = rows;
            const auto layout = whole_layer_tree::MakeCommunicationLayout(32, ranks);
            Communication good{0x10000, layout.bytes, 0x3000, 0x4000,
                               32, ranks - 1, ranks};
            assert(ValidCommunication(config, addresses, good));
            for (int test = 0; test < 8; ++test) {
                auto bad = good;
                switch (test) {
                    case 0: bad.capacity = rows - 1; break;
                    case 1: --bad.window_bytes; break;
                    case 2: bad.rank = ranks; break;
                    case 3: bad.ranks = 17; break;
                    case 4: ++bad.window; break;
                    case 5: bad.attention_sum = addresses.attention.output; break;
                    case 6: bad.output = addresses.bridge[ffn_bridge::kCombined]; break;
                    case 7: bad.output = bad.attention_sum; break;
                }
                assert(!ValidCommunication(config, addresses, bad));
                ++rejected;
            }
            ++cases;
        }
    }
    static_assert(whole_layer_tree::kCommunicationChunk == 2048);
    static_assert(whole_layer_tree::kCommunicationUbBytes == 45056);
    static_assert(sizeof(Config) == 664 && sizeof(Addresses) == 928);
    static_assert(sizeof(Communication) == 48);
    std::cout << "{\"communication_geometries\":" << cases
              << ",\"negative_controls\":" << rejected
              << ",\"communication_ub_bytes\":45056,\"pass\":true}\n";
}
