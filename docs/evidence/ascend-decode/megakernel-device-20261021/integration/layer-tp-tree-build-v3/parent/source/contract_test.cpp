#include "contract.h"

#include <cassert>
#include <iostream>

int main()
{
    using namespace layer_tp_chain;
    Addresses addresses{};
    addresses.attention.output = 0x1000;
    addresses.bridge[ffn_bridge::kCombined] = 0x2000;
    int cases = 0;
    for (int rows : {1, 2, 12, 32}) {
        for (int ranks : {2, 16}) {
            Config config{};
            config.indexed.attention.rows = rows;
            const auto bytes = whole_layer::MakeCommunicationLayout(32, ranks).bytes;
            Communication communication{0x10000, bytes, 0x3000, 0x4000,
                                        32, ranks - 1, ranks};
            assert(ValidCommunication(config, addresses, communication));
            auto wrong = communication;
            wrong.capacity = rows - 1;
            assert(!ValidCommunication(config, addresses, wrong));
            wrong = communication;
            --wrong.window_bytes;
            assert(!ValidCommunication(config, addresses, wrong));
            wrong = communication;
            wrong.rank = ranks;
            assert(!ValidCommunication(config, addresses, wrong));
            wrong = communication;
            wrong.attention_sum = addresses.attention.output;
            assert(!ValidCommunication(config, addresses, wrong));
            ++cases;
        }
    }
    std::cout << "{\"communication_geometries\":" << cases
              << ",\"kernel_argument_bytes\":1648,\"pass\":true}\n";
}
