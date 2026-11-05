#include "indexer_contract.h"

#include <cassert>
#include <cstddef>
#include <iostream>

int main()
{
    using namespace whole_layer_indexer;
    size_t cases = 0;
    for (bool quantized : {false, true}) {
        const uint64_t s1 = quantized ? 4 : 8;
        const uint64_t s2 = quantized ? 2048 : 512;
        const uint64_t matrix_rows = quantized ? s1 : 8 * kHeads;
        for (uint32_t rows = 1; rows <= 32; ++rows) {
            for (uint32_t blocks = 1; blocks <= 24; ++blocks) {
                const auto config = MakeConfig(rows, 73, 65, blocks, quantized);
                uint64_t cursor = 0;
                for (uint32_t core = 0; core < blocks; ++core) {
                    cursor += 2 * matrix_rows * s2 * sizeof(float);
                }
                assert(cursor == config.merge_offset);
                cursor += blocks * s1 * 4 * kTopK * sizeof(float);
                assert(cursor == config.params_offset);
                cursor += blocks * s1 * 2 * 16 * sizeof(int64_t);
                assert(cursor == config.weights_offset);
                if (quantized) {
                    cursor += blocks * 16 * (s1 * kHeads) * 2 * sizeof(uint16_t);
                }
                assert(cursor == config.bytes);
                assert(config.bytes % 512 == 0);
                assert(config.mm_bytes / blocks == (quantized ? 65536 : 1048576));
                assert(config.bytes / blocks == (quantized ? 205824 : 1312768));
                // Cache rows are assigned exactly once among all launched AIVs.
                for (uint32_t row = 0; row < rows; ++row) {
                    uint32_t owners = 0;
                    for (uint32_t aiv = 0; aiv < 2 * blocks; ++aiv) {
                        owners += row >= aiv && (row - aiv) % (2 * blocks) == 0;
                    }
                    assert(owners == 1);
                }
                ++cases;
            }
        }
    }
    std::cout << cases << " indexer workspace/cache-partition cases passed\n";
}
