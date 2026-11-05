#include "attention_norm_contract.h"

#include <cassert>
#include <iostream>

int main()
{
    using namespace whole_layer_attention;
    uint32_t cases = 0;
    for (uint32_t rows = 1; rows <= 32; ++rows) {
        for (uint32_t blocks = 1; blocks <= 24; ++blocks) {
            for (uint32_t pages : {1U, 2U, 4U, 65U}) {
                const auto attention = MakeConfig(rows, pages + 3, pages,
                    blocks, 0.0625F, 1e-5F, 1e-5F);
                const auto config = MakeNormConfig(attention, 1e-5F);
                assert(config.scale_offset >= rows * 6144ULL);
                assert(config.full_offset >= config.scale_offset + rows * 4ULL);
                assert(config.rstd_offset >= rows * 6144ULL * 2);
                assert(config.norm_temporary_bytes >= config.rstd_offset + rows * 4ULL);
                assert(config.full.bytes >= config.norm_temporary_bytes);
                assert(config.bytes == config.full_offset + config.full.bytes);
                assert(config.scale_offset % 512 == 0);
                assert(config.full_offset % 512 == 0);
                assert(config.rstd_offset % 512 == 0);
                ++cases;
            }
        }
    }
    std::cout << cases << " norm/attention lifetime cases passed\n";
}
