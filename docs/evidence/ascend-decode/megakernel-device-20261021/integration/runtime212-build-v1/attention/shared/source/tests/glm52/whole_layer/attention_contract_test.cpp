#include <cassert>
#include <cstdint>
#include <iostream>

#include "attention_full_contract.h"

int main()
{
    using namespace whole_layer_attention;
    uint64_t cases = 0;
    for (uint32_t rows = 1; rows <= 32; ++rows) {
        for (uint32_t blocks = 1; blocks <= 24; ++blocks) {
            for (uint32_t pages : {1U, 2U, 3U, 4U, 17U, 65U}) {
                const auto config = MakeConfig(rows, pages + 3, pages,
                    blocks, 0.0625F, 1e-5F, 1e-5F);
                assert(config.query_offset == 0);
                assert(config.rope_offset >= rows * 4ULL * 512 * 2);
                assert(config.work_offset >= config.rope_offset + rows * 4ULL * 64 * 2);
                assert(config.rope_offset % 512 == 0);
                assert(config.work_offset % 512 == 0);
                // Walk the vendor's actual full-quant Prolog workspace views.
                uint64_t end = rows * 576ULL * sizeof(int32_t);
                end += rows * 2048ULL * sizeof(int32_t);
                end += rows * 2048ULL * sizeof(int8_t);
                end += rows * 1024ULL * sizeof(int32_t);
                end += rows * 768ULL * sizeof(uint16_t);
                assert(end <= config.prolog_bytes);
                // Every SFA core owns two QK, P and PV tiles; the merged KV
                // and valid-ID views follow all cores' preceding tile views.
                uint64_t cursor = 0;
                for (uint32_t bytes : {4U, 2U}) {
                    for (uint32_t core = 0; core < blocks; ++core) {
                        cursor += 2ULL * config.mm_elements * bytes;
                    }
                }
                cursor += blocks * 2ULL * config.bmm_elements * 4;
                cursor += blocks * 512ULL * 576 * 4 * 2;
                cursor += blocks * 2ULL * 128 * 4 * 4;
                assert(cursor <= config.sfa_bytes);
                assert(config.bytes >= config.work_offset + cursor);
                assert(config.bytes >= config.work_offset + end);
                assert(config.s2_base <= pages * 128);
                assert(config.s2_base <= 512);
                assert(config.mm_elements >= 16 * config.s2_base);
                const auto full = MakeFullConfig(config);
                assert(full.arena_offset >= rows * 4ULL * 512 * 2);
                assert(full.arena_offset % 512 == 0);
                assert(full.bytes >= full.arena_offset + config.bytes);
                assert(full.bytes >= full.arena_offset + full.tail.bytes);
                assert(full.capacity == (rows <= 16 ? 16 : 32));
                ++cases;
            }
        }
    }
    std::cout << "attention workspace contracts: " << cases << " cases\n";
    for (uint32_t rows : {1U, 12U, 32U}) {
        const auto full = MakeFullConfig(MakeConfig(rows, 68, 65, 8,
                                                    0.0625F, 1e-5F, 1e-5F));
        std::cout << "M=" << rows << " full_scratch=" << full.bytes
                  << " separate_scratch_and_latent="
                  << full.attention.bytes + full.tail.bytes + full.arena_offset
                  << '\n';
    }
}
