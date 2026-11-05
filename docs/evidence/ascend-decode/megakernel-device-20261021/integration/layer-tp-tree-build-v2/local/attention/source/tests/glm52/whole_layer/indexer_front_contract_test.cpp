#include <cassert>
#include <cstdint>
#include <iostream>
#include <set>

#include "indexer_front_contract.h"

int main()
{
    using namespace indexer_front;
    for (int64_t rows = 1; rows <= 32; ++rows) {
        const auto c = MakeConfig(rows, 24, true);
        const auto bf16_scale = MakeConfig(rows, 24, true, true);
        assert(!c.query_scale_bf16 && bf16_scale.query_scale_bf16);
        assert(c.bytes == bf16_scale.bytes && c.system == bf16_scale.system);
        const uint64_t offsets[] = {c.query_acc, c.key_acc, c.query_projection,
            c.key_projection, c.key_normalized, c.query_rope, c.key_rope,
            c.norm_stats, c.query_hadamard, c.key_hadamard, c.system, c.bytes};
        const uint64_t sizes[] = {uint64_t(c.capacity * kQueryColumns * 4),
            uint64_t(c.capacity * kKeyColumns * 4),
            uint64_t(rows * kQueryColumns * 2), uint64_t(rows * kKeyColumns * 2),
            uint64_t(rows * kDim * 2), uint64_t(rows * kQueryColumns * 2),
            uint64_t(rows * kDim * 2), uint64_t(rows * 8),
            uint64_t(rows * kQueryColumns * 4), uint64_t(c.capacity * kDim * 4),
            kSystemBytes};
        for (unsigned region = 0; region < 11; ++region) {
            assert(offsets[region] % 512 == 0);
            assert(offsets[region] + sizes[region] <= offsets[region + 1]);
        }
        std::set<uint64_t> query_addresses, key_addresses;
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t head = 0; head < kHeads; ++head) {
                for (int64_t element = 0; element < kDim; ++element) {
                    const uint64_t address = ((head / 2) * c.capacity + row)
                        * kQueryTile + (head % 2) * kDim + element;
                    assert(address < uint64_t(c.capacity * kQueryColumns));
                    assert(query_addresses.insert(address).second);
                }
            }
            for (int64_t tile = 0; tile < 5; ++tile) {
                for (int64_t element = 0; element < kKeyTile; ++element) {
                    const uint64_t address = (tile * c.capacity + row)
                        * kKeyTile + element;
                    assert(address < uint64_t(c.capacity * kKeyColumns));
                    assert(key_addresses.insert(address).second);
                }
            }
        }
        assert(query_addresses.size() == uint64_t(rows * kQueryColumns));
        assert(key_addresses.size() == uint64_t(rows * kKeyColumns));
        const auto bf16 = MakeConfig(rows, 8, false);
        assert(bf16.bytes == c.bytes);
        std::cout << rows;
        for (const auto offset : offsets) {
            std::cout << ' ' << offset;
        }
        std::cout << '\n';
    }
}
