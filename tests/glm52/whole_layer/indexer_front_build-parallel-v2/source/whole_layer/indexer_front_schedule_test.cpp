#include <array>
#include <cassert>
#include <iostream>

#include "indexer_front_schedule.h"

int main()
{
    using namespace indexer_front;
    constexpr int64_t queries = kQueryColumns / kQueryTile;
    constexpr int64_t keys = kKeyColumns / kKeyTile;
    for (int64_t blocks = 1; blocks <= 64; ++blocks) {
        std::array<int, queries> query_writers{};
        std::array<int, keys> key_writers{};
        for (int64_t core = 0; core < blocks; ++core) {
            const auto owner = ProjectionOwnerFor(core, blocks);
            if (blocks >= queries + keys) {
                assert(owner.query_tile < 0 || owner.key_tile < 0);
            } else {
                assert(owner.query_tile == (core < queries ? core : -1));
                assert(owner.key_tile == (core < keys ? core : -1));
            }
            if (owner.query_tile >= 0) {
                for (auto tile = owner.query_tile; tile < queries;
                     tile += blocks) {
                    ++query_writers[tile];
                }
            }
            if (owner.key_tile >= 0) {
                for (auto tile = owner.key_tile; tile < keys; tile += blocks) {
                    ++key_writers[tile];
                }
            }
        }
        for (const auto writers : query_writers) {
            assert(writers == 1);
        }
        for (const auto writers : key_writers) {
            assert(writers == 1);
        }
    }
    std::cout << "64 launch geometries: disjoint complete projection owners\n";
}
