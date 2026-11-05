#ifndef TESTS_GLM52_INDEXER_FRONT_SCHEDULE_H
#define TESTS_GLM52_INDEXER_FRONT_SCHEDULE_H

#include "indexer_front_contract.h"

namespace indexer_front {

struct ProjectionOwner {
    int64_t query_tile;
    int64_t key_tile;
};

#if defined(__CCE__)
__host__ __aicore__
#endif
constexpr ProjectionOwner ProjectionOwnerFor(int64_t core, int64_t blocks)
{
    constexpr int64_t query_tiles = kQueryColumns / kQueryTile;
    constexpr int64_t key_tiles = kKeyColumns / kKeyTile;
    if (blocks >= query_tiles + key_tiles) {
        return {core < query_tiles ? core : -1,
                core >= query_tiles && core < query_tiles + key_tiles
                    ? core - query_tiles : -1};
    }
    return {core < query_tiles ? core : -1,
            core < key_tiles ? core : -1};
}

}  // namespace indexer_front

#endif
