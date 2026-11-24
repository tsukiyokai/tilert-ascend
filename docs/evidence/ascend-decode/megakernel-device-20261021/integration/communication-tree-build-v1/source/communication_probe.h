#ifndef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_PROBE_H
#define TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_PROBE_H

#include <cstdint>

namespace whole_layer_tree {

constexpr int64_t kCommunicationProbeSide = 16;
constexpr int64_t kCommunicationProbeTile = 256;

// One MIX launch: reduce BF16 partials, apply rank-local 16x16 projections
// on CUBE, narrow to BF16 on AIV, then reduce those dependent partials.
void LaunchCommunicationProbe(
    uint32_t blocks, void* stream, uint64_t sync_base, void* workspace,
    void* window, void* partial, void* weight, void* first_sum,
    void* product, void* next_partial, void* output, void* tiling,
    int64_t rows, int64_t capacity, int32_t rank, int32_t ranks);

}  // namespace whole_layer_tree
#endif
