#ifndef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_H
#define TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_H

#include <cstddef>
#include <cstdint>

namespace whole_layer_tree {

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define WHOLE_LAYER_COMMUNICATION_DEVICE __aicore__
#else
#define WHOLE_LAYER_COMMUNICATION_DEVICE
#endif

constexpr int64_t kCommunicationHidden = 6144;
constexpr int64_t kCommunicationMaximumRows = 32;
constexpr int32_t kCommunicationMaximumRanks = 16;
constexpr uint64_t kCommunicationCacheLine = 64;
constexpr int64_t kCommunicationChunk = 2048;
constexpr uint64_t kCommunicationUbBytes = kCommunicationChunk * 22;


struct CommunicationLayout {
    uint64_t epoch;
    uint64_t ready;
    uint64_t payload;
    uint64_t rank_bytes;
    uint64_t plane_bytes;
    uint64_t bytes;
};

WHOLE_LAYER_COMMUNICATION_DEVICE constexpr CommunicationLayout
MakeCommunicationLayout(int64_t capacity, int32_t ranks)
{
    CommunicationLayout layout{};
    layout.epoch = 0;
    layout.ready = kCommunicationCacheLine;
    layout.payload = layout.ready + 2 * ranks * kCommunicationCacheLine;
    layout.rank_bytes = capacity * kCommunicationHidden * sizeof(uint16_t);
    layout.plane_bytes = ranks * layout.rank_bytes;
    layout.bytes = layout.payload + 2 * layout.plane_bytes;
    return layout;
}

WHOLE_LAYER_COMMUNICATION_DEVICE constexpr bool
IsQuad4Active(int64_t capacity, int64_t rows)
{
#ifdef K11180_QUAD4_SIP_ALL
    return (capacity >= 2) && (rows > 0) && (rows <= (capacity >> 1));
#else
    (void)capacity;
    (void)rows;
    return false;
#endif
}

WHOLE_LAYER_COMMUNICATION_DEVICE constexpr CommunicationLayout
MakeQuad4Layout(int64_t capacity, int32_t ranks, int64_t rows)
{
    const int64_t half_cap = capacity >> 1;
    const int64_t q_cap = (rows <= 4 && half_cap >= 4) ? 4
        : ((rows <= 8 && half_cap >= 8) ? 8 : half_cap);
    CommunicationLayout layout{};
    layout.epoch = 0;
    layout.ready = kCommunicationCacheLine;
    layout.payload = layout.ready + 2 * ranks * kCommunicationCacheLine;
    layout.rank_bytes = static_cast<uint64_t>(q_cap) * kCommunicationHidden * sizeof(uint16_t);
    layout.plane_bytes = static_cast<uint64_t>(ranks) * layout.rank_bytes;
    layout.bytes = layout.payload + 4U * layout.plane_bytes;
    return layout;
}


struct CommunicationWindow {
    void* address = nullptr;
    uint64_t bytes = 0;
    int64_t capacity = 0;
    int32_t rank = -1;
    int32_t ranks = 0;
    int32_t device = -1;
};

CommunicationWindow AllocateCommunicationWindow(int64_t capacity,
                                                int32_t expected_rank,
                                                int32_t expected_ranks);
void ReleaseCommunicationWindow(CommunicationWindow& window);

static_assert(kCommunicationHidden % kCommunicationChunk == 0);
static_assert(MakeCommunicationLayout(32, 16).bytes == 12585024);

#undef WHOLE_LAYER_COMMUNICATION_DEVICE

}
#endif
