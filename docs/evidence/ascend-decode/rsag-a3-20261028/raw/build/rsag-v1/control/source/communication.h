#ifndef TRT_RSAG_CONTROL_COMMUNICATION_H
#define TRT_RSAG_CONTROL_COMMUNICATION_H

#include <cstddef>
#include <cstdint>
#include "layout.h"

namespace trt_rsag_control {

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

// Epoch, each sender's ready word, and payload occupy separate cache lines.
// Both ready and payload have two planes; they share the same epoch parity.
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
    layout.payload = trt_rsag_control_contract::Align512(layout.ready + 2 * ranks * kCommunicationCacheLine);
    layout.rank_bytes = capacity * kCommunicationHidden * sizeof(uint16_t);
    layout.plane_bytes = ranks * layout.rank_bytes;
    layout.bytes = layout.payload + 2 * layout.plane_bytes;
    return layout;
}

// One window belongs to one serialized device stream and its live graphs.
// All ranks allocate the same capacity in the same order before capture.
// Free it after every rank has stopped using it and before SHMEM finalization.
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
static_assert(MakeCommunicationLayout(32, 16).bytes == 12585472);

#undef WHOLE_LAYER_COMMUNICATION_DEVICE

}  // namespace trt_rsag_control
#endif
