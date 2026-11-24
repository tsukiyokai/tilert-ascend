#include "communication.h"

#include <stdexcept>

#include "acl/acl.h"
#include "shmem.h"

namespace trt_rsag_candidate {

CommunicationWindow AllocateCommunicationWindow(int64_t capacity,
                                                int32_t expected_rank,
                                                int32_t expected_ranks)
{
    if (capacity < 1 || capacity > kCommunicationMaximumRows
        || expected_ranks < 2 || expected_ranks > kCommunicationMaximumRanks
        || expected_rank < 0 || expected_rank >= expected_ranks) {
        throw std::invalid_argument("invalid whole-layer communication geometry");
    }
    if (aclshmemx_init_status() != ACLSHMEM_STATUS_IS_INITIALIZED
        || aclshmem_my_pe() != expected_rank
        || aclshmem_n_pes() != expected_ranks) {
        throw std::runtime_error("initialize the matching MTE SHMEM world first");
    }
    CommunicationWindow window;
    window.capacity = capacity;
    window.rank = expected_rank;
    window.ranks = expected_ranks;
    window.bytes = MakeCommunicationLayout(capacity, expected_ranks).bytes;
    if (aclrtGetDevice(&window.device) != ACL_SUCCESS) {
        throw std::runtime_error("cannot query the communication device");
    }
    window.address = aclshmem_align(512, window.bytes);
    if (window.address == nullptr) {
        throw std::runtime_error("SHMEM communication allocation failed");
    }
    if (!trt_rsag_candidate_contract::AlignedPayload(reinterpret_cast<uintptr_t>(window.address))
        || aclrtMemset(window.address, window.bytes, 0, window.bytes) != ACL_SUCCESS
        || aclrtSynchronizeDevice() != ACL_SUCCESS) {
        aclshmem_free(window.address);
        throw std::runtime_error("aligned SHMEM zero/synchronization failed");
    }
    // The driver must rendezvous all ranks after prepare, before any execute.
    return window;
}

void ReleaseCommunicationWindow(CommunicationWindow& window)
{
    if (window.address == nullptr) {
        return;
    }
    int32_t current_device = -1;
    if (aclrtGetDevice(&current_device) != ACL_SUCCESS
        || current_device != window.device
        || aclshmemx_init_status() != ACLSHMEM_STATUS_IS_INITIALIZED) {
        throw std::runtime_error("release the window in its live SHMEM device context");
    }
    aclshmem_free(window.address);
    window = {};
}

}  // namespace trt_rsag_candidate
