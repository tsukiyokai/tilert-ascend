#include "communication.h"
#include "communication_probe.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "acl/acl.h"
#include "lib/matmul/matmul_tiling.h"
#include "shmem.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* address, uint32_t* length);

namespace {

constexpr size_t kTilingBytes = 1024;
constexpr size_t kWorkspaceBytes = 16 * 1024 * 1024;
thread_local std::string last_error;

void Check(bool valid, const char* message)
{
    if (!valid) {
        throw std::runtime_error(message);
    }
}

struct Probe {
    trt_rsag_control::CommunicationWindow window;
    void* tiling = nullptr;
    void* workspace = nullptr;
    void* product = nullptr;
    void* next_partial = nullptr;
    uint64_t sync_base = 0;
    uint32_t blocks = 0;
    int64_t rows = 0;

    void Release()
    {
        trt_rsag_control::ReleaseCommunicationWindow(window);
        for (void* allocation : {tiling, workspace, product, next_partial}) {
            if (allocation != nullptr) {
                Check(aclrtFree(allocation) == ACL_SUCCESS,
                      "communication probe scratch release failed");
            }
        }
        tiling = workspace = product = next_partial = nullptr;
    }
};

template <typename Operation>
int Guard(Operation operation)
{
    try {
        operation();
        last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

void Allocate(void** address, size_t bytes)
{
    Check(aclrtMalloc(address, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
          "communication probe scratch allocation failed");
    Check(trt_rsag_control_contract::AlignedPayload(reinterpret_cast<uintptr_t>(*address)),
          "probe scratch requires 512-byte alignment");
}

void PrepareTiling(Probe& probe)
{
    matmul_tiling::MatmulApiTiling tiling;
    tiling.SetAType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_BF16, false);
    tiling.SetBType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_BF16, false);
    tiling.SetCType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT);
    constexpr int side = trt_rsag_control::kCommunicationProbeSide;
    tiling.SetShape(side, side, side);
    tiling.SetOrgShape(side, side, side);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1);
    AscendC::tiling::TCubeTiling cube;
    static_assert(sizeof(cube) <= kTilingBytes);
    Check(tiling.GetTiling(cube) == 0, "communication probe tiling failed");
    std::array<uint8_t, kTilingBytes> bytes{};
    std::memcpy(bytes.data(), &cube, sizeof(cube));
    Allocate(&probe.tiling, bytes.size());
    Check(aclrtMemcpy(probe.tiling, bytes.size(), bytes.data(), bytes.size(),
                       ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
          "communication probe tiling upload failed");
}

}  // namespace

// This C ABI belongs to the probe. Inputs are contiguous physical-ND tensors
// on the current device; the Python driver checks dtype and shape. Its owner
// and tensors remain alive through every queued launch and graph replay.
extern "C" const char* trt_rsag_control_probe_last_error()
{
    return last_error.c_str();
}

extern "C" int trt_rsag_control_probe_initialize(
    int32_t rank, int32_t ranks, const char* endpoint, uint64_t heap_bytes)
{
    return Guard([&] {
        Check(rank >= 0 && rank < ranks && ranks >= 2 && ranks <= 16,
              "communication probe requires 2..16 ranks");
        Check(endpoint != nullptr && endpoint[0] != '\0'
                  && std::strlen(endpoint) < ACLSHMEM_MAX_IP_PORT_LEN,
              "invalid SHMEM configuration-store endpoint");
        Check(aclshmemx_init_status() != ACLSHMEM_STATUS_IS_INITIALIZED,
              "the probe must own a fresh SHMEM context");
        aclshmemx_init_attr_t attributes{};
        aclshmemx_uniqueid_t unused_unique_id{};
        attributes.my_pe = rank;
        attributes.n_pes = ranks;
        attributes.local_mem_size = heap_bytes;
        std::strcpy(attributes.ip_port, endpoint);
        const int version = (1 << 16) + sizeof(attributes);
        attributes.option_attr = {version, ACLSHMEM_DATA_OP_MTE,
                                  120, 120, 120};
        attributes.comm_args = &unused_unique_id;
        Check(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes) == 0,
              "MTE SHMEM initialization failed");
    });
}

extern "C" int trt_rsag_control_probe_prepare(
    int64_t rows, int32_t rank, int32_t ranks, uint32_t blocks, void** owner)
{
    return Guard([&] {
        Check(owner != nullptr, "missing communication probe owner output");
        *owner = nullptr;
        auto probe = std::make_unique<Probe>();
        probe->rows = rows;
        probe->blocks = blocks;
        int32_t device = -1;
        int64_t cubes = 0, vectors = 0;
        Check(aclrtGetDevice(&device) == ACL_SUCCESS
                  && aclrtGetDeviceInfo(device, ACL_DEV_ATTR_AICORE_CORE_NUM,
                                         &cubes) == ACL_SUCCESS
                  && aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                         &vectors) == ACL_SUCCESS,
              "cannot query MIX residency");
        Check(blocks > 0 && blocks <= std::min(cubes, vectors / 2),
              "communication probe blocks exceed MIX residency");
        uint32_t length = 0;
        Check(rtGetC2cCtrlAddr(&probe->sync_base, &length) == 0
                  && probe->sync_base != 0,
              "cannot obtain the MIX synchronization address");
        try {
            probe->window = trt_rsag_control::AllocateCommunicationWindow(
                rows, rank, ranks);
            PrepareTiling(*probe);
            const size_t elements = rows * trt_rsag_control::kCommunicationHidden;
            Allocate(&probe->workspace, kWorkspaceBytes);
            Allocate(&probe->product, elements * sizeof(float));
            Allocate(&probe->next_partial, elements * sizeof(uint16_t));
        } catch (...) {
            probe->Release();
            throw;
        }
        *owner = probe.release();
    });
}

extern "C" int trt_rsag_control_probe_execute(
    void* owner, void* stream, void* partial, void* weight,
    void* first_sum, void* output)
{
    return Guard([&] {
        auto* probe = static_cast<Probe*>(owner);
        Check(probe != nullptr && stream != nullptr && partial != nullptr
                  && weight != nullptr && first_sum != nullptr && output != nullptr,
              "communication probe received a null launch argument");
        Check(trt_rsag_control_contract::AlignedPayload(reinterpret_cast<uintptr_t>(partial))
                  && trt_rsag_control_contract::AlignedPayload(reinterpret_cast<uintptr_t>(weight))
                  && trt_rsag_control_contract::AlignedPayload(reinterpret_cast<uintptr_t>(first_sum))
                  && trt_rsag_control_contract::AlignedPayload(reinterpret_cast<uintptr_t>(output)),
              "probe tensors require 512-byte alignment");
        trt_rsag_control::LaunchCommunicationProbe(
            probe->blocks, stream, probe->sync_base, probe->workspace,
            probe->window.address, partial, weight, first_sum, probe->product,
            probe->next_partial, output, probe->tiling, probe->rows,
            probe->window.capacity, probe->window.rank, probe->window.ranks);
    });
}

extern "C" int trt_rsag_control_probe_destroy(void* owner)
{
    return Guard([&] {
        auto* probe = static_cast<Probe*>(owner);
        Check(probe != nullptr, "communication probe owner is null");
        probe->Release();
        delete probe;
    });
}

extern "C" int trt_rsag_control_probe_finalize()
{
    return Guard([] {
        Check(aclshmem_finalize() == 0, "SHMEM finalization failed");
    });
}
