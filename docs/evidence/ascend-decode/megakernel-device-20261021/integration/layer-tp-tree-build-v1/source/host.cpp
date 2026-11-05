#include "contract.h"

#include <c10/core/DeviceGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "acl/acl.h"
#include "shmem.h"

namespace {

thread_local std::string last_error;
std::mutex context_mutex;
int world_device = -1;
int world_rank = -1;
int world_ranks = 0;
uint64_t next_handle = 1;

struct WindowOwner {
    whole_layer_tree::CommunicationWindow window;
    void* stream;
};
std::unordered_map<uint64_t, WindowOwner> windows;

void Check(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Operation>
int Guard(Operation operation)
{
    std::lock_guard<std::mutex> lock(context_mutex);
    try {
        operation();
        last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    } catch (...) {
        last_error = "Unknown TP layer host failure";
        return -2;
    }
}

WindowOwner& FindWindow(uint64_t handle)
{
    const auto found = windows.find(handle);
    Check(found != windows.end(), "The TP communication window is not live");
    return found->second;
}

}  // namespace

extern "C" const char* layer_tp_tree_last_error()
{
    return last_error.c_str();
}

extern "C" size_t layer_tp_tree_record_bytes(int record)
{
    switch (record) {
        case 0: return sizeof(layer_tp_tree_chain::Config);
        case 1: return sizeof(layer_tp_tree_chain::Addresses);
        case 2: return sizeof(layer_tp_tree_chain::Communication);
        default: return 0;
    }
}

extern "C" int layer_tp_tree_initialize(int device, int rank, int ranks,
    const char* endpoint, uint64_t heap_bytes)
{
    return Guard([&] {
        Check(device >= 0 && ranks >= 2 && ranks <= 16 && rank >= 0 &&
              rank < ranks && endpoint != nullptr && endpoint[0] != '\0' &&
              std::strlen(endpoint) < ACLSHMEM_MAX_IP_PORT_LEN &&
              heap_bytes >= whole_layer_tree::MakeCommunicationLayout(32, ranks).bytes,
              "Invalid TP SHMEM context geometry or endpoint");
        Check(world_device == -1 && windows.empty() &&
              aclshmemx_init_status() != ACLSHMEM_STATUS_IS_INITIALIZED,
              "The TP layer harness must own a fresh SHMEM context");
        const c10::OptionalDeviceGuard guard(
            c10::Device(c10::DeviceType::PrivateUse1, device));
        aclshmemx_init_attr_t attributes{};
        aclshmemx_uniqueid_t unused_unique_id{};
        attributes.my_pe = rank;
        attributes.n_pes = ranks;
        attributes.local_mem_size = heap_bytes;
        std::strcpy(attributes.ip_port, endpoint);
        const int version = (1 << 16) + sizeof(attributes);
        attributes.option_attr = {version, ACLSHMEM_DATA_OP_MTE, 120, 120, 120};
        attributes.comm_args = &unused_unique_id;
        Check(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes) == 0,
              "MTE SHMEM initialization failed");
        world_device = device;
        world_rank = rank;
        world_ranks = ranks;
    });
}

extern "C" int layer_tp_tree_window_create(int64_t capacity, uint64_t* handle)
{
    return Guard([&] {
        Check(handle != nullptr && capacity >= 1 && capacity <= 32 &&
              world_device >= 0, "Initialize TP SHMEM before creating a window");
        *handle = 0;
        const c10::OptionalDeviceGuard guard(
            c10::Device(c10::DeviceType::PrivateUse1, world_device));
        WindowOwner owner{};
        owner.stream = c10_npu::getCurrentNPUStream().stream();
        owner.window = whole_layer_tree::AllocateCommunicationWindow(
            capacity, world_rank, world_ranks);
        try {
            windows.emplace(next_handle, owner);
        } catch (...) {
            whole_layer_tree::ReleaseCommunicationWindow(owner.window);
            throw;
        }
        *handle = next_handle++;
    });
}

extern "C" int layer_tp_tree_window_info(uint64_t handle, uint64_t* fields,
    size_t count)
{
    return Guard([&] {
        Check(fields != nullptr && count == 6, "Expected six window fields");
        const auto& window = FindWindow(handle).window;
        fields[0] = reinterpret_cast<uint64_t>(window.address);
        fields[1] = window.bytes;
        fields[2] = window.capacity;
        fields[3] = window.rank;
        fields[4] = window.ranks;
        fields[5] = window.device;
    });
}

extern "C" int layer_tp_tree_submit(int device, uint64_t sync,
    const void* config_record, size_t config_bytes, const void* address_record,
    size_t address_bytes, bool indexed, uint64_t handle,
    uint64_t attention_sum, uint64_t output)
{
    return Guard([&] {
        using namespace layer_tp_tree_chain;
        Check(device >= 0 && device == world_device && sync != 0 &&
              config_record != nullptr && address_record != nullptr &&
              config_bytes == sizeof(Config) && address_bytes == sizeof(Addresses),
              "Invalid TP layer submission record");
        const auto& owner = FindWindow(handle);
        Config config;
        Addresses addresses;
        std::memcpy(&config, config_record, sizeof(config));
        std::memcpy(&addresses, address_record, sizeof(addresses));
        const auto& window = owner.window;
        const Communication communication{
            reinterpret_cast<uint64_t>(window.address), window.bytes,
            attention_sum, output, window.capacity, window.rank, window.ranks};
        Check(Valid(config, addresses, communication),
              "Invalid TP layer geometry or communication addresses");
        const c10::OptionalDeviceGuard guard(
            c10::Device(c10::DeviceType::PrivateUse1, device));
        // Submit queued TorchNPU producers before the direct AscendC launch.
        auto* stream = c10_npu::getCurrentNPUStream().stream();
        Check(stream == owner.stream,
              "A communication window belongs to its original serialized stream");
        const auto launch = indexed ? layer_tp_tree_indexed_launch : layer_tp_tree_shared_launch;
        Check(launch(stream, sync, &config, &addresses, &communication) == 0,
              "The TP layer launch rejected its metadata");
    });
}

// The caller destroys every graph and synchronizes all participating ranks
// before releasing windows. SHMEM allocations are collective in creation order.
extern "C" int layer_tp_tree_window_destroy(uint64_t handle)
{
    return Guard([&] {
        auto& owner = FindWindow(handle);
        const c10::OptionalDeviceGuard guard(
            c10::Device(c10::DeviceType::PrivateUse1, owner.window.device));
        whole_layer_tree::ReleaseCommunicationWindow(owner.window);
        windows.erase(handle);
    });
}

extern "C" int layer_tp_tree_finalize()
{
    return Guard([] {
        Check(world_device >= 0 && windows.empty(),
              "Release all TP windows before finalizing their SHMEM context");
        const c10::OptionalDeviceGuard guard(
            c10::Device(c10::DeviceType::PrivateUse1, world_device));
        Check(aclshmem_finalize() == 0, "SHMEM finalization failed");
        world_device = world_rank = -1;
        world_ranks = 0;
    });
}
