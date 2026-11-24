// Narrow C ABI for CANN 9.2 HIXL. ACL and device allocations belong to the caller.
#include <acl/acl_rt.h>
#include <hixl/hixl.h>

#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void checked(uint32_t status, const char* operation) {
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + " status=" +
                                 std::to_string(status));
    }
}

template <typename Function>
int guarded(Function function) noexcept {
    try {
        last_error.clear();
        function();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
    } catch (...) {
        last_error = "Unknown native exception";
    }
    return -1;
}

struct Region {
    uint64_t address;
    uint64_t length;
    hixl::MemHandle handle;
};

struct Engine {
    hixl::Hixl transport;
    aclrtContext context = nullptr;
    std::mutex mutex;
    std::vector<Region> regions;
    std::set<std::string> peers;

    void bind() {
        checked(aclrtSetCurrentContext(context), "aclrtSetCurrentContext");
    }

    bool contains(uint64_t address, uint64_t length) const {
        for (const auto& region : regions) {
            if (address >= region.address && length <= region.length &&
                address - region.address <= region.length - length) {
                return true;
            }
        }
        return false;
    }
};

Engine& engine(void* handle) {
    require(handle != nullptr, "Null HIXL handle");
    return *static_cast<Engine*>(handle);
}

void interval(uint64_t address, uint64_t length) {
    require(address != 0 && length != 0 &&
                length <= std::numeric_limits<uint64_t>::max() - address,
            "Invalid or overflowing memory interval");
}
}  // namespace

#define EXPORT extern "C" __attribute__((visibility("default")))

EXPORT const char* tilert_hixl_last_error() noexcept {
    return last_error.c_str();
}

EXPORT int tilert_hixl_open(const char* endpoint, int device, void** result) {
    return guarded([&] {
        require(endpoint != nullptr && result != nullptr && device >= 0,
                "Invalid HIXL open arguments");
        *result = nullptr;
        checked(aclrtSetDevice(device), "aclrtSetDevice");
        auto instance = std::make_unique<Engine>();
        checked(aclrtGetCurrentContext(&instance->context), "aclrtGetCurrentContext");
        // P and D on one die need distinct HCCL listeners.
        const std::string local(endpoint);
        const auto port = local.substr(local.rfind(':') + 1);
        const auto resources =
            "{\"comm_resource_config.listen_port\":\"" + port + "\"}";
        const std::map<hixl::AscendString, hixl::AscendString> options{
            {hixl::AscendString("GlobalResourceConfig"),
             hixl::AscendString(resources.c_str())},
        };
        checked(instance->transport.Initialize(hixl::AscendString(endpoint),
                                               options), "Initialize");
        *result = instance.release();
    });
}

EXPORT int tilert_hixl_register(void* handle, uint64_t address, uint64_t length) {
    return guarded([&] {
        auto& instance = engine(handle);
        std::lock_guard<std::mutex> lock(instance.mutex);
        interval(address, length);
        for (const auto& region : instance.regions) {
            if (address == region.address && length == region.length) {
                return;
            }
            require(address + length <= region.address ||
                        region.address + region.length <= address,
                    "Overlapping memory registrations");
        }
        instance.bind();
        hixl::MemDesc memory{};
        memory.addr = address;
        memory.len = length;
        hixl::MemHandle registration = nullptr;
        checked(instance.transport.RegisterMem(memory, hixl::MEM_DEVICE,
                                               registration), "RegisterMem");
        instance.regions.push_back({address, length, registration});
    });
}

EXPORT int tilert_hixl_write(void* handle, const char* remote,
                            const uint64_t* sources, const uint64_t* destinations,
                            const uint64_t* lengths, size_t count, int timeout_ms) {
    return guarded([&] {
        auto& instance = engine(handle);
        std::lock_guard<std::mutex> lock(instance.mutex);
        require(remote != nullptr && timeout_ms > 0, "Invalid write arguments");
        if (count == 0) {
            return;
        }
        require(sources && destinations && lengths, "Null descriptor array");
        std::vector<hixl::TransferOpDesc> operations;
        operations.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            interval(sources[index], lengths[index]);
            interval(destinations[index], lengths[index]);
            require(instance.contains(sources[index], lengths[index]),
                    "Source is outside registered local memory");
            operations.push_back({sources[index], destinations[index], lengths[index]});
        }
        instance.bind();
        const hixl::AscendString peer(remote);
        if (instance.peers.find(remote) == instance.peers.end()) {
            checked(instance.transport.Connect(peer, timeout_ms), "Connect");
            instance.peers.insert(remote);
        }
        // The transport API has no producer-stream argument. Publish completed bytes.
        checked(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice");
        const auto status = instance.transport.TransferSync(
            peer, hixl::WRITE, operations, timeout_ms);
        checked(status, "TransferSync");
    });
}

EXPORT int tilert_hixl_close(void* handle) {
    return guarded([&] {
        auto& instance = engine(handle);
        {
            std::lock_guard<std::mutex> lock(instance.mutex);
            instance.bind();
            while (!instance.peers.empty()) {
                const auto peer = instance.peers.begin();
                checked(instance.transport.Disconnect(hixl::AscendString(peer->c_str())),
                        "Disconnect");
                instance.peers.erase(peer);
            }
            while (!instance.regions.empty()) {
                checked(instance.transport.DeregisterMem(instance.regions.back().handle),
                        "DeregisterMem");
                instance.regions.pop_back();
            }
            instance.transport.Finalize();
        }
        delete &instance;
    });
}
