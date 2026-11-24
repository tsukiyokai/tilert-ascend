#include "host_metadata.h"

#include <c10/core/DeviceGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <exception>
#include <string>

namespace {
thread_local std::string error;
}

extern "C" size_t layer_local_config_bytes()
{
    return sizeof(layer_local_chain::Config);
}

extern "C" size_t layer_local_address_bytes()
{
    return sizeof(layer_local_chain::Addresses);
}

extern "C" const char* layer_local_last_error()
{
    return error.c_str();
}

extern "C" int layer_local_prepare_records(
    void* config_record, size_t config_bytes, void* address_record,
    size_t address_bytes, const int64_t* geometry, const float* scalars,
    const uint64_t* norm, const uint64_t* attention, uint64_t shared_topk,
    const uint64_t* post, const uint64_t* routed, const uint64_t* bridge,
    const uint64_t* body, const uint64_t* shared)
{
    error.clear();
    using namespace layer_local_chain;
    Config config{};
    Addresses addresses{};
    if (config_record == nullptr || address_record == nullptr ||
        config_bytes != sizeof(Config) || address_bytes != sizeof(Addresses) ||
        !MakeHostConfig(geometry, scalars, config) ||
        !MakeHostAddresses(config, norm, attention, shared_topk, post, routed,
                           bridge, body, shared, addresses)) {
        error = "Invalid local layer metadata, geometry or producer addresses";
        return -1;
    }
    std::memcpy(config_record, &config, sizeof(config));
    std::memcpy(address_record, &addresses, sizeof(addresses));
    return 0;
}

extern "C" int layer_local_submit(int device, uint64_t sync,
    const void* config_record, size_t config_bytes, const void* address_record,
    size_t address_bytes, bool indexed)
{
    error.clear();
    using namespace layer_local_chain;
    if (device < 0 || sync == 0 || config_record == nullptr ||
        address_record == nullptr || config_bytes != sizeof(Config) ||
        address_bytes != sizeof(Addresses)) {
        error = "Invalid local layer submission record";
        return -1;
    }
    Config config;
    Addresses addresses;
    std::memcpy(&config, config_record, sizeof(config));
    std::memcpy(&addresses, address_record, sizeof(addresses));
    try {
        const c10::OptionalDeviceGuard guard(
            c10::Device(c10::DeviceType::PrivateUse1, device));
        // The default stream() call submits queued TorchNPU producers before
        // the direct AscendC launch, including during ACL graph capture.
        auto* stream = c10_npu::getCurrentNPUStream().stream();
        const auto launch = indexed ? layer_local_indexed_launch
                                    : layer_local_shared_launch;
        const int status = launch(stream, sync, &config, &addresses);
        if (status != 0) {
            error = "The local layer launch rejected its metadata";
        }
        return status;
    } catch (const std::exception& exception) {
        error = exception.what();
        return -2;
    } catch (...) {
        error = "Unknown failure submitting the local layer";
        return -3;
    }
}
