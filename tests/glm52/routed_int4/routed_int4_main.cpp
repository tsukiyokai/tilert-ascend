/** Standalone A3 zero-row and hot-expert smoke test.
 *
 * Usage: routed_int4 [tokens=32] [device=0] [blocks=16]. This checks launches,
 * scratch bounds and zero quantization; use the Python fixture for numerical
 * parity against real W4A8 weights and nonzero, changing routes.
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "acl/acl.h"
#include "routed_int4_kernel.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace {
using namespace routed_int4;

void Check(aclError status, const char* operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(operation);
    }
}

class Arena {
 public:
    ~Arena()
    {
        for (void* pointer : allocations_) {
            aclrtFree(pointer);
        }
    }

    void* Zero(size_t bytes)
    {
        void* pointer = nullptr;
        Check(aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc");
        allocations_.push_back(pointer);
        Check(aclrtMemset(pointer, bytes, 0, bytes), "memset");
        return pointer;
    }

    template <class T>
    void* Copy(const std::vector<T>& values)
    {
        const size_t bytes = values.size() * sizeof(T);
        void* pointer = Zero(bytes);
        Check(aclrtMemcpy(pointer, bytes, values.data(), bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE), "upload");
        return pointer;
    }

 private:
    std::vector<void*> allocations_;
};

void Run(int64_t tokens, uint32_t blocks, aclrtStream stream)
{
    constexpr int64_t experts = kTopK;
    Arena arena;
    void* rows = arena.Zero(tokens * kHidden);
    void* gate_up = arena.Zero(experts * kHidden * kGateUp / 2);
    void* down = arena.Zero(experts * kIntermediate * kHidden / 2);
    auto* gu_bias = (float*)arena.Zero(experts * kGateUp * sizeof(float));
    auto* down_bias = (float*)arena.Zero(experts * kHidden * sizeof(float));
    auto* gu_scale = (float*)arena.Copy(std::vector<float>(experts * kGateUp, 1));
    auto* down_scale = (float*)arena.Copy(std::vector<float>(experts * kHidden, 1));
    auto* token_scale = (float*)arena.Copy(std::vector<float>(tokens, 1));
    std::vector<int32_t> counts(kMaxExperts), starts(kMaxExperts);
    std::vector<int32_t> sources(kMaxRoutes), slots(kMaxRoutes);
    std::vector<float> probabilities(kMaxRoutes);
    for (int64_t expert = 0; expert < experts; ++expert) {
        counts[expert] = tokens;
        starts[expert] = expert * tokens;
        for (int64_t token = 0; token < tokens; ++token) {
            const int64_t route = expert * tokens + token;
            sources[route] = token;
            slots[route] = expert;
            probabilities[route] = 1.0F / kTopK;
        }
    }
    for (int64_t expert = experts; expert < kMaxExperts; ++expert) {
        starts[expert] = tokens * kTopK;
    }
    void* group_count = arena.Copy(counts);
    void* group_start = arena.Copy(starts);
    void* source_row = arena.Copy(sources);
    void* source_slot = arena.Copy(slots);
    auto* row_weight = (float*)arena.Copy(probabilities);
    const int64_t stacked_rows = 2 * tokens * kTopK;
    void* stacked = arena.Zero(stacked_rows * kHidden / 2);
    void* partial = arena.Zero(stacked_rows * kGateUp * sizeof(int32_t));
    void* middle = arena.Zero(stacked_rows * kIntermediate / 2);
    auto* middle_scale = (float*)arena.Zero(tokens * kTopK * sizeof(float));
    void* wide = arena.Zero(stacked_rows * kHidden * sizeof(int32_t));
    void* routes = arena.Zero(tokens * kTopK * kHidden * sizeof(uint16_t));
    const size_t output_bytes = tokens * kHidden * sizeof(uint16_t);
    void* output = arena.Zero(output_bytes);
    Check(aclrtMemset(output, output_bytes, 0xFF, output_bytes), "poison output");
    void* workspace = arena.Zero(kMatmulWorkspaceBytes);
    std::vector<uint8_t> gu_tiling_host(kTilingBytes), down_tiling_host(kTilingBytes);
    if (routed_int4_build_tiling(kMaxStackedRows, kGateUp, kHidden, kGateUp,
                                 gu_tiling_host.data()) != 0
        || routed_int4_build_tiling(kMaxStackedRows, kHidden, kIntermediate, kHidden,
                                    down_tiling_host.data()) != 0) {
        throw std::runtime_error("tiling failed");
    }
    void* gu_tiling = arena.Copy(gu_tiling_host);
    void* down_tiling = arena.Copy(down_tiling_host);
    uint64_t sync_base = 0;
    uint32_t sync_length = 0;
    if (rtGetC2cCtrlAddr(&sync_base, &sync_length) != 0 || sync_base == 0) {
        throw std::runtime_error("rtGetC2cCtrlAddr failed");
    }
    routed_int4_launch(
        blocks, stream, sync_base, rows, gate_up, gu_bias, gu_scale, down,
        down_bias, down_scale, token_scale, group_count, group_start,
        source_row, source_slot, row_weight, stacked, partial, middle,
        middle_scale, wide, routes, output, gu_tiling, down_tiling, workspace,
        experts, tokens);
    Check(aclrtSynchronizeStream(stream), "kernel");
    std::vector<uint16_t> actual(tokens * kHidden);
    Check(aclrtMemcpy(actual.data(), output_bytes, output, output_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST), "read output");
    if (!std::all_of(actual.begin(), actual.end(),
                     [](uint16_t value) { return (value & 0x7FFF) == 0; })) {
        throw std::runtime_error("zero-row output differs from zero");
    }
    std::printf("Zero-row smoke passed: M=%ld, hot expert rows=%ld, blocks=%u\n",
                tokens, tokens, blocks);
}
}  // namespace

int main(int argc, char** argv)
{
    const int64_t tokens = argc > 1 ? std::atoll(argv[1]) : 32;
    const int32_t device = argc > 2 ? std::atoi(argv[2]) : 0;
    const uint32_t blocks = argc > 3 ? std::atoi(argv[3]) : 16;
    if (tokens < 1 || tokens > kMaxTokens || blocks < 1) {
        std::fprintf(stderr, "Expected tokens in [1,32] and positive blocks\n");
        return 2;
    }
    aclrtStream stream = nullptr;
    try {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "set device");
        int64_t cubes = 0;
        int64_t vectors = 0;
        Check(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_AICORE_CORE_NUM, &cubes),
              "query AIC count");
        Check(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectors),
              "query AIV count");
        if (blocks > std::min(cubes, vectors / 2)) {
            throw std::runtime_error("block count exceeds mixed-kernel residency");
        }
        Check(aclrtCreateStream(&stream), "create stream");
        Run(tokens, blocks, stream);
        Check(aclrtDestroyStream(stream), "destroy stream");
        stream = nullptr;
        Check(aclrtResetDevice(device), "reset device");
        Check(aclFinalize(), "finalize");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        return 1;
    }
    return 0;
}
