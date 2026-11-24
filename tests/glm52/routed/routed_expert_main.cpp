/**
 * Does one kernel do a layer's routed experts, and get them right?
 *
 * Every expert's whole chain -- gate and up projection, dequantisation,
 * SwiGLU, per-token quantisation, down projection, dequantisation to BF16 --
 * in one launch, with the experts spread over the cores and no grid barrier.
 * The reference computes both INT32 products exactly and rounds where the
 * device rounds.
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "acl/acl.h"
#include "routed_expert_kernel.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace {

constexpr int64_t HIDDEN = 6144;
constexpr int64_t HALF = 128;
constexpr int64_t GATE_UP = 2 * HALF;
#ifndef ROWS_PER_EXPERT
#define ROWS_PER_EXPERT 2
#endif
constexpr int64_t ROWS = ROWS_PER_EXPERT;
constexpr int64_t BLOCK_FLOATS = 8;
constexpr int64_t SCALE_STRIDE = ((ROWS + BLOCK_FLOATS - 1) / BLOCK_FLOATS)
                                 * BLOCK_FLOATS;
constexpr size_t TILING_BYTES = 256;

void Check(aclError status, const char* what)
{
    if (status != ACL_SUCCESS) {
        std::printf("%s failed: %d\n", what, status);
        std::exit(1);
    }
}

double Elapsed(const timespec& begin, const timespec& end)
{
    return (end.tv_sec - begin.tv_sec) * 1e6
           + (end.tv_nsec - begin.tv_nsec) / 1e3;
}

/** Round to IEEE half, which the device's INT8 cast goes through. */
float ToHalf(float value)
{
    const float magnitude = std::fabs(value);
    if (magnitude < 6.103515625e-05F) {
        return value;
    }
    int exponent = 0;
    std::frexp(magnitude, &exponent);
    const float quantum = std::ldexp(1.0F, exponent - 11);
    const float rounded = std::nearbyintf(magnitude / quantum) * quantum;
    return (value < 0.0F) ? -rounded : rounded;
}

uint16_t ToBf16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t low = bits & 0xFFFF;
    const uint32_t carry = (low > 0x8000)
                           || (low == 0x8000 && ((bits >> 16) & 1));
    return static_cast<uint16_t>((bits >> 16) + carry);
}

void* Hold(size_t bytes, const char* what)
{
    void* block = nullptr;
    Check(aclrtMalloc(&block, bytes, ACL_MEM_MALLOC_HUGE_FIRST), what);
    return block;
}

}  // namespace

int main(int argc, char** argv)
{
    const int64_t experts = (argc > 1) ? std::atoll(argv[1]) : 8;
    const int32_t device = (argc > 2) ? std::atoi(argv[2]) : 0;
    const uint32_t blocks = (argc > 3) ? std::atoi(argv[3]) : 4;
    const int repeats = (argc > 4) ? std::atoi(argv[4]) : 20;
    const int rounds = (argc > 5) ? std::atoi(argv[5]) : 3;

    std::vector<uint8_t> gu_tiling_host(TILING_BYTES, 0);
    std::vector<uint8_t> down_tiling_host(TILING_BYTES, 0);
    const int gu_status = routed_build_tiling(ROWS, GATE_UP, HIDDEN, GATE_UP,
                                              gu_tiling_host.data());
    const int down_status = routed_build_tiling(ROWS, HIDDEN, HALF, HIDDEN,
                                                down_tiling_host.data());
    if (gu_status != 0 || down_status != 0) {
        std::printf("tiling returned %d and %d\n", gu_status, down_status);
        return 1;
    }

    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    uint64_t sync_base = 0;
    uint32_t sync_length = 0;
    const int32_t sync_status = rtGetC2cCtrlAddr(&sync_base, &sync_length);
    if (sync_status != 0) {
        std::printf("rtGetC2cCtrlAddr returned %d\n", sync_status);
        return 1;
    }

    std::vector<int8_t> rows_host(experts * ROWS * HIDDEN);
    std::vector<int8_t> gu_host(experts * GATE_UP * HIDDEN);
    std::vector<int8_t> down_host(experts * HIDDEN * HALF);
    std::vector<float> gu_scale_host(experts * GATE_UP);
    std::vector<float> down_scale_host(experts * HIDDEN);
    std::vector<float> token_host(experts * SCALE_STRIDE, 0.0F);
    for (size_t index = 0; index < rows_host.size(); ++index) {
        rows_host[index] = static_cast<int8_t>((index % 37) - 18);
    }
    for (size_t index = 0; index < gu_host.size(); ++index) {
        gu_host[index] = static_cast<int8_t>((index % 23) - 11);
    }
    for (size_t index = 0; index < down_host.size(); ++index) {
        down_host[index] = static_cast<int8_t>((index % 29) - 14);
    }
    for (size_t index = 0; index < gu_scale_host.size(); ++index) {
        gu_scale_host[index] = 0.00031F + 0.0000007F * (index % 256);
    }
    for (size_t index = 0; index < down_scale_host.size(); ++index) {
        down_scale_host[index] = 0.00047F + 0.0000003F * (index % 6144);
    }
    for (int64_t expert = 0; expert < experts; ++expert) {
        for (int64_t row = 0; row < ROWS; ++row) {
            token_host[expert * SCALE_STRIDE + row] =
                0.021F + 0.003F * static_cast<float>(row)
                + 0.0005F * static_cast<float>(expert);
        }
    }

    void* rows = Hold(rows_host.size(), "rows");
    void* gate_up = Hold(gu_host.size(), "gate up");
    void* down = Hold(down_host.size(), "down");
    void* gu_scale = Hold(gu_scale_host.size() * sizeof(float), "gu scale");
    void* down_scale = Hold(down_scale_host.size() * sizeof(float), "d scale");
    void* token = Hold(token_host.size() * sizeof(float), "token");
    void* partial = Hold(experts * ROWS * GATE_UP * sizeof(int32_t), "part");
    void* middle = Hold(experts * ROWS * HALF, "middle");
    void* middle_scale = Hold(experts * SCALE_STRIDE * sizeof(float), "mid s");
    void* wide = Hold(experts * ROWS * HIDDEN * sizeof(int32_t), "wide");
    void* out = Hold(experts * ROWS * HIDDEN * sizeof(uint16_t), "out");
    void* gu_tiling = Hold(TILING_BYTES, "gu tiling");
    void* down_tiling = Hold(TILING_BYTES, "down tiling");

    const auto upload = [&](void* where, const void* what, size_t bytes,
                            const char* name) {
        Check(aclrtMemcpy(where, bytes, what, bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE), name);
    };
    upload(rows, rows_host.data(), rows_host.size(), "rows");
    upload(gate_up, gu_host.data(), gu_host.size(), "gate up");
    upload(down, down_host.data(), down_host.size(), "down");
    upload(gu_scale, gu_scale_host.data(),
           gu_scale_host.size() * sizeof(float), "gu scale");
    upload(down_scale, down_scale_host.data(),
           down_scale_host.size() * sizeof(float), "down scale");
    upload(token, token_host.data(), token_host.size() * sizeof(float),
           "token");
    upload(gu_tiling, gu_tiling_host.data(), TILING_BYTES, "gu tiling");
    upload(down_tiling, down_tiling_host.data(), TILING_BYTES, "down tiling");

    // A sentinel separates a wrong value from an unwritten one.
    const uint16_t SENTINEL = 0x7777;
    const std::vector<uint16_t> sentinel(experts * ROWS * HIDDEN, SENTINEL);
    upload(out, sentinel.data(), sentinel.size() * sizeof(uint16_t), "out");

    routed_expert_launch(blocks, stream, sync_base, rows, gate_up,
                         (float*)gu_scale, down, (float*)down_scale,
                         (float*)token, partial, middle, (float*)middle_scale,
                         wide, out, gu_tiling, down_tiling, experts);
    Check(aclrtSynchronizeStream(stream), "once");

    std::vector<uint16_t> out_host(experts * ROWS * HIDDEN);
    Check(aclrtMemcpy(out_host.data(), out_host.size() * sizeof(uint16_t),
                      out, out_host.size() * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST), "download out");

    int64_t wrong = 0;
    int64_t untouched = 0;
    std::vector<float> activation(HALF);
    std::vector<int8_t> code(HALF);
    for (int64_t expert = 0; expert < experts; ++expert) {
        for (int64_t row = 0; row < ROWS; ++row) {
            float largest = 0.0F;
            for (int64_t column = 0; column < HALF; ++column) {
                int32_t gate_product = 0;
                int32_t up_product = 0;
                for (int64_t inner = 0; inner < HIDDEN; ++inner) {
                    const int32_t one =
                        rows_host[(expert * ROWS + row) * HIDDEN + inner];
                    gate_product += one * gu_host[
                        (expert * GATE_UP + column) * HIDDEN + inner];
                    up_product += one * gu_host[
                        (expert * GATE_UP + HALF + column) * HIDDEN + inner];
                }
                const float of_row =
                    token_host[expert * SCALE_STRIDE + row];
                const float gate = static_cast<float>(gate_product)
                                   * gu_scale_host[expert * GATE_UP + column]
                                   * of_row;
                const float up = static_cast<float>(up_product)
                                 * gu_scale_host[expert * GATE_UP + HALF
                                                 + column] * of_row;
                const float value = gate / (1.0F + std::exp(-gate)) * up;
                activation[column] = value;
                const float magnitude = std::fabs(value);
                largest = (magnitude > largest) ? magnitude : largest;
            }
            const float quantum = largest / 127.0F;
            for (int64_t column = 0; column < HALF; ++column) {
                code[column] = static_cast<int8_t>(
                    std::nearbyintf(ToHalf(activation[column] / quantum)));
            }
            for (int64_t column = 0; column < HIDDEN; ++column) {
                int32_t product = 0;
                for (int64_t inner = 0; inner < HALF; ++inner) {
                    product += code[inner] * down_host[
                        (expert * HIDDEN + column) * HALF + inner];
                }
                const float want = static_cast<float>(product)
                                   * down_scale_host[expert * HIDDEN + column]
                                   * quantum;
                const uint16_t got =
                    out_host[(expert * ROWS + row) * HIDDEN + column];
                untouched += (got == SENTINEL);
                const int32_t slack = static_cast<int32_t>(got)
                                      - static_cast<int32_t>(ToBf16(want));
                wrong += (slack > 1 || slack < -1);
            }
        }
    }
    const int64_t total = experts * ROWS * HIDDEN;
    std::printf("experts %lld  blocks %u  rows a expert %lld\n",
                (long long)experts, blocks, (long long)ROWS);
    std::printf("against the host reference, 1 ulp allowed: %lld of %lld "
                "(%lld untouched)\n", (long long)wrong, (long long)total,
                (long long)untouched);

    for (int round = 0; round < rounds; ++round) {
        timespec begin;
        timespec end;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        for (int step = 0; step < repeats; ++step) {
            routed_expert_launch(blocks, stream, sync_base, rows, gate_up,
                                 (float*)gu_scale, down, (float*)down_scale,
                                 (float*)token, partial, middle,
                                 (float*)middle_scale, wide, out, gu_tiling,
                                 down_tiling, experts);
        }
        Check(aclrtSynchronizeStream(stream), "timing");
        clock_gettime(CLOCK_MONOTONIC, &end);
        const double each = Elapsed(begin, end) / repeats;
        std::printf("round %d  %8.3f us for %lld experts, %6.3f us an "
                    "expert\n", round, each, (long long)experts,
                    each / static_cast<double>(experts));
    }

    Check(aclrtDestroyStream(stream), "destroy");
    Check(aclrtResetDevice(device), "reset");
    Check(aclFinalize(), "finalize");
    return (wrong == 0) ? 0 : 1;
}
