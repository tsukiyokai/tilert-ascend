/** Every INT8 from -128 to 127, split, against the exact decomposition. */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "split_probe_kernel.h"

namespace {

#ifndef SPAN_K
#define SPAN_K 256
#endif
constexpr int64_t SPAN = SPAN_K;
#ifndef LOW_AT
#define LOW_AT 256
#endif
constexpr int64_t LOW_OFFSET = LOW_AT;

void Check(aclError status, const char* what)
{
    if (status != ACL_SUCCESS) {
        std::printf("%s failed: %d\n", what, status);
        std::exit(2);
    }
}

}  // namespace

int main(int argc, char** argv)
{
    const int32_t device = (argc > 1) ? std::atoi(argv[1]) : 0;
    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    void* source = nullptr;
    void* high_out = nullptr;
    void* low_out = nullptr;
    void* trace = nullptr;
    void* packed = nullptr;
    Check(aclrtMalloc(&source, SPAN, ACL_MEM_MALLOC_HUGE_FIRST), "source");
    Check(aclrtMalloc(&high_out, SPAN * 4, ACL_MEM_MALLOC_HUGE_FIRST), "high");
    Check(aclrtMalloc(&low_out, SPAN * 4, ACL_MEM_MALLOC_HUGE_FIRST), "low");
    Check(aclrtMalloc(&trace, SPAN * 4, ACL_MEM_MALLOC_HUGE_FIRST), "trace");
    Check(aclrtMalloc(&packed, (LOW_OFFSET + SPAN) / 2,
                      ACL_MEM_MALLOC_HUGE_FIRST), "packed");

    std::vector<int8_t> source_host(SPAN);
    for (int64_t index = 0; index < SPAN; ++index) {
        source_host[index] = static_cast<int8_t>((index * 37 + 11) % 256
                                                 - 128);
    }
    Check(aclrtMemcpy(source, SPAN, source_host.data(), SPAN,
                      ACL_MEMCPY_HOST_TO_DEVICE), "upload");

    split_probe_launch(stream, source, high_out, low_out, trace, packed);
    Check(aclrtSynchronizeStream(stream), "synchronize");

    std::vector<int32_t> high(SPAN);
    std::vector<int32_t> low(SPAN);
    std::vector<float> floored(SPAN);
    Check(aclrtMemcpy(high.data(), SPAN * 4, high_out, SPAN * 4,
                      ACL_MEMCPY_DEVICE_TO_HOST), "down high");
    Check(aclrtMemcpy(low.data(), SPAN * 4, low_out, SPAN * 4,
                      ACL_MEMCPY_DEVICE_TO_HOST), "down low");
    Check(aclrtMemcpy(floored.data(), SPAN * 4, trace, SPAN * 4,
                      ACL_MEMCPY_DEVICE_TO_HOST), "down trace");

    std::vector<uint8_t> packed_host((LOW_OFFSET + SPAN) / 2);
    Check(aclrtMemcpy(packed_host.data(), packed_host.size(), packed,
                      packed_host.size(), ACL_MEMCPY_DEVICE_TO_HOST),
          "down packed");
    {
        // Which nibble of a byte holds the earlier element?
        const auto read = [&](int64_t index, bool low_first) {
            const uint8_t byte = packed_host[index / 2];
            const bool take_low = (index % 2 == 0) == low_first;
            const uint8_t four = take_low ? (byte & 0x0F) : (byte >> 4);
            return (int)((four >= 8) ? (int)four - 16 : (int)four);
        };
        // The low half cycles every element where the high half is constant
        // in runs of sixteen, so only the low half can tell the two nibble
        // conventions apart.
        int64_t as_low_first = 0;
        int64_t as_high_first = 0;
        for (int64_t index = 0; index < SPAN; ++index) {
            as_low_first += (read(LOW_OFFSET + index, true) == low[index]);
            as_high_first += (read(LOW_OFFSET + index, false) == low[index]);
        }
        std::printf("packed low matches in %lld of %lld "
                    "(other convention %lld)\n", (long long)as_low_first,
                    (long long)SPAN, (long long)as_high_first);
        std::printf("  first 8 bytes:");
        for (int64_t at = 0; at < 8; ++at) {
            std::printf(" %02x", packed_host[LOW_OFFSET / 2 + at]);
        }
        std::printf("\n  first 8 INT32 lows:");
        for (int64_t at = 0; at < 8; ++at) {
            std::printf(" %d", low[at]);
        }
        std::printf("\n");
    }

    int64_t wrong = 0;
    int64_t unrepresentable = 0;
    for (int64_t index = 0; index < SPAN; ++index) {
        const int32_t x = source_host[index];
        const int32_t want_low = ((x + 8) % 16 + 16) % 16 - 8;
        const int32_t want_high = (x - want_low) / 16;
        if (want_high < -8 || want_high > 7) {
            ++unrepresentable;
            continue;
        }
        if (high[index] != want_high || low[index] != want_low) {
            if (wrong < 6) {
                std::printf("x %4d  want %3d,%3d  got %3d,%3d  "
                            "floor((x+8)/16) %g\n", x, want_high, want_low,
                            high[index], low[index], floored[index]);
            }
            ++wrong;
        }
    }
    std::printf("%lld of %lld wrong, %lld beyond two signed INT4\n",
                (long long)wrong, (long long)SPAN,
                (long long)unrepresentable);

    Check(aclrtDestroyStream(stream), "destroy");
    Check(aclrtResetDevice(device), "reset");
    Check(aclFinalize(), "finalize");
    return (wrong == 0) ? 0 : 1;
}
