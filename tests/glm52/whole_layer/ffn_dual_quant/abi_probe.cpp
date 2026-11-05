#include <cstdio>
#include <initializer_list>
#include <type_traits>

#include "source/descriptor.h"

namespace candidate = FFN_NAMESPACE;
using Launch = void (*)(void*, uint64_t, void*, void*, void*, void*, void*,
                       void**, void**, void**, routed_int4::pipeline::Input,
                       int64_t, int64_t, int64_t, float);
static_assert(std::is_same_v<decltype(&FFN_LAUNCH), Launch>);
static_assert(sizeof(routed_int4::pipeline::Input) == 112);
static_assert(candidate::kBridgeBuffers + candidate::kBodyBuffers == 33);

int main()
{
    void* weights[6];
    void* bridge[12];
    void* body[21];
    for (uint64_t i = 0; i < 6; ++i) {
        weights[i] = reinterpret_cast<void*>(0x1000 + i * 32);
    }
    for (uint64_t i = 0; i < 12; ++i) {
        bridge[i] = reinterpret_cast<void*>(0x2000 + i * 32);
    }
    for (uint64_t i = 0; i < 21; ++i) {
        body[i] = reinterpret_cast<void*>(0x3000 + i * 32);
    }
    routed_int4::pipeline::Input shared{
        0x20c0, 0x20e0, 3, 4, 5, 6, 0x3220, 8, 9,
        32, 11, 12, 13, 14};
    for (const int rows : {1, 2, 12, 32}) {
        for (const int groups : {16, 24}) {
            auto args = candidate::MakeBodyArguments(
                weights, bridge, body, shared, rows, groups);
            static_assert(sizeof(args) == 408);
            const auto* bytes = reinterpret_cast<const unsigned char*>(&args);
            for (size_t i = 0; i < sizeof(args); ++i) {
                std::printf("%02x", bytes[i]);
            }
            std::puts("");
        }
    }
}
