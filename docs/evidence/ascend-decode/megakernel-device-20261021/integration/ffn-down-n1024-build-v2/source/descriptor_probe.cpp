#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>

#include <descriptor.h>

int main()
{
    std::array<void*, 6> weights;
    std::array<void*, 12> bridge;
    std::array<void*, 21> scratch;
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = reinterpret_cast<void*>(0x100000000ULL + i * 4096);
    }
    for (size_t i = 0; i < bridge.size(); ++i) {
        bridge[i] = reinterpret_cast<void*>(0x200000000ULL + i * 4096);
    }
    for (size_t i = 0; i < scratch.size(); ++i) {
        scratch[i] = reinterpret_cast<void*>(0x300000000ULL + i * 4096);
    }
    const uint64_t words[] = {
        101, 103, 107, 109, 113, 127, 131, 137, 139, 16, 149, 151, 157, 163};
    routed_int4::pipeline::Input shared;
    static_assert(sizeof(words) == sizeof(shared));
    std::memcpy(&shared, words, sizeof(shared));
    for (int64_t rows = 1; rows <= 32; ++rows) {
        for (int64_t blocks : {16, 24}) {
            const auto descriptor = ffn_down_n1024::MakeBodyArguments(
                weights.data(), bridge.data(), scratch.data(), shared,
                rows, blocks);
            for (size_t i = 0; i < weights.size(); ++i) {
                assert(descriptor.weights[i] ==
                       reinterpret_cast<uint64_t>(weights[i]));
            }
            for (size_t i = 0; i < scratch.size(); ++i) {
                assert(descriptor.scratch[i] ==
                       reinterpret_cast<uint64_t>(scratch[i]));
            }
            assert(descriptor.dispatch.bf16_rows ==
                   reinterpret_cast<uint64_t>(bridge[0]));
            assert(descriptor.dispatch.expert_ids ==
                   reinterpret_cast<uint64_t>(bridge[5]));
            assert(descriptor.dispatch.probabilities ==
                   reinterpret_cast<uint64_t>(bridge[4]));
            assert(descriptor.dispatch.active_list ==
                   reinterpret_cast<uint64_t>(scratch[20]));
            assert(descriptor.dispatch.rows == 0);
            assert(descriptor.dispatch.scales == 0);
            assert(descriptor.dispatch.inverse == 0);
            assert(descriptor.dispatch.counts == 0);
            assert(std::memcmp(&descriptor.shared, &shared, sizeof(shared)) == 0);
            assert(descriptor.rows == rows && descriptor.blocks == blocks);
        }
    }
}
