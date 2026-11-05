#include "attention_full_contract.h"
#include "descriptor.h"

#include <array>
#include <cassert>
#include <cstring>
#include <utility>

template <size_t... Indices>
shared_attention_arguments::Portable Capture(
    const std::array<uint64_t, 27>& values, uint64_t offset,
    std::index_sequence<Indices...>)
{
    return shared_attention_arguments::Capture<uint64_t>(values[Indices]..., offset);
}

int main()
{
    using namespace whole_layer_attention_descriptor;
    std::array<uint64_t, 27> values{};
    for (size_t index = 0; index < values.size(); ++index) {
        values[index] = 0x123456780000ULL + index * 0x10000000ULL;
    }
    for (uint32_t rows = 1; rows <= 32; ++rows) {
        for (uint32_t blocks = 1; blocks <= 24; ++blocks) {
            const auto config = MakeFullConfig(MakeConfig(
                rows, 66, 64, blocks, 0.1F, 1e-6F, 1e-6F));
            const auto descriptor = Capture(values, config.arena_offset,
                                             std::make_index_sequence<27>{});
            std::array<uint64_t, 29> actual{};
            std::memcpy(actual.data(), &descriptor, sizeof(descriptor));
            for (size_t index = 0; index < 20; ++index) {
                assert(actual[index] == values[index]);
            }
            assert(actual[20] == values[24] + config.arena_offset);
            assert(actual[21] == values[24]);
            for (size_t index = 20; index < values.size(); ++index) {
                assert(actual[index + 2] == values[index]);
            }
        }
    }
}
