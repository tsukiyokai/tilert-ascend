#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include "split_contract.h"

namespace {

constexpr int kExperts = 256;
using Slots = std::array<int, 8>;

struct Routes {
    std::vector<Slots> choices;
    std::array<std::vector<int>, kExperts> groups;
    std::array<int, kExperts> counts{};
    std::array<int, kExperts> starts{};
    std::vector<int> sorted_original;
};

Routes MakeRoutes(int rows, int pattern, std::mt19937& generator)
{
    Routes result;
    result.choices.resize(rows);
    std::array<int, kExperts> experts;
    std::iota(experts.begin(), experts.end(), 0);
    for (int token = 0; token < rows; ++token) {
        if (pattern >= 2) {
            std::shuffle(experts.begin(), experts.end(), generator);
        }
        for (int slot = 0; slot < 8; ++slot) {
            const int expert = pattern == 0 ? slot : pattern == 1
                ? (token * 8 + slot) % kExperts : experts[slot];
            result.choices[token][slot] = expert;
            result.groups[expert].push_back(token * 8 + slot);
        }
    }
    int prefix = 0;
    for (int expert = 0; expert < kExperts; ++expert) {
        auto& group = result.groups[expert];
        // Sorting within an expert is intentionally not stable.
        std::shuffle(group.begin(), group.end(), generator);
        result.starts[expert] = prefix;
        result.counts[expert] = int(group.size());
        prefix += int(group.size());
        result.sorted_original.insert(result.sorted_original.end(),
                                      group.begin(), group.end());
    }
    return result;
}

bool ValidMetadata(const Routes& routes)
{
    const int rows = int(routes.choices.size());
    const int total = rows * 8;
    if (int(routes.sorted_original.size()) != total) {
        return false;
    }
    std::vector<int> seen(total, 0);
    int prefix = 0;
    for (int expert = 0; expert < kExperts; ++expert) {
        const int count = routes.counts[expert];
        if (routes.starts[expert] != prefix || count < 0 || count > rows
            || prefix + count > total) {
            return false;
        }
        for (int index = prefix; index < prefix + count; ++index) {
            const int original = routes.sorted_original[index];
            if (original < 0 || original >= total || seen[original]++ != 0
                || routes.choices[original / 8][original % 8] != expert) {
                return false;
            }
        }
        prefix += count;
    }
    return prefix == total;
}

void CheckPartition(const Routes& routes, int blocks)
{
    using namespace ffn_down_k128;
    assert(ValidMetadata(routes));
    const int rows = int(routes.choices.size());
    const int total = rows * 8;
    std::vector<int> inverse(total, -1);
    for (int route = 0; route < total; ++route) {
        inverse[routes.sorted_original[route]] = route;
    }
    using Signature = std::tuple<int, int, int>;
    const Signature unwritten{-1, -1, -1};
    std::vector<Signature> expected(2 * total * 3, unwritten);
    std::vector<Signature> actual(expected.size(), unwritten);
    std::vector<int> writes(expected.size(), 0);

    // Independent serial expert-major specification, not the new owner formula.
    int base = 0;
    for (const auto& group : routes.groups) {
        for (int row = 0; row < int(group.size()); ++row) {
            for (int half = 0; half < 2; ++half) {
                for (int tile = 0; tile < 3; ++tile) {
                    const int destination = 2 * base + half * int(group.size()) + row;
                    expected[destination * 3 + tile] =
                        {group[row] / 8, half, tile};
                }
            }
        }
        base += int(group.size());
    }
    for (int worker = 0; worker < 2 * blocks; ++worker) {
        for (int task = worker; task < rows * 3; task += 2 * blocks) {
            assert(SplitOwner(task, blocks) == worker);
            const int token = int(SplitToken(task));
            const int column = int(SplitColumn(task));
            assert(column >= 0 && column + kSplitTile <= kSplitHidden);
            assert(column % 256 == 0);
            for (int slot = 0; slot < 8; ++slot) {
                const int route = inverse[token * 8 + slot];
                const int expert = routes.choices[token][slot];
                const int begin = routes.starts[expert];
                const int count = routes.counts[expert];
                for (int half = 0; half < 2; ++half) {
                    const int destination = int(StackedRow(begin, count, route, half));
                    assert(destination >= 2 * begin && destination < 2 * (begin + count));
                    const int index = destination * 3 + column / int(kSplitTile);
                    assert(++writes[index] == 1);
                    actual[index] = {token, half, column / int(kSplitTile)};
                }
            }
        }
    }
    assert(actual == expected);
    assert(std::all_of(writes.begin(), writes.end(), [](int count) { return count == 1; }));
}

int Nibble(int code)
{
    return code & 15;
}

void CheckEveryPackedPair()
{
    for (int first = -128; first < 128; ++first) {
        for (int second = -128; second < 128; ++second) {
            std::array<int, 2> high{}, low{};
            const int values[] = {first, second};
            for (int element = 0; element < 2; ++element) {
                const float value = float(values[element]);
                const float h = std::floor(value * (1.0F / 16.0F));
                float l = h * -16.0F;
                l = l + value;
                l = l + -8.0F;
                high[element] = int(h);
                low[element] = int(l);
                assert(high[element] >= -8 && high[element] <= 7);
                assert(low[element] >= -8 && low[element] <= 7);
                assert(16 * high[element] + low[element] + 8 == values[element]);
            }
            const int high_byte = Nibble(high[0]) | (Nibble(high[1]) << 4);
            const int low_byte = Nibble(low[0]) | (Nibble(low[1]) << 4);
            assert(high_byte == (((first & 255) >> 4) | (((second & 255) >> 4) << 4)));
            assert(low_byte == (((first ^ 8) & 15) | (((second ^ 8) & 15) << 4)));
        }
    }
}

}  // namespace

int main()
{
    std::mt19937 generator(20261022);
    int configurations = 0;
    for (int rows = 1; rows <= 32; ++rows) {
        for (int pattern = 0; pattern < 34; ++pattern) {
            const auto routes = MakeRoutes(rows, pattern, generator);
            for (int blocks : {16, 24}) {
                CheckPartition(routes, blocks);
                ++configurations;
            }
        }
    }
    auto invalid = MakeRoutes(12, 0, generator);
    invalid.sorted_original[1] = invalid.sorted_original[0];
    assert(!ValidMetadata(invalid));
    invalid = MakeRoutes(12, 1, generator);
    invalid.sorted_original[0] = -1;
    assert(!ValidMetadata(invalid));
    invalid = MakeRoutes(12, 1, generator);
    std::swap(invalid.sorted_original[0], invalid.sorted_original[8]);
    assert(!ValidMetadata(invalid));
    CheckEveryPackedPair();
    std::cout << "{\"ownership_configurations\":" << configurations
              << ",\"packed_pairs\":65536,\"pass\":true}\n";
}
