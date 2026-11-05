#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "output_contract.h"

using namespace ffn_token_join;

int main()
{
    std::mt19937 random(20261022);
    for (int64_t rows = 1; rows <= kOutputMaximumRows; ++rows) {
        const int64_t routes = rows * kOutputTopK;
        for (int pattern = 0; pattern < 4; ++pattern) {
            std::vector<int32_t> experts(routes);
            for (int64_t row = 0; row < rows; ++row) {
                std::array<int32_t, 256> choices;
                std::iota(choices.begin(), choices.end(), 0);
                std::shuffle(choices.begin(), choices.end(), random);
                for (int64_t slot = 0; slot < kOutputTopK; ++slot) {
                    const int64_t original = row * kOutputTopK + slot;
                    experts[original] = pattern == 0 ? slot
                        : pattern == 1 ? original
                        : pattern == 2 ? (row + slot) % 9 : choices[slot];
                }
            }
            std::vector<int32_t> permutation(routes), inverse(routes);
            std::iota(permutation.begin(), permutation.end(), 0);
            std::shuffle(permutation.begin(), permutation.end(), random);
            std::stable_sort(permutation.begin(), permutation.end(),
                [&](int32_t left, int32_t right) {
                    return experts[left] < experts[right];
                });
            std::array<int32_t, 256> count{}, begin{};
            for (const auto expert : experts) {
                ++count[expert];
            }
            std::exclusive_scan(count.begin(), count.end(), begin.begin(), 0);
            for (int64_t route = 0; route < routes; ++route) {
                inverse[permutation[route]] = route;
            }
            std::vector<int32_t> dot_rows(2 * routes, 0);
            for (int64_t original = 0; original < routes; ++original) {
                const int64_t route = inverse[original];
                const int64_t expert = experts[original];
                const int64_t local = ExpertRow(route, begin[expert]);
                assert(permutation[route] == original);
                assert(local >= 0 && local < count[expert]);
                assert(count[expert] <= rows);
                const int64_t high = 2 * begin[expert] + local;
                const int64_t low = high + count[expert];
                assert(high >= 2 * begin[expert]);
                assert(low < 2 * (begin[expert] + count[expert]));
                ++dot_rows[high];
                ++dot_rows[low];
            }
            assert(std::all_of(dot_rows.begin(), dot_rows.end(),
                               [](int32_t visits) { return visits == 1; }));
            for (int64_t blocks : {16, 24}) {
                std::vector<int32_t> tasks(rows * kOutputTiles, 0);
                std::vector<int32_t> workers(2 * blocks, 0);
                for (int64_t worker = 0; worker < 2 * blocks; ++worker) {
                    for (int64_t task = worker; task < rows * kOutputTiles;
                         task += 2 * blocks) {
                        ++tasks[task];
                        ++workers[worker];
                        const int64_t token = OutputToken(task);
                        const int64_t column = OutputColumn(task);
                        assert(token >= 0 && token < rows);
                        assert(column >= 0 && column + kOutputTile <= kOutputHidden);
                        assert(column % kOutputTile == 0);
                        assert(OutputOwner(token, column / kOutputTile, blocks) == worker);
                    }
                }
                assert(std::all_of(tasks.begin(), tasks.end(),
                                   [](int32_t visits) { return visits == 1; }));
                assert(*std::max_element(workers.begin(), workers.end())
                       - *std::min_element(workers.begin(), workers.end()) <= 1);
                if (rows == 12 && blocks == 24) {
                    assert(std::all_of(workers.begin(), workers.end(),
                                       [](int32_t tasks) { return tasks == 1; }));
                }
            }
        }
    }
}
