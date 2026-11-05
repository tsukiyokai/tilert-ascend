#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>

#include "overlap_contract.h"

int main()
{
    using namespace ffn_down_batch;
    static_assert(IsDownTiling(64, 256, 128, 64, 6144, 128));
    static_assert(!IsDownTiling(32, 256, 128, 64, 6144, 128));
    static_assert(!IsDownTiling(64, 128, 128, 64, 6144, 128));
    for (int rows = 1; rows <= 32; ++rows) {
        for (int pattern = 0; pattern < 4; ++pattern) {
            std::array<std::vector<int>, 256> routes;
            for (int token = 0; token < rows; ++token) {
                for (int slot = 0; slot < 8; ++slot) {
                    const int expert = pattern == 0 ? slot
                        : pattern == 1 ? 8 * token + slot
                        : pattern == 2 ? (token + slot) % 256
                        : (13 * token + 29 * slot) % 256;
                    routes[expert].push_back(8 * token + slot);
                }
            }
            std::array<int, 256> start{};
            std::vector<int> active;
            int prefix = 0;
            for (int expert = 0; expert < 256; ++expert) {
                start[expert] = prefix;
                prefix += routes[expert].size();
                assert(routes[expert].size() <= static_cast<size_t>(rows));
                if (!routes[expert].empty()) {
                    active.push_back(expert);
                    std::reverse(routes[expert].begin(), routes[expert].end());
                }
            }
            assert(prefix == 8 * rows);
            for (int blocks : {16, 24}) {
                std::vector<int> written(8 * rows * 48, 0);
                for (int core = 0; core < blocks; ++core) {
                    int produced = 0;
                    std::array<int, 2> consumed{};
                    for (size_t ordinal = core; ordinal < active.size(); ordinal += blocks) {
                        const int expert = active[ordinal];
                        const int begin = start[expert];
                        const int count = routes[expert].size();
                        const int64_t expert_first = 2LL * begin * kDownHidden;
                        const int64_t expert_last = 2LL * (begin + count) * kDownHidden;
                        for (int tile = 0; tile < kDownTilesPerExpert; ++tile) {
                            ++produced;
                            for (int sub = 0; sub < 2; ++sub) {
                                ++consumed[sub];
                                const int column = tile * kDownTile + sub * kDownHalfTile;
                                for (int row = 0; row < count; ++row) {
                                    const int64_t high = expert_first + row * kDownHidden + column;
                                    const int64_t low = high + count * kDownHidden;
                                    assert(high >= expert_first);
                                    assert(high + kDownHalfTile <= expert_first + count * kDownHidden);
                                    assert(low >= expert_first + count * kDownHidden);
                                    assert(low + kDownHalfTile <= expert_last);
                                    assert(expert_last <= 2LL * 8 * rows * kDownHidden);
                                    const int original = routes[expert][row];
                                    ++written[original * 48 + 2 * tile + sub];
                                }
                            }
                        }
                    }
                    assert(consumed[0] == produced && consumed[1] == produced);
                }
                assert(std::all_of(written.begin(), written.end(), [](int n) { return n == 1; }));
            }
        }
    }
}
