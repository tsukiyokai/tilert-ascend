#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include "routed_int4_active_list.h"

namespace {
using namespace routed_int4;
using Counts = std::array<int32_t, kMaxExperts>;
using Call = std::tuple<int64_t, int64_t, int64_t, int64_t>;

struct List {
    std::array<int32_t, kActiveListElements> values;

    void SetValue(int64_t index, int32_t value)
    {
        assert(index >= 0 && index < kActiveListElements);
        values[index] = value;
    }
};

void BuildList(List& list, const Counts& counts)
{
    list.values.fill(0);
    int32_t size = 0;
    for (int64_t expert = 0; expert < kMaxExperts; ++expert) {
        size = AppendActiveExpert(list, size, expert, counts[expert]);
    }
    list.SetValue(0, size);
}

std::vector<int64_t> ExpertIds(const List& list)
{
    const auto size = list.values[0];
    assert(size >= 0 && size <= kMaxExperts);
    return {list.values.begin() + kActiveHeaderElements,
            list.values.begin() + kActiveHeaderElements + size};
}

std::vector<Call> MatrixCalls(const Counts& counts, const Counts& starts,
                              int64_t core, int64_t blocks)
{
    std::vector<Call> calls;
    int64_t ordinal = 0;
    int64_t next = core;
    for (int64_t expert = 0; expert < kMaxExperts; ++expert) {
        if (counts[expert] == 0) {
            continue;
        }
        if (ordinal++ != next) {
            continue;
        }
        next += blocks;
        calls.emplace_back(expert, counts[expert], starts[expert], 0);
    }
    return calls;
}

std::vector<Call> RowCalls(const Counts& counts, const Counts& starts,
                           const std::vector<int64_t>& experts,
                           int64_t worker, int64_t workers)
{
    std::vector<Call> calls;
    int64_t next = worker;
    for (int64_t expert : experts) {
        const int64_t begin = starts[expert];
        const int64_t count = counts[expert];
        if (next >= begin + count) {
            continue;
        }
        assert(next >= begin);
        calls.emplace_back(expert, count, begin, next - begin);
        do {
            next += workers;
        } while (next < begin + count);
    }
    return calls;
}

int64_t CheckOwners(List& list, const Counts& counts, int64_t tokens)
{
    BuildList(list, counts);
    const auto active = ExpertIds(list);
    std::vector<int64_t> all(kMaxExperts);
    std::iota(all.begin(), all.end(), 0);
    Counts starts{};
    int32_t prefix = 0;
    for (int64_t expert = 0; expert < kMaxExperts; ++expert) {
        assert(counts[expert] >= 0 && counts[expert] <= tokens);
        starts[expert] = prefix;
        prefix += counts[expert];
        assert(std::count(active.begin(), active.end(), expert)
               == (counts[expert] != 0));
    }
    assert(prefix == tokens * kTopK);
    assert(std::is_sorted(active.begin(), active.end()));
    for (int64_t index = 1; index < kActiveHeaderElements; ++index) {
        assert(list.values[index] == 0);
    }
    for (int64_t index = kActiveHeaderElements + active.size();
         index < kActiveListElements; ++index) {
        assert(list.values[index] == 0);
    }

    int64_t checks = 0;
    for (int64_t blocks : {16, 24}) {
        for (int64_t core = 0; core < blocks; ++core) {
            std::vector<Call> candidate;
            for (int64_t ordinal = core; ordinal < int64_t(active.size());
                 ordinal += blocks) {
                const auto expert = active[ordinal];
                candidate.emplace_back(expert, counts[expert], starts[expert], 0);
            }
            assert(candidate == MatrixCalls(counts, starts, core, blocks));
            ++checks;
        }
        for (int64_t workers : {blocks, 2 * blocks}) {
            std::vector<int> owners(tokens * kTopK);
            for (int64_t worker = 0; worker < workers; ++worker) {
                const auto candidate = RowCalls(counts, starts, active,
                                                worker, workers);
                assert(candidate == RowCalls(counts, starts, all,
                                              worker, workers));
                for (auto [expert, count, begin, first] : candidate) {
                    for (int64_t row = first; row < count; row += workers) {
                        ++owners[begin + row];
                    }
                }
                ++checks;
            }
            assert(std::all_of(owners.begin(), owners.end(),
                               [](int count) { return count == 1; }));
        }
    }
    return checks;
}

}  // namespace

int main()
{
    List list;
    std::mt19937 random(41021);
    int64_t cases = 0;
    int64_t checks = 0;
    for (int64_t tokens = 1; tokens <= kMaxTokens; ++tokens) {
        for (int pattern = 0; pattern < 16; ++pattern) {
            Counts counts{};
            for (int64_t token = 0; token < tokens; ++token) {
                std::array<int32_t, kMaxExperts> choices;
                std::iota(choices.begin(), choices.end(), 0);
                if (pattern >= 4) {
                    std::shuffle(choices.begin(), choices.end(), random);
                }
                for (int64_t slot = 0; slot < kTopK; ++slot) {
                    int64_t expert = choices[slot];
                    if (pattern == 1) {
                        expert = kMaxExperts - kTopK + slot;
                    } else if (pattern == 2 || pattern == 3) {
                        expert = (token * kTopK + slot + (pattern == 3 ? 127 : 0))
                            % kMaxExperts;
                    }
                    ++counts[expert];
                }
            }
            checks += CheckOwners(list, counts, tokens);
            ++cases;
        }
    }
    // Reuse one descriptor for the maximum union followed by a hot union.
    Counts dense;
    dense.fill(1);
    checks += CheckOwners(list, dense, 32);
    Counts hot{};
    std::fill(hot.end() - kTopK, hot.end(), 32);
    checks += CheckOwners(list, hot, 32);
    cases += 2;
    std::cout << "{\"cases\":" << cases
              << ",\"per_core_exact_call_sequence_checks\":" << checks
              << ",\"scratch_bytes\":" << kActiveListBytes
              << ",\"device_execution\":false}\n";
}
