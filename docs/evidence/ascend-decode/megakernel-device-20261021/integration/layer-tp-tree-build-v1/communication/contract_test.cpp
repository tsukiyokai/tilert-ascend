#include "tree_plan.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

uint32_t Bits(float value)
{
    uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float FromBits(uint32_t value)
{
    float result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float Add(float left, float right)
{
    volatile float result = left + right;
    return result;
}

// This is the installed round recurrence, independent of the leaf program.
template <class Value, class Operation>
Value Rounds(std::vector<Value> values, Operation operation)
{
    while (values.size() > 1) {
        size_t split = 1;
        while (split * 2 < values.size()) {
            split *= 2;
        }
        for (size_t index = 0; index + split < values.size(); ++index) {
            values[index] = operation(values[index], values[index + split]);
        }
        values.resize(split);
    }
    return values.front();
}

template <class Value, class Operation>
Value Program(const std::vector<Value>& values, Operation operation,
              int* observed_peak = nullptr)
{
    const int ranks = static_cast<int>(values.size());
    const auto plan = whole_layer_tree::GetTreePlan(ranks);
    std::array<Value, 5> stack{};
    std::array<bool, 16> loaded{};
    int depth = 0;
    int peak = 0;
    int adds = 0;
    for (int leaf = 0; leaf < ranks; ++leaf) {
        const auto step = whole_layer_tree::GetTreeStep(plan, leaf);
        assert(step.rank < static_cast<uint32_t>(ranks));
        assert(!loaded[step.rank]);
        loaded[step.rank] = true;
        assert(depth < 5);
        stack[depth++] = values[step.rank];
        peak = std::max(peak, depth);
        for (uint32_t index = 0; index < step.reductions; ++index) {
            assert(depth >= 2);
            stack[depth - 2] = operation(stack[depth - 2], stack[depth - 1]);
            --depth;
            ++adds;
        }
    }
    assert(depth == 1 && adds == ranks - 1);
    for (int rank = 0; rank < ranks; ++rank) {
        assert(loaded[rank]);
    }
    for (int leaf = ranks; leaf < 16; ++leaf) {
        const auto step = whole_layer_tree::GetTreeStep(plan, leaf);
        assert(step.rank == 0 && step.reductions == 0);
    }
    if (observed_peak != nullptr) {
        *observed_peak = peak;
    }
    return stack[0];
}

}  // namespace

int main()
{
    static_assert(sizeof(whole_layer_tree::TreePlan) == 16);
    static_assert(2048 * (2 + 5 * 4) == 45056);
    std::mt19937 random(20261022);
    std::uniform_int_distribution<int> significand(-255, 255);
    std::uniform_int_distribution<int> exponent(-30, 30);
    int maximum_peak = 0;
    int random_cases = 0;
    int special_cases = 0;
    auto expression = [](const std::string& left, const std::string& right) {
        return "(" + left + "+" + right + ")";
    };
    for (int ranks = 2; ranks <= 16; ++ranks) {
        std::vector<std::string> leaves;
        for (int rank = 0; rank < ranks; ++rank) {
            leaves.push_back(std::to_string(rank));
        }
        int peak = 0;
        assert(Program(leaves, expression, &peak) == Rounds(leaves, expression));
        maximum_peak = std::max(maximum_peak, peak);
        for (int iteration = 0; iteration < 256; ++iteration) {
            std::vector<float> values;
            for (int rank = 0; rank < ranks; ++rank) {
                const float value = std::ldexp(static_cast<float>(significand(random)),
                                               exponent(random));
                assert((Bits(value) & 65535U) == 0);
                values.push_back(value);
            }
            assert(Bits(Program(values, Add)) == Bits(Rounds(values, Add)));
            ++random_cases;
        }
        // CPU semantics only: signed zeros, BF16 subnormals and finite maxima.
        // Device atomic/vector equivalence requires the separate device gate.
        constexpr uint32_t samples[] = {
            0, 0x80000000U, 0x00010000U, 0x80010000U, 0x007f0000U,
            0x00800000U, 0x7f7f0000U, 0xff7f0000U, 0x3f800000U,
            0xbf800000U, 0x33000000U, 0xb3000000U,
        };
        for (uint32_t left : samples) {
            for (uint32_t right : samples) {
                std::vector<float> values(ranks, 0.0F);
                values[0] = FromBits(left);
                values[ranks - 1] = FromBits(right);
                const float actual = Program(values, Add);
                const float expected = Rounds(values, Add);
                assert(Bits(actual) == Bits(expected));
                ++special_cases;
            }
        }
    }
    assert(maximum_peak == 5);
    std::vector<float> witness3{1.0F, std::ldexp(1.0F, -25), -1.0F};
    assert(Bits(Program(witness3, Add)) == 0x33000000U);
    std::vector<float> witness16(16, 0.0F);
    witness16[0] = 1.0F;
    witness16[1] = witness16[3] = std::ldexp(1.0F, -25);
    witness16[2] = -1.0F;
    assert(Bits(Program(witness16, Add)) == 0x33800000U);
    std::cout << "{\"pass\":true,\"rank_counts\":15,\"random_cases\":"
              << random_cases << ",\"special_cpu_cases\":" << special_cases
              << ",\"max_live_slots\":" << maximum_peak
              << ",\"chunk\":2048,\"ub_bytes\":45056}\n";
}
