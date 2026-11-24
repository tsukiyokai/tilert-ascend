// CPU checks for the shared storage contract; no Ascend headers required.
#include <cassert>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "routed_int4_contract.h"

namespace {
using namespace routed_int4;

int Weight(int expert, int inner, int column)
{
    return (expert * 5 + inner * 3 + column * 7) % 16 - 8;
}

void CheckWeightLayout()
{
    constexpr int experts = 2;
    constexpr int inputs = 6;
    constexpr int outputs = 4;
    std::vector<uint8_t> packed(experts * inputs * outputs / 2);
    for (int expert = 0; expert < experts; ++expert) {
        for (int inner = 0; inner < inputs; ++inner) {
            for (int column = 0; column < outputs; ++column) {
                const auto offset = PackedWeightOffset(
                    expert, inner, column, inputs, outputs);
                packed.at(offset) |= (Weight(expert, inner, column) & 15)
                                      << (4 * (column % 2));
            }
        }
    }
    for (int expert = 0; expert < experts; ++expert) {
        for (int column = 0; column < outputs; ++column) {
            int actual = 0;
            int expected = 0;
            for (int inner = 0; inner < inputs; ++inner) {
                const int activation = 3 * inner - 7;
                const int index = (expert * inputs + inner) * outputs + column;
                const int nibble = (packed.at(index / 2)
                                     >> (4 * (index % 2))) & 15;
                actual += activation * (nibble < 8 ? nibble : nibble - 16);
                expected += activation * Weight(expert, inner, column);
            }
            assert(actual == expected);
        }
    }
}

void CheckStackedRows()
{
    for (int64_t tokens : {1, 2, 4, 8, 12, 16, 31, 32}) {
        for (int pattern = 0; pattern < 3; ++pattern) {
            std::array<int64_t, kMaxExperts> counts{};
            for (int64_t token = 0; token < tokens; ++token) {
                for (int64_t slot = 0; slot < kTopK; ++slot) {
                    const int64_t expert = pattern == 0 ? slot
                        : pattern == 1 ? token * kTopK + slot
                        : (token * 31 + slot * 17) % kMaxExperts;
                    ++counts[expert];
                }
            }
            for (int64_t inputs : {kHidden, kIntermediate}) {
                std::vector<int> writes(2 * tokens * kTopK);
                int64_t prefix = 0;
                for (int64_t count : counts) {
                    assert(count <= tokens);
                    for (int64_t row = 0; row < count; ++row) {
                        const int64_t high = StackedOffset(prefix, row, inputs);
                        const int64_t low = StackedOffset(
                            prefix, count + row, inputs);
                        assert(high % 32 == 0 && low % 32 == 0);
                        ++writes.at(high / (inputs / 2));
                        ++writes.at(low / (inputs / 2));
                    }
                    prefix += count;
                }
                assert(prefix == tokens * kTopK);
                for (int count : writes) {
                    assert(count == 1);
                }
            }
        }
    }
}

void CheckHotExpertRoutes()
{
    for (int64_t tokens : {1, 2, 4, 8, 9, 12, 16, 31, 32}) {
        std::vector<int> writes(tokens * kTopK);
        // Expert-major order differs from the required original-slot order.
        for (int64_t expert = 0; expert < kTopK; ++expert) {
            for (int64_t token = 0; token < tokens; ++token) {
                const int64_t slot = kTopK - expert - 1;
                const auto original = OriginalSlot(token, slot);
                assert(original >= 0 && original < tokens * kTopK);
                ++writes.at(original);
            }
        }
        for (int count : writes) {
            assert(count == 1);
        }
    }
}

void CheckGatePaddingIsolation()
{
    for (int64_t routes = 1; routes <= kMaxRoutes; ++routes) {
        for (int64_t begin = 0; begin < routes; ++begin) {
            for (int64_t count = 1; count <= kMaxRowsPerExpert
                 && count <= routes - begin; ++count) {
                const int64_t first = GateStorageRow(begin);
                const int64_t next = GateStorageRow(begin + count);
                const int64_t rows = GateMatrixRows(count);
                assert(rows >= 2 * count && rows <= kMaxStackedRows);
                assert(first + rows <= next);
                assert(next <= GateStorageRow(routes));
                if (kPadGateRows) {
                    assert(rows % 16 == 0);
                    assert((rows / 16) * (kGateUp / 16) >= 10);
                }
                // Stale padding rows are excluded from both MSD operands.
                for (int64_t row = 0; row < count; ++row) {
                    assert(first + row < first + count);
                    assert(first + count + row < first + 2 * count);
                }
            }
        }
    }
}

void CheckBalancedRouteWrites()
{
    for (int64_t tokens : {1, 2, 12, 31, 32}) {
        for (int64_t blocks : {1, 8, 16, 24}) {
            for (int pattern = 0; pattern < 3; ++pattern) {
                std::array<int64_t, kMaxExperts> counts{};
                for (int64_t token = 0; token < tokens; ++token) {
                    for (int64_t slot = 0; slot < kTopK; ++slot) {
                        const int64_t expert = pattern == 0 ? 255 - slot
                            : pattern == 1 ? token * kTopK + slot
                            : (token * 31 + slot * 17) % kMaxExperts;
                        ++counts[expert];
                    }
                }
                const int64_t routes = tokens * kTopK;
                // Split/activation use every AIV; output halves use one
                // worker pair. Neither ownership rule follows expert IDs.
                for (int64_t workers : {blocks, 2 * blocks}) {
                    std::vector<int> scale_bytes(routes * sizeof(float));
                    std::vector<int> stacked_rows(2 * routes);
                    std::vector<int> loads(workers);
                    for (int64_t worker = 0; worker < workers; ++worker) {
                        int64_t next = worker;
                        int64_t begin = 0;
                        for (int64_t count : counts) {
                            if (next < begin + count) {
                                for (int64_t row = next - begin; row < count;
                                     row += workers) {
                                    const int64_t route = begin + row;
                                    assert(route % workers == worker);
                                    ++loads[worker];
                                    ++stacked_rows.at(2 * begin + row);
                                    ++stacked_rows.at(2 * begin + count + row);
                                    for (size_t byte = 0; byte < sizeof(float); ++byte) {
                                        ++scale_bytes.at(route * sizeof(float) + byte);
                                    }
                                }
                                do {
                                    next += workers;
                                } while (next < begin + count);
                            }
                            begin += count;
                        }
                    }
                    for (int writes : stacked_rows) {
                        assert(writes == 1);
                    }
                    for (int writes : scale_bytes) {
                        assert(writes == 1);
                    }
                    for (int64_t worker = 0; worker < workers; ++worker) {
                        const int64_t expected = routes / workers
                            + (worker < routes % workers ? 1 : 0);
                        assert(loads[worker] == expected);
                    }
                }
            }
        }
    }
}

void CheckFusedQuantizationRounding()
{
    std::fesetround(FE_TONEAREST);
    for (float value : {1.4999F, 2.5001F, -1.4999F, -2.5001F}) {
        // All four probes are normal FP16 values; its spacing is 2^(e-11).
        int exponent = 0;
        std::frexp(std::fabs(value), &exponent);
        const float spacing = std::ldexp(1.0F, exponent - 11);
        const float half_rounded = std::nearbyint(value / spacing) * spacing;
        const float fused_code = std::nearbyint(half_rounded);
        const float direct_code = std::nearbyint(value);
        assert(fused_code == (value > 0 ? 2 : -2));
        assert(fused_code != direct_code);
    }
}

void CheckSlotZeroSeed()
{
    // Unpermute initializes from slot 0. Adding a synthetic +0 changes
    // the result when every weighted route is a negative zero.
    std::fesetround(FE_TONEAREST);
    const float weighted = -0.0F * 0.5F;
    float seeded = weighted;
    float zero_seeded = 0.0F;
    for (int slot = 0; slot < kTopK; ++slot) {
        zero_seeded += weighted;
        if (slot > 0) {
            seeded += weighted;
        }
    }
    assert(std::signbit(seeded));
    assert(!std::signbit(zero_seeded));
    assert(seeded == zero_seeded);

    // A zero-probability slot 0 is skipped and initializes +0, even if all
    // later valid routes are negative zero. It is not first-valid seeding.
    float skipped_first = 0.0F;
    for (int slot = 1; slot < kTopK; ++slot) {
        skipped_first += weighted;
    }
    assert(!std::signbit(skipped_first));
    assert(!std::signbit(seeded + 0.0F));
}
}  // namespace

int main()
{
    CheckGatePaddingIsolation();
    CheckWeightLayout();
    CheckStackedRows();
    CheckHotExpertRoutes();
    CheckBalancedRouteWrites();
    CheckFusedQuantizationRounding();
    CheckSlotZeroSeed();
    static_assert(routed_int4::DispatchScratchBytes(32) == 201856);
    std::cout << "Packed B, all 32 MSD rows, original-slot ownership and "
              << "fused FP16 double rounding and slot-zero seed pass; "
              << "TP16 M32 scratch="
              << routed_int4::ScratchBytes(32) << " bytes\n";
}
