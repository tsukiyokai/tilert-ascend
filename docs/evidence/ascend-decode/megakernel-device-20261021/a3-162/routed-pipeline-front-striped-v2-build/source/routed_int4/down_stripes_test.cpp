// Address and ownership checks for full-stride INT4 down-projection slices.
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include "routed_int4_down_stripes.h"

namespace {
using namespace routed_int4;

int64_t NZByteOffset(int64_t inner, int64_t column)
{
    return (((column / 64) * (kIntermediate / 16) + inner / 16) * 16
        + inner % 16) * 32 + (column % 64) / 2;
}

void CheckWeightSlices()
{
    // Check every byte and nibble in each N2048 slice against the full
    // physical matrix. The final K row and final stripe reach the bank end.
    for (bool nz : {false, true}) {
        std::vector<uint8_t> weights(kIntermediate * kHidden / 2);
        uint32_t state = 0x6d2b79f5;
        for (auto& byte : weights) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            byte = static_cast<uint8_t>(state);
        }
        for (int64_t stripe = 0; stripe < kDownStripes; ++stripe) {
            const int64_t column = stripe * kDownStripeColumns;
            const int64_t base = DownWeightColumnOffset(column, nz) / 2;
            for (int64_t inner = 0; inner < kIntermediate; ++inner) {
                for (int64_t local = 0; local < kDownStripeColumns; ++local) {
                    const int64_t full = nz ? NZByteOffset(inner, column + local)
                        : (inner * kHidden + column + local) / 2;
                    const int64_t sliced = base + (nz ? NZByteOffset(inner, local)
                        : (inner * kHidden + local) / 2);
                    assert(full == sliced);
                    const int shift = 4 * (local % 2);
                    assert(((weights.at(full) >> shift) & 15)
                           == ((weights.at(sliced) >> shift) & 15));
                }
            }
        }
    }
}

void CheckOutputOwnership()
{
    for (int64_t active : {1, 3, 8}) {
        for (int64_t count : {1, 2, 12, 31, 32}) {
            const int64_t rows = 2 * active * count;
            std::vector<uint8_t> writes(rows * kHidden);
            for (int64_t core = 0; core < kDownStripeBlocks; ++core) {
                const auto stripe = DownStripeFor(core, active);
                if (stripe.active_expert < 0) {
                    assert(core >= active * kDownStripes);
                    continue;
                }
                const int64_t expert_base = 2 * stripe.active_expert * count
                    * kHidden;
                for (int64_t row = 0; row < 2 * count; ++row) {
                    const int64_t first = expert_base + row * kHidden
                        + stripe.column;
                    for (int64_t column = 0; column < kDownStripeColumns;
                         ++column) {
                        ++writes.at(first + column);
                    }
                }
            }
            for (const auto count_writes : writes) {
                assert(count_writes == 1);
            }
        }
    }
}

void CheckEligibility()
{
    for (int64_t blocks = 1; blocks <= 24; ++blocks) {
        for (int64_t active = 0; active <= 256; ++active) {
            const bool expected = blocks == 24 && active >= 1 && active <= 8;
            assert(DownStripesEnabled(blocks, active) == expected);
        }
    }
    // Each high/low dot is bounded by K128 * (-8)*(-8), so independent
    // INT32 outputs cannot overflow or change any subsequent rounding.
    static_assert(kIntermediate * 64 == 8192);
    static_assert(kDownStripeColumns * kDownStripes == kHidden);
}

}  // namespace

int main()
{
    CheckWeightSlices();
    CheckOutputOwnership();
    CheckEligibility();
    std::cout << "Down stripe address, ownership and fallback contracts passed\n";
}
