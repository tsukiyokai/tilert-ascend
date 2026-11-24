#include <array>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "routed_int4_gu_splitk.h"

namespace {
using namespace routed_int4;

int64_t NZByteOffset(int64_t inner, int64_t column)
{
    return (((column / 64) * (kHidden / 16) + inner / 16) * 16
        + inner % 16) * 32 + (column % 64) / 2;
}

void CheckWeightSlices()
{
    for (int64_t part = 0; part < kGateParts; ++part) {
        const int64_t begin = part * kGatePartInner;
        for (int64_t inner = 0; inner < kGatePartInner; ++inner) {
            for (int64_t column = 0; column < kGateUp; ++column) {
                const auto nz = GatePartWeightOffset(part, true) / 2
                    + NZByteOffset(inner, column);
                assert(nz == NZByteOffset(begin + inner, column));
                const auto nd = GatePartWeightOffset(part, false) / 2
                    + (inner * kGateUp + column) / 2;
                assert(nd == ((begin + inner) * kGateUp + column) / 2);
                assert(nz < kHidden * kGateUp / 2);
                assert(nd < kHidden * kGateUp / 2);
            }
        }
    }
}

void CheckIntegerProducts()
{
    for (int row : {0, 1, 2, 7, 31, 63}) {
        for (int column : {0, 1, 63, 64, 127, 128, 255}) {
            int64_t full = 0;
            std::array<int32_t, kGateParts> parts{};
            for (int64_t inner = 0; inner < kHidden; ++inner) {
                const int32_t a = (row * 7 + inner * 3 + 5) % 16 - 8;
                const int32_t b = (column * 11 + inner * 13 + 9) % 16 - 8;
                full += int64_t(a) * b;
                parts[inner / kGatePartInner] += a * b;
            }
            const int32_t sum = (parts[0] + parts[1]) + parts[2];
            assert(sum == full);
            for (int32_t part : parts) {
                assert(std::abs(part) <= kGatePartInner * 64);
            }
        }
    }
    // Rounding each partial before summation changes an attainable dot:
    // 2049 = 32*(-8)*(-8) + 1; -2048 = 36*(-8)*7 + (-8)*4.
    std::fesetround(FE_TONEAREST);
    const std::array<int32_t, 3> witness{2049, -2048, 0};
    const int32_t exact = witness[0] + witness[1] + witness[2];
    const float prematurely_rounded = std::nearbyint(witness[0] / 2.0F) * 2.0F
        + std::nearbyint(witness[1] / 2.0F) * 2.0F;
    assert(exact == 1);
    assert(prematurely_rounded == 0.0F);
}

void CheckPlaneOwnership()
{
    for (int64_t tokens : {1, 2, 12, 31, 32}) {
        const int64_t routes = tokens * kTopK;
        const int64_t plane_rows = GateStorageRow(routes);
        const int64_t stride = GatePartStride(tokens, 24, 8);
        assert(stride == plane_rows * kGateUp);
        std::vector<int> produced(kGateParts * plane_rows);
        for (int64_t core = 0; core < 24; ++core) {
            const auto owner = GatePartFor(core, 8);
            for (int64_t row = 0; row < 2 * tokens; ++row) {
                const int64_t offset = owner.part * plane_rows
                    + 2 * owner.active_expert * tokens + row;
                ++produced.at(offset);
            }
        }
        for (int writes : produced) {
            assert(writes == 1);
        }
        std::vector<int> overwritten(plane_rows);
        for (int64_t worker = 0; worker < 48; ++worker) {
            for (int64_t route = worker; route < routes; route += 48) {
                const int64_t expert = route / tokens;
                const int64_t row = route % tokens;
                for (int64_t half : {0, 1}) {
                    const int64_t offset = 2 * expert * tokens + half * tokens + row;
                    for (int64_t part = 0; part < kGateParts; ++part) {
                        assert(produced.at(part * plane_rows + offset) == 1);
                    }
                    ++overwritten.at(offset);
                }
            }
        }
        for (int writes : overwritten) {
            assert(writes == 1);
        }
        for (int64_t blocks = 1; blocks <= 24; ++blocks) {
            for (int64_t active = 0; active <= 256; ++active) {
                const bool enabled = blocks == 24 && active > 0 && active <= 8;
                assert((GatePartStride(tokens, blocks, active) != 0) == enabled);
            }
        }
    }
}

}  // namespace

int main()
{
    CheckWeightSlices();
    CheckIntegerProducts();
    CheckPlaneOwnership();
    std::cout << "GU split-K address, integer rounding and ownership contracts passed\n";
}
