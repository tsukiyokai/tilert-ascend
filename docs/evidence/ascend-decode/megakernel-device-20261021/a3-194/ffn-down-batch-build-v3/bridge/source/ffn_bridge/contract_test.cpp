#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

#include "contract.h"

int main()
{
    using namespace ffn_bridge;
    int cases = 0;
    for (int64_t blocks : {16, 24}) {
        for (int64_t rows = 1; rows <= kMaximumRows; ++rows) {
            std::array<int, kMaximumRows> writes{};
            for (int64_t vector = 0; vector < 2 * blocks; ++vector) {
                if (vector < rows) {
                    assert(RowOwner(vector, blocks) == vector);
                    ++writes[vector];
                }
            }
            for (int64_t row = 0; row < rows; ++row) {
                assert(writes[row] == 1);
            }
            std::array<int, kExperts> columns{};
            for (int64_t core = 0; core < blocks; ++core) {
                for (int64_t tile = core; tile < kExperts / kGateTile; tile += blocks) {
                    for (int64_t column = tile * kGateTile;
                         column < (tile + 1) * kGateTile; ++column) {
                        ++columns[column];
                        const auto last_b = (kHidden - 1) * kExperts + column;
                        const auto last_c = (rows - 1) * kExperts + column;
                        assert(last_b < kHidden * kExperts);
                        assert(last_c < rows * kExperts);
                    }
                }
            }
            for (int writes : columns) {
                assert(writes == 1);
            }
            const auto explicit_bytes = 4 * rows * kHidden * 2
                + rows * kHidden + rows * kExperts * 4 + 2 * rows * 4
                + 2 * rows * kTopK * 4 + kTilingBytes + kSystemBytes;
            assert(ScratchBytes(rows) == explicit_bytes);
            ++cases;
        }
    }
    assert(!ValidGeometry(0, 24) && !ValidGeometry(33, 24));
    assert(!ValidGeometry(12, 8));
    std::cout << "64 geometry cases passed; M32 scratch bytes "
              << ScratchBytes(32) << '\n';
}
