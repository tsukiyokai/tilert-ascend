/** Host-only inspection of the exact routed MatmulApiTiling result. */
#include <array>
#include <cstring>
#include <iostream>

#include "routed_int4_kernel.h"
#include "lib/matmul/matmul_tiling.h"

int main()
{
    for (int64_t layout : {routed_int4::kWeightND, routed_int4::kWeightNZ}) {
        for (int projection : {0, 1}) {
            const int columns = projection == 0
                ? routed_int4::kGateUp : routed_int4::kHidden;
            const int inner = projection == 0
                ? routed_int4::kHidden : routed_int4::kIntermediate;
            std::array<uint8_t, routed_int4::kTilingBytes> bytes{};
            if (routed_int4_build_tiling(routed_int4::kMaxStackedRows,
                                         columns, inner, columns,
                                         bytes.data(), layout) != 0) {
                return 1;
            }
            AscendC::tiling::TCubeTiling cube;
            std::memcpy(&cube, bytes.data(), sizeof(cube));
            std::cout << "{\"layout\":" << layout
                      << ",\"projection\":" << projection
                      << ",\"M\":" << cube.M << ",\"N\":" << cube.N
                      << ",\"Ka\":" << cube.Ka << ",\"Kb\":" << cube.Kb
                      << ",\"singleCoreM\":" << cube.singleCoreM
                      << ",\"singleCoreN\":" << cube.singleCoreN
                      << ",\"singleCoreK\":" << cube.singleCoreK
                      << ",\"baseM\":" << cube.baseM
                      << ",\"baseN\":" << cube.baseN
                      << ",\"baseK\":" << cube.baseK
                      << ",\"depthA1\":" << cube.depthA1
                      << ",\"depthB1\":" << cube.depthB1
                      << ",\"stepM\":" << cube.stepM
                      << ",\"stepN\":" << cube.stepN
                      << ",\"stepKa\":" << cube.stepKa
                      << ",\"stepKb\":" << cube.stepKb
                      << ",\"iterateOrder\":" << cube.iterateOrder
                      << ",\"shareL1Size\":" << cube.shareL1Size
                      << ",\"shareL0CSize\":" << cube.shareL0CSize
                      << ",\"shareUbSize\":" << cube.shareUbSize
                      << ",\"dbL0A\":" << cube.dbL0A
                      << ",\"dbL0B\":" << cube.dbL0B
                      << ",\"dbL0C\":" << cube.dbL0C
                      << "}\n";
        }
    }
}
