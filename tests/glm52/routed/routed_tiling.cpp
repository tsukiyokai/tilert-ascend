/**
 * The cube tilings, in their own translation unit.
 *
 * `matmul_tiling.h` is host-only and pulls in the operator registry's types,
 * so it stays away from the file `bisheng` compiles. Take the
 * `AscendC::tiling::TCubeTiling` overload of `GetTiling`: the `optiling` one
 * needs `TilingDef` symbols that live in `register`.
 */
#include <cstring>

#include "lib/matmul/matmul_tiling.h"

extern "C" int routed_build_tiling(int rows, int columns, int inner,
                                   int width, void* buffer)
{
    matmul_tiling::MatmulApiTiling tiling;
    tiling.SetAType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT8, false);
    tiling.SetBType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT8, true);
    tiling.SetCType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT32);
    tiling.SetShape(rows, columns, inner);
    tiling.SetOrgShape(rows, width, inner);
    tiling.SetBias(false);
    // Fixed tiles, because the cost is scalar and the scalar work is per
    // iteration inside `IterateAll`: fewer, larger tiles mean fewer address
    // computations. 128 by 512 of INT8 fills L0B exactly at 64 KB, which is
    // the corner -- measured against the library's own choice, 41.677 µs a
    // layer rather than 49.692, with 128 by 256 at 42.095, 256 by 128 at
    // 49.874 and 64 by 512 at 56.410. Equal compute per byte does not
    // predict it: [16,256,256] has the same cycle count and fewer bytes a
    // block, and measures 49.494. What decides is the iteration count, K
    // over baseK -- twelve passes at 512 against twenty-four at 256 -- which
    // is what a scalar-bound kernel pays for.
    tiling.SetFixSplit(16, 128, 512);
    tiling.SetBufferSpace(-1, -1, -1);

    AscendC::tiling::TCubeTiling cube;
    const int64_t status = tiling.GetTiling(cube);
    if (status != 0) {
        return static_cast<int>(status);
    }
    std::memcpy(buffer, &cube, sizeof(cube));
    return 0;
}
