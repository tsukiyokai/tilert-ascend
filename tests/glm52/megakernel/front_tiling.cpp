/**
 * The cube tiling, in its own translation unit.
 *
 * `matmul_tiling.h` is host-only and pulls in the operator registry's types,
 * so it stays away from the file `bisheng` compiles.
 */
#include <cstring>

#include "front_contract.h"
#include "lib/matmul/matmul_tiling.h"

// `base_m`, `base_n` and `base_k` fix the cube's tile; -1 leaves the choice to
// the tiling API, which at twelve rows beats every explicit tile measured.
extern "C" int megakernel_front_build_tiling(int rows, int columns, int width,
                                             int reduction, int base_m,
                                             int base_n, int base_k,
                                             void* buffer)
{
#ifdef MEGAKERNEL_FRONT_BF16
    constexpr auto element = matmul_tiling::DataType::DT_BF16;
    constexpr auto accumulator = matmul_tiling::DataType::DT_FLOAT;
#else
    constexpr auto element = matmul_tiling::DataType::DT_INT8;
    constexpr auto accumulator = matmul_tiling::DataType::DT_INT32;
#endif
    matmul_tiling::MatmulApiTiling tiling;
    tiling.SetAType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND, element, false);
    tiling.SetBType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND, element, false);
    tiling.SetCType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND, accumulator);
    tiling.SetShape(rows, columns, reduction);
    tiling.SetOrgShape(rows, width, reduction);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1);
    if (base_m > 0 && base_n > 0 && base_k > 0) {
        tiling.SetFixSplit(base_m, base_n, base_k);
    }

    AscendC::tiling::TCubeTiling cube;
    static_assert(sizeof(cube) <= megakernel_front::kTilingStride,
                  "Increase the tiling stride for this CANN version");
    const int64_t status = tiling.GetTiling(cube);
    if (status != 0) {
        return static_cast<int>(status);
    }
    std::memcpy(buffer, &cube, sizeof(cube));
    return 0;
}
