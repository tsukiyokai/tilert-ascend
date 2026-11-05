/**
 * The cube tiling, in its own translation unit.
 *
 * `matmul_tiling.h` is host-only and pulls in the operator registry's types,
 * so it is kept away from the file `bisheng` compiles.
 */
#include <cstring>

#include "lib/matmul/matmul_tiling.h"

namespace {

constexpr int32_t GATE_UP = 256;
constexpr int32_t HIDDEN = 6144;

}  // namespace

extern "C" int mix_build_tiling(int rows, int columns, int inner,
                                void* buffer)
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
    // A block projects one contiguous run of channels at a time; the full
    // 256-wide shape is what the strides are against.
    // A block projects one contiguous run of output columns at a time; the
    // full output width is what the strides are against. The gate-and-up
    // projection is 6144 deep into 256 columns, the down projection 128 deep
    // into 6144.
    const int32_t width = (inner == HIDDEN) ? GATE_UP : HIDDEN;
    tiling.SetShape(rows, columns, inner);
    tiling.SetOrgShape(rows, width, inner);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1);

    AscendC::tiling::TCubeTiling cube;
    const int64_t status = tiling.GetTiling(cube);
    if (status != 0) {
        return static_cast<int>(status);
    }
    std::memcpy(buffer, &cube, sizeof(cube));
    return 0;
}
