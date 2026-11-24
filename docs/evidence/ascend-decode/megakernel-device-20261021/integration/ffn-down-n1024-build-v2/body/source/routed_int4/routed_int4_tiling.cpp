/**
 * The cube tilings, in their own translation unit.
 *
 * `matmul_tiling.h` is host-only and pulls in the operator registry's types,
 * so it stays away from the file `bisheng` compiles. Take the
 * `AscendC::tiling::TCubeTiling` overload of `GetTiling`: the `optiling` one
 * needs `TilingDef` symbols that live in `register`.
 */
#include <cstring>

#include "routed_int4_kernel.h"
#include "../../../source/down_tiling_contract.h"
#include "lib/matmul/matmul_tiling.h"

extern "C" int routed_int4_build_tiling(int rows, int columns, int inner,
                                   int width, void* buffer, int64_t weight_layout)
{
    if (!routed_int4::IsWeightLayout(weight_layout)) {
        return -1;
    }
    matmul_tiling::MatmulApiTiling tiling;
    // All dimensions count INT4 elements, including NZ's 64-element C0.
    tiling.SetAType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT4, false);
    tiling.SetBType(matmul_tiling::TPosition::GM,
                    weight_layout == routed_int4::kWeightNZ
                        ? matmul_tiling::CubeFormat::NZ
                        : matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT4, false);
    tiling.SetCType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
#ifdef ROUTED_FIXPIPE_HALF
                    matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetDequantType(matmul_tiling::DequantType::TENSOR);
#else
                    matmul_tiling::DataType::DT_INT32);
#endif
    tiling.SetShape(rows, columns, inner);
    tiling.SetOrgShape(rows, width, inner);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1);

    const bool is_down = columns == routed_int4::kHidden &&
                         inner == routed_int4::kIntermediate;
    if (is_down) {
        if (rows != routed_int4::kMaxStackedRows || width != routed_int4::kHidden) {
            return -2;
        }
        const int split_status = tiling.SetFixSplit(32, 1024, 64);
        if (split_status != 0) {
            return split_status;
        }
    }

    AscendC::tiling::TCubeTiling cube;
    static_assert(sizeof(cube) <= routed_int4::kTilingBytes,
                  "Increase the shared tiling allocation for this CANN version");
    const int64_t status = tiling.GetTiling(cube);
    if (status != 0) {
        return static_cast<int>(status);
    }
    if (is_down && !ffn_down_n1024::IsExpectedDownTiling(cube)) {
        return -3;
    }
    std::memcpy(buffer, &cube, sizeof(cube));
    return 0;
}
