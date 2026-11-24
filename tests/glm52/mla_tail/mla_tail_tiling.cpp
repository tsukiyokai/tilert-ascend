#include <cstring>

#include "lib/matmul/matmul_tiling.h"
#include "mla_tail.h"

namespace {

int BuildProjection(int rows, int columns, int inner,
                    matmul_tiling::DataType operand_type,
                    matmul_tiling::DataType output_type, void* buffer)
{
    matmul_tiling::MatmulApiTiling tiling;
    tiling.SetAType(matmul_tiling::TPosition::GM,
                   matmul_tiling::CubeFormat::ND, operand_type, false);
    tiling.SetBType(matmul_tiling::TPosition::GM,
                   matmul_tiling::CubeFormat::ND, operand_type, false);
    tiling.SetCType(matmul_tiling::TPosition::GM,
                   matmul_tiling::CubeFormat::ND, output_type);
    tiling.SetShape(rows, columns, inner);
    tiling.SetOrgShape(rows, columns, inner);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1);
    AscendC::tiling::TCubeTiling cube;
    static_assert(sizeof(cube) <= mla_tail::kCubeTilingBytes,
                  "increase the serialized cube tiling stride");
    const int64_t status = tiling.GetTiling(cube);
    if (status != 0) {
        return static_cast<int>(status);
    }
    std::memcpy(buffer, &cube, sizeof(cube));
    return 0;
}

}  // namespace

extern "C" int mla_tail_build_tiling(int capacity, void* buffer, int heads)
{
    if ((capacity != 16 && capacity != 32) || (heads != 4 && heads != 8)) {
        return -1;
    }
    const int projection_input = heads * static_cast<int>(mla_tail::kValue);
    const int value_tile = projection_input / static_cast<int>(mla_tail::kValueTiles);
    std::memset(buffer, 0, mla_tail::kTilingBytes);
    int status = BuildProjection(
        capacity, value_tile, mla_tail::kLatent,
        matmul_tiling::DataType::DT_BF16,
        matmul_tiling::DataType::DT_FLOAT, buffer);
    if (status != 0) {
        return status;
    }
    return BuildProjection(
        capacity, mla_tail::kOutputTile, projection_input,
        matmul_tiling::DataType::DT_INT8,
        matmul_tiling::DataType::DT_INT32,
        static_cast<uint8_t*>(buffer) + mla_tail::kCubeTilingBytes);
}
