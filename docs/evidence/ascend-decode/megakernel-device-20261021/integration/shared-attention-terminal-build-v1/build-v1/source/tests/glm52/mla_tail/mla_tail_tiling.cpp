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
    static_assert(sizeof(cube) <= mla_tail_attention_terminal::kCubeTilingBytes,
                  "increase the serialized cube tiling stride");
    const int64_t status = tiling.GetTiling(cube);
    if (status != 0) {
        return static_cast<int>(status);
    }
    std::memcpy(buffer, &cube, sizeof(cube));
    return 0;
}

}  // namespace

extern "C" int mla_tail_attention_terminal_build_tiling(int capacity, void* buffer)
{
    if (capacity != 16 && capacity != 32) {
        return -1;
    }
    std::memset(buffer, 0, mla_tail_attention_terminal::kTilingBytes);
    int status = BuildProjection(
        capacity, mla_tail_attention_terminal::kValueTile, mla_tail_attention_terminal::kLatent,
        matmul_tiling::DataType::DT_BF16,
        matmul_tiling::DataType::DT_FLOAT, buffer);
    if (status != 0) {
        return status;
    }
    return BuildProjection(
        capacity, mla_tail_attention_terminal::kOutputTile, mla_tail_attention_terminal::kProjectionInput,
        matmul_tiling::DataType::DT_INT8,
        matmul_tiling::DataType::DT_INT32,
        static_cast<uint8_t*>(buffer) + mla_tail_attention_terminal::kCubeTilingBytes);
}
