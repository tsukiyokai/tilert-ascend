#include <cstring>

#include "lib/matmul/matmul_tiling.h"
#include "indexer_front_contract.h"

namespace {

int Projection(int rows, int columns, int inner,
               matmul_tiling::DataType input_type,
               matmul_tiling::DataType output_type, void* buffer)
{
    matmul_tiling::MatmulApiTiling tiling;
    tiling.SetAType(matmul_tiling::TPosition::GM,
                   matmul_tiling::CubeFormat::ND, input_type, false);
    tiling.SetBType(matmul_tiling::TPosition::GM,
                   matmul_tiling::CubeFormat::ND, input_type, false);
    tiling.SetCType(matmul_tiling::TPosition::GM,
                   matmul_tiling::CubeFormat::ND, output_type);
    tiling.SetShape(rows, columns, inner);
    tiling.SetOrgShape(rows, columns, inner);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1);
    AscendC::tiling::TCubeTiling cube;
    static_assert(sizeof(cube) <= indexer_front::kTilingStride);
    const int status = tiling.GetTiling(cube);
    if (status == 0) {
        std::memcpy(buffer, &cube, sizeof(cube));
    }
    return status;
}

}  // namespace

extern "C" int indexer_front_build_tiling(int capacity, void* buffer)
{
    using namespace indexer_front;
    using matmul_tiling::DataType;
    if (capacity != 16 && capacity != 32) {
        return -1;
    }
    std::memset(buffer, 0, kTilingBytes);
    int status = Projection(capacity, kQueryTile, kQueryRank,
                            DataType::DT_INT8, DataType::DT_INT32, buffer);
    if (status == 0) {
        status = Projection(capacity, kKeyTile, kHidden, DataType::DT_BF16,
            DataType::DT_FLOAT, static_cast<uint8_t*>(buffer) + kTilingStride);
    }
    if (status == 0) {
        status = Projection(kHadamardRows, kDim, kDim, DataType::DT_BF16,
            DataType::DT_FLOAT, static_cast<uint8_t*>(buffer) + 2 * kTilingStride);
    }
    return status;
}
