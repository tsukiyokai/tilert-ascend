#include "shared_w8.h"

#include <cstring>
#include "acl/acl.h"
#include "lib/matmul/matmul_tiling.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t*, uint32_t*);

extern "C" int shared_w8_grouped_quant()
{
    return shared_w8::kGroupedQuant;
}

namespace {
int BuildProjection(int rows, int columns, int inner, void* buffer)
{
    using namespace matmul_tiling;
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32);
    tiling.SetShape(rows, columns, inner);
    tiling.SetOrgShape(rows, columns, inner);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1);
    AscendC::tiling::TCubeTiling cube;
    static_assert(sizeof(cube) <= shared_w8::kTilingStride);
    const int status = tiling.GetTiling(cube);
    if (status != 0) {
        return status;
    }
    std::memcpy(buffer, &cube, sizeof(cube));
    return 0;
}
}  // namespace

extern "C" int shared_w8_prepare(int capacity, int blocks, void* tiling,
                                  uint64_t* sync_base)
{
    if ((capacity != 16 && capacity != 32) || blocks < 1
        || tiling == nullptr || sync_base == nullptr) {
        return -1;
    }
    int32_t device = 0;
    int64_t cubes = 0, vectors = 0;
    if (aclrtGetDevice(&device) != ACL_SUCCESS
        || aclrtGetDeviceInfo(device, ACL_DEV_ATTR_AICORE_CORE_NUM, &cubes)
            != ACL_SUCCESS
        || aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectors)
            != ACL_SUCCESS
        || blocks > cubes || 2 * blocks > vectors) {
        return -2;
    }
    uint32_t length = 0;
    if (rtGetC2cCtrlAddr(sync_base, &length) != 0 || *sync_base == 0) {
        return -3;
    }
    std::memset(tiling, 0, 2 * shared_w8::kTilingStride);
    const int status = BuildProjection(
        capacity, shared_w8::kGateTile, shared_w8::kInnerTile, tiling);
    if (status != 0) {
        return status;
    }
    return BuildProjection(capacity, shared_w8::kDownTile,
        shared_w8::kIntermediate,
        static_cast<uint8_t*>(tiling) + shared_w8::kTilingStride);
}

extern "C" uint64_t shared_w8_workspace_field(int capacity, int field)
{
    if ((capacity != 16 && capacity != 32) || field < 0 || field > 6) {
        return 0;
    }
    const auto layout = shared_w8::MakeLayout(capacity);
    const uint64_t fields[]{layout.staged, layout.partial, layout.middle,
        layout.scales, layout.down, layout.system, layout.bytes};
    return fields[field];
}
