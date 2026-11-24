#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>

int main(int argc, char **argv)
{
    const int m = argc > 1 ? std::stoi(argv[1]) : 32;
    const int n = argc > 2 ? std::stoi(argv[2]) : 1024;
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910_9391");
    if (!platform) return 2;
    using namespace matmul_tiling;
    MultiCoreMatmulTiling mm(*platform);
    mm.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT4, false);
    mm.SetBType(TPosition::GM, CubeFormat::NZ, DataType::DT_INT4, false);
    mm.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    mm.SetBias(false);
    mm.SetOrgShape(m, 6144, 128);
    mm.SetShape(m, n, 128);
    mm.SetFixSplit(m, n, 128);
    optiling::TCubeTiling result;
    if (mm.GetTiling(result) == -1) return 3;
#define FIELD(name) std::cout << #name << '=' << result.get_##name() << '\n';
    FIELD(usedCoreNum) FIELD(M) FIELD(N) FIELD(Ka) FIELD(Kb)
    FIELD(singleCoreM) FIELD(singleCoreN) FIELD(singleCoreK)
    FIELD(baseM) FIELD(baseN) FIELD(baseK)
    FIELD(depthA1) FIELD(depthB1) FIELD(stepM) FIELD(stepN)
    FIELD(stepKa) FIELD(stepKb) FIELD(dbL0A) FIELD(dbL0B) FIELD(dbL0C)
    FIELD(shareL1Size) FIELD(shareL0CSize) FIELD(shareUbSize)
    FIELD(isBias) FIELD(iterateOrder) FIELD(transLength) FIELD(shareMode)
    FIELD(batchM) FIELD(batchN) FIELD(singleBatchM) FIELD(singleBatchN)
    FIELD(depthAL1CacheUB) FIELD(depthBL1CacheUB)
}
