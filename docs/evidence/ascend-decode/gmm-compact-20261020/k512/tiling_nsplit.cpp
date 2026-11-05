#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>
int main(int argc, char **argv) {
    const int m = std::stoi(argv[1]), bk = std::stoi(argv[2]), n = std::stoi(argv[3]);
    auto *p = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910_9391");
    if (!p) return 2;
    using namespace matmul_tiling;
    MatmulApiTiling mm(*p);
    mm.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT4);
    mm.SetBType(TPosition::GM, CubeFormat::NZ, DataType::DT_INT4);
    mm.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    mm.SetBias(false);
    mm.SetShape(m,n,6144);
    mm.SetFixSplit(m,n,bk);
    mm.SetOrgShape(256,256,6144);
    mm.SetBufferSpace(-1,-1,-1);
    optiling::TCubeTiling r;
    int rc=mm.GetTiling(r);
    std::cout<<"rc="<<rc<<'\n';
    if (rc==-1) return 3;
#define F(n) std::cout<<#n<<'='<<r.get_##n()<<'\n';
    F(M) F(N) F(Ka) F(Kb) F(singleCoreM) F(singleCoreN) F(singleCoreK)
    F(baseM) F(baseN) F(baseK) F(depthA1) F(depthB1) F(stepM) F(stepN)
    F(stepKa) F(stepKb) F(dbL0A) F(dbL0B) F(dbL0C)
    F(shareL1Size) F(shareL0CSize) F(shareUbSize) F(iterateOrder) F(transLength)
}
