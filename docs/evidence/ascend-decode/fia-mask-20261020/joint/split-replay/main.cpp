#include "split_core.h"
#include <iostream>
#include <climits>
int main() {
    for (unsigned cores : {16,20,24}) {
        optiling::BaseInfo b{};
        b.bSize=1;b.n2Size=1;b.gSize=4;b.s1Size=8;b.s2Size=8192;
        b.isS1G=true;b.attenMaskFlag=true;b.sparseMode=0;
        b.preToken=INT_MAX;b.nextToken=INT_MAX;
        optiling::SplitParam p{};p.mBaseSize=256;p.s2BaseSize=512;p.gS1BaseSizeOfFd=8;
        optiling::SplitResult r{cores,2};optiling::SplitCore(cores,b,p,r);
        std::cout << "cores=" << cores << " used=" << r.usedCoreNum << " fdHeads=" << r.numOfFdHead << " maxS2Split=" << r.maxS2SplitNum << " fdVectors=" << r.usedVecNumOfFd << " maxCost=" << r.maxCost << '\n';
        for(unsigned i=0;i<r.usedCoreNum;++i) std::cout << "end[" << i << "]=(" << r.bN2End[i] << ',' << r.gS1End[i] << ',' << r.s2End[i] << ") ";
        std::cout << '\n';
        for(unsigned i=0;i<r.numOfFdHead;++i) std::cout << "FD[" << i << "]=(" << r.fdRes.bN2IdxOfFdHead[i] << ',' << r.fdRes.gS1IdxOfFdHead[i] << ",S2=" << r.fdRes.s2SplitNumOfFdHead[i] << ",M=" << r.fdRes.gS1SplitNumOfFdHead[i] << ") ";
        std::cout << '\n';
    }
}
