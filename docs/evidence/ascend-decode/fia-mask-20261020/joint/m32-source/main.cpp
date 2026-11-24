#include "split_core.h"
#include <climits>
#include <iostream>
int main() {
    for (unsigned width : {8, 32}) {
        for (unsigned capacity : {8192, 9344}) {
            optiling::BaseInfo b{};
            b.bSize=1; b.n2Size=1; b.gSize=4;
            b.s1Size=width; b.s2Size=capacity;
            b.isS1G=true; b.attenMaskFlag=true; b.sparseMode=0;
            b.preToken=INT_MAX; b.nextToken=INT_MAX;
            optiling::SplitParam p{};
            p.mBaseSize=256; p.s2BaseSize=512; p.gS1BaseSizeOfFd=8;
            optiling::SplitResult r{24,2};
            optiling::SplitCore(24,b,p,r);
            std::cout << "width=" << width << " capacity=" << capacity
                      << " usedCoreNum=" << r.usedCoreNum
                      << " numOfFdHead=" << r.numOfFdHead
                      << " maxS2SplitNum=" << r.maxS2SplitNum
                      << " usedVecNumOfFd=" << r.usedVecNumOfFd << '\n';
            for (unsigned i=0;i<r.usedCoreNum;++i)
                std::cout << "end[" << i << "]=" << r.bN2End[i] << ','
                          << r.gS1End[i] << ',' << r.s2End[i] << ' ';
            std::cout << '\n';
            for (unsigned i=0;i<r.numOfFdHead;++i)
                std::cout << "FD[" << i << "]=" << r.fdRes.bN2IdxOfFdHead[i]
                          << ',' << r.fdRes.gS1IdxOfFdHead[i]
                          << ",S2=" << r.fdRes.s2SplitNumOfFdHead[i]
                          << ",GS1=" << r.fdRes.gS1SplitNumOfFdHead[i] << ' ';
            std::cout << '\n';
        }
    }
}
