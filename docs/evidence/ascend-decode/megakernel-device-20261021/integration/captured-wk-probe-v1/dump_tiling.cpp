// CPU-only decoding of the accepted indexer front tiling object's output.
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "lib/matmul/matmul_tiling.h"

extern "C" int indexer_front_build_tiling(int capacity, void* buffer);

int main()
{
    for (int capacity : {16, 32}) {
        std::array<uint8_t, 1536> bytes{};
        if (indexer_front_build_tiling(capacity, bytes.data()) != 0) {
            return 1;
        }
        AscendC::tiling::TCubeTiling c{};
        static_assert(sizeof(c) <= 512);
        std::memcpy(&c, bytes.data() + 512, sizeof(c));
        std::cout << "{\"capacity\":" << capacity
                  << ",\"M\":" << c.M << ",\"N\":" << c.N
                  << ",\"Ka\":" << c.Ka << ",\"Kb\":" << c.Kb
                  << ",\"singleCoreM\":" << c.singleCoreM
                  << ",\"singleCoreN\":" << c.singleCoreN
                  << ",\"singleCoreK\":" << c.singleCoreK
                  << ",\"baseM\":" << c.baseM
                  << ",\"baseN\":" << c.baseN
                  << ",\"baseK\":" << c.baseK
                  << ",\"depthA1\":" << c.depthA1
                  << ",\"depthB1\":" << c.depthB1
                  << ",\"stepM\":" << c.stepM
                  << ",\"stepN\":" << c.stepN
                  << ",\"stepKa\":" << c.stepKa
                  << ",\"stepKb\":" << c.stepKb
                  << ",\"iterateOrder\":" << c.iterateOrder
                  << ",\"shareL1Size\":" << c.shareL1Size
                  << ",\"shareL0CSize\":" << c.shareL0CSize
                  << ",\"shareUbSize\":" << c.shareUbSize
                  << ",\"dbL0A\":" << c.dbL0A
                  << ",\"dbL0B\":" << c.dbL0B
                  << ",\"dbL0C\":" << c.dbL0C
                  << "}\n";
    }
}
