/** Call the compiled host tiler without initializing a device. */
#include <cstdint>
#include <iostream>

#include "lib/matmul/matmul_tiling.h"

extern "C" int routed_int4_build_tiling(int, int, int, int, void*, int64_t);

int main()
{
    for (int layout : {0, 1}) {
        for (int projection : {0, 1}) {
            const int columns = projection == 0 ? 256 : 6144;
            const int inner = projection == 0 ? 6144 : 128;
            AscendC::tiling::TCubeTiling cube{};
            const int status = routed_int4_build_tiling(
                64, columns, inner, columns, &cube, layout);
            std::cout << "{\"projection\":" << projection
                      << ",\"layout\":" << layout << ",\"status\":" << status;
#define FIELD(name) std::cout << ",\"" #name "\":" << cube.name
            FIELD(M); FIELD(N); FIELD(Ka); FIELD(Kb);
            FIELD(singleCoreM); FIELD(singleCoreN); FIELD(singleCoreK);
            FIELD(baseM); FIELD(baseN); FIELD(baseK);
            FIELD(depthA1); FIELD(depthB1);
            FIELD(stepM); FIELD(stepN); FIELD(stepKa); FIELD(stepKb);
            FIELD(iterateOrder); FIELD(shareL1Size); FIELD(shareL0CSize); FIELD(shareUbSize);
            FIELD(dbL0A); FIELD(dbL0B); FIELD(dbL0C);
#undef FIELD
            std::cout << "}\n";
            if (status != 0) return 1;
        }
    }
}
