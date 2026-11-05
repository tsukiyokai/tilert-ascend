/** Call the compiled host tiler without initializing a device. */
#include <cstdint>
#include <fstream>
#include <iomanip>

#include "routed_int4_kernel.h"
#include "lib/matmul/matmul_tiling.h"

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    std::ofstream output(argv[1]);
    if (!output) return 3;
    for (int layout : {0, 1}) {
        for (int projection : {0, 1}) {
            const int columns = projection == 0 ? 256 : 6144;
            const int inner = projection == 0 ? 6144 : 128;
            AscendC::tiling::TCubeTiling cube{};
            const int status = routed_int4_build_tiling(
                64, columns, inner, columns, &cube, layout);
            output << "{\"projection\":" << projection
                      << ",\"layout\":" << layout << ",\"status\":" << status;
#define FIELD(name) output << ",\"" #name "\":" << cube.name
            FIELD(M); FIELD(N); FIELD(Ka); FIELD(Kb);
            FIELD(singleCoreM); FIELD(singleCoreN); FIELD(singleCoreK);
            FIELD(baseM); FIELD(baseN); FIELD(baseK);
            FIELD(depthA1); FIELD(depthB1);
            FIELD(stepM); FIELD(stepN); FIELD(stepKa); FIELD(stepKb);
            FIELD(iterateOrder); FIELD(shareL1Size); FIELD(shareL0CSize); FIELD(shareUbSize);
            FIELD(dbL0A); FIELD(dbL0B); FIELD(dbL0C);
#undef FIELD
            output << ",\"raw_hex\":\"";
            const auto* raw = reinterpret_cast<const unsigned char*>(&cube);
            for (size_t i = 0; i < sizeof(cube); ++i) {
                output << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<unsigned>(raw[i]);
            }
            output << std::dec << "\"}\n";
            if (status != 0) return 1;
        }
    }
}
