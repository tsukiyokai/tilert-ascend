#include <cassert>
#include <cstddef>
#include <initializer_list>

#include "contract.h"

int main()
{
    for (int64_t rows = 1; rows <= 32; ++rows) {
        for (int64_t blocks : {16, 24}) {
            assert(ffn_bridge::ValidGeometry(rows, blocks));
            assert(ffn_bridge::Capacity(rows) == routed_int4::pipeline::Capacity(rows));
            // Top-k IDs and probabilities contain exactly the M*8 routes
            // consumed by the frozen input-front; all rows have an AIV owner.
            assert(rows * ffn_bridge::kTopK <= routed_int4::kMaxRoutes);
            assert(rows <= 2 * blocks);
            assert(ffn_bridge::RowOwner(rows - 1, blocks) == rows - 1);
        }
    }
    assert(ffn_active::kBridgeBuffers + ffn_active::kBodyBuffers == 32);
}
