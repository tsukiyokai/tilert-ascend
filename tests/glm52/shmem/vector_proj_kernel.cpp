/**
 * A decode projection with no cube engine at all.
 *
 * One `Axpy` multiplies 128 weight elements into one output row, so four rows
 * cost four instructions per 128 weights. The whole projection is 12.58 M
 * weight elements, which is 8192 instructions per core across 48 cores —
 * roughly 4.6 µs — against 15.6 µs to read 24 MiB at the 1540 GB/s this die
 * delivers. The vector unit is therefore not the constraint at decode's row
 * count, and the cube is not what this shape needs.
 *
 * That matters because the vendor's W4A8 pipeline pays for the cube twice,
 * splitting the activation into two int4 halves and passing them through
 * global memory so the cube has operands it can accept, which is where 5.6× of
 * its slack lives.
 *
 * Nothing here synchronises across cores: each core owns a block of output
 * columns and never reads another's.
 */
#include "vector_proj_kernel.h"
#include "kernel_operator.h"

using namespace AscendC;

// One Axpy covers 128 halves, so a block of that width keeps every
// instruction full.
constexpr int64_t BLOCK_COLUMNS = 128;
constexpr int64_t TILE_ROWS = 256;
constexpr int64_t MAX_ROWS = 4;

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void vector_proj_kernel(
    __gm__ half* activation, __gm__ half* weight, __gm__ half* out, int rows, int inner, int cols, int rounds,
    int banks)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> weight_tile;
    TBuf<TPosition::VECCALC> activation_copy;
    TBuf<TPosition::VECCALC> output;
    pipe.InitBuffer(weight_tile, TILE_ROWS * BLOCK_COLUMNS * sizeof(half));
    pipe.InitBuffer(activation_copy, MAX_ROWS * 2048 * sizeof(half));
    pipe.InitBuffer(output, MAX_ROWS * BLOCK_COLUMNS * sizeof(half));
    LocalTensor<half> tile = weight_tile.Get<half>();
    LocalTensor<half> local_activation = activation_copy.Get<half>();
    LocalTensor<half> accumulator = output.Get<half>();

    const int64_t core = GetBlockIdx();
    const int64_t cores = GetBlockNum();
    const int64_t blocks = cols / BLOCK_COLUMNS;
    if (core >= blocks) {
        return;
    }

    for (int round = 0; round < rounds; ++round) {
        // Rotating the bank defeats L2, so the measurement is of the weight
        // stream rather than of a cache a real round would have evicted.
        const int64_t bank = round % banks;
        __gm__ half* bank_weight = weight + bank * static_cast<int64_t>(inner) * cols;

        for (int64_t block = core; block < blocks; block += cores) {
            const int64_t column = block * BLOCK_COLUMNS;
            Duplicate(accumulator, (half)0.0, MAX_ROWS * BLOCK_COLUMNS);
            for (int row = 0; row < rows; ++row) {
                GlobalTensor<half> source;
                source.SetGlobalBuffer(activation + row * inner, inner);
                DataCopy(local_activation[row * inner], source, inner);
            }
            SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            for (int64_t depth = 0; depth < inner; depth += TILE_ROWS) {
                const int64_t span = (inner - depth < TILE_ROWS) ? inner - depth : TILE_ROWS;
                // One strided copy lifts a [span, BLOCK_COLUMNS] window out of a
                // row-major weight whose rows are `cols` wide.
                DataCopyParams shape;
                shape.blockCount = (uint16_t)span;
                shape.blockLen = (uint16_t)(BLOCK_COLUMNS * sizeof(half) / 32);
                shape.srcStride = (uint16_t)((cols - BLOCK_COLUMNS) * sizeof(half) / 32);
                shape.dstStride = 0;
                GlobalTensor<half> weights;
                weights.SetGlobalBuffer(bank_weight + depth * cols + column);
                DataCopy(tile, weights, shape);
                SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

                for (int64_t step = 0; step < span; ++step) {
                    for (int row = 0; row < rows; ++row) {
                        const half scalar = local_activation.GetValue(row * inner + depth + step);
                        Axpy(accumulator[row * BLOCK_COLUMNS], tile[step * BLOCK_COLUMNS], scalar, BLOCK_COLUMNS);
                    }
                }
                PipeBarrier<PIPE_V>();
            }

            for (int row = 0; row < rows; ++row) {
                GlobalTensor<half> sink;
                sink.SetGlobalBuffer(out + row * cols + column, BLOCK_COLUMNS);
                DataCopy(sink, accumulator[row * BLOCK_COLUMNS], BLOCK_COLUMNS);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
}

void vector_proj_demo(uint32_t block_dim, void* stream, half* activation, half* weight, half* out, int rows, int inner,
                      int cols, int rounds, int banks)
{
    vector_proj_kernel<<<block_dim, nullptr, stream>>>(activation, weight, out, rows, inner, cols, rounds, banks);
}
