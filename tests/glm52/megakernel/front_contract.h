#ifndef TESTS_GLM52_MEGAKERNEL_FRONT_CONTRACT_H
#define TESTS_GLM52_MEGAKERNEL_FRONT_CONTRACT_H

#include <cstdint>

// One dependency level of a layer. A decode matmul's cost follows its reduction
// length, paid once an invocation, while its output axis is spread across cores
// for free -- so every projection that reads the same activation belongs in one
// invocation over their concatenated output.
namespace megakernel_front {

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define MEGAKERNEL_FRONT_DEVICE __aicore__
#else
#define MEGAKERNEL_FRONT_DEVICE
#endif

constexpr int64_t kMaxParts = 8;
constexpr int64_t kMaxRows = 32;
constexpr int64_t kMaxBlocks = 24;
constexpr int64_t kColumnGranularity = 32;
// A weight row must start on a 512-byte boundary or every core's column window
// straddles one. Padding the first level's 3040 columns to 3072 takes it from
// 48.12 to 38.43 microseconds; see megakernel-front-20261022.
constexpr int64_t kStrideBytes = 512;
constexpr uint64_t kTilingStride = 512;
constexpr uint64_t kSystemBytes = 16ULL << 20;
// The vector stage's published total and the chunks it sums a pass; each chunk
// is 1024 elements, so a pass is kVectorChunks * 1024. One cache line a core,
// because a flush writes the whole line and neighbours lose each other's value.
constexpr int64_t kWitnessStride = kStrideBytes / sizeof(float) / 16;  // 8
constexpr int64_t kMaxWitness = kMaxBlocks * 2 * kWitnessStride;
constexpr int64_t kVectorChunks = 72;
constexpr int64_t kMaxSkewSteps = 16;
constexpr int64_t kMaxVectorOps = 16;

constexpr int64_t AlignedColumns(int64_t columns, int64_t element_bytes)
{
    return (columns * element_bytes + kStrideBytes - 1) / kStrideBytes
        * kStrideBytes / element_bytes;
}

// Every core's slice has the same aligned stride.
MEGAKERNEL_FRONT_DEVICE constexpr int64_t SliceStride(int64_t columns,
                                                      int64_t blocks)
{
    const int64_t even = (columns + blocks - 1) / blocks;
    return (even + kColumnGranularity - 1) / kColumnGranularity
        * kColumnGranularity;
}

// A core's contiguous share of `columns`, aligned so the cube sees whole
// fractal blocks. Cores past the end receive no work.
struct Slice {
    int64_t offset, width;
};

MEGAKERNEL_FRONT_DEVICE constexpr Slice SliceFor(int64_t columns,
                                                 int64_t blocks, int64_t index)
{
    const int64_t stride = SliceStride(columns, blocks);
    const int64_t offset = index * stride;
    if (offset >= columns) {
        return {columns, 0};
    }
    const int64_t remaining = columns - offset;
    return {offset, remaining < stride ? remaining : stride};
}

// A step's share of a core's slice, aligned so the cube still sees whole
// fractal blocks. The last step takes the remainder.
MEGAKERNEL_FRONT_DEVICE constexpr Slice StepShare(const Slice& whole,
                                                  int64_t step, int64_t steps)
{
    if (whole.width == 0 || steps < 1) {
        return {whole.offset, 0};
    }
    const int64_t even = (whole.width + steps - 1) / steps;
    const int64_t aligned =
        (even + kColumnGranularity - 1) / kColumnGranularity
        * kColumnGranularity;
    const int64_t start = step * aligned;
    if (start >= whole.width) {
        return {whole.offset + whole.width, 0};
    }
    const int64_t remaining = whole.width - start;
    return {whole.offset + start, remaining < aligned ? remaining : aligned};
}

struct Config {
    int64_t rows, blocks;
    // The shared activation's width, which is this level's reduction length.
    int64_t reduction;
    // The projections reading it, and the row stride of their concatenation.
    int64_t part_count;
    int64_t part_columns[kMaxParts];
    int64_t weight_stride;
    // The operand's width in bytes, which fixes the row-stride alignment.
    int64_t element_bytes;
    // Vector passes over the activation, run on the AIV half while the cube
    // works. Sweeping this says how much vector work the cube hides for free.
    int64_t vector_passes;
    // Grid barriers to issue around the work. A skewed pipeline pays one a
    // step, so this prices the step.
    int64_t barriers;
    // Vector operations applied to each chunk. `fused_ops` applies them all to
    // one load; otherwise each operation loads and stores the chunk itself, so
    // the same arithmetic costs that many passes.
    int64_t vector_ops;
    int64_t fused_ops;
    // Steps the work is cut into. Zero runs the cube and the vector halves one
    // after the other, which is what a phase barrier gives. More than one
    // interleaves step s of the cube with step s-1 of the vector, so the two
    // halves run together with one barrier a step.
    int64_t skew_steps;
};

MEGAKERNEL_FRONT_DEVICE constexpr int64_t TotalColumns(const Config& config)
{
    int64_t total = 0;
    for (int64_t part = 0; part < config.part_count; ++part) {
        total += config.part_columns[part];
    }
    return total;
}

MEGAKERNEL_FRONT_DEVICE constexpr int64_t PartOffset(const Config& config,
                                                     int64_t part)
{
    int64_t offset = 0;
    for (int64_t earlier = 0; earlier < part; ++earlier) {
        offset += config.part_columns[earlier];
    }
    return offset;
}

struct Addresses {
    uint64_t activation, weight, product, tiling, system, witness;
};

// Host-side validation before a launch. It sums the parts itself rather than
// sharing the device helper, which bisheng would not let it call.
constexpr bool Valid(const Config& config, const Addresses& addresses)
{
    if (config.rows < 1 || config.rows > kMaxRows
        || config.blocks < 1 || config.blocks > kMaxBlocks
        || config.reduction < 1
        || config.part_count < 1 || config.part_count > kMaxParts
        || config.weight_stride * config.element_bytes % kStrideBytes != 0
        || config.vector_passes < 0 || config.barriers < 0
        || config.skew_steps < 0 || config.skew_steps > kMaxSkewSteps
        || config.vector_ops < 1 || config.vector_ops > kMaxVectorOps) {
        return false;
    }
    int64_t columns = 0;
    for (int64_t part = 0; part < config.part_count; ++part) {
        if (config.part_columns[part] < 1) {
            return false;
        }
        columns += config.part_columns[part];
    }
    return config.weight_stride >= columns
        && addresses.activation != 0 && addresses.weight != 0
        && addresses.product != 0 && addresses.tiling != 0
        && addresses.system != 0;
}

}  // namespace megakernel_front

extern "C" int megakernel_front_launch(void* stream, uint64_t sync,
    const megakernel_front::Config* config,
    const megakernel_front::Addresses* addresses);

extern "C" int megakernel_front_build_tiling(int rows, int columns, int width,
                                             int reduction, int base_m,
                                             int base_n, int base_k,
                                             void* buffer);

#endif
