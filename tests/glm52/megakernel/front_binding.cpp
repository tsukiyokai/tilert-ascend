/** Torch entry for one dependency level of a layer. */
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <vector>

#include "front_contract.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace {
using namespace megakernel_front;

int64_t SyncBase()
{
    uint64_t address = 0;
    uint32_t length = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&address, &length) == 0 && address != 0,
                "rtGetC2cCtrlAddr failed");
    return static_cast<int64_t>(address);
}

#ifdef MEGAKERNEL_FRONT_BF16
constexpr at::ScalarType kElement = at::kBFloat16;
constexpr at::ScalarType kAccumulator = at::kFloat;
constexpr int64_t kElementBytes = 2;
#else
constexpr at::ScalarType kElement = at::kChar;
constexpr at::ScalarType kAccumulator = at::kInt;
constexpr int64_t kElementBytes = 1;
#endif

Config Describe(int64_t rows, int64_t blocks, int64_t reduction,
                at::IntArrayRef parts, int64_t weight_stride,
                int64_t vector_passes, int64_t barriers,
                int64_t skew_steps, int64_t vector_ops,
                int64_t fused_ops)
{
    TORCH_CHECK(parts.size() >= 1 && parts.size() <= kMaxParts,
                "a level holds 1 to ", kMaxParts, " projections");
    Config config{rows, blocks, reduction,
                  static_cast<int64_t>(parts.size()), {}, weight_stride,
                  kElementBytes, vector_passes, barriers, vector_ops,
                  fused_ops, skew_steps};
    for (size_t part = 0; part < parts.size(); ++part) {
        config.part_columns[part] = parts[part];
    }
    return config;
}

int64_t AlignedWidth(int64_t columns)
{
    return AlignedColumns(columns, kElementBytes);
}

at::Tensor BuildTiling(int64_t rows, int64_t blocks, int64_t reduction,
                       at::IntArrayRef parts, int64_t weight_stride,
                       int64_t base_m, int64_t base_n, int64_t base_k)
{
    const Config config = Describe(rows, blocks, reduction, parts,
                                   weight_stride, 0, 0, 0, 1, 1);
    const Slice widest = SliceFor(TotalColumns(config), blocks, 0);
    auto host = at::empty({static_cast<int64_t>(kTilingStride)},
                          at::dtype(at::kByte));
    const int status = megakernel_front_build_tiling(
        static_cast<int>(rows), static_cast<int>(widest.width),
        static_cast<int>(weight_stride), static_cast<int>(reduction),
        static_cast<int>(base_m), static_cast<int>(base_n),
        static_cast<int>(base_k), host.data_ptr());
    TORCH_CHECK(status == 0, "megakernel_front_build_tiling failed: ", status);
    return host;
}

void Forward(const at::Tensor& activation, const at::Tensor& weight,
             const at::Tensor& product, const at::Tensor& tiling,
             const at::Tensor& system, const at::Tensor& witness,
             int64_t blocks, at::IntArrayRef parts, int64_t vector_passes,
             int64_t barriers, int64_t skew_steps, int64_t vector_ops,
             int64_t fused_ops, int64_t sync_base)
{
    TORCH_CHECK(activation.dim() == 2 && weight.dim() == 2
                && product.dim() == 2, "two-dimensional operands");
    const Config config = Describe(activation.size(0), blocks,
                                   activation.size(1), parts, weight.size(1),
                                   vector_passes, barriers, skew_steps,
                                   vector_ops, fused_ops);
    const int64_t columns = TotalColumns(config);
    TORCH_CHECK(weight.size(0) == config.reduction,
                "weight must reduce over ", config.reduction, " rows");
    TORCH_CHECK(product.size(0) == config.rows
                && product.size(1) == weight.size(1),
                "product must match the weight's row stride");
    TORCH_CHECK(weight.size(1) >= columns,
                "weight needs at least ", columns, " columns");
    TORCH_CHECK(activation.scalar_type() == kElement
                && weight.scalar_type() == kElement
                && product.scalar_type() == kAccumulator,
                "this build takes ", kElement, " operands and a ",
                kAccumulator, " product");
    TORCH_CHECK(witness.numel() >= kMaxWitness
                && witness.scalar_type() == at::kFloat,
                "the witness holds ", kMaxWitness, " floats");
    for (const at::Tensor& owner : {activation, weight, product, tiling,
                                    system, witness}) {
        TORCH_CHECK(owner.is_contiguous() && owner.device().is_privateuseone(),
                    "every owner must be contiguous and on the device");
    }

    const Addresses addresses{
        reinterpret_cast<uint64_t>(activation.data_ptr()),
        reinterpret_cast<uint64_t>(weight.data_ptr()),
        reinterpret_cast<uint64_t>(product.data_ptr()),
        reinterpret_cast<uint64_t>(tiling.data_ptr()),
        reinterpret_cast<uint64_t>(system.data_ptr()),
        reinterpret_cast<uint64_t>(witness.data_ptr())};
    TORCH_CHECK(Valid(config, addresses), "invalid level or addresses");

    const c10::DeviceGuard guard(activation.device());
    const int status = megakernel_front_launch(
        c10_npu::getCurrentNPUStream().stream(),
        static_cast<uint64_t>(sync_base), &config, &addresses);
    TORCH_CHECK(status == 0, "megakernel_front_launch failed: ", status);
}

TORCH_LIBRARY_FRAGMENT(megakernel_front, m)
{
    m.def("sync_base() -> int");
    m.def("aligned_width(int columns) -> int");
    m.def("element_bytes() -> int");
    m.def("witness_floats() -> int");
    m.def("vector_chunks() -> int");
    m.def("witness_stride() -> int");
    m.def("build_tiling(int rows, int blocks, int reduction, int[] parts, "
          "int weight_stride, int base_m, int base_n, int base_k) -> Tensor");
    m.def("forward(Tensor activation, Tensor weight, Tensor product, "
          "Tensor tiling, Tensor system, Tensor witness, int blocks, "
          "int[] parts, int vector_passes, int barriers, int skew_steps, "
          "int vector_ops, int fused_ops, int sync_base) -> ()");
}

TORCH_LIBRARY_IMPL(megakernel_front, CompositeExplicitAutograd, m)
{
    m.impl("sync_base", TORCH_FN(SyncBase));
    m.impl("aligned_width", TORCH_FN(AlignedWidth));
    m.impl("element_bytes", [] { return kElementBytes; });
    m.impl("witness_floats", [] { return kMaxWitness; });
    m.impl("vector_chunks", [] { return kVectorChunks; });
    m.impl("witness_stride", [] { return kWitnessStride; });
    m.impl("build_tiling", TORCH_FN(BuildTiling));
}

TORCH_LIBRARY_IMPL(megakernel_front, PrivateUse1, m)
{
    m.impl("forward", TORCH_FN(Forward));
}

}  // namespace
