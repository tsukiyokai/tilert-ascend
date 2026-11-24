// Storage-preserving INT32 carrier descriptors for the pinned W4A8 NZ banks.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/aten/common/from_blob.h>
#include <torch_npu/csrc/core/NPUStorageImpl.h>

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace {
using Dimensions = std::vector<int64_t>;
using Description = c10::Dict<std::string, Dimensions>;

torch_npu::NPUStorageImpl* Storage(const at::Tensor& value)
{
    TORCH_CHECK(value.device().type() == c10::DeviceType::PrivateUse1,
                "Expected an NPU tensor");
    return static_cast<torch_npu::NPUStorageImpl*>(value.storage().unsafeGetStorageImpl());
}

Dimensions Copy(at::IntArrayRef values)
{
    return Dimensions(values.begin(), values.end());
}

void SetDescription(at::Tensor& value, at::IntArrayRef physical, aclFormat format)
{
    auto& description = Storage(value)->npu_desc_;
    description.base_sizes_ = value.sizes();
    description.base_strides_ = value.strides();
    description.storage_sizes_ = physical;
    description.base_offset_ = 0;
    description.origin_format_ = ACL_FORMAT_ND;
    description.npu_format_ = format;
    description.data_type_ = value.dtype();
    Storage(value)->set_resizable(false);
}

at::Tensor OwnedAlias(const at::Tensor& owner, at::IntArrayRef sizes,
                      at::IntArrayRef strides, const at::TensorOptions& options)
{
    // The new StorageImpl owns this closure; its Tensor copy owns the original
    // allocation. The original StorageImpl and descriptor are never modified.
    std::function<void(void*)> retain_owner = [owner](void*) {};
    return at_npu::native::from_blob(owner.data_ptr(), sizes, strides, 0,
                                     retain_owner, options, owner.device());
}

at::Tensor CanonicalAlias(const at::Tensor& source)
{
    const auto* storage = Storage(source);
    const auto& description = storage->npu_desc_;
    TORCH_CHECK(source.scalar_type() == at::kInt && source.dim() == 3
                && source.size(0) == 256 && source.is_contiguous()
                && source.storage_offset() == 0,
                "Expected a complete contiguous INT32 [256,K,N/8] carrier");
    const int64_t inner = source.size(1);
    const int64_t columns = source.size(2) * 8;
    TORCH_CHECK((inner == 6144 && columns == 256)
                || (inner == 128 && columns == 6144),
                "Only the pinned TP16 W4A8 GU/Down bank geometries are supported");
    const std::array<int64_t, 5> canonical{256, columns / 64, inner / 16, 16, 8};
    const std::array<int64_t, 5> old_storage{256, columns / 64, inner / 16, 16, 32};
    const std::array<int64_t, 3> old_base{256, inner, columns / 2};
    const std::array<int64_t, 3> old_strides{inner * columns / 2, columns / 2, 1};
    const auto required_bytes = static_cast<size_t>(256 * inner * columns / 2);
    TORCH_CHECK(description.npu_format_ == ACL_FORMAT_FRACTAL_NZ
                && source.storage().nbytes() == required_bytes
                && source.nbytes() == required_bytes,
                "Expected the exact complete aligned NZ allocation");
    const bool already_canonical = description.storage_sizes_ == at::IntArrayRef(canonical)
        && description.base_sizes_ == source.sizes()
        && description.base_strides_ == source.strides()
        && description.data_type_ == source.dtype();
    if (already_canonical) return source;
    TORCH_CHECK(description.storage_sizes_ == at::IntArrayRef(old_storage)
                && description.base_sizes_ == at::IntArrayRef(old_base)
                && description.base_strides_ == at::IntArrayRef(old_strides)
                && description.data_type_ == caffe2::TypeMeta::Make<int8_t>(),
                "The source does not have the observed INT8-owner dtype-view descriptor");
    auto result = OwnedAlias(source, source.sizes(), source.strides(), source.options());
    TORCH_CHECK(result.storage().unsafeGetStorageImpl() != source.storage().unsafeGetStorageImpl(),
                "The alias must have an independent StorageImpl");
    SetDescription(result, canonical, ACL_FORMAT_FRACTAL_NZ);
    TORCH_CHECK(result.data_ptr() == source.data_ptr()
                && result.storage().nbytes() == source.storage().nbytes(),
                "The alias must retain the exact physical byte span");
    return result;
}

at::Tensor RawStorage(const at::Tensor& source)
{
    Storage(source);
    TORCH_CHECK(source.storage_offset() == 0, "Raw storage requires offset zero");
    const std::array<int64_t, 1> shape{static_cast<int64_t>(source.storage().nbytes())};
    const std::array<int64_t, 1> stride{1};
    auto result = OwnedAlias(source, shape, stride, source.options().dtype(at::kByte));
    SetDescription(result, shape, ACL_FORMAT_ND);
    return result;
}

Description Describe(const at::Tensor& value)
{
    const auto* storage = Storage(value);
    const auto& description = storage->npu_desc_;
    Description result;
    result.insert("logical_shape", Copy(value.sizes()));
    result.insert("logical_stride", Copy(value.strides()));
    result.insert("base_sizes", Copy(description.base_sizes_));
    result.insert("base_strides", Copy(description.base_strides_));
    result.insert("storage_sizes", Copy(description.storage_sizes_));
    result.insert("format", Dimensions{description.npu_format_});
    result.insert("origin_format", Dimensions{description.origin_format_});
    result.insert("descriptor_scalar_type", Dimensions{static_cast<int64_t>(
        c10::typeMetaToScalarType(description.data_type_))});
    result.insert("logical_scalar_type", Dimensions{static_cast<int64_t>(value.scalar_type())});
    result.insert("storage_offset", Dimensions{value.storage_offset()});
    result.insert("storage_bytes", Dimensions{static_cast<int64_t>(value.storage().nbytes())});
    result.insert("data_ptr", Dimensions{reinterpret_cast<int64_t>(value.data_ptr())});
    result.insert("storage_impl", Dimensions{reinterpret_cast<int64_t>(storage)});
    result.insert("tensor_impl_use_count", Dimensions{static_cast<int64_t>(value.use_count())});
    return result;
}
}  // namespace

TORCH_LIBRARY(tilert_nz_storage, library)
{
    library.def("canonical_alias(Tensor(a) source) -> Tensor(a)");
    library.def("raw_storage(Tensor(a) source) -> Tensor(a)");
    library.def("describe(Tensor value) -> Dict(str, int[])");
}

TORCH_LIBRARY_IMPL(tilert_nz_storage, PrivateUse1, library)
{
    library.impl("canonical_alias", TORCH_FN(CanonicalAlias));
    library.impl("raw_storage", TORCH_FN(RawStorage));
    library.impl("describe", TORCH_FN(Describe));
}
