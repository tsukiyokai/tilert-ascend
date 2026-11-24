"""Bind vLLM-Ascend cache views to the GLM-5.2 split-region wire layout.

This module reads tensor metadata only. It does not synchronize producers,
register memory, or prove that a prefix has been written. The caller owns
those steps and must enforce TP16, PP1, PCP1, and DCP1 at engine setup.
"""

from dataclasses import dataclass
import re

from .ascend_layout import (
    FULL_INDEX_LAYERS,
    INT8_INDEX_LAYERS,
    NUM_LAYERS,
    PAGE_SIZE,
    cache_layout,
    page_transfer_plan,
    validate_layout,
)


_LAYER_NAME = re.compile(
    r"(?:^|\.)layers\.(\d+)\.self_attn\.(attn|indexer\.k_cache)$"
)


@dataclass(frozen=True)
class BoundRegion:
    name: str
    layer_name: str
    group_id: int
    tensor: object
    data_ptr: int
    page_stride_bytes: int
    block_count: int
    row_bytes: int
    storage_ptr: int
    storage_bytes: int
    device_id: int


@dataclass(frozen=True)
class SourceBinding:
    layout: dict
    regions: dict[str, BoundRegion]
    device_id: int

    def registration_ranges(self):
        """Return backing allocations retained by the bound tensors."""
        return list(dict.fromkeys(
            (region.storage_ptr, region.storage_bytes)
            for region in self.regions.values()
        ))

    def transfer_plan(
        self, block_ids_per_group, destination_ptr, prefix_tokens, *,
        mtp_prefix_tokens=None, destination_layout=None,
        parent_hidden_ptr=None,
    ):
        """Resolve each region's request-local scheduler block table."""
        blocks = {}
        for name, region in self.regions.items():
            try:
                blocks[name] = block_ids_per_group[region.group_id]
            except (IndexError, KeyError) as error:
                raise ValueError(
                    f"{name}: missing block table for group {region.group_id}"
                ) from error
        return page_transfer_plan(
            self.layout if destination_layout is None else destination_layout,
            {name: region.data_ptr for name, region in self.regions.items()},
            blocks, destination_ptr, prefix_tokens,
            source_page_strides={
                name: region.page_stride_bytes
                for name, region in self.regions.items()
            },
            source_block_counts={
                name: region.block_count
                for name, region in self.regions.items()
            },
            mtp_prefix_tokens=mtp_prefix_tokens,
            parent_hidden_ptr=parent_hidden_ptr,
        )


def _layer_groups(kv_cache_config):
    result = {}
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        if group.host_resident or not group.enable_kv_transfer:
            raise ValueError("Ascend PD requires transferable device groups")
        for name in group.layer_names:
            if name in result:
                raise ValueError(f"{name}: belongs to multiple cache groups")
            spec = group.kv_cache_spec
            if hasattr(spec, "kv_cache_specs"):
                if name not in spec.kv_cache_specs:
                    raise ValueError(f"{name}: missing per-layer cache spec")
                spec = spec.kv_cache_specs[name]
            result[name] = (group_id, spec)
    return result


def _check_spec(name, spec, layer, is_indexer):
    if spec.block_size != PAGE_SIZE or spec.num_kv_heads != 1:
        raise ValueError(f"{name}: requires 128-token, single-head pages")
    if getattr(spec, "store_on_host", False):
        raise ValueError(f"{name}: host cache is not a device source")
    if is_indexer:
        quantized = layer in INT8_INDEX_LAYERS
        if (
            spec.head_size != 128
            or spec.scale_dim != int(quantized)
            or spec.cache_sparse_li_c8 != quantized
            or getattr(spec, "cache_sparse_li_c4", False)
            or spec.sfa_dcp_replicated_indexer_size != 1
        ):
            raise ValueError(f"{name}: indexer recipe or replication mismatch")
    elif (
        spec.head_size != 576
        or getattr(spec, "uses_packed_sfa_main_cache", False)
        or getattr(spec, "tokens_per_state", 1) != 1
    ):
        raise ValueError(f"{name}: requires uncompressed BF16 MLA cache")


def _bind_region(name, layer_name, group_id, tensor, layout):
    expected = layout["regions"][name]
    dtype = str(tensor.dtype).removeprefix("torch.")
    element_bytes = int(tensor.element_size())
    if dtype != expected["dtype"]:
        raise ValueError(f"{name}: dtype {dtype} != {expected['dtype']}")
    if element_bytes != (1 if dtype == "int8" else 2):
        raise ValueError(f"{name}: incorrect element size")
    width = expected["row_bytes"] // element_bytes
    shape = tuple(int(size) for size in tensor.shape)
    strides = tuple(int(step) for step in tensor.stride())
    if (
        len(shape) != 4 or shape[0] <= 0
        or shape[1:] != (PAGE_SIZE, 1, width)
        or strides[1:] != (width, width, 1)
    ):
        raise ValueError(f"{name}: requires block-first dense inner pages")
    page_bytes = PAGE_SIZE * expected["row_bytes"]
    page_stride = strides[0] * element_bytes
    if page_stride < page_bytes:
        raise ValueError(f"{name}: source pages overlap")
    if tensor.device.type not in ("npu", "privateuseone"):
        raise ValueError(f"{name}: requires NPU device memory")
    pointer = int(tensor.data_ptr())
    storage = tensor.untyped_storage()
    storage_ptr = int(storage.data_ptr())
    storage_bytes = int(storage.nbytes())
    last_end = pointer + (shape[0] - 1) * page_stride + page_bytes
    if (
        storage_ptr <= 0 or pointer < storage_ptr
        or last_end > storage_ptr + storage_bytes
    ):
        raise ValueError(f"{name}: tensor view exceeds its backing allocation")
    return BoundRegion(
        name, layer_name, group_id, tensor, pointer, page_stride, shape[0],
        expected["row_bytes"], storage_ptr, storage_bytes,
        int(tensor.device.index),
    )


def _regions_overlap(left, right):
    """Compare page intervals without treating padding as tensor contents."""
    left_page = PAGE_SIZE * left.row_bytes
    right_page = PAGE_SIZE * right.row_bytes
    if left.page_stride_bytes == right.page_stride_bytes:
        stride = left.page_stride_bytes
        delta = right.data_ptr - left.data_ptr
        first = (delta - left_page) // stride + 1
        last = (delta + right_page - 1) // stride
        return max(first, 1 - right.block_count) <= min(
            last, left.block_count - 1,
        )
    left_block = right_block = 0
    while left_block < left.block_count and right_block < right.block_count:
        left_start = left.data_ptr + left_block * left.page_stride_bytes
        right_start = right.data_ptr + right_block * right.page_stride_bytes
        if left_start + left_page <= right_start:
            left_block += 1
        elif right_start + right_page <= left_start:
            right_block += 1
        else:
            return True
    return False


def bind_vllm_caches(kv_caches, kv_cache_config, *, layout=None):
    """Bind all 79 MLA pairs, 22 index keys, and 16 scales explicitly.

    The input is the dictionary passed to KVConnector.register_kv_caches.
    MTP must have its actual layer number 78; missing caches are errors.
    Tensor shape, dtype, stride, device, storage bounds, group membership,
    and per-layer cache spec are checked. Parallel sizes and valid prefix
    contents are not represented by KVCacheConfig and remain caller inputs.
    """
    layout = cache_layout() if layout is None else layout
    validate_layout(layout)
    groups = _layer_groups(kv_cache_config)
    regions = {}
    for layer_name, tensors in kv_caches.items():
        match = _LAYER_NAME.search(layer_name)
        if match is None or layer_name not in groups:
            raise ValueError(f"{layer_name}: unknown cache layer or group")
        layer = int(match.group(1))
        is_indexer = match.group(2) == "indexer.k_cache"
        if not 0 <= layer < NUM_LAYERS:
            raise ValueError(f"{layer_name}: unexpected layer number")
        if is_indexer and layer not in FULL_INDEX_LAYERS:
            raise ValueError(f"{layer_name}: shared indexer owns no cache")
        group_id, spec = groups[layer_name]
        _check_spec(layer_name, spec, layer, is_indexer)
        if is_indexer:
            names = [f"index.{layer}"]
            if layer in INT8_INDEX_LAYERS:
                names.append(f"index_scale.{layer}")
        else:
            names = [f"mla_latent.{layer}", f"mla_rope.{layer}"]
        if (
            not isinstance(tensors, (tuple, list))
            or len(tensors) != len(names)
        ):
            raise ValueError(
                f"{layer_name}: expected {len(names)} cache views"
            )
        for name, tensor in zip(names, tensors):
            if name in regions:
                raise ValueError(f"{name}: duplicate layer cache")
            regions[name] = _bind_region(
                name, layer_name, group_id, tensor, layout,
            )
    missing = set(layout["regions"]) - regions.keys()
    if missing:
        raise ValueError(f"missing Ascend cache regions: {sorted(missing)}")
    devices = {region.device_id for region in regions.values()}
    if len(devices) != 1:
        raise ValueError("one rank's cache must reside on one NPU")
    allocations = {}
    for region in regions.values():
        peers = allocations.setdefault(region.storage_ptr, [])
        for peer in peers:
            if region.storage_bytes != peer.storage_bytes:
                raise ValueError("inconsistent size for shared backing")
            if _regions_overlap(region, peer):
                raise ValueError(
                    f"cache regions alias: {region.name}/{peer.name}"
                )
        peers.append(region)
    return SourceBinding(layout, regions, devices.pop())
