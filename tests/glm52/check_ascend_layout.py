"""Check split Ascend KV geometry and vLLM bindings using metadata doubles."""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType, SimpleNamespace


class Storage:
    def __init__(self, pointer, size):
        self.pointer, self.size = pointer, size

    def data_ptr(self):
        return self.pointer

    def nbytes(self):
        return self.size


class Tensor:
    def __init__(self, storage, width, dtype, offset=0, page_stride=None):
        self.storage, self.offset = storage, offset
        self.dtype = f"torch.{dtype}"
        self.device = SimpleNamespace(type="npu", index=3)
        self.shape = (96, 128, 1, width)
        self.element_bytes = 1 if dtype == "int8" else 2
        stride = page_stride or 128 * width * self.element_bytes
        self.strides = (stride // self.element_bytes, width, width, 1)

    def data_ptr(self):
        return self.storage.pointer + self.offset

    def element_size(self):
        return self.element_bytes

    def stride(self):
        return self.strides

    def untyped_storage(self):
        return self.storage


def load_modules(root):
    package = ModuleType("ascend_layout_check")
    package.__path__ = [str(root / "tilert/pd_vllm")]
    sys.modules[package.__name__] = package
    modules = []
    for name in ("ascend_layout", "ascend_source"):
        path = root / f"tilert/pd_vllm/{name}.py"
        spec = importlib.util.spec_from_file_location(
            f"{package.__name__}.{name}", path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def cache_name(layer, indexer=False):
    prefix = "mtp.model" if layer == 78 else "model"
    suffix = "indexer.k_cache" if indexer else "attn"
    return f"{prefix}.layers.{layer}.self_attn.{suffix}"


def fixture(layout_module, *, strided=False):
    groups = [SimpleNamespace(
        layer_names=[], kv_cache_spec=SimpleNamespace(kv_cache_specs={}),
        host_resident=False, enable_kv_transfer=True,
    ) for _ in range(5)]
    caches = {}
    next_pointer = 2**40

    def allocate(nbytes):
        nonlocal next_pointer
        next_pointer += 2**28
        return Storage(next_pointer, nbytes)

    def add(layer, indexer, tensors, group_id):
        name = cache_name(layer, indexer)
        quantized = indexer and layer in layout_module.INT8_INDEX_LAYERS
        spec = SimpleNamespace(
            block_size=128, num_kv_heads=1,
            head_size=128 if indexer else 576,
            scale_dim=int(quantized), cache_sparse_li_c8=quantized,
            cache_sparse_li_c4=False, sfa_dcp_replicated_indexer_size=1,
            uses_packed_sfa_main_cache=False, tokens_per_state=1,
        )
        caches[name] = tensors
        group = groups[group_id]
        group.layer_names.append(name)
        group.kv_cache_spec.kv_cache_specs[name] = spec

    for layer in range(79):
        if strided and layer % 2 == 0:
            stride = 128 * 576 * 2 + 512
            storage = allocate(96 * stride)
            latent = Tensor(storage, 512, "bfloat16", page_stride=stride)
            rope = Tensor(
                storage, 64, "bfloat16", offset=128 * 512 * 2,
                page_stride=stride,
            )
        else:
            latent = Tensor(allocate(96 * 128 * 1024), 512, "bfloat16")
            rope = Tensor(allocate(96 * 128 * 128), 64, "bfloat16")
        add(layer, False, (latent, rope), int(layer == 78))
    for layer in layout_module.FULL_INDEX_LAYERS:
        quantized = layer in layout_module.INT8_INDEX_LAYERS
        dtype, size = ("int8", 1) if quantized else ("bfloat16", 2)
        tensors = (Tensor(allocate(96 * 128 * 128 * size), 128, dtype),)
        if quantized:
            tensors += (Tensor(allocate(96 * 128 * 2), 1, "float16"),)
        group_id = 4 if layer == 78 else 3 if quantized else 2
        add(layer, True, tensors, group_id)
    return caches, SimpleNamespace(kv_cache_groups=groups)


def addresses_match(binding, tables, destination, prefix, plan):
    sources, destinations, lengths = plan
    if len(sources) != len(destinations) or len(sources) != len(lengths):
        return False
    descriptor = 0
    for name, geometry in binding.layout["regions"].items():
        region = binding.regions[name]
        for token in range(prefix):
            dest = destination + geometry["offset"] + token * region.row_bytes
            while (
                descriptor < len(lengths)
                and destinations[descriptor] + lengths[descriptor] <= dest
            ):
                descriptor += 1
            if descriptor == len(lengths):
                return False
            source = (
                region.data_ptr
                + tables[region.group_id][token // 128]
                * region.page_stride_bytes
                + token % 128 * region.row_bytes
            )
            if (
                not destinations[descriptor] <= dest
                or dest + region.row_bytes
                > destinations[descriptor] + lengths[descriptor]
                or sources[descriptor] + dest - destinations[descriptor]
                != source
            ):
                return False
    return True


def source_evidence(root, files):
    if root is None:
        return None
    return {
        "root": str(root),
        "revision": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in files
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm-ascend-root", type=Path)
    parser.add_argument("--vllm-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    geometry, source = load_modules(root)
    layout = geometry.cache_layout(9219)
    regions = layout["regions"]
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    def rejected(name, operation):
        try:
            operation()
        except (ValueError, TypeError):
            check(name, True)
        else:
            check(name, False)

    check("capacity_page_rounding", layout["capacity_tokens"] == 9344)
    check("tp16_layout", layout["num_ranks"] == 16)
    check("layout_version", layout["version"] == 2)
    check("full_indexer_layers", geometry.FULL_INDEX_LAYERS == (
        0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38,
        42, 46, 50, 54, 58, 62, 66, 70, 74, 78,
    ))
    check("int8_indexer_layers", geometry.INT8_INDEX_LAYERS == {
        6, 10, 14, 18, 22, 26, 30, 34,
        38, 42, 46, 50, 54, 58, 62, 66,
    })
    for kind, count in (
        ("mla_latent", 79), ("mla_rope", 79), ("index", 22),
        ("index_scale", 16),
    ):
        check(f"{kind}.count", sum(
            name.startswith(f"{kind}.") for name in regions
        ) == count)
    check("mtp_index_bfloat16", regions["index.78"]["dtype"] == "bfloat16")
    check("shared_index_absent", all(
        f"index.{layer}" not in regions for layer in (3, 4, 5, 7, 77)
    ))
    check("capacity_bytes", layout["nbytes"] == 9344 * 94624)
    row_bytes = sum(region["row_bytes"] for region in regions.values())
    check("8k_prefix_bytes", row_bytes * 8192 == 775159808)
    check("contiguous_regions", all(
        left["offset"] + left["nbytes"] == right["offset"]
        for left, right in zip(
            list(regions.values()), list(regions.values())[1:],
        )
    ))
    geometry.validate_layout(json.loads(json.dumps(layout)))
    check("json_layout_roundtrip", True)
    for strided in (False, True):
        caches, config = fixture(geometry, strided=strided)
        binding = source.bind_vllm_caches(caches, config, layout=layout)
        label = "page_strided" if strided else "contiguous"
        check(f"{label}.device", binding.device_id == 3)
        check(f"{label}.regions", len(binding.regions) == 196)
        check(f"{label}.storage_dedup", len(binding.registration_ranges())
              == (156 if strided else 196))
        check(f"{label}.tensor_lifetime", all(
            region.tensor is caches[region.layer_name][
                int(name.startswith(("mla_rope.", "index_scale.")))
            ] for name, region in binding.regions.items()
        ))
        for prefix in (257, 512, 8192):
            tables = [
                [3 + group, 4 + group, *range(8 + group, 70 + group)]
                for group in range(5)
            ]
            destination = 2**48
            plan = binding.transfer_plan(tables, destination, prefix)
            check(f"{label}.{prefix}.bytes", sum(plan[2])
                  == prefix * row_bytes)
            check(f"{label}.{prefix}.every_row_address", addresses_match(
                binding, tables, destination, prefix, plan,
            ))
        if not strided:
            tables = [list(range(64)) for _ in range(5)]
            plan = binding.transfer_plan(tables, 2**48, 8192)
            check("contiguous.8k_coalescing", len(plan[2]) == 196)

    baseline = fixture(geometry, strided=True)

    def reject_change(name, change):
        caches, config = copy.deepcopy(baseline)
        change(caches, config)
        rejected(name, lambda: source.bind_vllm_caches(
            caches, config, layout=layout,
        ))

    for name in (cache_name(78), cache_name(78, True), cache_name(0, True)):
        reject_change(f"missing.{name}", lambda caches, _: caches.pop(name))
    c8_name = cache_name(6, True)
    reject_change("missing.c8_scale", lambda caches, _: caches.update({
        c8_name: caches[c8_name][:1],
    }))
    reject_change("wrong.c8_dtype", lambda caches, _: setattr(
        caches[c8_name][0], "dtype", "torch.bfloat16",
    ))
    reject_change("wrong.scale_dtype", lambda caches, _: setattr(
        caches[c8_name][1], "dtype", "torch.float32",
    ))
    reject_change("wrong.inner_stride", lambda caches, _: setattr(
        caches[cache_name(0)][0], "strides", (74000, 576, 512, 1),
    ))
    reject_change("wrong.storage_bounds", lambda caches, _: setattr(
        caches[cache_name(0)][0].storage, "size", 1,
    ))
    reject_change("wrong.cpu_memory", lambda caches, _: setattr(
        caches[cache_name(0)][0].device, "type", "cpu",
    ))
    reject_change("wrong.multiple_devices", lambda caches, _: setattr(
        caches[cache_name(0)][0].device, "index", 4,
    ))
    reject_change("wrong.host_group", lambda _, config: setattr(
        config.kv_cache_groups[0], "host_resident", True,
    ))
    reject_change("wrong.nontransferable_group", lambda _, config: setattr(
        config.kv_cache_groups[0], "enable_kv_transfer", False,
    ))
    reject_change("missing.group", lambda _, config:
                  config.kv_cache_groups[0].layer_names.remove(cache_name(0)))
    reject_change("alias.index_cache", lambda caches, _: caches.update({
        cache_name(1, True): caches[cache_name(0, True)],
    }))
    reject_change("alias.mla_components", lambda caches, _: setattr(
        caches[cache_name(0)][1], "offset", 0,
    ))
    for field, value in (("block_size", 64), ("num_kv_heads", 2)):
        reject_change(f"wrong.spec.{field}", lambda _, config: setattr(
            config.kv_cache_groups[0].kv_cache_spec.kv_cache_specs[
                cache_name(0)
            ], field, value,
        ))
    reject_change("wrong.indexer_dcp", lambda _, config: setattr(
        config.kv_cache_groups[3].kv_cache_spec.kv_cache_specs[c8_name],
        "sfa_dcp_replicated_indexer_size", 2,
    ))
    binding = source.bind_vllm_caches(*baseline, layout=layout)
    for name, tables in (
        ("missing", []), ("short", [[0]] * 5),
        ("negative", [[0, -1]] * 5), ("out_of_bounds", [[0, 96]] * 5),
    ):
        rejected(f"block_table.{name}", lambda: binding.transfer_plan(
            tables, 2**48, 256,
        ))
    for tokens in (0, 9345):
        rejected(f"prefix.{tokens}", lambda: binding.transfer_plan(
            [list(range(96))] * 5, 2**48, tokens,
        ))
    corrupt = copy.deepcopy(layout)
    corrupt["version"] = 1
    rejected("reject_combined_wire_version", lambda:
             geometry.validate_layout(corrupt))
    check("no_torch_import", "torch" not in sys.modules)
    check("no_torch_npu_import", "torch_npu" not in sys.modules)
    files = (
        "tilert/pd_vllm/ascend_layout.py",
        "tilert/pd_vllm/ascend_source.py",
        "tests/glm52/check_ascend_layout.py",
    )
    report = {
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "command": [sys.executable, *sys.argv],
        "host": platform.node(), "python": sys.version,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in files
        },
        "vllm_ascend": source_evidence(args.vllm_ascend_root, (
            "vllm_ascend/worker/model_runner_v1.py",
            "vllm_ascend/worker/v2/attn_utils.py",
            "vllm_ascend/attention/sfa_v1.py",
            "vllm_ascend/core/kv_cache_interface.py",
        )),
        "vllm": source_evidence(args.vllm_root, (
            "vllm/v1/kv_cache_interface.py",
            "vllm/model_executor/models/deepseek_mtp.py",
        )),
        "allocated_bytes_per_rank": layout["nbytes"],
        "8k_transfer_bytes_per_rank": row_bytes * 8192,
        "regions": len(regions),
        "checks_total": len(checks),
        "checks_failed": sum(not item["passed"] for item in checks),
        "scope": "Host stdlib address plans with tensor metadata doubles",
        "pending": [
            "Binding real vLLM-Ascend NPU tensors",
            "Producer completion and valid main/MTP prefix lengths",
            "Caller enforcement of TP16, PP1, PCP1, and DCP1",
            "HIXL registration, WRITE, and receiver cache consumption",
        ],
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        key: report[key] for key in ("status", "checks_total", "checks_failed")
    }))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
