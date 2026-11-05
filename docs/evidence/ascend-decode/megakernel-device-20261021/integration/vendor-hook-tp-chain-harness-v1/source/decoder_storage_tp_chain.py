"""Normalize W4A8 carrier descriptors before the two-rank local component chain."""

import hashlib
import json
from pathlib import Path
import traceback

from tp_chain import run_tp_chain as original_forward


def raw_digest(torch, value):
    """Read every physical byte, including padding, through an independent ND view."""
    raw = torch.ops.tilert_nz_storage.raw_storage(value)
    result = hashlib.sha256()
    for start in range(0, raw.numel(), 16 << 20):
        data = raw[start:start + (16 << 20)].cpu()
        result.update(data.numpy().tobytes())
    return result.hexdigest()


def stable_descriptor(value):
    return {name: item for name, item in value.items()
            if name != "tensor_impl_use_count"}


def run_decoder_forward(torch, layers, config, state, args):
    library = Path(__file__).with_name("nz_alias.so")
    expected = json.loads(Path(__file__).with_name("nz-alias-manifest.json").read_text())
    if hashlib.sha256(library.read_bytes()).hexdigest() != expected["library_sha256"]:
        raise RuntimeError("The frozen storage-alias helper changed")
    torch.ops.load_library(str(library))
    report = {
        "scope": "One-time INT32 NZ carrier descriptors before the two-rank eager component chain",
        "library": str(library), "library_sha256": expected["library_sha256"],
        "helper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "additional_hbm_bytes": 0, "raw_hash_chunk_bytes": 16 << 20,
        "owner_contract": "Each independent alias StorageImpl retains the original Tensor in its deleter closure",
        "banks": [], "completed": False,
    }
    destination = args.output / "nz-storage-correction.json"

    def flush():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    flush()
    try:
        for index, layer in zip((6, 7), layers):
            routed = layer.mlp.experts.routed_experts
            for name in ("w13_weight", "w2_weight"):
                original = getattr(routed, name)
                before = dict(torch.ops.tilert_nz_storage.describe(original))
                original_sha = raw_digest(torch, original)
                alias = torch.ops.tilert_nz_storage.canonical_alias(original)
                source_after = dict(torch.ops.tilert_nz_storage.describe(original))
                after = dict(torch.ops.tilert_nz_storage.describe(alias))
                if stable_descriptor(before) != stable_descriptor(source_after):
                    raise RuntimeError("The source owner's descriptor was modified")
                if after["storage_impl"] == before["storage_impl"]:
                    raise RuntimeError("This fixture must exercise an independent corrected StorageImpl")
                for field in ("data_ptr", "storage_bytes", "logical_shape", "logical_stride",
                              "storage_offset", "format", "logical_scalar_type"):
                    if after[field] != before[field]:
                        raise RuntimeError(f"The alias changed {field}")
                if source_after["tensor_impl_use_count"][0] <= before["tensor_impl_use_count"][0]:
                    raise RuntimeError("The alias did not retain an additional original Tensor owner")
                alias_sha = raw_digest(torch, alias)
                if alias_sha != original_sha:
                    raise RuntimeError("The complete raw NZ bytes changed")
                repeated = torch.ops.tilert_nz_storage.canonical_alias(alias)
                if repeated.untyped_storage()._cdata != alias.untyped_storage()._cdata:
                    raise RuntimeError("Canonical descriptors must be idempotent")
                parameter = torch.nn.Parameter(alias, requires_grad=False)
                parameter.__dict__.update(original.__dict__)
                setattr(routed, name, parameter)
                parameter_descriptor = dict(torch.ops.tilert_nz_storage.describe(parameter))
                if stable_descriptor(parameter_descriptor) != stable_descriptor(after):
                    raise RuntimeError("Parameter installation changed the corrected descriptor")
                report["banks"].append({
                    "layer": index, "parameter": name, "before": before,
                    "source_after": source_after, "after": after,
                    "installed_parameter": parameter_descriptor,
                    "raw_sha256_before": original_sha, "raw_sha256_after": alias_sha,
                    "raw_bytes_equal": True, "new_device_allocation_bytes": 0,
                })
                flush()
        torch.npu.synchronize()
        report["completed"] = True
        flush()
    except Exception:
        report["error"] = traceback.format_exc()
        flush()
        raise
    return original_forward(torch, layers, config, state, args)
