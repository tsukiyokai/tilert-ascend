"""Compose the accepted active-expert body with the frozen FFN bridge."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


FFN_LIBRARY = "3673c2a73b2f4b5a4a93b3c2543a4aca35b45773d6a53fc043f27e59a2482655"
ACTIVE_KERNEL = "b231ec19c7b822cc952cbb1d455be8670a79834388edcab1091a23a2328988ec"
RENAMES = {
    "routed_shared_front_active_build_routed_tiling": "ffn_active_body_build_routed_tiling",
    "routed_shared_front_active_build_tiling": "ffn_active_body_build_shared_tiling",
    "routed_shared_front_active_launch": "ffn_active_unused_body_launch",
    "routed_shared_front_active_fused": "ffn_active_unused_body_kernel",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_helper(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one active-body seam: {before!r}")
    return source.replace(before, after, 1)


def copy_sources(parent, hashes, destination):
    for name, expected in hashes.items():
        source = parent / name
        if digest(source) != expected:
            raise ValueError(f"Frozen source changed: {source}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def prepare(ffn, active, output):
    manifest = json.loads((ffn / "manifest.json").read_text())
    active_manifest = json.loads((active / "build-manifest.json").read_text())
    if manifest["artifacts"]["ffn_composed.so"] != FFN_LIBRARY:
        raise ValueError("Use the accepted complete-FFN build-v2 parent")
    if (active_manifest["namespace"] != "tilert_pipeline_front_active"
            or active_manifest["scratch_tensors"] != 21
            or active_manifest["own_sources"]["routed_int4/routed_int4_kernel.cpp"]
            != ACTIVE_KERNEL):
        raise ValueError("Use the accepted active-expert enumeration parent")
    helpers = Path(__file__).resolve().parent.parent
    extractor_path = helpers / "routed_body/prepare.py"
    host_path = helpers / "ffn_composed/prepare.py"
    extractor = load_helper(extractor_path, "active_body_extraction")
    host = load_helper(host_path, "active_host_validation")
    output.mkdir(parents=True, exist_ok=False)
    copy_sources(ffn, manifest["sources"], output)
    copy_sources(active / "source", active_manifest["own_sources"], output / "body/source")
    own = output / "body/source/routed_int4"
    callable_body, _, numerical_hash = extractor.extract(
        (own / "routed_int4_kernel.cpp").read_text())
    (own / "routed_body_native.h").write_text(callable_body)
    validators, removed = host.host_seam(
        (own / "routed_int4_binding.cpp").read_text(),
        "ffn_active_body_host", ("routed_int4_launch",))
    (own / "host_validate.h").write_text(validators)
    contract = output / "source/contract.h"
    contract.write_text(replace_once(contract.read_text(),
        "static_assert(kBridgeBuffers == 12 && kBodyBuffers == 20);",
        "static_assert(kBridgeBuffers == 12 && kBodyBuffers == 21);"))
    kernel = output / "source/kernel.cpp"
    kernel.write_text(replace_once(kernel.read_text(),
        "    DispatchInput dispatch;",
        "    DispatchInput dispatch;\n"
        "    dispatch.active_list = (uint64_t)body.values[pipeline::kActiveList];"))
    for path in output.rglob("*"):
        if path.suffix not in (".h", ".cpp"):
            continue
        text = path.read_text().replace("ffn_composed", "ffn_active")
        for before, after in RENAMES.items():
            text = text.replace(before, after)
        path.write_text(text)
    for name in ("prepare.py", "build.py", "prepare_check.py", "README.md"):
        shutil.copy2(Path(__file__).with_name(name), output / "source" / name)
    for path, name in ((extractor_path, "extract_body.py"), (host_path, "extract_host.py")):
        shutil.copy2(path, output / "source" / name)
    report = {
        "parent_ffn_library_sha256": FFN_LIBRARY,
        "active_parent_kernel_sha256": ACTIVE_KERNEL,
        "active_numerical_body_sha256": numerical_hash,
        "parent_manifest_sha256": digest(ffn / "manifest.json"),
        "active_parent_manifest_sha256": digest(active / "build-manifest.json"),
        "removed_body_launch": removed,
        "changes": ["accepted ascending nonempty-expert enumeration",
                    "caller-owned pipe with unchanged active-body statements",
                    "one extra INT32[264] scratch tensor at body slot 20"],
        "body_internal_barriers": 6,
        "sources": {str(p.relative_to(output)): digest(p)
                    for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (output / "source-transforms.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(ffn / "manifest.json", output / "parent-manifest.json")
    shutil.copy2(ffn / "commands.jsonl", output / "parent-commands.jsonl")
    shutil.copy2(active / "build-manifest.json", output / "active-parent-manifest.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ffn", type=Path)
    parser.add_argument("active", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.ffn.resolve(), args.active.resolve(), args.output.resolve())
