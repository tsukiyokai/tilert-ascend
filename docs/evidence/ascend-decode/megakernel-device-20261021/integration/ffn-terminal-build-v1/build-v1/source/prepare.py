"""Apply body argument substitution to the frozen active-expert FFN."""

import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil


PARENT_LIBRARY = (
    "6a6241da7d7a5e2e4a2ba9396d20b902f0ca19fa2e4af224a7e7da41a3dc6513"
)
PARENT_CHECKER = (
    "4444cbe1d6362bcbe4e58f8d77e5acfd83b108ce8381a034484848b30d24fd1d"
)
DESCRIPTOR_TRANSFORM = (
    "41f2edfe75f5bb6aa24eded9fc4277450ac8310551cb4997f77f5426b4e2781b"
)
NAMESPACE = "ffn_active_descriptor"
BODY = "body/source/routed_int4/routed_body_native.h"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_transform():
    path = Path(__file__).resolve().parent.parent / "ffn_descriptor/prepare.py"
    if digest(path) != DESCRIPTOR_TRANSFORM:
        raise ValueError("The validated descriptor transform changed")
    spec = importlib.util.spec_from_file_location("descriptor_transform", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def rename(source):
    source = re.sub(r"ffn_(?:active|descriptor|composed)", NAMESPACE, source)
    return re.sub(r"FFN_(?:ACTIVE|DESCRIPTOR|COMPOSED)",
                  "FFN_ACTIVE_DESCRIPTOR", source)


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"]["ffn_active.so"] != PARENT_LIBRARY:
        raise ValueError("Use the validated active FFN build-v1")
    for name, expected in manifest["sources"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"Frozen active source changed: {name}")
    checker = parent / "harness-v1/check.py"
    if digest(checker) != PARENT_CHECKER:
        raise ValueError("The active FFN numerical checker changed")
    transform, helper_path = load_transform()
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["sources"]:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, destination)

    before_body = (parent / BODY).read_text()
    body, numerical, _ = transform.bind_body(before_body)
    (output / BODY).write_text(body)
    before_kernel = (parent / "source/kernel.cpp").read_text()
    # The host/kernel binding seam has the same structure in both parents.
    kernel = transform.bind_kernel(
        before_kernel.replace("ffn_active", "ffn_composed"))
    (output / "source/kernel.cpp").write_text(kernel)
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(rename(path.read_text()))
    for name in ("descriptor.h", "prepare.py", "build.py",
                 "test_prepare.py", "descriptor_probe.cpp", "README.md"):
        shutil.copy2(Path(__file__).with_name(name), output / "source" / name)
    shutil.copy2(helper_path, output / "source/descriptor_transform.py")
    (output / "check.py").write_text(checker.read_text().replace(
        "torch.ops.tilert_ffn_active", "torch.ops.tilert_ffn_active_descriptor"))
    patches = []
    for name, before in ((BODY, before_body),
                         ("source/kernel.cpp", before_kernel)):
        patches.extend(difflib.unified_diff(
            before.splitlines(keepends=True),
            (output / name).read_text().splitlines(keepends=True),
            fromfile=f"parent/{name}", tofile=f"candidate/{name}"))
    (output / "descriptor.patch").write_text("".join(patches))
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "parent_checker_sha256": PARENT_CHECKER,
        "descriptor_transform_sha256": DESCRIPTOR_TRANSFORM,
        "descriptor_bytes": 408,
        "additional_kernel_parameter_bytes": 64,
        "additional_gm_bytes_vs_active_parent": 0,
        "scratch_tensors": {"bridge": 12, "body": 21},
        "body_statement_sha256": hashlib.sha256(numerical.encode()).hexdigest(),
        "body_statement_equivalence": (
            "Exact after reversing the validated parameter substitutions"
        ),
        "body_parameter_bindings": transform.BINDINGS,
        "sources": {str(p.relative_to(output)): digest(p)
                    for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (output / "source-transforms.json").write_text(
        json.dumps(report, indent=2) + "\n")
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.output.resolve())
