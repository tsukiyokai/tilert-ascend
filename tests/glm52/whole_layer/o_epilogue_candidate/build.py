#!/usr/bin/env python3
"""Build an isolated O-projection scale-order candidate from frozen sources."""

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


TAIL = Path("tests/glm52/mla_tail/mla_tail_native.h")
OLD_CALL = "        DequantizeOutputRows(pipe, accumulators, output_scale, scales, bias,"
NEW_CALL = (
    "        DequantizeOutputRows<ScaleOrder::CombinedScales>(\n"
    "            pipe, accumulators, output_scale, scales, bias,"
)
PARENT_LIBRARIES = {
    "indexed": "5d5e8ddb6527bcf50ee4e90507723d2b37cd66ea5f7a4734f7ce5995b13e46f7",
    "shared": "245ae243c779594664ebe80ce91b5f84bc133457d47da56734261bce2347ee0f",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def verify(path, expected):
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual}")


def source_hashes(root):
    return {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def workspace_path(value, workspace):
    return workspace / Path(value).relative_to("/workspace")


def check_shared_parent(shared, indexed):
    """Allow only the already accepted optional epilogue template extension."""
    shared_prefix, shared_tail = shared.split(
        "__aicore__ inline void DequantizeOutputRows(", 1)
    indexed_prefix, indexed_tail = indexed.split("enum class ScaleOrder", 1)
    shared_prefix = shared_prefix.replace(
        "CANN per-token epilogues apply channel scale before token scale.",
        "Preserve the channel-then-token order of this projection's epilogue.",
    )
    if shared_prefix.rstrip() != indexed_prefix.rstrip():
        raise RuntimeError("Shared parent differs outside the epilogue helper")
    if shared_tail.split("__aicore__ inline void Forward(", 1)[1] != (
        indexed_tail.split("__aicore__ inline void Forward(", 1)[1]
    ):
        raise RuntimeError("The parent Forward implementations differ")


def prepare(args):
    parent_path = args.workspace / "logs/attention-torch210-v2/manifest.json"
    parent = json.loads(parent_path.read_text())
    if parent["torch"] != "2.10.0+cpu" or parent["torch_npu"] != "2.10.0.post2":
        raise RuntimeError("Expected the accepted Torch 2.10 binding build")
    indexed = next(b for b in parent["builds"] if b["mode"] == "indexed")
    indexed_root = workspace_path(indexed["source_root"], args.workspace)
    indexed_tail = (indexed_root / TAIL).read_text()
    verify(indexed_root / TAIL, indexed["source_hashes"][str(TAIL)])
    if indexed_tail.count(OLD_CALL) != 1:
        raise RuntimeError("Expected one default epilogue call in Forward")
    candidate_tail = indexed_tail.replace(OLD_CALL, NEW_CALL)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), args.output / "build.py")
    shutil.copyfile(parent_path, args.output / "parent-manifest.json")
    common = args.output / "common"
    common.mkdir()
    (common / "mla_tail_native.h").write_text(candidate_tail)
    report = {
        "scope": "O projection only: FP32(channel * token) then FP32(acc * scale)",
        "parent_manifest_sha256": digest(parent_path),
        "builder_sha256": digest(Path(__file__)),
        "common_tail_sha256": digest(common / "mla_tail_native.h"),
        "added_epilogue_ub_bytes": 1024,
        "actual_runtime_tiling_key": None,
        "builds": [],
    }
    for original in parent["builds"]:
        mode = original["mode"]
        if original["library_sha256"] != PARENT_LIBRARIES[mode]:
            raise RuntimeError(f"Unexpected parent library for {mode}")
        old_output = args.workspace / "logs/attention-torch210-v2" / mode
        old_library = old_output / Path(original["library"]).name
        verify(old_library, PARENT_LIBRARIES[mode])
        source_root = workspace_path(original["source_root"], args.workspace)
        for relative, expected in original["source_hashes"].items():
            verify(source_root / relative, expected)
        before = (source_root / TAIL).read_text()
        if mode == "shared":
            check_shared_parent(before, indexed_tail)
        output = args.output / mode
        snapshot = output / "source"
        shutil.copytree(source_root, snapshot)
        (snapshot / TAIL).write_text(candidate_tail)
        difference = difflib.unified_diff(
            before.splitlines(keepends=True),
            candidate_tail.splitlines(keepends=True),
            fromfile=f"accepted-{mode}/{TAIL}", tofile=f"candidate-{mode}/{TAIL}",
        )
        (output / "tail.patch").write_text("".join(difference))
        stem = "attention_indexed" if mode == "indexed" else "attention_full"
        reused = {f"{stem}_binding.o": original["binding_sha256"]}
        reused.update({name: item["sha256"]
                       for name, item in original["reused_objects"].items()
                       if not name.endswith("_kernel.o")})
        for name, expected in reused.items():
            verify(old_output / name, expected)
            shutil.copyfile(old_output / name, output / name)
        old_source = workspace_path(original["accepted_library"], args.workspace)
        old_provenance = old_source.parent / "provenance.json"
        shutil.copyfile(old_provenance, output / "parent-provenance.json")
        report["builds"].append({
            "mode": mode, "stem": stem,
            "parent_library": str(old_library),
            "parent_library_sha256": PARENT_LIBRARIES[mode],
            "source_parent": str(source_root),
            "source_hashes": source_hashes(snapshot),
            "parent_source_hashes": original["source_hashes"],
            "reused_objects": reused,
            "library_name": old_library.name,
        })
    save_json(args.output / "manifest.json", report)


def compile_kernels(args):
    report_path = args.output / "manifest.json"
    report = json.loads(report_path.read_text())
    for item in report["builds"]:
        output = args.output / item["mode"]
        snapshot = output / "source"
        if source_hashes(snapshot) != item["source_hashes"]:
            raise RuntimeError("Frozen source identity changed")
        provenance = json.loads((output / "parent-provenance.json").read_text())
        for relative, expected in provenance["vendor_headers"].items():
            verify(args.vendor / relative, expected)
        command = ["bash", str(snapshot / "tests/glm52/whole_layer/attention_build.sh"),
                   str(output)]
        env = dict(os.environ)
        for name in tuple(env):
            if name.startswith("ATTENTION_") or name == "SANITIZE":
                del env[name]
        env.update({"CANN_ROOT": str(args.cann),
                    "OPS_TRANSFORMER_ROOT": str(args.vendor),
                    "ATTENTION_KERNEL_ONLY": "1", "PYTHON_BIN": "python3"})
        env["ATTENTION_INDEXED" if item["mode"] == "indexed" else "ATTENTION_FULL"] = "1"
        with (output / "kernel-build.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        item["kernel_build_command"] = command
        item["kernel_build_env"] = {
            key: value for key, value in env.items()
            if key.startswith("ATTENTION_") or key in (
                "CANN_ROOT", "OPS_TRANSFORMER_ROOT", "PYTHON_BIN")
        }
        item["kernel_sha256"] = digest(output / f"{item['stem']}_kernel.o")
        save_json(report_path, report)


def link_and_load(args):
    import torch
    import torch_npu

    if torch.npu.is_initialized():
        raise RuntimeError("CPU link check requires an uninitialized NPU")
    if torch.__version__ != "2.10.0+cpu" or torch_npu.__version__ != "2.10.0.post2":
        raise RuntimeError("Use the recorded Torch 2.10 image")
    devices = [str(p) for p in Path("/dev").glob("davinci*")]
    if devices:
        raise RuntimeError(f"CPU link container has NPU device nodes: {devices}")
    report_path = args.output / "manifest.json"
    report = json.loads(report_path.read_text())
    parent = json.loads((args.output / "parent-manifest.json").read_text())
    for item, original in zip(report["builds"], parent["builds"], strict=True):
        output = args.output / item["mode"]
        for name, expected in item["reused_objects"].items():
            verify(output / name, expected)
        verify(output / f"{item['stem']}_kernel.o", item["kernel_sha256"])
        library = output / item["library_name"]
        command = [str(output / Path(value).name)
                   if value.endswith((".o", ".so")) else value
                   for value in original["link_command"]]
        with (output / "link.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        torch.ops.load_library(str(library))
        item["link_command"] = command
        item["library_sha256"] = digest(library)
        save_json(report_path, report)
    report["schemas"] = {
        name: str(getattr(torch.ops.tilert, name).default._schema)
        for name in parent["schemas"]
    }
    if report["schemas"] != parent["schemas"]:
        raise RuntimeError("Candidate schemas differ from accepted schemas")
    report["cpu_load"] = {
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "npu_initialized": torch.npu.is_initialized(), "device_nodes": devices,
    }
    if report["cpu_load"]["npu_initialized"]:
        raise RuntimeError("CPU check initialized the NPU")
    save_json(report_path, report)
    print(json.dumps({"builds": report["builds"],
                      "cpu_load": report["cpu_load"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "kernels", "link"))
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cann", type=Path,
                        default=Path("/opt/ascend/cann-9.2.0"))
    parser.add_argument("--vendor", type=Path,
                        default=Path("/workspace/attention_vendor"))
    args = parser.parse_args()
    {"prepare": prepare, "kernels": compile_kernels,
     "link": link_and_load}[args.stage](args)


if __name__ == "__main__":
    main()
