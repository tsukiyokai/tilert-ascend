"""Keep full-attention joins for callable use and drain locally at kernel exit."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c"
LAYER = "tests/glm52/whole_layer/"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one frozen source seam: {before!r}")
    return source.replace(before, after, 1)


def terminal_full(source):
    source = replace_once(source, "__aicore__ inline void FullForward(\n",
                          "template <bool Terminal = false>\n"
                          "__aicore__ inline void FullForward(\n")
    return replace_once(source, "    EndPhase(pipe);\n", """    if constexpr (Terminal) {
        // The global task boundary orders consumers in subsequent kernels.
        // Callable use still requires every output producer to join here.
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    } else {
        EndPhase(pipe);
    }
""")


def terminal_kernel(source):
    return replace_once(source, "    FullForward(pipe, config, args,",
                        "    FullForward<true>(pipe, config, args,")


def private_names(source):
    return (source.replace("whole_layer_attention", "whole_layer_attention_terminal")
            .replace("whole_layer_full_attention", "whole_layer_full_attention_terminal")
            .replace("namespace mla_tail {", "namespace mla_tail_attention_terminal {")
            .replace("mla_tail::", "mla_tail_attention_terminal::")
            .replace("mla_tail_build_tiling", "mla_tail_attention_terminal_build_tiling")
            .replace("TORCH_LIBRARY_FRAGMENT(tilert,", "TORCH_LIBRARY_FRAGMENT(tilert_attention_terminal,")
            .replace("TORCH_LIBRARY_IMPL(tilert,", "TORCH_LIBRARY_IMPL(tilert_attention_terminal,"))


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    shared = next(item for item in manifest["builds"] if item["mode"] == "shared")
    if shared["library_sha256"] != PARENT_LIBRARY:
        raise ValueError("Use the frozen runtime212 shared-attention parent")
    parent_source = parent / "shared/source"
    for name, expected in shared["source_hashes"].items():
        if digest(parent_source / name) != expected:
            raise ValueError(f"Frozen parent source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    source_root = output / "source"
    transforms = {
        LAYER + "attention_full_native.h": terminal_full,
        LAYER + "attention_full_kernel.cpp": terminal_kernel,
    }
    changes = []
    for name in shared["source_hashes"]:
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        old = (parent_source / name).read_text()
        new = transforms[name](old) if name in transforms else old
        if old != new:
            changes.append("".join(difflib.unified_diff(
                old.splitlines(keepends=True), new.splitlines(keepends=True),
                fromfile="parent/" + name, tofile="candidate/" + name)))
        target.write_text(private_names(new) if target.suffix in (".cpp", ".h") else new)
        shutil.copymode(parent_source / name, target)
    (output / "terminal.patch").write_text("".join(changes))
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "scope": "Standalone global entry; default callable retains final EndPhase",
        "additional_gm_bytes": 0,
        "additional_ub_bytes": 0,
        "sources": {str(path.relative_to(source_root)): digest(path)
                    for path in sorted(source_root.rglob("*")) if path.is_file()},
    }
    (output / "source-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent, args.output)
