"""Extract the validated rank-local FFN with a caller-owned pipe."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "3673c2a73b2f4b5a4a93b3c2543a4aca35b45773d6a53fc043f27e59a2482655"
INITIALIZE = """    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync);
    SetSysWorkspace(bridge.values[ffn_bridge::kSystem]);
    TPipe pipe;
"""
CALL = """    Forward<format>(pipe, input, residual, gamma, gate, bias,
        w, bridge, body, shared, rows, blocks, epsilon);
"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(source):
    start = source.index("template <CubeFormat format>\n")
    body_start = source.index("\n{\n", start) + 3
    end = source.index("\n}\n}  // namespace ffn_composed_device", body_start)
    declaration = source[start:body_start]
    body = source[body_start:end]
    if not body.startswith(INITIALIZE):
        raise ValueError("Unexpected FFN entry initialization")
    numerical = body[len(INITIALIZE):]
    if not numerical.rstrip().endswith("ffn_bridge::device::EndPhase(pipe);"):
        raise ValueError("Callable FFN must retain its final publication barrier")
    callable_declaration = declaration.replace(
        "__global__ __mix__(1, 2) __schedmode__(1) __aicore__\n",
        "__aicore__ inline\n", 1).replace(
        "void Kernel(uint64_t sync,", "void Forward(TPipe& pipe,", 1)
    header = ("#ifndef TILERT_FFN_CALLABLE_NATIVE_H\n"
              "#define TILERT_FFN_CALLABLE_NATIVE_H\n\n"
              + source[:start] + callable_declaration + numerical
              + "\n}\n}  // namespace ffn_composed_device\n\n#endif\n")
    wrapper = ('#include "native.h"\n\nnamespace ffn_composed_device {\n'
               + declaration + INITIALIZE + CALL + source[end:])
    return header, wrapper, numerical


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"]["ffn_composed.so"] != PARENT_LIBRARY:
        raise ValueError("Use the validated composed build-v2 parent")
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in manifest["sources"].items():
        source = parent / name
        if digest(source) != expected:
            raise ValueError(f"Frozen parent source changed: {name}")
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    kernel = output / "source/kernel.cpp"
    header, wrapper, statements = extract(kernel.read_text())
    kernel.with_name("native.h").write_text(header)
    kernel.write_text(wrapper)
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(path.read_text().replace(
                "ffn_composed", "ffn_callable").replace(
                "FFN_COMPOSED", "FFN_CALLABLE"))
    if statements not in kernel.with_name("native.h").read_text():
        raise ValueError("Symbol changes altered an original numerical statement")
    for name in ("prepare.py", "build.py", "README.md", "test_prepare.py"):
        shutil.copy2(Path(__file__).with_name(name), output / "source" / name)
    parent_checker = parent / "harness-v1/check.py"
    if parent_checker.is_file():
        checker = parent_checker.read_text()
        (output / "check.py").write_text(checker.replace(
            "torch.ops.tilert_ffn_composed", "torch.ops.tilert_ffn_callable"))
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "parent_kernel_sha256": manifest["sources"]["source/kernel.cpp"],
        "parent_statement_sha256": hashlib.sha256(statements.encode()).hexdigest(),
        "callable_statement_sha256": hashlib.sha256(statements.encode()).hexdigest(),
        "statement_bytes": len(statements.encode()),
        "symbol_rename": {"ffn_composed": "ffn_callable",
                          "FFN_COMPOSED": "FFN_CALLABLE"},
        "entry_contract": "Caller initializes SoC/sync once, selects bridge system workspace, and supplies a drained/reset TPipe",
        "return_contract": "Original final EndPhase publishes FFN outputs and resets TPipe; FFN TP reduction remains outside",
        "sources": {str(p.relative_to(output)): digest(p)
                    for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (output / "source-transforms.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "parent-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.output.resolve())
