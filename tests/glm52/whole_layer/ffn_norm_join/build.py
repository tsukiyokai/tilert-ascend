"""Build the private norm-join FFN with its frozen parent's CANN commands."""

import argparse
import json
from pathlib import Path
import subprocess

from prepare import NAMESPACE, PARENT_LIBRARY, digest, prepare


def build(parent, output):
    if digest(parent / "ffn_original_shared_gu_static.so") != PARENT_LIBRARY:
        raise ValueError("The frozen parent library changed")
    report = prepare(parent, output)
    commands = [json.loads(line) for line in
                (parent / "commands.jsonl").read_text().splitlines()]
    kernel = next(item["argv"] for item in commands if item["step"] == "kernel")
    original = str(Path(kernel[kernel.index("-o") + 1]).parent)
    if "-DSHARED_W8_DYNAMIC_PERFORMANCE_QUANT=1" not in kernel:
        raise ValueError("Use the CANN key30013 shared-activation parent")
    for command in commands:
        argv = [arg.replace(original, str(output)).replace(
            "/ffn_original_shared_gu_static.so", "/" + NAMESPACE + ".so")
            for arg in command["argv"]]
        if any("ROUTED_ENABLE_PROFILE" in arg for arg in argv):
            raise ValueError("Use the uninstrumented parent")
        with (output / "commands.jsonl").open("a") as log:
            log.write(json.dumps({"step": command["step"], "argv": argv}) + "\n")
        with (output / (command["step"] + ".log")).open("w") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    for name, expected in report["sources"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Source changed during compilation: {name}")
    import torch
    import torch_npu

    torch.ops.load_library(str(parent / "ffn_original_shared_gu_static.so"))
    torch.ops.load_library(str(output / (NAMESPACE + ".so")))
    schemas = {}
    for name in ("prepare", "forward_out"):
        old_operator = getattr(torch.ops.tilert_ffn_original_shared_gu_static,
                               name)
        new_operator = getattr(getattr(torch.ops, "tilert_" + NAMESPACE), name)
        old = str(old_operator.default._schema)
        new = str(new_operator.default._schema)
        if new != old.replace("ffn_original_shared_gu_static", NAMESPACE):
            raise RuntimeError(f"The public schema changed: {name}")
        schemas[name] = new
    if torch.npu.is_initialized():
        raise RuntimeError("The CPU build initialized NPU")
    report.update({
        "namespace": "tilert_" + NAMESPACE,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "parent_co_load": True, "npu_initialized": False, "schemas": schemas,
        "artifacts": {path.name: digest(path) for path in output.iterdir()
                      if path.suffix in (".so", ".o")},
        "device_validation": "not run",
    })
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": True, "library_sha256":
                     report["artifacts"][NAMESPACE + ".so"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.parent.resolve(), args.output.resolve())
