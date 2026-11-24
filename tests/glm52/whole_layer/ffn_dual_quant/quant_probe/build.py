"""Build a direct two-quantizer differential from a frozen candidate tree."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    parent, output = args.parent.resolve(), args.output.resolve()
    manifest = json.loads((parent / "manifest.json").read_text())
    for name, expected in manifest["sources"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"Frozen quantizer source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(parent / "source", output / "source")
    shutil.copytree(parent / "bridge", output / "bridge")
    shutil.copytree(parent / "body", output / "body")
    own = output / "probe"
    own.mkdir()
    for name in ("kernel.cpp", "binding.cpp", "build.py", "check.py", "test_check.py"):
        shutil.copyfile(Path(__file__).with_name(name), own / name)
    commands = [json.loads(line) for line in
                (parent / "commands.jsonl").read_text().splitlines()]
    identities = {str(path.relative_to(output)): digest(path)
                  for path in output.rglob("*") if path.is_file()}
    compiled = []
    for stage in ("kernel", "binding", "link"):
        original = next(item["argv"] for item in commands if item["step"] == stage)
        argv = [item.replace(str(parent), str(output)) for item in original]
        if stage != "link":
            argv[argv.index("-c") + 1] = str(own / (stage + ".cpp"))
            argv.extend(("-I", str(output)))
        else:
            argv = [item for item in argv if not item.endswith(
                ("bridge-tiling.o", "routed-tiling.o", "shared-tiling.o"))]
            argv[argv.index("-o") + 1] = str(output / "quant_probe.so")
        compiled.append({"step": stage, "argv": argv})
        with (output / (stage + ".log")).open("w") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    import torch
    import torch_npu

    torch.ops.load_library(str(parent / "ffn_dual_quant.so"))
    torch.ops.load_library(str(output / "quant_probe.so"))
    if torch.npu.is_initialized():
        raise RuntimeError("The CPU probe build initialized NPU")
    for name, expected in identities.items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Source changed during build: {name}")
    result = {
        "parent_library_sha256": manifest["artifacts"]["ffn_dual_quant.so"],
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "scope": "Quantization only; fixed legal route metadata, no norm or TopK",
        "sources": identities, "commands": compiled,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "npu_initialized": False,
        "schema": str(torch.ops.tilert_ffn_dual_quant_probe.forward_out.default._schema),
        "artifacts": {name: digest(output / name)
                      for name in ("kernel.o", "binding.o", "quant_probe.so")},
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": True, "library_sha256":
                     result["artifacts"]["quant_probe.so"]}), flush=True)


if __name__ == "__main__":
    main()
