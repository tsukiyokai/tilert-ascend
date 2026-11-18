"""Build isolated CANN/SHMEM control and RSAG probes from frozen source."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from prepare import digest


EXPORTS = ("initialize", "prepare", "execute", "destroy", "finalize", "last_error")


def build(output, cann, shmem, library, compiler):
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["sources"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    report = {"cann": str(cann), "shmem": str(shmem),
              "shmem_library": {"path": str(library), "sha256": digest(library)},
              "prepared_manifest_sha256": digest(output / "manifest.json"),
              "arms": {}}
    for arm in ("control", "candidate"):
        source = output / arm / "source"
        destination = output / arm / "build"
        if destination.exists():
            raise FileExistsError(destination)
        environment = {**os.environ, "CANN_ROOT": str(cann),
                       "SHMEM_SOURCE": str(shmem), "SHMEM_LIBRARY": str(library),
                       "BUILD_DIR": str(destination), "BISHENG_BIN": str(compiler)}
        command = ["bash", str(source / "communication_build.sh")]
        with (output / arm / "build.log").open("w") as log:
            subprocess.run(command, env=environment, check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        artifact = destination / "communication_probe.so"
        symbols = subprocess.run(["nm", "-D", "--defined-only", str(artifact)],
                                 capture_output=True, text=True, check=True).stdout
        (destination / "exports.txt").write_text(symbols)
        names = {line.split()[-1] for line in symbols.splitlines()}
        prefix = manifest["arms"][arm]["c_prefix"]
        for export in EXPORTS:
            if prefix + export not in names:
                raise ValueError(f"Missing private C ABI: {prefix + export}")
            for foreign in ("communication_probe_", "communication_tree_probe_",
                            "trt_rsag_" + ("candidate" if arm == "control"
                                           else "control") + "_probe_"):
                if foreign + export in names:
                    raise ValueError("Foreign C ABI leaked into the private DSO")
        report["arms"][arm] = {
            "command": command, "c_prefix": prefix,
            "library": {"path": str(artifact), "sha256": digest(artifact)},
            "artifacts": {str(path.relative_to(output)): digest(path)
                          for path in sorted(destination.iterdir())
                          if path.is_file()},
        }
        (output / "build-report.json").write_text(
            json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--shmem", type=Path, required=True)
    parser.add_argument("--shmem-library", type=Path, required=True)
    parser.add_argument("--compiler", type=Path)
    args = parser.parse_args()
    cann = args.cann.resolve()
    print(json.dumps(build(
        args.prepared.resolve(), cann, args.shmem.resolve(),
        args.shmem_library.resolve(),
        args.compiler.resolve() if args.compiler else cann / "bin/bisheng",
    ), indent=2))


if __name__ == "__main__":
    main()
