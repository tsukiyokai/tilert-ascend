"""Run actual patched gather methods against instrumented host memory."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from prepare import HERE, RELATIVE, VECTOR, digest, method, prepare


def execute(original, candidate, directory):
    methods = ("GetRealS2Idx", "GetKeyGmOffset", "MergeKv")
    extracted = []
    for service, source in (("OriginalService", original),
                            ("StagedService", candidate)):
        names = methods if service == "OriginalService" else (
            "StageMetadata", *methods
        )
        extracted.extend(
            method(source, name).replace("SFAVectorService", service)
            for name in names
        )
    (directory / "extracted.h").write_text("\n\n".join(extracted) + "\n")
    binary = directory / "host-check"
    compilation = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
         "-Wno-sign-compare", "-I", str(directory),
         str(HERE / "host_check.cpp"), "-o", str(binary)],
        capture_output=True, text=True, check=True,
    )
    if compilation.stderr:
        raise RuntimeError(compilation.stderr)
    return subprocess.run([str(binary)], capture_output=True, text=True)


def check(source):
    with tempfile.TemporaryDirectory(prefix="trt-sfa-metadata-") as temporary:
        directory = Path(temporary)
        output = directory / "overlay"
        manifest = prepare(source, output)
        original = (source / RELATIVE / VECTOR).read_text()
        candidate = (output / RELATIVE / VECTOR).read_text()
        result = execute(original, candidate, directory)
        if result.returncode:
            raise AssertionError(result.stderr)
        report = {"host_result": result.stdout.strip(),
                  "overlay_sha256": manifest["overlay_sha256"],
                  "device_validation": "pending", "negative_controls": {}}
        report["check_source_sha256"] = {
            name: digest(HERE / name) for name in (
                "check.py", "host_check.cpp", "prepare.py", "metadata.patch"
            )
        }
        report["host_compiler"] = subprocess.run(
            ["g++", "--version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0]
        mutations = {
            "live_ub_overlap": (
                "v0ValidSizeBuff.Get<int32_t>()[512]",
                "v0ValidSizeBuff.Get<int32_t>()[128]",
            ),
            "page_buffer_overrun": (
                "v0ValidSizeBuff.Get<int32_t>()[1024]",
                "v0ValidSizeBuff.Get<int32_t>()[2040]",
            ),
            "missing_dma_wait": (
                "        WaitFlag<AscendC::HardEvent::MTE2_S>(1);",
                "",
            ),
            "stale_fallback_state": (
                "    metadataReady_ = false;",
                "",
            ),
            "wrong_split_offset": (
                "metadataTopk_.GetValue(topkGmIdx - metadataBegin_)",
                "metadataTopk_.GetValue(topkGmIdx)",
            ),
        }
        for name, (before, after) in mutations.items():
            if candidate.count(before) != 1:
                raise AssertionError(f"Mutation anchor changed: {name}")
            result = execute(original, candidate.replace(before, after),
                             directory)
            if result.returncode == 0:
                raise AssertionError(f"Fault not detected: {name}")
            report["negative_controls"][name] = result.stderr.strip()
        try:
            prepare(source, output)
        except FileExistsError:
            report["existing_destination_rejected"] = True
        else:
            raise AssertionError("Preparation overwrote an existing overlay")
        for name, sha in manifest["source_sha256"].items():
            if digest(source / RELATIVE / name) != sha:
                raise AssertionError("Source tree changed")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not shutil.which("g++"):
        parser.error("g++ is required to execute the extracted C++ methods")
    report = check(args.source.resolve())
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
