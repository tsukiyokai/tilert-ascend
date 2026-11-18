"""Prepare four private CANN SFA headers with staged gather metadata."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
RELATIVE = Path("attention/sparse_flash_attention/op_kernel/arch22")
VECTOR = "sparse_flash_attention_service_vector_mla.h"
SOURCE_HASHES = {
    "sparse_flash_attention_common.h":
        "d3f5204b5602ca3aa677812103162d504b2d220415f683df6e7767d140bad632",
    "sparse_flash_attention_kernel_mla.h":
        "f6c6b092ccfe09e65e9fa371b29de153d800918290ef2a08f799f0ac53f2bacd",
    "sparse_flash_attention_service_cube_mla.h":
        "63db752f6168d38029392d331c60891efe40975e3e175136d1e00e647aa74aaf",
    VECTOR:
        "681ff5278aa25abaa2741fbbd9528aedba19394548f9d98be069f71046b341bd",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method(source, name):
    """Extract a complete instantiated service method, including its body."""
    marker = "SFAVectorService<SFAT>::" + name + "("
    start = source.index(marker)
    start = source.rfind("template <typename SFAT>", 0, start)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def prepare(source, output):
    observed = {
        name: digest(source / RELATIVE / name) for name in SOURCE_HASHES
    }
    if observed != SOURCE_HASHES:
        raise ValueError("SFA source differs from the audited version")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / RELATIVE
    destination.mkdir(parents=True)
    for name in SOURCE_HASHES:
        shutil.copy2(source / RELATIVE / name, destination / name)
    patch = HERE / "metadata.patch"
    result = subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", "-p1",
         "--input", str(patch)],
        cwd=output, capture_output=True, text=True, check=True,
    )
    (output / "patch.log").write_text(result.stdout + result.stderr)
    original = (source / RELATIVE / VECTOR).read_text()
    candidate = (destination / VECTOR).read_text()
    preserved = (
        "InitBuffers", "CopyInKv", "CopyInSingleKv", "CopyOutMrgeResult",
        "GetKeyRopeGmOffset", "ProcessVec1L", "ProcessVec2L", "RowDivs",
    )
    for name in preserved:
        if method(original, name) != method(candidate, name):
            raise ValueError(f"Unexpected arithmetic or buffer change: {name}")
    suffix = "SFAVectorService<SFAT>::ProcessVec1L("
    if original.split(suffix, 1)[1] != candidate.split(suffix, 1)[1]:
        raise ValueError("Post-gather arithmetic changed")
    report = {
        "source_root": str(source), "overlay_root": str(output),
        "source_sha256": observed,
        "overlay_sha256": {
            name: digest(destination / name) for name in SOURCE_HASHES
        },
        "patch_sha256": digest(patch),
        "prepare_sha256": digest(Path(__file__)),
        "preserved_methods": list(preserved),
        "additional_ub_bytes": 0,
        "metadata_ub_bytes": {"topk": [2048, 3072],
                              "page_table": [4096, 8192]},
        "extra_event": "paired AIV MTE2_S(1), no cross-core event",
        "device_validation": "pending",
    }
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source.resolve(), args.output.resolve()),
                     indent=2))


if __name__ == "__main__":
    main()
