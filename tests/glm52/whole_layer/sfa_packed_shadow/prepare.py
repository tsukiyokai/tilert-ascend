"""Create a private, hash-checked SFA packed-cache include overlay."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
RELATIVE = Path("attention/sparse_flash_attention/op_kernel/arch22")
ORIGINAL = {
    "sparse_flash_attention_common.h":
        "d3f5204b5602ca3aa677812103162d504b2d220415f683df6e7767d140bad632",
    "sparse_flash_attention_kernel_mla.h":
        "f6c6b092ccfe09e65e9fa371b29de153d800918290ef2a08f799f0ac53f2bacd",
    "sparse_flash_attention_service_cube_mla.h":
        "63db752f6168d38029392d331c60891efe40975e3e175136d1e00e647aa74aaf",
    "sparse_flash_attention_service_vector_mla.h":
        "681ff5278aa25abaa2741fbbd9528aedba19394548f9d98be069f71046b341bd",
}
VECTOR = "sparse_flash_attention_service_vector_mla.h"
PATCHED_VECTOR = "f5620780b20befe34f493c34ab3fc409272a36f1af5c1d1c7e567d0848291cd8"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output):
    for name, expected in ORIGINAL.items():
        path = source / RELATIVE / name
        if digest(path) != expected:
            raise ValueError(f"unsupported SFA source identity: {path}")
    if output.exists():
        raise FileExistsError(f"choose a new overlay directory: {output}")
    destination = output / RELATIVE
    destination.mkdir(parents=True)
    for name in ORIGINAL:
        shutil.copy2(source / RELATIVE / name, destination / name)
    patch = HERE / "packed_gather.patch"
    result = subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", "-p1",
         "--input", str(patch)], cwd=output, text=True,
        capture_output=True, check=True)
    (output / "patch.log").write_text(result.stdout + result.stderr)
    expected = {**ORIGINAL, VECTOR: PATCHED_VECTOR}
    for name, sha in expected.items():
        if digest(destination / name) != sha:
            raise ValueError(f"unexpected patched source identity: {name}")
        if digest(source / RELATIVE / name) != ORIGINAL[name]:
            raise ValueError(f"input source changed while preparing: {name}")
    before = (source / RELATIVE / VECTOR).read_text()
    after = (destination / VECTOR).read_text()
    suffix = "// b s1 k\n"
    if before.split(suffix, 1)[1] != after.split(suffix, 1)[1]:
        raise ValueError("MergeKv or subsequent arithmetic changed")
    report = {
        "source_root": str(source), "overlay_root": str(output),
        "original_headers": ORIGINAL, "overlay_headers": expected,
        "patch_sha256": digest(patch), "prepare_sha256": digest(Path(__file__)),
        "macro": "SFA_PACKED_SHADOW_KV",
        "key_argument": "BF16 ND [physical pages,128,576] packed shadow",
        "untouched_suffix": "MergeKv and all subsequent vector arithmetic",
        "geometry": {"kv_heads": 1, "latent": 512, "rope": 64,
                     "sparse_block_size": 1, "page_size": 128,
                     "s2_base": 512, "preload": 2},
        "vector_ub_bytes": 178 * 1024,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.source.resolve(), args.output.resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
