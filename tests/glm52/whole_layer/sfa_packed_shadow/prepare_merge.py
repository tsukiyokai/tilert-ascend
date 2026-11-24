"""Create the independent packed-gather/packed-merge SFA include overlay."""

import argparse
import json
from pathlib import Path
import re
import subprocess

import prepare as gather


HERE = Path(__file__).resolve().parent
CUBE = "sparse_flash_attention_service_cube_mla.h"
PATCHED = {
    gather.VECTOR: "5c65d6676b856a1e0ef26598b9a9cecfe1ab27db7139d922765ecca5fd6a8847",
    CUBE: "a99ad8dab2ac1156c5635a4072de58d7338018d94d1aa67d3f98fb6d3acbfc38",
}


def block(text, marker):
    start = text.index(marker)
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return start, end


def function(text, signature):
    start, end = block(text, signature)
    return text[start:end]


def omit_block(text, marker):
    start, end = block(text, marker)
    return text[:start] + "<load/store layout block>" + text[end:]


def verify_arithmetic(before_vector, after_vector, before_cube, after_cube):
    vector_suffix = "__aicore__ inline void SFAVectorService<SFAT>::ProcessVec1L"
    if before_vector.split(vector_suffix, 1)[1] != after_vector.split(vector_suffix, 1)[1]:
        raise ValueError("post-merge vector arithmetic changed")
    merge = "void SFAVectorService<SFAT>::MergeKv("
    zero = "if (unlikely(s2GmStartOffset + mte2Size < s2GmLimit))"
    if (omit_block(function(before_vector, merge), zero) !=
            omit_block(function(after_vector, merge), zero)):
        raise ValueError("MergeKv ordering or valid-count publication changed")
    event = re.compile(r"(?:SetFlag|WaitFlag|CrossCoreSetFlag|CrossCoreWaitFlag|PipeBarrier)"
                       r"\s*<[^;]+;", re.MULTILINE)
    for before, after in ((before_vector, after_vector), (before_cube, after_cube)):
        if event.findall(before) != event.findall(after):
            raise ValueError("a synchronization primitive changed")
    for name in ("ComputeMm1", "ComputeMm2"):
        signature = f"void SFAMatmulService<SFAT>::{name}("
        before = function(before_cube, signature)
        after = function(after_cube, signature)
        marker = "if constexpr (TEMPLATE_MODE == V_TEMPLATE)"
        if omit_block(before, marker) != omit_block(after, marker):
            raise ValueError(f"{name} arithmetic or task order changed")
    signature = "void SFAMatmulService<SFAT>::ComputeMm1NoRope("
    if function(before_cube, signature) != function(after_cube, signature):
        raise ValueError("the excluded no-RoPE path changed")


def prepare(source, output):
    gather_report = gather.prepare(source, output)
    destination = output / gather.RELATIVE
    before_vector = (destination / gather.VECTOR).read_text()
    before_cube = (destination / CUBE).read_text()
    (output / "manifest.json").rename(output / "gather-manifest.json")
    patch = HERE / "packed_merge.patch"
    result = subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "--input", str(patch)],
        cwd=output, text=True, capture_output=True, check=True)
    (output / "merge-patch.log").write_text(result.stdout + result.stderr)
    expected = {**gather.ORIGINAL, **PATCHED}
    for name, sha in expected.items():
        if gather.digest(destination / name) != sha:
            raise ValueError(f"unexpected packed-merge source identity: {name}")
        if gather.digest(source / gather.RELATIVE / name) != gather.ORIGINAL[name]:
            raise ValueError(f"original vendor source changed: {name}")
    verify_arithmetic(before_vector, (destination / gather.VECTOR).read_text(),
                      before_cube, (destination / CUBE).read_text())
    report = {
        "variant": "packed-gather-and-merge-v2",
        "source_root": str(source), "overlay_root": str(output),
        "original_headers": gather.ORIGINAL,
        "gather_headers": gather_report["overlay_headers"],
        "overlay_headers": expected,
        "patches_sha256": {
            "packed_gather.patch": gather.digest(HERE / "packed_gather.patch"),
            "packed_merge.patch": gather.digest(patch)},
        "prepare_sha256": gather.digest(Path(__file__)),
        "gather_prepare_sha256": gather.digest(HERE / "prepare.py"),
        "macro": "SFA_PACKED_SHADOW_KV",
        "key_argument": "BF16 ND [physical pages,128,576] packed shadow",
        "merge_workspace": "four BF16 [512,576] row-major planes per AIC group",
        "geometry": gather_report["geometry"], "vector_ub_bytes": 178 * 1024,
        "preserved": ["pair order and causal compaction", "all event sequences",
                      "L1 destination offsets and ND-to-NZ dimensions",
                      "L0 load/MMAD/Fixpipe instructions and order",
                      "post-merge vector arithmetic and all original scratch sizes"],
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
