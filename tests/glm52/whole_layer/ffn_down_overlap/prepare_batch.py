"""Batch four published down tiles in an isolated frozen overlap child."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "1d4220181c3a0235109d663aa348c4b435d9e5f91bdb8e82e0201e82c8214822"
NATIVE = "source/overlap_native.h"
CONTRACT = "source/overlap_contract.h"
CONSUMER = "__aicore__ inline void ConsumeDown("
OLD_LOOP = """        for (int64_t tile = 0; tile < ffn_down_overlap::kDownTilesPerExpert; ++tile) {
            CrossCoreWaitFlag(ffn_down_overlap::kDownReadyFlag);
            const int64_t column = tile * ffn_down_overlap::kDownTile + half;
            StoreRoutes(held, count, begin, wide + 2 * begin * kHidden,
                correction + expert * kHidden, scales + expert * kHidden,
                middle_scales + begin, routes, column,
                column + ffn_down_overlap::kDownHalfTile, tokens);
            // StoreRoutes drains each route store before reusing its UB source.
            CrossCoreSetFlag<2, PIPE_MTE3>(ffn_down_overlap::kDownFreeFlag);
        }
"""
NEW_LOOP = """        for (int64_t tile = 0; tile < ffn_down_overlap::kDownTilesPerExpert;
             tile += ffn_down_overlap::kDownConsumerBatchTiles) {
            for (int64_t ready = 0; ready < ffn_down_overlap::kDownConsumerBatchTiles;
                 ++ready) {
                CrossCoreWaitFlag(ffn_down_overlap::kDownReadyFlag);
            }
            const int64_t column = tile * ffn_down_overlap::kDownTile + half;
            StoreRoutes(held, count, begin, wide + 2 * begin * kHidden,
                correction + expert * kHidden, scales + expert * kHidden,
                middle_scales + begin, routes, column,
                column + ffn_down_overlap::kDownConsumerHalfSpan, tokens);
            // Both workers complete all four tiles before releasing their credits.
            for (int64_t free = 0; free < ffn_down_overlap::kDownConsumerBatchTiles;
                 ++free) {
                CrossCoreSetFlag<2, PIPE_MTE3>(ffn_down_overlap::kDownFreeFlag);
            }
        }
"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one frozen batch seam: {before!r}")
    return source.replace(before, after, 1)


def rename(source):
    return source.replace("ffn_down_overlap", "ffn_down_batch").replace(
        "FFN_DOWN_OVERLAP", "FFN_DOWN_BATCH")


def bind_native(source):
    source = replace_once(source,
        "    const int64_t half = GetSubBlockIdx() * ffn_down_overlap::kDownHalfTile;",
        "    const int64_t half = GetSubBlockIdx() * ffn_down_overlap::kDownConsumerHalfSpan;")
    source = replace_once(source, OLD_LOOP, NEW_LOOP)
    return replace_once(source, CONSUMER,
        "static_assert(ffn_down_overlap::kDownConsumerHalfSpan <= kVectorTile);\n\n"
        + CONSUMER)


def bind_contract(source):
    source = replace_once(source, "constexpr int64_t kDownCredits = 4;\n",
        "constexpr int64_t kDownCredits = 4;\n"
        "constexpr int64_t kDownConsumerBatchTiles = 4;\n"
        "constexpr int64_t kDownConsumerHalfSpan = kDownTile * kDownConsumerBatchTiles / 2;\n")
    return replace_once(source, "static_assert(kDownHalfTile == 128);\n",
        "static_assert(kDownHalfTile == 128);\n"
        "static_assert(kDownConsumerHalfSpan == 512);\n"
        "static_assert(kDownConsumerBatchTiles <= kDownCredits);\n"
        "static_assert(kDownTilesPerExpert % kDownConsumerBatchTiles == 0);\n"
        "static_assert(kDownConsumerHalfSpan % kDownTile == 0);\n")


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"]["ffn_down_overlap.so"] != PARENT_LIBRARY:
        raise ValueError("Use the frozen down-overlap build-v2 parent")
    for name, expected in manifest["sources"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"Frozen parent source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["sources"]:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, destination)
    changes = []
    for name, transform in ((NATIVE, bind_native), (CONTRACT, bind_contract)):
        target = output / name
        before = target.read_text()
        after = transform(before)
        target.write_text(after)
        changes.extend(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile="parent/" + name, tofile="candidate/" + name))
    (output / "down-batch.patch").write_text("".join(changes))
    for path in output.rglob("*"):
        if path.suffix in (".cpp", ".h"):
            path.write_text(rename(path.read_text()))
    (output / "check.py").write_text(rename((parent / "check.py").read_text()))
    for name in ("prepare_batch.py", "build_batch.py", "test_batch.py",
                 "batch_ownership_probe.cpp", "BATCH.md"):
        shutil.copy2(Path(__file__).with_name(name), output / "source" / name)
    shutil.copy2(parent / "manifest.json", output / "parent-overlap-manifest.json")
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "composed_ancestor_library_sha256": manifest["parent_library_sha256"],
        "public_scratch_count": 32,
        "additional_gm_bytes": 0,
        "additional_ub_bytes": 0,
        "maximum_resident_ub_bytes": 112352,
        "body_barriers": 6,
        "outer_end_phases": 2,
        "down_base_shape": [64, 256, 128],
        "down_single_shape": [64, 6144, 128],
        "credit_window": 4,
        "consumer_batch_tiles": 4,
        "consumer_columns_per_worker": 512,
        "ready_flag": 8,
        "free_flag": 9,
        "counter_mode": 2,
        "producer_change": "none",
        "system_workspace_behavior": "Original calls and initial bridge selection retained",
        "sources": {str(path.relative_to(output)): digest(path)
                    for path in sorted(output.rglob("*")) if path.is_file()},
    }
    (output / "source-transforms.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.output.resolve())
