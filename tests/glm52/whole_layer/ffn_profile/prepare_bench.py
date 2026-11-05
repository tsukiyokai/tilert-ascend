"""Derive a two-clock comparison from the frozen matched-address FFN gate."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from prepare import replace_once


PARENT_BENCH = "15ed8ad01e71e34703fb3628403f067568a0aabcd3db08d6a74d5402a592eb58"
PARENT_RUNNER = "4318d35211dd18ddddfed629909841234cc7613e886b633ac64ca13051913d13"
SPARSE_LIBRARY = "601e73587b826f3b7d2484c98710293e5a61568258092597f15332a400875fa8"


def prepare_bench(parent, output):
    for name, expected in (("bench.py", PARENT_BENCH),
                           ("variant_bench_run.sh", PARENT_RUNNER)):
        if hashlib.sha256((parent / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Use the frozen variant-timing-v3 {name}")
    source = (parent / "bench.py").read_text()
    source = replace_once(source, "import argparse\n",
                          "import argparse\nimport decode\n")
    begin = source.index("        records = actual.tolist()\n")
    end = source.index("\n    def checked(", begin)
    source = source[:begin] + '''        records = actual.tolist()
        samples = decode.read_sparse_trace(records, args.rows, args.blocks)
        windows = [(marks[4] - marks[3]) / decode.TICKS_PER_US
                   for marks in samples]
        return {"valid_worker_traces": len(records),
                "clock_reads_per_worker": 2, "sampled_marks": [3, 4],
                "body_setup_us_by_worker": windows}

''' + source[end:]
    source = replace_once(source,
        '        "arm_names": {"fused": args.candidate_kind, "separate": "base_one_launch"},',
        '        "arm_names": {"fused": args.candidate_kind, "separate": "base_one_launch"},\n'
        '        "profile_schema": {"clock_reads_per_worker": 2, "sampled_marks": [3, 4],\n'
        '                           "decoder_sha256": digest(Path(decode.__file__))},')
    compile(source, "bench.py", "exec")
    runner = (parent / "variant_bench_run.sh").read_text()
    runner = replace_once(runner,
        "cb88ddf9bc72997838d80855a138cbeea3e2018a0ff8a31e658983e09c991a6c",
        SPARSE_LIBRARY)
    runner = replace_once(runner,
        'candidate="$task_root/tests/glm52/whole_layer/ffn_$kind/build-v1/ffn_$kind.so"',
        '[[ "$kind" == profile ]] || exit 2\n'
        'candidate="$task_root/tests/glm52/whole_layer/ffn_profile/build-sparse-v1/ffn_profile.so"')
    output.mkdir(parents=True, exist_ok=False)
    (output / "bench.py").write_text(source)
    (output / "variant_bench_run.sh").write_text(runner)
    for name in ("decode.py", "prepare_bench.py"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    report = {
        "parent_bench_sha256": PARENT_BENCH,
        "parent_runner_sha256": PARENT_RUNNER,
        "candidate_library_sha256": SPARSE_LIBRARY,
        "changes": ["Frozen sparse profile library path and identity",
                    "Sparse timestamp-layout validation and reporting"],
        "preserved": ["C ABI argument setup and common tensor addresses",
                      "Graph capture, event timing and alternating arm order",
                      "Numerical comparisons and fresh-write poisoning"],
        "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare_bench(args.parent.resolve(), args.output.resolve())
