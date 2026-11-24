"""Add trace validation to the frozen complete-FFN numerical checker."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from prepare import replace_once


PARENT_CHECKER = "d5c1f9f3214a1461d0aff29e166ec781d894c6bd991369bbac27f80baed3c9bb"


def prepare_check(parent, output, sparse=False):
    content = parent.read_bytes()
    if hashlib.sha256(content).hexdigest() != PARENT_CHECKER:
        raise ValueError("Use the frozen complete-FFN harness-v1 checker")
    source = content.decode().replace("tilert_ffn_composed", "tilert_ffn_profile")
    reader = "read_sparse_trace" if sparse else "read_trace"
    summary = "summarize_sparse" if sparse else "summarize"
    clock_reads = 2 if sparse else 13
    scope = ("Instrumented local FFN; two-clock body-setup window, no TP or latency acceptance"
             if sparse else
             "Instrumented local FFN; per-worker phase windows, no TP or latency acceptance")
    source = replace_once(source, "import argparse\n", "import argparse\nimport decode\n")
    source = replace_once(source,
        '"scope": "One rank-local FFN launch; TP and model latency excluded",',
        f'"scope": "{scope}",')
    source = replace_once(source,
        '"timing": False, "cases": [], "pass": False,',
        '"timing": False, "cases": [], "pass": False,\n'
        f'        "phase_profiles": [], "clock_reads_per_worker": {clock_reads},\n'
        '        "decoder_sha256": digest(Path(decode.__file__)),')
    source = replace_once(source,
        "                    for iteration in range(args.replays):",
        "                    trace_samples, active_counts = [], []\n"
        "                    for iteration in range(args.replays):")
    source = replace_once(source,
        "                        actual = all_values(torch, cb, cr, rows)",
        "                        actual = all_values(torch, cb, cr, rows)\n"
        f"                        trace_samples.append(decode.{reader}(\n"
        "                            cb[-1].cpu().tolist(), rows, blocks))\n"
        "                        active_counts.append(sum(\n"
        "                            count != 0 for count in cr[12].cpu().tolist()))")
    source = replace_once(source,
        "                    for graph in captures.values():\n"
        "                        graph.reset()",
        '                    report["phase_profiles"].append({\n'
        '                        "rows": rows, "blocks": blocks,\n'
        '                        "active_experts": active_counts,\n'
        '                        "samples_ticks": trace_samples,\n'
        f'                        "summary": decode.{summary}(trace_samples, blocks)}})\n'
        "                    persist()\n"
        "                    for graph in captures.values():\n"
        "                        graph.reset()")
    output.mkdir(parents=True, exist_ok=False)
    (output / "check.py").write_text(source)
    shutil.copy2(Path(__file__).with_name("decode.py"), output / "decode.py")
    manifest = {
        "parent_checker_sha256": PARENT_CHECKER,
        "sampled_marks": [3, 4] if sparse else list(range(13)),
        "numerical_contract": "Unchanged fixtures, graph captures, changing inputs and 30 raw-byte boundaries",
        "changes": ["profile namespace", "per-worker trace identity and monotonicity",
                    "trace samples and active expert counts after each synchronized replay"],
        "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in output.iterdir() if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()
    prepare_check(args.parent.resolve(), args.output.resolve(), args.sparse)
