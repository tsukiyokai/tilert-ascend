"""Create an explicitly derived M1 view for the existing layer-major reader."""

import argparse
import hashlib
import json
import sys
from importlib import util
from pathlib import Path

import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--full", type=Path, required=True)
parser.add_argument("--capture-case", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--prompt-output", type=Path, required=True)
args = parser.parse_args()
sys.path[:0] = [str(args.source), str(args.source / "tests/glm52")]
spec = util.spec_from_file_location(
    "frozen_precision_entry", args.source / "experiments/cann-precision/run.py"
)
entry = util.module_from_spec(spec)
spec.loader.exec_module(entry)
summary = json.loads((args.full / "summary.json").read_text())
assert summary["capture_complete"] and summary["status"] == "fail"
manifest = json.loads((args.capture_case / "manifest.json").read_text())
args.prompt_output.write_text(json.dumps(manifest["prompt_token_ids"]) + "\n")
args.output.mkdir(parents=True, exist_ok=False)
receipts = []
for rank in range(16):
    report_path = args.full / f"rank-{rank}.json"
    raw = report_path.read_bytes()
    report = json.loads(raw)
    original_sha = hashlib.sha256(raw).hexdigest()
    expected = next(
        value for value in summary["ranks"] if value["rank"] == rank
    )
    assert original_sha == expected["report_sha256"]
    source_path = args.full / f"rank-{rank}/serial-00.pt"
    assert digest(source_path) == report["trace_sha256"]["serial-00.pt"]
    wide = torch.load(source_path, map_location="cpu", weights_only=True)
    row = entry.single_row(wide, 0, choices=True)
    fields = (
        "token",
        "position",
        "committed_position",
        "hidden_states",
        "cache_rows",
        "indices",
    )
    derived = {name: row[name] for name in fields}
    assert type(derived["token"]) is int and derived["position"] == 8192
    folder = args.output / f"rank-{rank}"
    folder.mkdir()
    output = folder / "step-08192.pt"
    torch.save(derived, output)
    identity = {
        name: report[name]
        for name in (
            "rank",
            "checkpoint_sha256",
            "inputs",
            "model_tp_reduction",
        )
    }
    identity.update(
        status=report["status"],
        trace_sha256={output.name: digest(output)},
        view_scope="Actual tensor row projection and filename adaptation; original numerical failure retained",
        view_fields=list(fields),
        original_report_sha256=original_sha,
        original_trace_sha256=report["trace_sha256"]["serial-00.pt"],
    )
    (args.output / f"rank-{rank}.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )
    receipts.append(
        {"rank": rank, "derived_sha256": identity["trace_sha256"][output.name]}
    )
(args.output / "view.json").write_text(
    json.dumps(
        {
            "source_summary_sha256": digest(args.full / "summary.json"),
            "source_case_precision_passed": summary["case_precision_passed"],
            "script_sha256": digest(__file__),
            "ranks": receipts,
        },
        indent=2,
    )
    + "\n"
)
print(
    json.dumps(
        {"rank_views": len(receipts), "source_case_status": summary["status"]}
    )
)
