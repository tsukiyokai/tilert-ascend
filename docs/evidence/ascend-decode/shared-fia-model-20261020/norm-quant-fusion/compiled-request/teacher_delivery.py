"""Compare saved request delivery with the frozen 8K teacher window."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report", type=Path)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source / "tests/glm52"))
from frozen_teacher import load_teacher


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path("/workspace")
capture = root / "offline/prefix8k-allocation-20261017/capture/case-00"
path = root / "experiments/shared-fia-20261020/teacher-32.json"
report_path = args.report
report = json.loads(report_path.read_text())
teacher = load_teacher(path, capture, report["width"], report["capacity"])
manifest = json.loads((capture / "manifest.json").read_text())
assert report["inputs"]["prompt_sha256"] == manifest["prompt_sha256"]
assert report["inputs"]["capture_manifest_sha256"] == digest(capture / "manifest.json")
expected = [teacher["first_token_id"], *teacher["expected_next_tokens"]]
assert len(expected) == 33
requests = report["requests"]
assert len(requests) == report["warmup"] + report["measured_requests"]
rows = []
for number, request in enumerate(requests):
    tokens = request["tokens"]
    first = next((i for i, (a, b) in enumerate(zip(tokens, expected))
                  if a != b), None)
    complete = len(tokens) >= len(expected)
    rows.append({
        "request": number, "warmup": request["warmup"],
        "compared_tokens": min(len(tokens), len(expected)),
        "first_difference": first,
        "actual": tokens[first] if first is not None else None,
        "expected": expected[first] if first is not None else None,
        "delivery_equal": tokens == request["returned_tokens"],
        "passed": complete and first is None
        and tokens == request["returned_tokens"],
    })
result = {
    "passed": bool(rows) and all(row["passed"] for row in rows),
    "scope": "One frozen prompt, first 33 delivered outputs; not the eight-prompt release gate",
    "teacher_sha256": digest(path),
    "capture_manifest_sha256": digest(capture / "manifest.json"),
    "report_sha256": digest(report_path), "requests": rows,
    "script_sha256": digest(Path(__file__)),
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
raise SystemExit(0 if result["passed"] else 1)
