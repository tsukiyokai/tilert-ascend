"""Summarize the existing fixed-trajectory replay; no proposal policy here."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path("/workspace/offline/lookup-cpu-20261019/wide-frontier-20261020")
RAW = ROOT / "raw.json"
raw = json.loads(RAW.read_text())
inputs = raw["source_sha256"]
record_path = next(Path(path) for path in inputs if path.endswith("rank-0.json"))
record = json.loads(record_path.read_text())
requests = [item["tokens"] for item in record["requests"] if not item["warmup"]]
assert len({tuple(tokens) for tokens in requests}) == 1
tokens = requests[0]
committed = len(tokens) - 1
baseline_rounds = next(item["rounds"] for item in raw["results"]
                       if item["width"] == 32)
results = []
schedules = {}
for item in raw["results"]:
    width = item["width"]
    indexed = item["arms"]["indexed"][0]
    scan = item["arms"]["scan"][0]
    assert indexed["signature_sha256"] == scan["signature_sha256"]
    rounds = indexed["rounds"]
    position = 1
    details = []
    for step in rounds:
        assert step["position"] == position
        padded = step["proposal"] + [tokens[position - 1]] * (
            width - 1 - len(step["proposal"]))
        assert len(padded) == width - 1
        matched = 0
        for proposed, actual in zip(padded, tokens[position:]):
            if proposed != actual:
                break
            matched += 1
        assert step["accepted"] == min(matched + 1, len(tokens) - position)
        assert 1 <= step["accepted"] <= width
        details.append((position, step["accepted"], matched))
        position += step["accepted"]
    assert position == len(tokens)
    assert sum(step["accepted"] for step in rounds) == committed
    schedule = [(step["position"], step["accepted"]) for step in rounds]
    schedules[width] = schedule
    results.append({
        "width": width,
        "rounds": len(rounds),
        "committed_tokens": committed,
        "A_committed_per_round": committed / len(rounds),
        "verification_rows": width * len(rounds),
        "verification_rows_per_committed_token": width * len(rounds) / committed,
        "round_cost_ratio_vs_M32_strict_win_upper_bound": baseline_rounds / len(rounds),
        "max_committed_in_round": max(step["accepted"] for step in rounds),
        "accepted_count_histogram": dict(sorted(Counter(
            step["accepted"] for step in rounds).items())),
        "last_round": {"position": details[-1][0],
                       "accepted": details[-1][1],
                       "matched_drafts": details[-1][2],
                       "remaining_before_commit": len(tokens) - details[-1][0]},
        "indexed_reverse_scan_signature_equal": True,
        "signature_sha256": indexed["signature_sha256"],
        "accepted_schedule_sha256": hashlib.sha256(
            json.dumps(schedule).encode()).hexdigest(),
    })
assert schedules[64] == schedules[128] == schedules[256]
summary = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "CPU acceptance replay on one old free-running trajectory; not frozen-teacher acceptance, actual wide-model routes, or NPU timing",
    "policy": {"shortest": 1, "longest": 8,
               "selection": "longest suffix then most recent known continuation",
               "context": "prompt plus all committed tokens including pending",
               "padding": "repeat pending to fill width minus one",
               "commit": "matched drafts plus bonus, clipped to remaining output budget",
               "verification_rows": "fixed width per round including the last clipped commit"},
    "prompt_tokens": raw["prompt_tokens"],
    "output_tokens": len(tokens),
    "prefill_output_tokens_excluded_from_A": 1,
    "committed_tokens": committed,
    "identical_nonwarmup_requests": len(requests),
    "independent_inputs": 1,
    "input_and_source_sha256": inputs,
    "raw_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
    "summary_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "M64_M128_M256_same_commit_boundaries": True,
    "win_condition": "mean complete round cost T_M / T_32 < R_32 / R_M; equality only breaks even, with common setup excluded",
    "results": results,
    "decision": "Acceptance saturates at M64 on this trajectory. Close M64-plus extrapolation from these data; M128/M256 add no acceptance and M64 allows only 2.083333 percent higher mean round cost than M32.",
    "limits": ["No NPU was used and production width contracts were not changed.",
               "The reference trajectory is an older unaccepted free-running stream, not model correctness evidence.",
               "No actual expert unions, kernel costs, or TPOT predictions can be inferred.",
               "Each width starts fresh and advances at its own accepted boundaries; future tokens are used only for acceptance comparison."],
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({key: summary[key] for key in (
    "raw_sha256", "summary_source_sha256", "results")}, indent=2))
