"""Compare a guarded source cursor with frozen suffix lookup on CPU."""

from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "lookup_draft.py"
RECORD = Path("/workspace/experiments/lookup-20261019/chain-w8/rank-0.json")
MANIFEST = Path("/workspace/offline/prefix8k-allocation-20261017/capture/"
                "case-00/manifest.json")
EXPECTED_RECORD = "3aa6cce191de24a06a1131f45db3c93f30716ffe5024ad0c23ca4f8c7524d671"
EXPECTED_MANIFEST = "b01cceb2d2e30564807ce9efc441ad535999b56b73d0c333e93bb18694ccda74"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location("frozen_lookup", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
LookupDrafter = module.LookupDrafter


def latest_source(drafter, count):
    context = drafter.context
    for size in range(drafter.longest, drafter.shortest - 1, -1):
        if len(context) <= size:
            continue
        source = drafter._latest.get(tuple(context[-size:]))
        if source is not None and context[source:source + count]:
            return source, size
    return None, 0


class CursorLookup:
    """Carry only a source whose whole draft and observed bonus agreed."""

    def __init__(self, context):
        self.lookup = LookupDrafter(context, longest=8, shortest=1)
        self.cursor = None

    def propose(self, count):
        context = self.lookup.context
        baseline = self.lookup.propose(count)
        source, size = latest_source(self.lookup, count)
        assert baseline == ([] if source is None else
                            context[source:source + count])
        cursor_ready = (self.cursor is not None and self.cursor < len(context)
                        and context[self.cursor - 1] == context[-1])
        if cursor_ready:
            run = context[self.cursor:self.cursor + count]
            return run, self.cursor, {
                "cursor_used": True, "suffix_size": size,
                "baseline_source": source,
                "differs_from_latest": run != baseline,
            }
        self.cursor = None
        return baseline, source, {
            "cursor_used": False, "suffix_size": size,
            "baseline_source": source, "differs_from_latest": False,
        }

    def commit(self, source, proposal, accepted, matched, width):
        context = self.lookup.context
        observed_length = len(context)
        self.cursor = None
        reason = "not_full_accept"
        if matched == width - 1 and len(accepted) == width:
            reason = "padding_in_full_accept"
            if source is not None and len(proposal) == width - 1:
                bonus_source = source + len(proposal)
                reason = "source_bonus_unobserved"
                if bonus_source < observed_length:
                    reason = "source_bonus_mismatch"
                    if context[bonus_source] == accepted[-1]:
                        self.cursor = bonus_source + 1
                        reason = "cursor_armed"
        self.lookup.commit(accepted)
        return reason


def matching_prefix(proposal, teacher):
    matched = 0
    for proposed, actual in zip(proposal, teacher):
        if proposed != actual:
            break
        matched += 1
    return matched


def replay(prompt, tokens, width, use_cursor):
    drafter = CursorLookup(prompt + tokens[:1])
    position = 1
    rounds = []
    reasons = Counter()
    while position < len(tokens):
        assert drafter.lookup.context == prompt + tokens[:position]
        observed_length = len(drafter.lookup.context)
        if use_cursor:
            proposal, source, details = drafter.propose(width - 1)
        else:
            proposal = drafter.lookup.propose(width - 1)
            source, size = latest_source(drafter.lookup, width - 1)
            details = {"cursor_used": False, "suffix_size": size,
                       "baseline_source": source,
                       "differs_from_latest": False}
        if source is not None:
            assert 0 <= source < observed_length
            assert proposal == drafter.lookup.context[source:source + width - 1]
            assert source + len(proposal) <= observed_length
        pending = tokens[position - 1]
        padded = proposal + [pending] * (width - 1 - len(proposal))
        assert len(padded) == width - 1
        # Future teacher access begins after the complete proposal is fixed.
        matched = matching_prefix(padded, tokens[position:])
        accepted_count = min(matched + 1, len(tokens) - position)
        accepted = tokens[position:position + accepted_count]
        reason = drafter.commit(source, proposal, accepted, matched, width)
        reasons[reason] += 1
        rounds.append({
            "position": position, "observed_context_length": observed_length,
            "source": source, "proposal": proposal, "padding": len(padded) - len(proposal),
            "matched": matched, "accepted": accepted_count,
            "budget_clipped": accepted_count < matched + 1,
            "cursor_next_guard": reason, **details,
        })
        position += accepted_count
    assert sum(item["accepted"] for item in rounds) == len(tokens) - 1
    return {
        "rounds": len(rounds), "verified_rows": len(rounds) * width,
        "new_tokens_committed": len(tokens) - 1,
        "cursor_rounds": sum(item["cursor_used"] for item in rounds),
        "cursor_proposal_rows": sum(len(item["proposal"]) for item in rounds
                                    if item["cursor_used"]),
        "cursor_changed_proposal_rounds": sum(item["differs_from_latest"]
                                              for item in rounds),
        "cursor_accepted_tokens": sum(item["accepted"] for item in rounds
                                      if item["cursor_used"]),
        "cursor_full_accept_rounds": sum(item["cursor_used"] and
                                         item["matched"] == width - 1
                                         for item in rounds),
        "guard_counts": dict(reasons), "raw_rounds": rounds,
    }


def counterexamples():
    prefix = list(range(1, 9))
    common = list(range(20, 28))
    history = prefix + common + [30, 40, 41] + common + [31, 42, 43] + prefix
    policy = CursorLookup(history)
    run, source, _ = policy.propose(7)
    assert run == common[:7]
    assert policy.commit(source, run, common, 7, 8) == "cursor_armed"
    run, _, details = policy.propose(7)
    assert details["cursor_used"] and details["differs_from_latest"]
    assert run[0] == 30 and policy.lookup.propose(7)[0] == 31
    # Equal committed prefix permits both next-token outcomes. The cursor is
    # predictive evidence, never a guarantee or a bypass of target verification.
    return {"history": history, "accepted": common,
            "cursor_next": 30, "latest_next": 31,
            "cursor_can_help_if_target_next": 30,
            "cursor_can_hurt_if_target_next": 31}


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    assert digest(RECORD) == EXPECTED_RECORD
    assert digest(MANIFEST) == EXPECTED_MANIFEST
    record = json.loads(RECORD.read_text())
    prompt = json.loads(MANIFEST.read_text())["prompt_token_ids"]
    requests = [item["tokens"] for item in record["requests"] if not item["warmup"]]
    assert len({tuple(tokens) for tokens in requests}) == 1
    tokens = requests[0]
    assert len(prompt) == 8192 and len(tokens) == 1024
    report = {
        "scope": "One frozen 8192+1024 stream; CPU acceptance replay only",
        "server": platform.node(), "python": platform.python_version(),
        "npu_execution": False, "model_timing": False,
        "source_commit": "d49ed4f", "longest": 8, "shortest": 1,
        "source_hashes": {str(path): digest(path)
                          for path in (SOURCE, Path(__file__), RECORD, MANIFEST)},
        "contract": ("Context contains pending exactly once; count=width-1. "
                     "Pad short proposals with pending; always charge full width; "
                     "clip final correction or bonus to output budget. Cursor needs "
                     "all width-1 real proposals accepted and bonus equal to an "
                     "already-observed source successor, checked before commit."),
        "worktree_note": "Uncommitted pending-argument interface excluded",
        "counterexample": counterexamples(), "results": [],
    }
    for width, expected_rounds in ((8, 228), (16, 171), (32, 147)):
        baseline = replay(prompt, tokens, width, False)
        assert baseline["rounds"] == expected_rounds
        candidate = replay(prompt, tokens, width, True)
        item = {"width": width, "baseline": baseline, "cursor": candidate}
        report["results"].append(item)
        print(json.dumps({"width": width,
                          "baseline": {k: v for k, v in baseline.items() if k != "raw_rounds"},
                          "cursor": {k: v for k, v in candidate.items() if k != "raw_rounds"}}),
              flush=True)
    (ROOT / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
