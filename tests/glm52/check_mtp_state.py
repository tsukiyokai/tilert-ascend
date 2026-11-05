"""Check MTP control with an independent sequential greedy CPU oracle."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import sys

from tilert.models.glm_5_2_ascend.mtp_state import MtpState


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def consume(digest, token, position):
    return (digest * 131 + token * 17 + position + 3) % 1000000007


def predict(digest):
    return 1000 + digest % 150000


class Oracle:
    """Advance one real greedy token at a time, without prefix-match math."""

    def __init__(self, prefix, budget, eos=()):
        self.tokens = []
        self.hidden = []
        self.rows = []
        self.digest = 0
        for token in prefix:
            self._consume(token)
        self.pending = predict(self.digest)
        self.outputs = [self.pending] if budget else []
        self.budget = budget
        self.eos = frozenset(eos)

    def _consume(self, token):
        position = len(self.tokens)
        self.digest = consume(self.digest, token, position)
        hidden = ("target", position, self.digest)
        self.tokens.append(token)
        self.hidden.append(hidden)
        self.rows.append((token, position, hidden))

    def round(self, candidates):
        digest = self.digest
        pending = self.pending
        start = len(self.tokens)
        expected = []
        consumed = []
        for offset, candidate in enumerate(candidates):
            if candidate != pending:
                break
            consumed.append(pending)
            digest = consume(digest, pending, start + offset)
            pending = predict(digest)
            expected.append(pending)
        natural = len(expected)
        emitted = []
        kept = []
        for token, successor in zip(consumed, expected):
            if len(self.outputs) == self.budget:
                break
            self._consume(token)
            self.pending = successor
            self.outputs.append(successor)
            emitted.append(successor)
            kept.append(token)
            if successor in self.eos:
                break
        return tuple(emitted), tuple(kept), natural

    def draft_row(self, position):
        token = (self.tokens[position + 1] if position + 1 < len(self.tokens)
                 else self.pending)
        parent = self.hidden[position]
        return (token, position, parent)


class Cache:
    """Store provenance labels, including tentative rows beyond validity."""

    def __init__(self, oracle, draft_length):
        self.main = list(oracle.rows)
        self.main_length = len(self.main)
        self.draft = [] if draft_length is None else [
            oracle.draft_row(index) for index in range(draft_length)
        ]
        self.draft_length = draft_length

    @staticmethod
    def write(rows, position, value):
        if position == len(rows):
            rows.append(value)
        else:
            rows[position] = value

    def verify(self, plan, oracle, candidates, predictions, hidden):
        for index, token in enumerate(candidates):
            position = plan.main_start + index
            self.write(self.main, position, (token, position, hidden[index]))
        if plan.draft_verify_start is None:
            return
        if plan.draft_seed_position is not None:
            position = plan.draft_seed_position
            self.write(self.draft, position, oracle.draft_row(position))
        for index, token in enumerate(candidates[1:], start=1):
            position = plan.main_start + index - 1
            self.write(self.draft, position,
                       (token, position, ("chained", index - 1)))
        for index, token in enumerate(predictions):
            position = plan.draft_verify_start + index
            self.write(self.draft, position, (token, position, hidden[index]))

    def commit(self, main, draft):
        self.main_length = main.retained
        if draft is not None:
            self.draft_length = draft.retained

    def check(self, oracle, start=0):
        require(self.main_length == len(oracle.tokens), "main length")
        require(self.main[start:self.main_length] == oracle.rows[start:],
                "main token or hidden provenance")
        if self.draft_length is not None:
            for position in range(start, self.draft_length):
                require(self.draft[position] == oracle.draft_row(position),
                        f"MTP token/position/parent_hidden at {position}")


def make_case(width=4, prefix_length=5, budget=100, capacity=128,
              seeded=False, eos=()):
    prefix = [11 + index % 97 for index in range(prefix_length)]
    oracle = Oracle(prefix, budget, eos)
    state = MtpState(width, capacity, budget, eos)
    draft_length = prefix_length - int(not seeded) if width > 1 else None
    state.reset(prefix_length, oracle.pending, draft_length)
    return state, oracle, Cache(oracle, draft_length)


def batch(oracle, width, matches):
    candidates = [oracle.pending]
    predictions, hidden = [], []
    digest = oracle.digest
    start = len(oracle.tokens)
    for index in range(width):
        position = start + index
        digest = consume(digest, candidates[index], position)
        hidden.append(("target", position, digest))
        successor = predict(digest)
        predictions.append(successor)
        if index + 1 < width:
            candidates.append(successor if matches[index]
                              else (successor + 1) % 154880)
    return tuple(candidates), tuple(predictions), tuple(hidden)


def run_round(state, oracle, cache, matches, full_check=True):
    plan = state.begin()
    before = len(oracle.tokens)
    candidates, predictions, hidden = batch(oracle, plan.width, matches)
    cache.verify(plan, oracle, candidates, predictions, hidden)
    expected, consumed, natural = oracle.round(candidates)
    result = state.resolve(plan, candidates, predictions)
    require(result.accepted_tokens == expected, "emitted tokens")
    require(result.consumed_tokens == consumed, "consumed tokens")
    require(result.natural_accepted == natural, "natural accepted span")
    require(result.matched_drafts == natural - 1, "matched draft count")
    require(result.pending_token == oracle.pending, "next pending token")
    require(state.emitted_count == len(oracle.outputs), "output count")
    require(result.main.retained == before + len(consumed), "main commit")
    require(result.main.discard == (before + len(consumed), plan.main_end),
            "main rollback interval")
    require(result.main_hidden.row == len(consumed) - 1, "target hidden row")
    require(result.main_hidden.position == len(oracle.tokens) - 1,
            "target hidden position")
    if result.draft is not None:
        require(result.draft.retained == result.main.retained,
                "equal lengths, MTP embedding is one token ahead")
        require(result.draft.discard == (
            result.main.retained, plan.draft_end
        ), "MTP rollback interval")
        require(result.draft_hidden.row == len(consumed) - 1,
                "verified MTP hidden row")
        require(result.draft_hidden.position == len(oracle.tokens) - 1,
                "verified MTP hidden position")
    cache.commit(result.main, result.draft)
    cache.check(oracle, 0 if full_check else before)
    require(state.main_length == cache.main_length, "state main length")
    require(state.draft_length == cache.draft_length, "state MTP length")
    return plan, result


def expect_value_error(operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def exhaustive():
    rounds = 0
    spans = set()
    for width in (1, 2, 4):
        for length in (1, 127, 128, 8191):
            for seeded in ((False,) if width == 1 else (False, True)):
                for matches in product((False, True), repeat=width - 1):
                    state, oracle, cache = make_case(
                        width, length, capacity=length + 16, seeded=seeded
                    )
                    plan, result = run_round(state, oracle, cache, matches)
                    expected_seed = (
                        length - 1 if width > 1 and not seeded else None
                    )
                    require(plan.draft_seed_position == expected_seed,
                            "P-1 versus P seed")
                    spans.add((width, result.committed))
                    rounds += 1
    expected_spans = {(width, count) for width in (1, 2, 4)
                      for count in range(1, width + 1)}
    require(spans == expected_spans, "all accept positions")
    return {"rounds": rounds, "accepted_spans": sorted(spans)}


def stop_boundaries():
    cases = 0
    for width in (1, 2, 4):
        for remaining in range(1, width + 1):
            state, oracle, cache = make_case(width, budget=remaining + 1)
            _, result = run_round(state, oracle, cache, (True,) * 3)
            require(result.committed == remaining, "budget clipping")
            require(result.finish_reason == "length", "budget stop")
            cases += 1
        _, initial, _ = make_case(width)
        _, predictions, _ = batch(initial, width, (True,) * 3)
        for position, eos in enumerate(predictions):
            state, oracle, cache = make_case(width, eos=(eos,))
            _, result = run_round(state, oracle, cache, (True,) * 3)
            require(result.committed == predictions.index(eos) + 1,
                    f"EOS slot {position}")
            require(result.finish_reason == "eos", "EOS stop")
            expect_value_error(state.begin)
            cases += 1
        for budget in (0, 1):
            state, _, _ = make_case(width, budget=budget)
            require(state.emitted_count == budget, "initial output budget")
            require(state.finish_reason == "length", "initial budget stop")
            expect_value_error(state.begin)
            cases += 1
        _, initial, _ = make_case(width)
        state, _, _ = make_case(width, eos=(initial.pending,))
        require(state.finish_reason == "eos", "first output EOS")
        expect_value_error(state.begin)
        cases += 1
    _, initial, _ = make_case()
    candidates, predictions, _ = batch(initial, 4, (False, True, True))
    for eos in (candidates[1], predictions[-1]):
        require(eos != predictions[0], "distinct rejected-branch EOS")
        state, oracle, cache = make_case(eos=(eos,))
        _, result = run_round(state, oracle, cache, (False, True, True))
        require(result.committed == 1, "rejected-branch EOS acceptance")
        require(result.finish_reason is None, "rejected-branch EOS ignored")
        cases += 1
    return {"cases": cases}


def capacity_boundaries():
    cases = 0
    for width in (1, 2, 4):
        for room in range(7):
            state, oracle, cache = make_case(
                width, 127, capacity=127 + room
            )
            if room == 0:
                require(state.finish_reason == "capacity", "full cache")
                expect_value_error(state.begin)
            else:
                plan, result = run_round(state, oracle, cache, (True,) * 3)
                require(plan.main_end <= state.capacity, "main capacity")
                if plan.draft_end is not None:
                    require(plan.draft_end <= state.capacity, "MTP capacity")
                if room == 1:
                    require(plan.width == 1, "last slot verifies one input")
                    require((result.draft is not None) == (width > 1),
                            "MTP last row uses the same physical position")
                    require(result.finish_reason == "capacity", "last slot")
            cases += 1
    return {"cases": cases, "page_size": 128}


def reset_abort_and_errors():
    state, oracle, cache = make_case()
    plan = state.begin()
    values = batch(oracle, plan.width, (False,) * 3)
    cache.verify(plan, oracle, *values)
    expect_value_error(state.begin)
    expect_value_error(lambda: state.resolve(plan, (0,), (0,)))
    wrong = (values[0][0] + 1,) + values[0][1:]
    expect_value_error(lambda: state.resolve(plan, wrong, values[1]))
    main, draft = state.abort(plan)
    cache.commit(main, draft)
    cache.check(oracle)
    require(state.statistics()["rounds"] == 0, "abort does not commit")
    expect_value_error(lambda: state.resolve(plan, *values[:2]))
    fresh = state.begin()
    state.reset(5, oracle.pending, 4)
    expect_value_error(lambda: state.resolve(fresh, *values[:2]))
    _, first = run_round(state, oracle, cache, (True,) * 3)
    state.reset(5, predict(Oracle([11 + i for i in range(5)], 100).digest), 4)
    replay = Oracle([11 + i for i in range(5)], 100)
    _, second = run_round(state, replay, Cache(replay, 4), (True,) * 3)
    require(first == second, "reset replays result")
    expect_value_error(lambda: MtpState(3, 128, 10))
    expect_value_error(lambda: MtpState(True, 128, 10))
    expect_value_error(lambda: MtpState(4, 0, 10))
    expect_value_error(lambda: MtpState(4, 128, -1))
    expect_value_error(lambda: state.reset(5, 10, 3))
    expect_value_error(lambda: state.reset(5, 10, 6))
    expect_value_error(lambda: state.reset(5, 154880, 4))
    expect_value_error(lambda: MtpState(1, 128, 10).reset(5, 10, 5))
    require(state.main_length == 9, "invalid reset preserves request")
    return {"reset_replay": True, "abort_preserves_prefix": True}


def parent_hidden_detection():
    state, oracle, cache = make_case()
    run_round(state, oracle, cache, (True,) * 3)
    position = state.draft_length - 1
    token, row_position, _ = cache.draft[position]
    cache.draft[position] = (token, row_position, ("chained", position - 1))
    require(token == oracle.draft_row(position)[0], "same token retained")
    try:
        cache.check(oracle)
    except AssertionError:
        return {"same_token_wrong_parent_detected": True}
    raise AssertionError("oracle missed chained hidden in committed MTP KV")


def alternating_rounds():
    state, oracle, cache = make_case(budget=50000, capacity=50000)
    counts = {1: 0, 4: 0}
    for index in range(10000):
        matches = (bool(index % 2),) * 3
        _, result = run_round(state, oracle, cache, matches, full_check=False)
        counts[result.committed] += 1
    cache.check(oracle)
    require(counts == {1: 5000, 4: 5000}, "alternating accept/reject")
    require(state.statistics()["committed"] == 25000, "total commits")
    require(state.statistics()["emitted"] == 25001, "first output counted")
    return {"counts": counts, "statistics": state.statistics()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = []
    for operation in (
        exhaustive, stop_boundaries, capacity_boundaries,
        reset_abort_and_errors, parent_hidden_detection, alternating_rounds,
    ):
        try:
            detail = operation()
            checks.append({"name": operation.__name__, "passed": True,
                           "detail": detail})
        except Exception as error:
            checks.append({"name": operation.__name__, "passed": False,
                           "error": f"{type(error).__name__}: {error}"})
    require("torch" not in sys.modules, "stdlib runner imported Torch")
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / (
        "tilert/models/glm_5_2_ascend/mtp_state.py"
    )]
    for path in paths:
        ast.parse(path.read_text(), feature_version=(3, 11))
        require(all(len(line) <= 79 for line in path.read_text().splitlines()),
                f"line exceeds 79 columns: {path.name}")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "python": sys.version,
        "command": [sys.executable, *sys.argv],
        "scope": "Greedy control and labelled cache semantics on CPU",
        "target_conditioned_mtp": {
            "row": "position p uses target hidden[p] and token[p+1]",
            "chained_hidden": "must be overwritten before retaining KV",
            "main_and_mtp_commit_lengths": "both P+A",
            "prefill_seed": "MTP length P-1 needs seed at P-1; P is seeded",
            "vllm_commit": "c4df37dfcc7f72ca41cd43801a468526ecf99a2b",
            "vllm_ascend_commit":
                "f558b58c177916424ae9dcf9bb8e66c0b9031236",
            "source_anchors": {
                "vllm_ascend/spec_decode/eagle_proposer.py": "10-12",
                "vllm/v1/spec_decode/llm_base_proposer.py": "113-120",
                "vllm_ascend/spec_decode/llm_base_proposer.py":
                    "1801-1838, 2033-2131",
            },
            "corrected_assumption":
                "The previous report used token position p+1 as cache row; "
                "the fixed vLLM source keeps the parent position p.",
        },
        "device_execution": False, "model_numerics": False,
        "performance_measured": False, "torch_imported": False,
        "python_311_syntax": True,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest() for path in paths
        },
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"],
                      "checks": len(checks), "report": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
