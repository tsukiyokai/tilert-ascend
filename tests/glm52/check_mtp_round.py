"""Execute real MTP orchestration against independent scalar cache models."""

import argparse
from dataclasses import dataclass
import hashlib
import io
from itertools import product
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.models.glm_5_2_ascend.mtp_round import (  # noqa: E402
    MtpRoundExecutor,
)


def advance_hash(previous, token):
    return (previous * 1000003 + token + 17) % 2147483647


def next_token(context_hash, length):
    return (context_hash * 19 + length * 7) % 251 + 1


class Token(int):
    """Model a device scalar: the executor must snapshot it before reuse."""

    def clone(self):
        return self

    def item(self):
        return int(self)


@dataclass(frozen=True)
class Hidden:
    kind: str
    context_hash: int
    length: int


class ScalarTarget:
    def __init__(self, prompt, capacity):
        self.capacity = capacity
        self.position = 0
        self.rows = []
        self.verify(prompt)

    def verify(self, tokens, full_logits=False):
        predictions, hidden = [], []
        previous = self.rows[self.position - 1][1] if self.position else 0
        for token in tokens:
            previous = advance_hash(previous, token)
            entry = (token, previous)
            if self.position < len(self.rows):
                self.rows[self.position] = entry
            else:
                self.rows.append(entry)
            self.position += 1
            hidden.append(Hidden("target", previous, self.position))
            predictions.append(Token(next_token(previous, self.position)))
        return SimpleNamespace(token_ids=predictions, last_hidden=hidden)

    def host_tokens(self, *groups):
        return tuple(tuple(int(token) for token in group) for group in groups)

    def truncate(self, length):
        if not 0 <= length <= self.position:
            raise ValueError("Target truncate exceeds written rows")
        self.position = length


class ScalarDraft:
    def __init__(self, target, prefix_length, first_token, mode):
        self.capacity = target.capacity
        self.position = prefix_length
        self.rows = []
        self.calls = []
        self.mode = mode
        for row in range(prefix_length):
            token = (target.rows[row + 1][0]
                     if row + 1 < target.position else first_token)
            parent = Hidden("target", target.rows[row][1], row + 1)
            self.rows.append((token, parent))

    def truncate(self, length):
        if not 0 <= length <= self.position:
            raise ValueError("Draft truncate exceeds written rows")
        self.position = length

    def forward(
        self, tokens, parent_hidden, full_logits=False, external_indices=None,
    ):
        if len(tokens) != len(parent_hidden):
            raise ValueError("MTP tokens and parents have different lengths")
        start = self.position
        self.calls.append((start, tuple(tokens), tuple(parent_hidden),
                           external_indices))
        predictions, recycled, indices = [], [], []
        for row, (token, parent) in enumerate(zip(tokens, parent_hidden)):
            entry = (token, parent)
            if self.position < len(self.rows):
                self.rows[self.position] = entry
            else:
                self.rows.append(entry)
            self.position += 1
            digest = advance_hash(parent.context_hash, token)
            length = parent.length + 1
            predicted = next_token(digest, length)
            wrong = self.mode == "reject" or (
                self.mode == "mixed" and (self.position % 4 == 1)
            )
            predictions.append(Token(predicted + int(wrong)))
            recycled.append(Hidden("draft", digest, length))
            indices.append(
                tuple(range(min(self.position, 2048)))
                if external_indices is None else external_indices[row]
            )
        return SimpleNamespace(token_ids=predictions,
                               recycle_hidden=recycled, indices=indices)


class Cell:
    """A mutable token slot, as a graphed draft returns reused storage.

    ``clone`` is the only way to detach a value, so an executor that keeps
    the slot itself among its candidates sees the next chain step rewrite a
    proposal it already made.
    """

    def __init__(self, value):
        self.value = value

    def clone(self):
        return Cell(self.value)

    def item(self):
        return int(self.value)

    def __int__(self):
        return int(self.value)

    def __index__(self):
        return int(self.value)

    def __eq__(self, other):
        return int(self.value) == int(other)

    def __hash__(self):
        return hash(int(self.value))

    def __add__(self, other):
        return int(self.value) + int(other)

    __radd__ = __add__

    def __mod__(self, other):
        return int(self.value) % int(other)


class RecycledDraft(ScalarDraft):
    """Hand every forward the same cells, refilled in place."""

    def __init__(self, target, prefix_length, first_token, mode):
        super().__init__(target, prefix_length, first_token, mode)
        self.cells = []

    def forward(self, tokens, parent_hidden, **kwargs):
        result = super().forward(tokens, parent_hidden, **kwargs)
        while len(self.cells) < len(result.token_ids):
            self.cells.append(Cell(0))
        for cell, token in zip(self.cells, result.token_ids):
            cell.value = int(token)
        result.token_ids = self.cells[:len(result.token_ids)]
        return result


def autoregressive(prompt, first, count, capacity, eos=()):
    digest = 0
    for token in prompt:
        digest = advance_hash(digest, token)
    output, rows, token = [], list(prompt), first
    while len(output) < count:
        output.append(token)
        if token in eos or len(output) == count or len(rows) == capacity:
            break
        rows.append(token)
        digest = advance_hash(digest, token)
        token = next_token(digest, len(rows))
    return output, rows


def make_executor(width, prefix, missing_seed, mode, budget=64, room=100,
                  draft_class=ScalarDraft):
    prompt = tuple((row * 7 + 3) % 251 for row in range(prefix))
    target = ScalarTarget(prompt, prefix + room)
    parent = Hidden("target", target.rows[-1][1], prefix)
    first = next_token(parent.context_hash, prefix)
    draft = draft_class(target, prefix - int(missing_seed), first, mode)
    executor = MtpRoundExecutor(target, draft, width, budget)
    executor.start(first, [parent])
    return executor, prompt, first


class MtpRoundChecks(unittest.TestCase):
    boundary_coverage = None

    def test_normal_import_does_not_load_torch(self):
        self.assertNotIn("torch", sys.modules)

    def test_outputs_and_committed_cache_match_serial_oracle(self):
        for width in (2, 4):
            for prefix in (1, 127, 128, 8192):
                for missing in (False, True):
                    for mode in ("accept", "reject", "mixed"):
                        with self.subTest(width=width, prefix=prefix,
                                          missing=missing, mode=mode):
                            executor, prompt, first = make_executor(
                                width, prefix, missing, mode
                            )
                            output = [first]
                            while executor.state.finish_reason is None:
                                result = executor.advance()
                                output.extend(result.commit.accepted_tokens)
                                self.check_committed_cache(executor, prompt,
                                                           output)
                            expected, rows = autoregressive(
                                prompt, first, 64, executor.target.capacity
                            )
                            self.assertEqual(output, expected)
                            valid = executor.target.rows[
                                :executor.target.position
                            ]
                            self.assertEqual([row[0] for row in valid], rows)

    def test_a_recycled_draft_buffer_does_not_cost_acceptance(self):
        """A reused output slot must not rewrite an already-made proposal."""
        for width in (2, 4):
            with self.subTest(width=width):
                plain, prompt, first = make_executor(width, 128, False,
                                                     "accept")
                recycled = make_executor(width, 128, False, "accept",
                                         draft_class=RecycledDraft)[0]
                expected, rounds = [first], 0
                while plain.state.finish_reason is None:
                    expected.extend(plain.advance().commit.accepted_tokens)
                    rounds += 1
                actual, recycled_rounds = [first], 0
                while recycled.state.finish_reason is None:
                    actual.extend(
                        recycled.advance().commit.accepted_tokens
                    )
                    recycled_rounds += 1
                self.assertEqual(actual, expected)
                self.assertEqual(recycled_rounds, rounds)

    def check_committed_cache(self, executor, prompt, output):
        target, draft, state = executor.target, executor.draft, executor.state
        self.assertEqual(target.position, state.main_length)
        self.assertEqual(draft.position, state.draft_length)
        self.assertEqual(target.position, draft.position)
        self.assertEqual(
            [entry[0] for entry in target.rows[:target.position]],
            list(prompt) + output[:-1],
        )
        for position in range(len(prompt) - 1, target.position):
            token, parent = draft.rows[position]
            expected_hash = target.rows[position][1]
            self.assertEqual(parent, Hidden("target", expected_hash,
                                            position + 1))
            self.assertEqual(token, next_token(expected_hash, position + 1))

    def test_chained_rows_share_seed_indices_and_are_recomputed(self):
        executor, prompt, first = make_executor(4, 1, True, "accept", budget=8)
        result = executor.advance()
        calls = executor.draft.calls
        self.assertEqual([call[0] for call in calls], [0, 1, 2, 1])
        self.assertEqual(calls[1][3], [(0,)])
        self.assertEqual(calls[2][3], [(0,)])
        self.assertIsNone(calls[3][3])
        self.assertTrue(all(parent.kind == "draft" for parent in calls[1][2]))
        self.assertTrue(all(parent.kind == "target" for parent in calls[3][2]))
        self.check_committed_cache(
            executor, prompt, [first, *result.commit.accepted_tokens]
        )

    def test_first_rejection_commits_one_main_and_one_mtp_row(self):
        executor, prompt, first = make_executor(4, 128, False, "reject")
        result = executor.advance()
        self.assertEqual(result.commit.matched_drafts, 0)
        self.assertEqual(result.commit.committed, 1)
        self.assertEqual(executor.target.position, 129)
        self.assertEqual(executor.draft.position, 129)
        self.check_committed_cache(
            executor, prompt, [first, *result.commit.accepted_tokens]
        )

    def test_capacity_closes_with_width_one(self):
        for room in (1, 2, 3, 4, 7):
            executor, prompt, first = make_executor(
                4, 127, True, "accept", budget=32, room=room
            )
            output = [first]
            while executor.state.finish_reason is None:
                output.extend(executor.advance().commit.accepted_tokens)
            expected, _ = autoregressive(prompt, first, 32,
                                          executor.target.capacity)
            self.assertEqual(output, expected)
            self.assertEqual(executor.target.position, 127 + room)
            self.assertEqual(executor.draft.position, 127 + room)

    def test_zero_and_one_output_launch_no_round(self):
        for budget in (0, 1):
            executor, _, _ = make_executor(4, 128, True, "accept", budget)
            self.assertIsNotNone(executor.state.finish_reason)
            self.assertEqual(executor.draft.calls, [])
            with self.assertRaises(ValueError):
                executor.advance()

    def test_eos_truncates_accepted_span(self):
        executor, prompt, first = make_executor(4, 127, True, "accept")
        oracle, _ = autoregressive(prompt, first, 4, executor.target.capacity)
        executor.state.eos_tokens = frozenset((oracle[2],))
        result = executor.advance()
        self.assertEqual([first, *result.commit.accepted_tokens], oracle[:3])
        self.assertEqual(result.commit.finish_reason, "eos")
        self.assertEqual(executor.target.position, 129)
        self.assertEqual(executor.draft.position, 129)

    def test_eos_budget_capacity_cross_product(self):
        widths = (2, 4)
        prefixes = (1, 127, 128)
        missing_seeds = (False, True)
        modes = ("accept", "reject", "mixed")
        budgets = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        capacities_left = (0, 1, 2, 3, 4, 5)
        coverage = {"fixtures": 0, "rounds": 0}
        type(self).boundary_coverage = coverage
        spans, reasons = set(), set()
        combinations = product(widths, prefixes, missing_seeds, modes,
                               budgets, capacities_left)
        for width, prefix, missing, mode, budget, room in combinations:
            _, prompt, first = make_executor(
                width, prefix, missing, mode, budget, room
            )
            unrestricted, _ = autoregressive(prompt, first, 7, prefix + 7)
            eos_choices = ((), *((token,) for token in unrestricted[:5]))
            for eos in eos_choices:
                with self.subTest(width=width, prefix=prefix,
                                  missing=missing, mode=mode, budget=budget,
                                  room=room, eos=eos):
                    base, prompt, first = make_executor(
                        width, prefix, missing, mode, budget, room
                    )
                    executor = MtpRoundExecutor(
                        base.target, base.draft, width, budget, eos
                    )
                    executor.start(first, base.seed_parent_hidden)
                    output = [first] if budget else []
                    while executor.state.finish_reason is None:
                        result = executor.advance()
                        output.extend(result.commit.accepted_tokens)
                        self.check_committed_cache(executor, prompt, output)
                        spans.add((result.commit.width,
                                   result.commit.committed))
                        coverage["rounds"] += 1
                    expected, rows = autoregressive(
                        prompt, first, budget, prefix + room, eos
                    )
                    self.assertEqual(output, expected)
                    actual = executor.target.rows[:executor.target.position]
                    self.assertEqual([entry[0] for entry in actual], rows)
                    reasons.add(executor.state.finish_reason)
                    coverage["fixtures"] += 1
        coverage.update(
            accepted_spans=sorted(spans), finish_reasons=sorted(reasons),
            widths=widths, prefixes=prefixes, missing_seeds=missing_seeds,
            modes=modes, budgets=budgets, capacities_left=capacities_left,
            eos_choices="none or each of the first five serial output IDs",
        )
        self.assertEqual(coverage["fixtures"], 11664)
        self.assertEqual(spans, {(width, count) for width in (1, 2, 4)
                                 for count in range(1, width + 1)})
        self.assertEqual(reasons, {"capacity", "eos", "length"})

    def test_missing_parent_marks_failed_round(self):
        executor, _, first = make_executor(4, 128, True, "accept")
        executor.start(first, None)
        with self.assertRaises(ValueError):
            executor.advance()
        self.assertTrue(executor.failed)
        with self.assertRaises(RuntimeError):
            executor.advance()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MtpRoundChecks)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    sources = [Path(__file__), *(ROOT / path for path in (
        "tilert/models/glm_5_2_ascend/mtp_round.py",
        "tilert/models/glm_5_2_ascend/mtp_state.py",
    ))]
    report = {
        "status": "pass" if result.wasSuccessful() else "fail",
        "scope": "CPU scalar MTP orchestration/cache provenance; no CANN",
        "python": platform.python_version(), "tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "torch_imported": "torch" in sys.modules,
        "boundary_enumeration": MtpRoundChecks.boundary_coverage,
        "source_sha256": {str(path.relative_to(ROOT)):
                          hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        "test_output": stream.getvalue(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(stream.getvalue())
    print(json.dumps({key: report[key] for key in (
        "status", "tests", "failures", "errors", "torch_imported",
    )}))
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())
