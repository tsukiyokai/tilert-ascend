"""Greedy speculation must decode the serial stream, at every width.

A verification width is a throughput choice. Whatever a drafter proposes,
a round commits the target's own predictions, so the emitted tokens must
be the ones plain greedy decoding produces -- the same at width 2 as at
width 32. A width that changes the output is not a faster decode but a
different one, and the acceptance it reports is then the drafter reading
its own corruption.

Two targets stand in for the model. One predicts from the whole history,
where a suffix drafter almost never matches; the other predicts from the
last two tokens, so its greedy stream loops and a drafter matches nearly
every row. The looping target is the regime the 8K agentic-coding fixture
produced, and it is where a width-dependent output hides best, because
acceptance stays high either way.

Each check has a control: an engine known to be wrong in one respect, so
that a passing check is evidence rather than a tautology.
"""

import argparse
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

from tilert.models.glm_5_2_ascend.automaton_draft import (  # noqa: E402
    AutomatonDrafter,
)
from tilert.models.glm_5_2_ascend.lookup_draft import (  # noqa: E402
    LookupDrafter,
)
from tilert.models.glm_5_2_ascend.mtp_round import (  # noqa: E402
    MtpRoundExecutor,
)
from tilert.models.glm_5_2_ascend.tree_draft import TreeDrafter  # noqa: E402


PROMPT = tuple((row * 7 + 3) % 23 + 1 for row in range(256))
OUTPUT_TOKENS = 96
WIDTHS = (2, 4, 6, 8, 12, 16, 24, 32)
DRAFTERS = {
    "lookup": LookupDrafter,
    "automaton": AutomatonDrafter,
    "tree": TreeDrafter,
}


class WholeHistoryRule:
    """A prediction from every token decoded so far, which never repeats."""

    start = 0

    @staticmethod
    def advance(state, token):
        return (state * 1000003 + token + 17) % 2147483647

    @staticmethod
    def predict(state, length):
        return (state * 19 + length * 7) % 251 + 1


class BigramRule:
    """A prediction from the last two tokens, so the stream cycles fast.

    Two tokens of state give the shortest cycle of the three rules, and a
    suffix drafter matches a short cycle almost every row. This is the rule
    that makes the invariance checks exercise deep acceptance rather than
    the bonus token alone.
    """

    start = (0, 0)

    @staticmethod
    def advance(state, token):
        return (state[1], token)

    @staticmethod
    def predict(state, length):
        return (state[0] * 31 + state[1] * 7) % 29 + 1


class RecentHistoryRule:
    """A prediction from the last four tokens, which cycles more slowly.

    Four is the shortest window that punishes a key taken from the wrong
    branch: a bigram's state is washed out by the next token, so a bigram
    rule cannot tell a gathered accepted path from an ungathered one and
    would leave the tree rounds without a control.
    """

    start = (0, 0, 0, 0)

    @staticmethod
    def advance(state, token):
        return (*state[1:], token)

    @staticmethod
    def predict(state, length):
        value = 0
        for token in state:
            value = value * 37 + token
        return value % 29 + 1


RULES = (WholeHistoryRule, BigramRule, RecentHistoryRule)


class ScalarTarget:
    """A rule's rows, honouring a tree's ancestry when one is installed.

    ``verify`` is told each row's depth and the mask is told each row's
    parent, which is how the engine divides one fact between rope and
    attention. A target that chained tree rows regardless would hide the
    defect these checks exist to find, so a tree round that installs no
    ancestry fails here rather than quietly decoding a chain.
    """

    device = None

    def __init__(self, rule, prompt, capacity):
        self.rule = rule
        self.capacity = capacity
        self.position = 0
        self.rows = []
        self.lineage = None
        self.verify(tuple(prompt))

    def install_mask(self, lineage, main_start, device):
        self.lineage = tuple(lineage)

    def verify(self, tokens, full_logits=False, depths=None):
        start = self.position
        parents = self._parents(len(tokens), depths)
        predictions, hidden = [], []
        for row, token in enumerate(tokens):
            depth = row if depths is None else depths[row]
            state = self.rule.advance(self._state(parents[row]), int(token))
            self._write(start + row, (int(token), state))
            predictions.append(self.rule.predict(state, start + depth + 1))
            hidden.append(state)
        self.position = start + len(tokens)
        return SimpleNamespace(token_ids=predictions, last_hidden=hidden)

    def _parents(self, count, depths):
        """Absolute row each verification row continues; -1 is the prefix."""
        start = self.position
        if depths is None:
            return [start + row - 1 for row in range(count)]
        if self.lineage is None or len(self.lineage) != count:
            raise ValueError("A tree round must install its own ancestry")
        return [start + parent if parent >= 0 else start - 1
                for parent in self.lineage]

    def _state(self, row):
        return self.rows[row][1] if row >= 0 else self.rule.start

    def _write(self, row, entry):
        while len(self.rows) <= row:
            self.rows.append(None)
        self.rows[row] = entry

    def host_tokens(self, *groups):
        return tuple(tuple(int(token) for token in group) for group in groups)

    def truncate(self, length):
        if not 0 <= length <= self.position:
            raise ValueError("Truncation exceeds the written rows")
        self.position = length

    def compact(self, start, rows):
        """Gather an accepted path's rows into contiguous slots."""
        gathered = [self.rows[start + row] for row in rows]
        for offset, entry in enumerate(gathered):
            self.rows[start + offset] = entry


class LeakyTarget(ScalarTarget):
    """A target whose rows all predict from the candidate block's last row.

    This is what a verification mask admitting future rows would give, and
    it is the suspicion the invariance checks are aimed at. Under it the
    width decides what is decoded, so a check that passes on ``ScalarTarget``
    and fails here is testing the property it claims to test.
    """

    def verify(self, tokens, full_logits=False, depths=None):
        result = super().verify(tokens, full_logits, depths)
        if len(tokens) > 1:
            state = self._state(self.position - 1)
            result.token_ids = [self.rule.predict(state, self.position)
                                for _ in tokens]
        return result


class UncompactedTarget(ScalarTarget):
    """A target leaving an accepted path's rows where the tree put them.

    The accepted path is a chain only once its rows are gathered, so an
    engine that skips the gather carries another branch's keys into the
    next round. It is the control for the tree rounds.
    """

    def compact(self, start, rows):
        pass


def serial_decode(rule, prompt, count):
    """What plain greedy decoding emits, one token per forward pass."""
    state = rule.start
    for token in prompt:
        state = rule.advance(state, token)
    length = len(prompt)
    output = []
    token = rule.predict(state, length)
    while len(output) < count:
        output.append(token)
        state = rule.advance(state, token)
        length += 1
        token = rule.predict(state, length)
    return output


def decode(rule, drafter_name, width, target_class=ScalarTarget,
           prompt=PROMPT, count=OUTPUT_TOKENS):
    """Run one request through the real executor and a host drafter."""
    target = target_class(rule, prompt, len(prompt) + count + 64)
    first = rule.predict(target.rows[len(prompt) - 1][1], len(prompt))
    drafter = DRAFTERS[drafter_name](list(prompt))
    executor = MtpRoundExecutor(target, None, width, count, drafter=drafter)
    if isinstance(drafter, TreeDrafter):
        executor.tree_mask = target.install_mask
    executor.start(first, None)
    output = [first]
    while executor.state.finish_reason is None:
        output.extend(executor.advance().commit.accepted_tokens)
    return output, executor.state.statistics()


class RoundInvariance(unittest.TestCase):

    def test_every_width_and_drafter_decodes_the_serial_stream(self):
        for rule, name, width in product(RULES, DRAFTERS, WIDTHS):
            with self.subTest(rule=rule.__name__, drafter=name, width=width):
                output, _ = decode(rule, name, width)
                self.assertEqual(output,
                                 serial_decode(rule, PROMPT, OUTPUT_TOKENS))

    def test_a_leaking_verification_mask_changes_what_is_decoded(self):
        """The control: the check above can see a row reading its future."""
        output, _ = decode(RecentHistoryRule, "automaton", 12, LeakyTarget)
        self.assertNotEqual(output, serial_decode(RecentHistoryRule, PROMPT,
                                                  OUTPUT_TOKENS))

    def test_an_ungathered_accepted_path_changes_what_is_decoded(self):
        """The control for the tree rounds, where the gather is the risk."""
        output, _ = decode(RecentHistoryRule, "tree", 12, UncompactedTarget)
        self.assertNotEqual(output, serial_decode(RecentHistoryRule, PROMPT,
                                                  OUTPUT_TOKENS))

    def test_the_looping_target_commits_more_than_the_bonus_token(self):
        """Otherwise the invariance checks never exercise a deep match."""
        _, looping = decode(BigramRule, "automaton", 12)
        self.assertGreater(looping["mean_committed_per_round"], 2)

    def test_a_tree_keeps_inside_the_round_budget(self):
        """A node past the width has no verification row to be read from."""
        for width in WIDTHS:
            with self.subTest(width=width):
                drafter = TreeDrafter(list(PROMPT))
                tokens, _ = drafter.propose_tree(width - 1)
                self.assertLessEqual(len(tokens), width - 1)

    def test_tree_with_one_cache_row_only_verifies_the_pending_token(self):
        for rule, width, stop in product(RULES, WIDTHS, (False, True)):
            with self.subTest(rule=rule.__name__, width=width, stop=stop):
                expected = serial_decode(rule, PROMPT, 2)
                target = ScalarTarget(rule, PROMPT, len(PROMPT) + 1)
                drafter = TreeDrafter(list(PROMPT))

                def reject_empty_tree(count):
                    self.fail(f"Capacity tail asked for {count} tree nodes")

                drafter.propose_tree = reject_empty_tree
                executor = MtpRoundExecutor(
                    target, None, width, 2,
                    eos_tokens=(expected[1],) if stop else (),
                    drafter=drafter,
                )
                executor.tree_mask = target.install_mask
                executor.start(expected[0], None)
                result = executor.advance()
                self.assertEqual(result.candidates, (expected[0],))
                self.assertEqual(result.parents, (-1,))
                self.assertEqual(result.commit.accepted_tokens,
                                 (expected[1],))
                self.assertEqual(result.commit.width, 1)
                self.assertEqual(target.position, target.capacity)
                self.assertEqual([row[0] for row in target.rows],
                                 [*PROMPT, expected[0]])
                self.assertEqual(result.commit.finish_reason,
                                 "eos" if stop else "length")
                self.assertFalse(executor.failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RoundInvariance)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    sources = [Path(__file__), *(ROOT / path for path in (
        "tilert/models/glm_5_2_ascend/mtp_round.py",
        "tilert/models/glm_5_2_ascend/mtp_state.py",
        "tilert/models/glm_5_2_ascend/tree_draft.py",
        "tilert/models/glm_5_2_ascend/tree_resolve.py",
    ))]
    report = {
        "status": "pass" if result.wasSuccessful() else "fail",
        "scope": "CPU width invariance of greedy speculation; no CANN",
        "python": platform.python_version(), "tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "torch_imported": "torch" in sys.modules,
        "widths": list(WIDTHS), "drafters": sorted(DRAFTERS),
        "rules": [rule.__name__ for rule in RULES],
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
