"""Check real TP16 MTP command dispatch with independent scalar caches."""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.models.glm_5_2_ascend.session import PrefixRequest  # noqa: E402
from tilert.models.glm_5_2_ascend.tp_runtime import (  # noqa: E402
    CoordinatedBackend, RankExecutor,
)
from tilert.pd_vllm.ascend_cache import (  # noqa: E402
    AscendRankEngine, RankPrefix,
)


def consume(previous, token):
    return (previous * 1000003 + token + 17) % 2147483647


def predict(digest, length):
    return (digest * 19 + length * 7) % 251 + 1


class Token(int):
    """Model a device scalar: the executor must snapshot it before reuse."""

    def clone(self):
        return self

    def item(self):
        return int(self)


@dataclass(frozen=True)
class Hidden:
    kind: str
    digest: int
    length: int


class ScalarCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.rows = []
        self.position = 0
        self.truncations = []

    def inject(self, rows):
        self.rows = list(rows)
        self.position = len(rows)

    def reset(self):
        self.rows = []
        self.position = 0

    def truncate(self, length):
        if not 0 <= length <= self.position:
            raise ValueError("Cannot retain unwritten scalar rows")
        self.truncations.append((self.position, length))
        self.position = length

    def append(self, row):
        if self.position == self.capacity:
            raise ValueError("Scalar cache is full")
        if self.position < len(self.rows):
            self.rows[self.position] = row
        else:
            self.rows.append(row)
        self.position += 1


class ScalarTarget(ScalarCache):
    def verify(self, tokens, full_logits=False):
        previous = self.rows[self.position - 1][1] if self.position else 0
        predictions, hidden = [], []
        for token in tokens:
            previous = consume(previous, token)
            self.append((token, previous))
            hidden.append(Hidden("target", previous, self.position))
            predictions.append(Token(predict(previous, self.position)))
        return SimpleNamespace(token_ids=predictions, last_hidden=hidden)

    def host_tokens(self, *groups):
        return tuple(tuple(int(token) for token in group) for group in groups)


class ScalarDraft(ScalarCache):
    def __init__(self, capacity, mode):
        super().__init__(capacity)
        self.mode = mode

    def forward(
        self, tokens, parent_hidden, full_logits=False, external_indices=None,
    ):
        predictions, recycled, indices = [], [], []
        for row, (token, parent) in enumerate(zip(tokens, parent_hidden)):
            if parent.length != self.position + 1:
                raise ValueError("MTP parent has the wrong physical position")
            kind = "target" if external_indices is None else "draft"
            if parent.kind != kind:
                raise ValueError("MTP parent has the wrong hidden-state kind")
            self.append((token, parent))
            digest = consume(parent.digest, token)
            length = parent.length + 1
            predicted = predict(digest, length)
            wrong = self.mode == "reject" or (
                self.mode == "mixed" and self.position % 4 == 1
            )
            predictions.append(Token((predicted % 251) + 1 if wrong
                                   else predicted))
            recycled.append(Hidden("draft", digest, length))
            indices.append((self.position - 1,) if external_indices is None
                           else external_indices[row])
        return SimpleNamespace(token_ids=predictions,
                               recycle_hidden=recycled, indices=indices)


def load_prefix(request, rank):
    prompt = request.metadata["prompt"]
    if len(prompt) != request.seq_len or rank not in range(16):
        raise ValueError("Wrong rank-local prefix descriptor")
    rows, mtp, digest = [], [], 0
    for position, token in enumerate(prompt):
        digest = consume(digest, token)
        rows.append((token, digest))
        if position + 1 < len(prompt):
            parent = Hidden("target", digest, position + 1)
            mtp.append((prompt[position + 1], parent))
    return RankPrefix(rows[:-1], mtp, prompt[-1], len(prompt))


class FanoutChannel:
    """Deliver each real command to all workers without an HCCL transport."""

    def __init__(self, workers):
        self.workers = workers
        self.commands = []
        self.results = []

    def send(self, command):
        self.commands.append(deepcopy(command))
        self.results = [worker.execute(deepcopy(command))
                        for worker in self.workers]


def make_backend(mode="accept", capacity=64):
    ranks = []
    for rank in range(16):
        engine = AscendRankEngine(ScalarTarget(capacity),
                                 ScalarDraft(capacity, mode))
        ranks.append(RankExecutor(engine, rank, load_prefix))
    channel = FanoutChannel(ranks[1:])
    return CoordinatedBackend(ranks[0], channel), ranks, channel


def inject(backend, prompt=(3, 10, 17), request_id="request"):
    backend.inject(PrefixRequest(request_id, len(prompt), {"prompt": prompt}))
    return backend.executor.engine.first_token_id


def serial_oracle(prompt, count, capacity, eos=()):
    rows, digest = [], 0
    for token in prompt:
        digest = consume(digest, token)
        rows.append((token, digest))
    output = [predict(digest, len(rows))] if count else []
    while output and len(output) < count and len(rows) < capacity:
        if output[-1] in eos:
            break
        digest = consume(digest, output[-1])
        rows.append((output[-1], digest))
        output.append(predict(digest, len(rows)))
    return output, rows


class MtpTpChecks(unittest.TestCase):
    def assert_no_more_commands(self, backend, channel):
        self.assertTrue(backend.failed)
        count = len(channel.commands)
        operations = (backend.reset, backend.close, backend.advance_mtp,
                      lambda: backend.finish_mtp(backend.position))
        for operation in operations:
            with self.assertRaisesRegex(RuntimeError, "restart all sixteen"):
                operation()
        self.assertEqual(len(channel.commands), count)

    def assert_rank_prefixes(self, ranks, expected):
        for rank in ranks:
            target, draft = rank.engine.target, rank.engine.draft
            self.assertEqual(target.rows[:target.position], expected)
            for position, (token, parent) in enumerate(
                draft.rows[:draft.position]
            ):
                self.assertEqual(parent, Hidden(
                    "target", expected[position][1], position + 1,
                ))
                self.assertEqual(token, predict(parent.digest, parent.length)
                                 if position + 1 == len(expected)
                                 else expected[position + 1][0])

    def test_normal_import_does_not_load_torch(self):
        self.assertNotIn("torch", sys.modules)

    def test_sixteen_ranks_accept_reject_and_commit_identically(self):
        for mode in ("accept", "reject", "mixed"):
            for width in (2, 4):
                with self.subTest(mode=mode, width=width):
                    backend, ranks, channel = make_backend(mode)
                    first = inject(backend)
                    backend.start_mtp(first, 17, width, ())
                    output, counts = [first], []
                    while True:
                        executed = backend.advance_mtp()
                        self.assertTrue(all(
                            item == executed for item in channel.results
                        ))
                        output.extend(executed.commit.accepted_tokens)
                        counts.append(executed.commit.committed)
                        if executed.commit.finish_reason is not None:
                            break
                    expected, rows = serial_oracle((3, 10, 17), 17, 64)
                    self.assertEqual(output, expected)
                    self.assertEqual(backend.position, 3 + sum(counts))
                    self.assert_rank_prefixes(ranks, rows)
                    backend.finish_mtp(backend.position)
                    self.assertTrue(all(rank.engine._mtp is None
                                        for rank in ranks))
                    self.assertNotIn("step", [command["operation"]
                                              for command in channel.commands])

    def test_eos_budget_and_capacity_finish_at_same_rank_positions(self):
        for capacity, budget, eos_at in ((64, 3, None), (5, 20, None),
                                         (64, 20, 2)):
            with self.subTest(capacity=capacity, budget=budget, eos=eos_at):
                backend, ranks, _ = make_backend(capacity=capacity)
                first = inject(backend)
                full, _ = serial_oracle((3, 10, 17), 8, 64)
                stops = () if eos_at is None else (full[eos_at],)
                backend.start_mtp(first, budget, 4, stops)
                output = [first]
                while True:
                    result = backend.advance_mtp()
                    output.extend(result.commit.accepted_tokens)
                    if result.commit.finish_reason is not None:
                        break
                expected, rows = serial_oracle(
                    (3, 10, 17), budget, capacity, stops,
                )
                self.assertEqual(output, expected)
                self.assert_rank_prefixes(ranks, rows)

    def test_partial_delivery_trims_and_allows_a_fresh_request(self):
        for delivered in range(5):
            with self.subTest(delivered=delivered):
                backend, ranks, channel = make_backend()
                first = inject(backend)
                backend.start_mtp(first, 20, 4, ())
                result = backend.advance_mtp()
                self.assertEqual(result.commit.committed, 4)
                visible = [first, *result.commit.accepted_tokens[:delivered]]
                backend.finish_mtp(3 + delivered)
                expected, rows = serial_oracle((3, 10, 17), len(visible), 64)
                self.assertEqual(visible, expected)
                self.assert_rank_prefixes(ranks, rows)
                self.assertTrue(all(rank.serial == len(channel.commands)
                                    for rank in ranks))
                fresh = inject(backend, (7, 19), "next")
                backend.start_mtp(fresh, 5, 2, ())
                new_round = backend.advance_mtp()
                self.assertEqual(new_round.commit.number, 1)
                self.assertTrue(all(rank.request_id == "next"
                                    for rank in ranks))

    def test_finishing_without_a_round_preserves_missing_seed(self):
        backend, ranks, _ = make_backend()
        first = inject(backend)
        backend.start_mtp(first, 20, 4, ())
        backend.finish_mtp(3)
        self.assertTrue(all((rank.engine.target.position,
                             rank.engine.draft.position) == (3, 2)
                            for rank in ranks))

    def test_finish_does_not_truncate_unchanged_lengths(self):
        backend, ranks, _ = make_backend()
        first = inject(backend)
        backend.start_mtp(first, 20, 4, ())
        backend.advance_mtp()
        before = [(len(rank.engine.target.truncations),
                   len(rank.engine.draft.truncations)) for rank in ranks]
        backend.finish_mtp(backend.position)
        after = [(len(rank.engine.target.truncations),
                  len(rank.engine.draft.truncations)) for rank in ranks]
        self.assertEqual(before, after)

    def test_step_and_mtp_cannot_mix_in_either_order(self):
        for finish in (False, True):
            with self.subTest(finish=finish):
                backend, _, channel = make_backend()
                first = inject(backend)
                backend.start_mtp(first, 8, 4, ())
                if finish:
                    backend.finish_mtp(3)
                with self.assertRaises(RuntimeError):
                    backend.step(first)
                self.assert_no_more_commands(backend, channel)
        backend, _, channel = make_backend()
        first = inject(backend)
        backend.step(first)
        with self.assertRaises(RuntimeError):
            backend.start_mtp(first, 8, 4, ())
        self.assert_no_more_commands(backend, channel)

    def test_wrong_first_token_or_terminal_start_is_rejected(self):
        for wrong_token, budget, eos in ((True, 8, False), (False, 1, False),
                                        (False, 8, True)):
            with self.subTest(wrong_token=wrong_token, budget=budget, eos=eos):
                backend, _, channel = make_backend()
                first = inject(backend)
                token = first + int(wrong_token)
                stops = (first,) if eos else ()
                with self.assertRaises(ValueError):
                    backend.start_mtp(token, budget, 4, stops)
                self.assert_no_more_commands(backend, channel)

    def test_serial_position_and_request_identity_are_checked(self):
        backend, ranks, _ = make_backend()
        first = inject(backend)
        backend.start_mtp(first, 8, 4, ())
        executor = ranks[15]
        command = {"serial": 3, "position": 3, "request_id": "request",
                   "operation": "advance_mtp"}
        for changed in ({"serial": 4}, {"position": 4},
                        {"request_id": "older"}):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    executor.execute({**command, **changed})
                self.assertEqual(executor.serial, 2)
                self.assertEqual(executor.engine.position, 3)

    def test_mtp_commands_require_an_injected_request_and_start(self):
        _, ranks, _ = make_backend()
        executor = ranks[0]
        for operation in ("start_mtp", "advance_mtp", "finish_mtp"):
            with self.subTest(operation=operation):
                with self.assertRaises(RuntimeError):
                    executor.execute({"serial": 1, "position": 0,
                                      "operation": operation})
        backend, _, channel = make_backend()
        inject(backend)
        with self.assertRaises(RuntimeError):
            backend.advance_mtp()
        self.assert_no_more_commands(backend, channel)

    def test_uninjected_broadcast_failure_disables_the_backend(self):
        backend, _, channel = make_backend()
        with self.assertRaisesRegex(RuntimeError, "no injected request"):
            backend.advance_mtp()
        self.assert_no_more_commands(backend, channel)

    def test_finish_rejects_prompt_loss_or_unwritten_rows(self):
        for retained in (2, 8, True, 3.0):
            with self.subTest(retained=retained):
                backend, _, channel = make_backend()
                first = inject(backend)
                backend.start_mtp(first, 8, 4, ())
                backend.advance_mtp()
                with self.assertRaises(ValueError):
                    backend.finish_mtp(retained)
                self.assert_no_more_commands(backend, channel)

    def test_rank_zero_start_failure_rejects_reset_without_broadcast(self):
        backend, ranks, channel = make_backend()
        first = inject(backend)
        original = ranks[0].engine.start_mtp

        def fail_after_start(*args):
            original(*args)
            raise RuntimeError("rank zero start failed")

        with patch.object(ranks[0].engine, "start_mtp", fail_after_start):
            with self.assertRaisesRegex(RuntimeError, "rank zero start"):
                backend.start_mtp(first, 8, 4, ())
        self.assertEqual([rank.serial for rank in ranks], [1] + [2] * 15)
        self.assert_no_more_commands(backend, channel)

    def test_rank_zero_finish_failure_rejects_reset_without_broadcast(self):
        backend, ranks, channel = make_backend()
        first = inject(backend)
        backend.start_mtp(first, 8, 4, ())
        backend.advance_mtp()
        original = ranks[0].engine.finish_mtp

        def fail_after_finish(*args):
            original(*args)
            raise RuntimeError("rank zero finish failed")

        with patch.object(ranks[0].engine, "finish_mtp", fail_after_finish):
            with self.assertRaisesRegex(RuntimeError, "rank zero finish"):
                backend.finish_mtp(4)
        self.assertEqual([rank.serial for rank in ranks], [3] + [4] * 15)
        self.assert_no_more_commands(backend, channel)

    def test_worker_failure_disables_further_commands(self):
        backend, ranks, channel = make_backend()
        first = inject(backend)
        with patch.object(ranks[8].engine, "start_mtp",
                          side_effect=RuntimeError("worker failed")):
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                backend.start_mtp(first, 8, 4, ())
        self.assertEqual([rank.serial for rank in ranks],
                         [1] + [2] * 7 + [1] * 8)
        self.assert_no_more_commands(backend, channel)

    def test_send_failure_disables_further_commands(self):
        backend, _, channel = make_backend()
        inject(backend)
        with patch.object(channel, "send",
                          side_effect=RuntimeError("send failed")) as send:
            with self.assertRaisesRegex(RuntimeError, "send failed"):
                backend.reset()
            self.assert_no_more_commands(backend, channel)
            self.assertEqual(send.call_count, 1)

    def test_reset_clears_round_and_seed_on_every_rank(self):
        backend, ranks, _ = make_backend()
        first = inject(backend)
        backend.start_mtp(first, 8, 4, ())
        backend.advance_mtp()
        backend.reset()
        self.assertTrue(all(
            rank.request_id is None and rank.engine.position == 0
            and rank.engine.draft.position == 0 and rank.engine._mtp is None
            and rank.engine.parent_hidden is None
            and rank.engine.first_token_id is None for rank in ranks
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MtpTpChecks)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    sources = [Path(__file__), *(ROOT / name for name in (
        "tilert/models/glm_5_2_ascend/tp_runtime.py",
        "tilert/models/glm_5_2_ascend/mtp_round.py",
        "tilert/models/glm_5_2_ascend/mtp_state.py",
        "tilert/pd_vllm/ascend_cache.py",
    ))]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if result.wasSuccessful() else "fail",
        "scope": "TP16 Python command and MTP state; scalar cache arithmetic",
        "excluded": "HCCL, HIXL, Torch, NPU numerics and latency",
        "python": platform.python_version(), "hostname": platform.node(),
        "command": [sys.executable, *sys.argv], "ranks": 16,
        "tests": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "torch_imported": "torch" in sys.modules,
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
