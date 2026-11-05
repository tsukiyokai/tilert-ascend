"""Exhaust the two-consumer credit protocol and compact down-address ownership."""

from pathlib import Path
import subprocess
import tempfile
import unittest


def explore(total, window=4, drain=True, require_both=True):
    """State is published tiles, both completed consumers, consumed free credits."""
    initial = (0, 0, 0, 0)
    pending = [initial]
    seen = {initial}
    terminal = []
    violations = []
    while pending:
        state = pending.pop()
        published, first, second, waited = state
        joined = min(first, second) if require_both else max(first, second)
        if (published - min(first, second) > window
                or max(published - first, published - second, joined - waited) > 15):
            violations.append(state)
        next_states = []
        if published < total and (published < window or waited > published - window):
            next_states.append((published + 1, first, second, waited))
        if first < published:
            next_states.append((published, first + 1, second, waited))
        if second < published:
            next_states.append((published, first, second + 1, waited))
        needs_credit = published < total and published >= window and waited <= published - window
        tail_credit = drain and published == total and waited < total
        if joined > waited and (needs_credit or tail_credit):
            next_states.append((published, first, second, waited + 1))
        if not next_states:
            terminal.append(state)
        for after in next_states:
            if after not in seen:
                seen.add(after)
                pending.append(after)
    return seen, terminal, violations


class DownCreditContract(unittest.TestCase):
    def test_both_vector_workers_are_required_before_releasing_credit(self):
        states, terminal, bad = explore(24)
        self.assertIn((4, 4, 0, 0), states)
        self.assertNotIn((5, 4, 0, 1), states)
        self.assertEqual(terminal, [(24, 24, 24, 24)])
        self.assertEqual(bad, [])
        _, _, wrong_join = explore(24, require_both=False)
        self.assertTrue(wrong_join)

    def test_every_valid_expert_count_drains_ready_and_free_counters(self):
        for experts in range(17):
            total = 24 * experts
            with self.subTest(experts=experts):
                _, terminal, bad = explore(total)
                self.assertEqual(terminal, [(total, total, total, total)])
                self.assertEqual(bad, [])

    def test_skipped_drain_leaves_exactly_four_credits_for_the_next_replay(self):
        for total in (1, 2, 4, 24, 48):
            _, terminal, _ = explore(total, drain=False)
            self.assertEqual(terminal, [(total, total, total, max(0, total - 4))])

    def test_zero_and_sparse_experts_have_matching_continuous_tile_sequences(self):
        fixtures = ([0] * 256, [1] * 256,
                    [12 if i < 8 else 0 for i in range(256)],
                    [1 if i % 31 == 0 else 0 for i in range(256)])
        for counts in fixtures:
            for blocks in (16, 24):
                for core in range(blocks):
                    ordinal, next_owner = 0, core
                    producer = []
                    for expert, count in enumerate(counts):
                        if count == 0:
                            continue
                        owned = ordinal == next_owner
                        ordinal += 1
                        if owned:
                            next_owner += blocks
                            producer.extend((expert, tile) for tile in range(24))
                    active = [expert for expert, count in enumerate(counts) if count]
                    expected = [(expert, tile) for expert in active[core::blocks]
                                for tile in range(24)]
                    self.assertEqual(producer, expected)
                    self.assertEqual(len(producer), 24 * len(active[core::blocks]))

    def test_cpp_ownership_and_both_compact_partial_addresses(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "ownership"
            subprocess.run([
                "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                str(Path(__file__).with_name("ownership_probe.cpp")),
                "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
