"""Check four-credit waiting, complete batch publication and frozen source scope."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from prepare_batch import (CONSUMER, CONTRACT, NATIVE, bind_contract, bind_native,
                           digest, prepare, rename)


def explore(total, window=4, batch=4, require_both=True, drain=True):
    """Track publications, free waits and each worker's ready/store/ack counts."""
    initial = (0, 0, 0, 0, 0, 0, 0, 0)
    pending = [initial]
    seen = {initial}
    terminal = []
    violations = []
    while pending:
        state = pending.pop()
        published, waited, read0, done0, ack0, read1, done1, ack1 = state
        joined = min(ack0, ack1) if require_both else max(ack0, ack1)
        if (published - min(ack0, ack1) > window
                or max(published - read0, published - read1,
                       ack0 - waited, ack1 - waited) > 15):
            violations.append(state)
        following = []
        if published < total and (published < window or waited > published - window):
            after = list(state)
            after[0] += 1
            following.append(tuple(after))
        needs_credit = published < total and published >= window and waited <= published - window
        tail_credit = drain and published == total and waited < total
        if joined > waited and (needs_credit or tail_credit):
            after = list(state)
            after[1] += 1
            following.append(tuple(after))
        for offset in (2, 5):
            read, done, ack = state[offset:offset + 3]
            if ack == done and read < published and read < done + batch:
                after = list(state)
                after[offset] += 1
                following.append(tuple(after))
            if ack == done and read == done + batch:
                after = list(state)
                after[offset + 1] += batch
                following.append(tuple(after))
            if ack < done:
                after = list(state)
                after[offset + 2] += 1
                following.append(tuple(after))
        if not following:
            terminal.append(state)
        for after in following:
            if after not in seen:
                seen.add(after)
                pending.append(after)
    return seen, terminal, violations


class DownBatchContract(unittest.TestCase):
    parent = Path(os.environ.get("FFN_DOWN_BATCH_PARENT",
        "/tmp/tilert-megakernel-validation-20261021/ffn-down-overlap-build-v2"))

    def test_both_consumers_and_all_four_publications_precede_reuse(self):
        states, terminal, bad = explore(24)
        self.assertIn((4, 0, 4, 4, 4, 0, 0, 0), states)
        self.assertNotIn((5, 1, 4, 4, 4, 0, 0, 0), states)
        self.assertEqual(terminal, [(24,) * 8])
        self.assertEqual(bad, [])
        self.assertTrue(explore(24, require_both=False)[2])

    def test_window_smaller_than_batch_has_a_concrete_deadlock(self):
        _, terminal, _ = explore(24, window=3)
        self.assertEqual(terminal, [(3, 0, 3, 0, 0, 3, 0, 0)])

    def test_zero_to_maximum_owned_experts_drain_all_counters(self):
        for experts in (0, 1, 2, 3, 11, 16):
            total = experts * 24
            with self.subTest(experts=experts):
                _, terminal, bad = explore(total)
                self.assertEqual(terminal, [(total,) * 8])
                self.assertEqual(bad, [])
        _, terminal, _ = explore(24, drain=False)
        self.assertEqual(terminal, [(24, 20, 24, 24, 24, 24, 24, 24)])

    def test_every_batch_ends_inside_its_expert_and_ownership_matches(self):
        fixtures = ([0] * 256, [1] * 256,
                    [12 if index < 8 else 0 for index in range(256)],
                    [1 if index % 31 == 0 else 0 for index in range(256)])
        for counts in fixtures:
            active = [expert for expert, count in enumerate(counts) if count]
            for blocks in (16, 24):
                for core in range(blocks):
                    produced = [(expert, tile) for expert in active[core::blocks]
                                for tile in range(24)]
                    batches = [produced[index:index + 4]
                               for index in range(0, len(produced), 4)]
                    self.assertEqual([item for batch in batches for item in batch], produced)
                    self.assertTrue(all(len(batch) == 4 and len({e for e, _ in batch}) == 1
                                        for batch in batches))
                    self.assertTrue(all(batch[0][1] % 4 == 0 for batch in batches))

    def test_actual_cpp_addresses_cover_both_partial_planes_and_all_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "overlap_contract.h").write_text(rename(bind_contract(
                (self.parent / CONTRACT).read_text())))
            binary = root / "ownership"
            subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(root), str(Path(__file__).with_name("batch_ownership_probe.cpp")),
                "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_producer_matmul_credit_and_drain_source_are_unchanged(self):
        original = (self.parent / NATIVE).read_text()
        changed = bind_native(original)
        prefix = changed.split(CONSUMER)[0].replace(
            "static_assert(ffn_down_overlap::kDownConsumerHalfSpan <= kVectorTile);\n\n", "")
        self.assertEqual(prefix, original.split(CONSUMER)[0])
        self.assertNotIn("InitBuffer", changed)
        self.assertNotIn("Reset", changed)
        self.assertIn("mm.GetTensorC(c, 0, false);", changed)

    def test_only_consumer_and_constants_change_beyond_private_names(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            report = prepare(self.parent, output)
            parent_manifest = json.loads((self.parent / "manifest.json").read_text())
            for name in parent_manifest["sources"]:
                if name in (NATIVE, CONTRACT):
                    continue
                old = (self.parent / name).read_bytes()
                expected = rename(old.decode()).encode() if (
                    Path(name).suffix in (".cpp", ".h") or name == "check.py") else old
                self.assertEqual((output / name).read_bytes(), expected, name)
            self.assertEqual(report["public_scratch_count"], 32)
            self.assertEqual(report["additional_gm_bytes"], 0)
            self.assertEqual(report["additional_ub_bytes"], 0)
            self.assertEqual(report["body_barriers"], 6)
            self.assertEqual(report["outer_end_phases"], 2)
            for name, expected in report["sources"].items():
                self.assertEqual(digest(output / name), expected)


if __name__ == "__main__":
    unittest.main()
