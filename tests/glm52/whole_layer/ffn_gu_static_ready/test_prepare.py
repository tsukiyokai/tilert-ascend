"""Check expert ownership, two-consumer credits and unchanged FFN arithmetic."""

import json
import os
from pathlib import Path
import random
import tempfile
import unittest

from prepare import BODY, PARENT_LIBRARY, digest, prepare, rename, transform


PARENT = Path(os.environ.get(
    "FFN_GU_STATIC_READY_PARENT",
    "/tmp/tilert-megakernel-validation-20261021/ffn-original-shared-gu-static-build-v1"))


def credit_states(total, require_both=True, drain=True):
    pending, seen = [(0, 0, 0, 0)], {(0, 0, 0, 0)}
    terminal, violations = [], []
    while pending:
        published, first, second, waited = state = pending.pop()
        joined = min(first, second) if require_both else max(first, second)
        if (published - min(first, second) > 4
                or max(published - first, published - second, joined - waited) > 15):
            violations.append(state)
        after = []
        if published < total and (published < 4 or waited > published - 4):
            after.append((published + 1, first, second, waited))
        if first < published:
            after.append((published, first + 1, second, waited))
        if second < published:
            after.append((published, first, second + 1, waited))
        needed = published < total and published >= 4 and waited <= published - 4
        tail = drain and published == total and waited < total
        if joined > waited and (needed or tail):
            after.append((published, first, second, waited + 1))
        if not after:
            terminal.append(state)
        for value in after:
            if value not in seen:
                seen.add(value)
                pending.append(value)
    return terminal, violations


class GateReadyContract(unittest.TestCase):
    def test_two_consumers_drain_every_supported_expert_sequence(self):
        for total in range(17):
            terminal, violations = credit_states(total)
            self.assertEqual(terminal, [(total, total, total, total)])
            self.assertEqual(violations, [])
        self.assertTrue(credit_states(16, require_both=False)[1])
        self.assertEqual(credit_states(16, drain=False)[0], [(16, 16, 16, 12)])

    def test_paired_workers_cover_every_route_once_and_share_producer_order(self):
        rng = random.Random(20261022)
        for rows in (1, 2, 12, 32):
            fixtures = [[list(range(8)) for _ in range(rows)],
                        [rng.sample(range(256), 8) for _ in range(rows)],
                        [[(8 * token + slot) % 256 for slot in range(8)]
                         for token in range(rows)]]
            for choices in fixtures:
                counts = [sum(expert in row for row in choices) for expert in range(256)]
                active = [expert for expert, count in enumerate(counts) if count]
                for blocks in (16, 24):
                    seen = []
                    for core in range(blocks):
                        owned = active[core::blocks]
                        for subblock in (0, 1):
                            consumed = active[((2 * core + subblock) // 2)::blocks]
                            self.assertEqual(consumed, owned)
                            seen.extend((expert, row) for expert in consumed
                                        for row in range(subblock, counts[expert], 2))
                    expected = [(expert, row) for expert in active
                                for row in range(counts[expert])]
                    self.assertEqual(sorted(seen), expected)
                    self.assertEqual(len(seen), len(set(seen)))

    def test_original_arithmetic_helpers_and_down_tail_are_unchanged(self):
        source = (PARENT / BODY).read_text()
        changed = transform(source)
        begin = source.index("// int4b_t is a storage wrapper")
        end = source.index('#include "../../../source/output_native.h"')
        self.assertIn(source[begin:end], changed)
        self.assertEqual(source[source.index("    timing.Barrier(3);"):],
                         changed[changed.index("    timing.Barrier(3);"):])
        self.assertEqual(source.count("timing.Barrier("), changed.count("timing.Barrier("))
        self.assertIn("if (!overlap_gate)", changed)
        self.assertLess(changed.index("ConsumeGate(held"), changed.index("    timing.Barrier(2);"))
        self.assertLess(changed.index("DrainGate(gate_sequence)"), changed.index("    timing.Barrier(2);"))
        self.assertGreater(changed.index("    shared.Down(pipe)"), changed.index("    timing.Barrier(2);"))

    def test_frozen_non_gu_sources_and_public_contract_remain_identical(self):
        manifest = json.loads((PARENT / "manifest.json").read_text())
        self.assertEqual(manifest["artifacts"]["ffn_original_shared_gu_static.so"], PARENT_LIBRARY)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            report = prepare(PARENT, output)
            for name in manifest["sources"]:
                if name == BODY or name.startswith("candidate_tools/") or name == "parent-manifest.json":
                    continue
                target = output / ("parent-check.py" if name == "check.py" else name)
                original = (PARENT / name).read_bytes()
                if target.suffix in (".h", ".cpp"):
                    original = rename(original.decode()).encode()
                self.assertEqual(target.read_bytes(), original, name)
            self.assertEqual(report["public_scratch_count"], 33)
            self.assertEqual(report["body_barriers"], 5)
            self.assertEqual(report["additional_gm_bytes"], 0)
            guards = [path.read_text().splitlines()[0] for path in
                      (output / "source").glob("*.h")]
            self.assertEqual(len(guards), len(set(guards)))
            for name, identity in report["sources"].items():
                self.assertEqual(digest(output / name), identity, name)


if __name__ == "__main__":
    unittest.main()
