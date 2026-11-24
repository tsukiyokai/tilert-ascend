"""Check the isolated scheduling patch against the frozen FFN numerical body."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from prepare import (BODY, BODY_MARKER, PHASE_FIVE, PHASE_FOUR, PHASE_SIX,
                     TILING, TILING_GUARD, bind_body, bind_tiling, prepare, rename)


class DownSourceContract(unittest.TestCase):
    parent = Path(os.environ.get("FFN_DOWN_OVERLAP_PARENT", str(
        Path(__file__).resolve().parents[4]
        / "docs/evidence/ascend-decode/megakernel-device-20261021"
        / "a3-194/ffn-composed-build-v2")))

    def test_all_math_helpers_prefix_tail_and_global_joins_are_preserved(self):
        original = (self.parent / BODY).read_text()
        candidate = bind_body(original)
        without_includes = candidate.replace(
            '\n#include "../../../source/overlap_contract.h"', "").replace(
            '#include "../../../source/overlap_native.h"\n\n', "")
        self.assertEqual(without_includes.split(PHASE_FOUR, 1)[0],
                         original.split(PHASE_FOUR, 1)[0])
        self.assertEqual(candidate.split(PHASE_SIX, 1)[1],
                         original.split(PHASE_SIX, 1)[1])
        self.assertEqual(candidate.count("timing.Barrier("), 5)
        self.assertIn(PHASE_FIVE + PHASE_SIX, candidate)
        for name in ("routed_int4_front.h", "routed_int4_dispatch.h"):
            self.assertEqual((self.parent / Path(BODY).with_name(name)).read_text().count(
                "timing.Barrier(0);"), 1)
        phase = candidate.split(PHASE_FOUR, 1)[1].split(PHASE_FIVE, 1)[0]
        self.assertEqual(phase.count("int64_t down_sequence = 0;"), 1)
        self.assertLess(phase.index("down_sequence = 0"), phase.index("for (int64_t expert"))
        self.assertLess(phase.index("shared.Store(pipe)"), phase.index("ConsumeDown("))
        self.assertLess(phase.index("DrainDownCredits("), phase.index("ConsumeDown("))
        native = Path(__file__).with_name("overlap_native.h").read_text()
        self.assertIn("ROUTED_FIXPIPE_HALF", native)
        self.assertIn("ROUTED_ENABLE_PROFILE", native)
        self.assertIn("mm.GetTensorC(c, 0, false);", native)
        self.assertNotIn("pipe.Reset", native)
        self.assertNotIn("InitBuffer", native)
        # Default IterateAll performs the same shape setup and full-GM copy.
        prefix = original.split("    const int64_t matrix_rows =", 1)[1]
        prefix = prefix.split("#ifdef ROUTED_FIXPIPE_HALF", 1)[0]
        prefix = "    const int64_t matrix_rows =" + prefix
        prefix = prefix.replace("pad_rows ? GateMatrixRows(count) : 2 * count", "2 * count")
        self.assertIn(prefix, native)
        self.assertEqual(without_includes.split(BODY_MARKER, 1)[0],
                         original.split(BODY_MARKER, 1)[0])

    def test_host_tiling_guard_changes_no_tiler_inputs_or_validators(self):
        original = (self.parent / TILING).read_text()
        candidate = bind_tiling(original)
        restored = candidate.replace(
            '\n#include "../../../source/overlap_contract.h"', "").replace(TILING_GUARD, "")
        self.assertEqual(restored, original)
        self.assertLess(candidate.index("tiling.GetTiling(cube)"),
                        candidate.index("IsDownTiling("))
        self.assertLess(candidate.index("IsDownTiling("), candidate.index("std::memcpy"))

    def test_public_abi_owners_allocation_and_checker_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            report = prepare(self.parent, output)
            for name in ("source/contract.h", "source/binding.cpp", "source/kernel.cpp",
                         "body/source/routed_int4/host_validate.h",
                         "bridge/source/ffn_bridge/host_validate.h",
                         "body/source/routed_int4/routed_int4_front.h",
                         "body/source/routed_int4/routed_int4_dispatch.h",
                         "body/source/routed_int4/routed_int4_pipeline_native.h"):
                self.assertEqual((output / name).read_text(),
                                 rename((self.parent / name).read_text()))
            expected = (self.parent / "harness-v1/check.py").read_text().replace(
                "torch.ops.tilert_ffn_composed", "torch.ops.tilert_ffn_down_overlap")
            self.assertEqual((output / "check.py").read_text(), expected)
            self.assertEqual(report["public_scratch_count"], 32)
            self.assertEqual(report["additional_gm_bytes"], 0)
            self.assertEqual(report["additional_ub_bytes"], 0)
            self.assertEqual(report["maximum_resident_ub_bytes"], 112352)
            self.assertEqual(report["body_barriers"], 6)
            self.assertEqual(report["outer_end_phases"], 2)
            for name, expected in report["sources"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)

    def test_unverified_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent"
            parent.mkdir()
            (parent / "manifest.json").write_text(json.dumps({
                "artifacts": {"ffn_composed.so": "unverified"}}))
            with self.assertRaisesRegex(ValueError, "validated uninstrumented"):
                prepare(parent, Path(directory) / "candidate")


if __name__ == "__main__":
    unittest.main()
