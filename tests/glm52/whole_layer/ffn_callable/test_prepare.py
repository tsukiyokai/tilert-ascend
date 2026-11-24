"""Check the FFN extraction against the device-validated parent source."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from prepare import INITIALIZE, extract, prepare


class CallableSeam(unittest.TestCase):
    parent = (Path(__file__).resolve().parents[4]
              / "docs/evidence/ascend-decode/megakernel-device-20261021"
              / "a3-194/ffn-composed-build-v2")

    def setUp(self):
        if not self.parent.is_dir():
            self.skipTest("Frozen composed parent is unavailable")

    def test_every_original_statement_and_final_barrier_survive(self):
        original = (self.parent / "source/kernel.cpp").read_text()
        header, wrapper, statements = extract(original)
        self.assertIn(INITIALIZE + statements, original)
        self.assertIn(statements, header)
        self.assertNotIn("InitSocState", header)
        self.assertNotIn("SetSyncBaseAddr", header)
        self.assertNotIn("TPipe pipe;", header)
        self.assertEqual(header.count("ffn_bridge::device::EndPhase(pipe);"), 2)
        self.assertTrue(statements.rstrip().endswith("ffn_bridge::device::EndPhase(pipe);"))
        self.assertIn(INITIALIZE, wrapper)
        self.assertEqual(wrapper.count("Forward<format>"), 1)

    def test_host_checks_change_only_namespace_and_c_symbols(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "callable"
            report = prepare(self.parent, output)
            for name in ("source/binding.cpp", "source/contract.h",
                         "bridge/source/ffn_bridge/host_validate.h",
                         "body/source/routed_int4/host_validate.h"):
                expected = (self.parent / name).read_text().replace(
                    "ffn_composed", "ffn_callable").replace(
                    "FFN_COMPOSED", "FFN_CALLABLE")
                self.assertEqual((output / name).read_text(), expected)
            body = extract((self.parent / "source/kernel.cpp").read_text())[2]
            self.assertEqual(report["callable_statement_sha256"],
                             hashlib.sha256(body.encode()).hexdigest())
            self.assertIn(body, (output / "source/native.h").read_text())
            for name, expected in report["sources"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)

    def test_parent_identity_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent"
            parent.mkdir()
            (parent / "manifest.json").write_text(json.dumps({
                "artifacts": {"ffn_composed.so": "unverified"}}))
            with self.assertRaisesRegex(ValueError, "validated composed"):
                prepare(parent, Path(directory) / "output")


if __name__ == "__main__":
    unittest.main()
