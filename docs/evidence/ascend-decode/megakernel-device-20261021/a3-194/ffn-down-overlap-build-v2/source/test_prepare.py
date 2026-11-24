"""Verify the composition's frozen seams without compiling or launching NPU code."""

from pathlib import Path
import tempfile
import unittest

from prepare import bridge_body, host_seam, prepare


class CompositionSeams(unittest.TestCase):
    bridge = Path("/tmp/tilert-megakernel-validation-20261021/ffn-bridge-builds/native-v2")
    body = (Path(__file__).resolve().parents[4]
            / "docs/evidence/ascend-decode/megakernel-device-20261021"
              "/a3-194/routed-body-build-v2")

    def setUp(self):
        if not self.bridge.is_dir() or not self.body.is_dir():
            self.skipTest("Frozen source parents are unavailable")

    def test_host_checks_are_retained_without_launch(self):
        source = (self.body / "source/routed_int4/routed_int4_binding.cpp").read_text()
        actual, removed = host_seam(source, "private_body", ("routed_int4_launch",))
        original = source.split("\n}  // namespace\n", 1)[0] + "\n}  // namespace\n"
        restored = actual.replace("namespace private_body {\n", "namespace {\n", 1)
        marker = "\n\n    return scratch[kOutput];"
        restored = restored.replace(marker, "\n" + removed[0] + marker[1:], 1)
        self.assertEqual(restored, original)

    def test_bridge_statements_keep_three_boundaries(self):
        source = (self.bridge / "source/ffn_bridge/kernel.cpp").read_text()
        actual, _ = bridge_body(source)
        begin = source.index("    whole_layer_norm::device::Forward(pipe,")
        end = source.index("\n}\n", begin)
        self.assertIn(source[begin:end], actual)
        self.assertEqual(actual.count("device::EndPhase(pipe);"), 3)
        self.assertNotIn("SetSysWorkspace", actual)

    def test_manifest_freezes_both_parents_and_seams(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "build"
            report = prepare(self.bridge, self.body, output)
            self.assertIn("body/source/routed_int4/host_validate.h", report["sources"])
            self.assertIn("bridge/source/ffn_bridge/body_native.h", report["sources"])
            self.assertEqual(len(report["removed_launch_statements"]["bridge"]), 3)
            source = (output / "body/source/routed_int4/host_validate.h").read_text()
            self.assertNotIn("TORCH_LIBRARY", source)
            self.assertNotIn("getCurrentNPUStream", source)


if __name__ == "__main__":
    unittest.main()
