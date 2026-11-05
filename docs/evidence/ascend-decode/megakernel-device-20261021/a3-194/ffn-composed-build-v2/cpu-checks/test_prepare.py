"""Verify the composition's frozen seams without compiling or launching NPU code."""

from pathlib import Path
import importlib.util
import tempfile
import unittest

from prepare import bridge_body, host_seam, prepare


class CompositionSeams(unittest.TestCase):
    evidence = (Path(__file__).resolve().parents[4]
                / "docs/evidence/ascend-decode/megakernel-device-20261021")
    bridge = evidence / "integration/ffn-bridge-build-v2/native-v2"
    body = evidence / "a3-194/routed-body-build-v2"

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

    def test_shared_decoder_excludes_padding_and_keeps_last_valid_row(self):
        import torch

        path = Path(__file__).with_name("check.py")
        spec = importlib.util.spec_from_file_location("composition_check", path)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        for rows in (1, 12, 17, 32):
            capacity = 16 if rows <= 16 else 32
            partial, _, _, down = checker.shared_layout(capacity)
            payload = torch.zeros(down + capacity * 6144 * 4, dtype=torch.uint8)
            scratch = [torch.empty(0)] * 20
            scratch[17] = payload
            values = checker.body_values(torch, scratch, rows)
            before = values["body.shared_gate_int32"].clone()
            address = partial // 4 + (((5 * 4 + 3) * capacity + rows - 1) * 64 + 63)
            payload.view(torch.int32)[address] = 1701
            after = checker.body_values(torch, scratch, rows)["body.shared_gate_int32"]
            self.assertEqual(int(after[5, 3, rows - 1, 63]), 1701)
            self.assertEqual(int((after != before).sum()), 1)
            if rows < capacity:
                padding = partial // 4 + rows * 64
                expected = after.clone()
                payload.view(torch.int32)[padding:padding + 64] = -12345
                actual = checker.body_values(torch, scratch, rows)["body.shared_gate_int32"]
                self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
