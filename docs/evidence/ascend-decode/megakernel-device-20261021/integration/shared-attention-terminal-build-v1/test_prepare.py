"""Protect callable completion and the frozen numerical and host contracts."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from prepare import LAYER, prepare, private_names, terminal_full, terminal_kernel


PARENT = Path(os.environ.get(
    "ATTENTION_TERMINAL_PARENT", "/tmp/tilert-runtime212-build-v1/attention"))


class TerminalContract(unittest.TestCase):
    def test_default_callable_keeps_final_join(self):
        original = (PARENT / "shared/source" / LAYER / "attention_full_native.h").read_text()
        transformed = terminal_full(original)
        self.assertIn("template <bool Terminal = false>", transformed)
        self.assertIn("} else {\n        EndPhase(pipe);\n    }", transformed)
        self.assertIn("AscendC::PipeBarrier<PIPE_ALL>();", transformed)
        self.assertIn("AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();", transformed)
        old_body = original.split("{\n", 2)[2].split("    EndPhase(pipe);", 1)[0]
        new_body = transformed.split("{\n", 2)[2].split("    if constexpr", 1)[0]
        self.assertEqual(old_body, new_body)

    def test_only_global_entry_selects_terminal_completion(self):
        original = (PARENT / "shared/source" / LAYER / "attention_full_kernel.cpp").read_text()
        transformed = terminal_kernel(original)
        self.assertEqual(transformed.replace("FullForward<true>(", "FullForward("), original)
        suffix = transformed.split("FullForward<true>(", 1)[1].split("\n}", 1)[0]
        self.assertEqual(suffix.count(";"), 1)

    def test_all_other_numerical_and_host_sources_are_unchanged(self):
        manifest = json.loads((PARENT / "manifest.json").read_text())
        shared = next(item for item in manifest["builds"] if item["mode"] == "shared")
        changed = {LAYER + "attention_full_native.h", LAYER + "attention_full_kernel.cpp"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            prepare(PARENT, output)
            for name in shared["source_hashes"]:
                if name in changed:
                    continue
                old = (PARENT / "shared/source" / name).read_text()
                expected = private_names(old) if Path(name).suffix in (".cpp", ".h") else old
                self.assertEqual((output / "source" / name).read_text(), expected, name)

    def test_unexpected_completion_seam_is_rejected(self):
        source = (PARENT / "shared/source" / LAYER / "attention_full_native.h").read_text()
        with self.assertRaises(ValueError):
            terminal_full(source.replace("    EndPhase(pipe);\n", ""))
        with self.assertRaises(ValueError):
            terminal_full(source + "    EndPhase(pipe);\n")


if __name__ == "__main__":
    unittest.main()
