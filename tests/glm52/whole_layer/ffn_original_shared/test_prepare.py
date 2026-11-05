"""Protect complete-FFN layout, synchronization and the activation seam."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from prepare import NATIVE, UP_BIAS, bind_activation, prepare, rename


PARENT = Path(os.environ.get(
    "FFN_ORIGINAL_SHARED_PARENT", "/tmp/tilert-megakernel-validation-20261021/"
    "ffn-active-token-join-runtime-v1/build-v1"))


class SharedActivationContract(unittest.TestCase):
    def test_compiled_sources_outside_activation_are_preserved(self):
        manifest = json.loads((PARENT / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            report = prepare(PARENT, output)
            for name in manifest["sources"]:
                if Path(name).suffix not in (".h", ".cpp") or name == NATIVE:
                    continue
                self.assertEqual((output / name).read_text(),
                                 rename((PARENT / name).read_text()), name)
            self.assertEqual(report["body_barriers"], 5)

    def test_input_staging_projection_and_publication_are_preserved(self):
        original = (PARENT / NATIVE).read_text()
        changed = bind_activation(original)
        marker = "__aicore__ inline void Activate("
        self.assertEqual(original.split(marker)[0], changed.split(marker)[0])
        publish = "        Fence<HardEvent::V_MTE3>();\n"
        self.assertEqual(original[original.rindex(publish):],
                         changed[changed.rindex(publish):])

    def test_ambiguous_or_missing_source_seams_are_rejected(self):
        original = (PARENT / NATIVE).read_text()
        for invalid in (original.replace(UP_BIAS, ""), original + UP_BIAS):
            with self.assertRaises(ValueError):
                bind_activation(invalid)


if __name__ == "__main__":
    unittest.main()
