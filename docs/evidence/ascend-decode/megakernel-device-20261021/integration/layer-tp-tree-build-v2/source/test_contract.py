"""Check exact source substitution and the retained ABI before compilation."""

import ast
import json
from pathlib import Path
import unittest

from prepare import (RENAMED_FILES, TREE_FILES, digest, private_names, verify)


ROOT = Path(__file__).resolve().parents[1]


class IntegrationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "manifest.json").read_text())
        cls.parent = json.loads((ROOT / "parent/manifest.json").read_text())
        cls.tree = json.loads((ROOT / "tree/manifest.json").read_text())

    def test_frozen_inputs(self):
        verify(ROOT, self.manifest["sources"])

    def test_layer_changes_are_only_private_names(self):
        for name in RENAMED_FILES:
            with self.subTest(name=name):
                original = (ROOT / "parent" / name).read_text()
                self.assertEqual((ROOT / name).read_text(), private_names(original))

    def test_all_other_parent_sources_are_identical(self):
        changed = set(RENAMED_FILES) | {"source/build.py", "source/prepare.py"}
        changed.update("communication/" + name for name in TREE_FILES[:3])
        for name, expected in self.parent["sources"].items():
            if name not in changed:
                with self.subTest(name=name):
                    self.assertEqual(digest(ROOT / name), expected)

    def test_communication_files_are_device_accepted(self):
        for name in TREE_FILES:
            with self.subTest(name=name):
                self.assertEqual(digest(ROOT / "communication" / name),
                                 self.tree["sources"]["source/" + name])

    def test_python_owner_contract_is_unchanged(self):
        original = (ROOT / "parent/host.py").read_text()
        candidate = (ROOT / "source/host.py").read_text()
        self.assertEqual(candidate.replace("layer_tp_tree_", "layer_tp_"), original)
        self.assertEqual(
            ast.dump(ast.parse(candidate.replace("layer_tp_tree_", "layer_tp_"))),
            ast.dump(ast.parse(original)))

    def test_reducer_allocation_matches_phase_budget(self):
        self.assertEqual(self.manifest["communication_ub_bytes"], 2048 * (2 + 5 * 4))
        self.assertEqual(self.manifest["record_bytes"], [664, 928, 48])
        self.assertEqual(self.manifest["kernel_argument_bytes"], 8 + 664 + 928 + 48)


if __name__ == "__main__":
    unittest.main()
