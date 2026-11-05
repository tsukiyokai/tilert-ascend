"""Check the N1024 host record and unchanged local-layer dataflow."""

import ast
import hashlib
import json
from pathlib import Path
import unittest

import torch

from down_tiling import EXPECTED_HEX, EXPECTED_SHA, check_down_tiling


class N1024LayerContract(unittest.TestCase):
    def test_complete_tiling_rejects_mutation_and_prior_geometries(self):
        value = torch.tensor(list(bytes.fromhex(EXPECTED_HEX)), dtype=torch.uint8)
        self.assertEqual(check_down_tiling(torch, value)["sha256"], EXPECTED_SHA)
        for offset in (0, 36, 40, 199):
            changed = value.clone()
            changed[offset] ^= 1
            with self.assertRaisesRegex(RuntimeError, "host tiler"):
                check_down_tiling(torch, changed)
        changed = value.clone()
        changed[36:40] = torch.tensor([0, 2, 0, 0], dtype=torch.uint8)
        with self.assertRaisesRegex(RuntimeError, "host tiler"):
            check_down_tiling(torch, changed)

    def test_609_and_explicit_topk_copy_are_retained(self):
        root = Path(__file__).resolve().parent
        source = (root / "local_chain.py").read_text()
        parent = (root / "parent-local_chain.py").read_text()

        def definitions(text):
            tree = ast.parse(text)
            result = {}
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    result[node.name] = ast.dump(node, include_attributes=False)
                if isinstance(node, ast.ClassDef):
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef):
                            result[node.name + "." + method.name] = ast.dump(
                                method, include_attributes=False)
            return result

        before, after = definitions(parent), definitions(source)
        for name in ("attention_addresses", "publish_topk", "successor_inputs",
                     "LayerCase.separate", "LayerCase.combined", "LayerCase.owners"):
            self.assertEqual(before[name], after[name], name)
        self.assertIn("609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f", source)
        self.assertIn("if number == 6:\n                            publish_topk(", source)
        self.assertNotIn("and arm ==", source)
        self.assertIn("case.body[20].fill_(1)", source)
        self.assertIn("case.body[20][0].zero_()", source)

    def test_all_embedded_device_sources_match_and_identities_are_frozen(self):
        root = Path(__file__).resolve().parent
        compatibility = json.loads((root / "k128-source-compatibility.json").read_text())
        self.assertEqual(len(compatibility["normalized_unchanged_files"]), 48)
        self.assertEqual(len(compatibility["embedded_ffn_files_identical"]), 49)
        self.assertEqual(compatibility["changed_files"],
                         ["body/source/routed_int4/routed_int4_tiling.cpp"])
        for line in (root / "frozen-inputs.sha256").read_text().splitlines():
            expected, name = line.split("  ", 1)
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(),
                             expected, name)


if __name__ == "__main__":
    unittest.main()
