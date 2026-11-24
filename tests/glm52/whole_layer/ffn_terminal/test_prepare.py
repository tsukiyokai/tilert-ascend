"""Protect FFN publication boundaries, references and host launch contract."""

import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

from prepare import (
    KERNEL, NEW_EXIT, OLD_EXIT, bind_checker, bind_kernel, prepare, rename,
)


PARENT = Path(os.environ.get(
    "FFN_TERMINAL_PARENT", "/tmp/tilert-megakernel-validation-20261021/"
    "ffn-active-token-join-runtime-v1/build-v1"))


class TerminalContract(unittest.TestCase):
    def test_only_final_global_completion_changes(self):
        original = (PARENT / KERNEL).read_text()
        transformed = bind_kernel(original)
        self.assertEqual(transformed.replace(NEW_EXIT, OLD_EXIT), original)
        self.assertIn(NEW_EXIT + "\n}\n", transformed)
        for invalid in (original.replace(OLD_EXIT, ""), original + OLD_EXIT):
            with self.assertRaises(ValueError):
                bind_kernel(invalid)

    def test_callable_math_and_all_host_arguments_are_unchanged(self):
        manifest = json.loads((PARENT / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            prepare(PARENT, output)
            for name in manifest["sources"]:
                if Path(name).suffix not in (".h", ".cpp") or name == KERNEL:
                    continue
                self.assertEqual((output / name).read_text(),
                                 rename((PARENT / name).read_text()), name)
            body = output / "body/source/routed_int4/routed_body_native.h"
            self.assertIn("#ifdef ROUTED_ENABLE_PROFILE\n#error", body.read_text())

    def test_reference_chains_and_queued_input_producers_are_preserved(self):
        original = ast.parse((PARENT / "check.py").read_text())
        generated = bind_checker((PARENT / "check.py").read_text())
        candidate = ast.parse(generated)

        def function(tree, name):
            return next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == name)

        for name in ("separate", "production", "all_values", "body_values"):
            self.assertEqual(ast.dump(function(original, name)),
                             ast.dump(function(candidate, name)), name)
        composed = function(candidate, "composed")
        self.assertEqual(len(composed.body), 2)
        self.assertEqual(ast.unparse(composed.body[0].value.args[0]), "hidden.clone()")
        self.assertEqual(ast.unparse(composed.body[0].value.args[1]), "residual.clone()")
        self.assertEqual(ast.unparse(composed.body[1].body[0]), "destination.copy_(value)")
        self.assertIn("for value in (*terminal_outputs, *consumed):", generated)
        self.assertIn("value.fill_(float('nan'))", generated)


if __name__ == "__main__":
    unittest.main()
