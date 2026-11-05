"""Check reference mapping and the byte-exact acceptance gate on CPU."""

import ast
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

from prepare_check import LAYER, prepare, transform


BUILD = Path(os.environ.get('SHARED_DESCRIPTOR_BUILD',
    '/tmp/tilert-megakernel-validation-20261021/shared-attention-descriptor-build-v1'))


class CheckerContract(unittest.TestCase):
    def test_fixture_updates_are_preserved(self):
        source = (BUILD / 'source' / LAYER / 'attention_full_check.py').read_text()
        original = ast.parse(source)
        candidate = ast.parse(transform(source))
        for name in ('update', 'tail_fixture'):
            old = next(node for node in ast.walk(original)
                       if isinstance(node, ast.FunctionDef) and node.name == name)
            new = next(node for node in ast.walk(candidate)
                       if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(old), ast.dump(new))

    def test_both_full_calls_share_the_parameter_contract(self):
        source = (BUILD / 'source' / LAYER / 'attention_full_check.py').read_text()
        tree = ast.parse(transform(source))
        calls = []
        for name in ('full', 'separate'):
            function = next(node for node in ast.walk(tree)
                            if isinstance(node, ast.FunctionDef) and node.name == name)
            call = function.body[0].value
            self.assertIsInstance(call, ast.Call)
            self.assertEqual(call.func.attr, 'shared_full_attention_out')
            args = [ast.unparse(value) for value in call.args]
            calls.append([value.replace('full_', 'arm_').replace('separate_', 'arm_')
                          for value in args])
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(len(calls[0]) + 8, 32)  # Nine core weights share one starred argument.

    def test_empty_replay_and_signed_zero_fail_byte_gate(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'checker'
            prepare(BUILD, output)
            spec = importlib.util.spec_from_file_location('descriptor_metric', output / 'attention_check.py')
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            expected = torch.zeros(4, 16, dtype=torch.bfloat16)
            empty_replay = torch.full_like(expected, float('nan'))
            self.assertFalse(helper.metric(torch, empty_replay, expected)['pass'])
            check = helper.metric(torch, -expected, expected)
            self.assertTrue(check['pass'])
            self.assertFalse(check['bitwise_equal'])
            generated = (output / 'check.py').read_text()
            self.assertIn('value["pass"] and value["bitwise_equal"]', generated)


if __name__ == '__main__':
    unittest.main()
