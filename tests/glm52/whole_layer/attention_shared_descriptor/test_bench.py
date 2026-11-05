"""Validate the paired caller against the frozen Torch operator schema."""

import ast
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

import torch

import bench


BUILD = Path(os.environ.get(
    'SHARED_DESCRIPTOR_BUILD',
    '/tmp/tilert-megakernel-validation-20261021/shared-attention-descriptor-build-v1'))
SOURCE = BUILD / 'source/tests/glm52/whole_layer'


class PairedContract(unittest.TestCase):
    def test_actual_schema_and_same_objects_in_both_arms(self):
        source = (SOURCE / 'attention_full_binding.cpp').read_text()
        declaration = re.search(
            r'library\.def\(("shared_full_attention_out.*?);', source, re.S).group(1)
        schema_text = ''.join(ast.literal_eval(value) for value in
                              re.findall(r'"[^"\n]*"', declaration))
        names = [arg.name for arg in torch._C.parse_schema(schema_text).arguments]
        fields = {'x': 'token_x', 'x_scale': 'token_scale', 'cache': 'kv_cache',
                  'rope': 'rope_cache', 'sync': 'sync_base'}
        state_names = ('x', 'x_scale', 'sin', 'cos', 'slots', 'cache', 'rope',
                       'topk', 'query_lengths', 'kv_lengths', 'page_table',
                       'scratch', 'tiling', 'output', 'blocks', 'sync')
        state = SimpleNamespace(**{
            name: torch.zeros(1) for name in state_names})
        bank = {'main': [torch.zeros(1) for _ in bench.MAIN],
                'tail': [torch.zeros(1) for _ in range(3)], 'bias': None}
        expected = {fields.get(name, name): getattr(state, name)
                    for name in state_names}
        expected.update(zip(bench.MAIN, bank['main']))
        expected.update(zip(('value_weight', 'output_weight', 'output_scale'),
                            bank['tail']))
        expected.update(bias=None, scale=1 / 16, epsilon_cq=1e-5, epsilon_ckv=1e-5)
        values = bench.call_arguments(state, bank)
        self.assertEqual(len(names), 32)
        self.assertEqual(len(values), len(names))
        for name, value in zip(names, values):
            if torch.is_tensor(value):
                self.assertIs(value, expected[name], name)
            else:
                self.assertEqual(value, expected[name], name)

        received = {'control': [], 'candidate': []}
        operations = {
            arm: SimpleNamespace(shared_full_attention_out=
                                 lambda *args, arm=arm: received[arm].append(args))
            for arm in received
        }
        arguments = [bench.call_arguments(state, bank) for _ in range(16)]
        calls = bench.bind_calls(arguments, operations)
        for arm in calls:
            for call in calls[arm]:
                call()
        for index in range(16):
            for original, control, candidate in zip(
                    arguments[index], received['control'][index],
                    received['candidate'][index]):
                self.assertIs(original, control)
                self.assertIs(original, candidate)

    def test_empty_replay_and_signed_zero_fail_acceptance(self):
        helper, _ = bench.load_helpers(SOURCE)
        expected = {'output': torch.zeros(4, 16, dtype=torch.bfloat16)}
        for actual in (torch.full_like(expected['output'], float('nan')),
                       -expected['output']):
            with self.assertRaises(RuntimeError):
                bench.compare(torch, helper, {'output': actual}, expected)
        checks = bench.compare(torch, helper, expected, expected)
        self.assertTrue(checks['output']['bitwise_equal'])


if __name__ == '__main__':
    unittest.main()
