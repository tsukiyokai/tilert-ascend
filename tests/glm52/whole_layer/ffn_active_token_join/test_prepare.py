"""Check the actual frozen arithmetic, ownership and descriptor boundaries."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from prepare import BODY, BODY_MARKER, STAGE_FIVE, prepare, rename


ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
PARENT = Path(os.environ.get('FFN_COMBINED_PARENT',
    str(ROOT / 'ffn-active-descriptor-runtime-v1/build-v1')))
JOIN = Path(os.environ.get('FFN_COMBINED_JOIN',
    str(ROOT / 'ffn-token-join-build-v1')))


class CompositionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.output = Path(cls.directory.name) / 'candidate'
        cls.report = prepare(PARENT, JOIN, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_active_projection_prefix_is_unchanged(self):
        def prefix(source):
            start = source.index('\n{\n', source.index(BODY_MARKER)) + 3
            return source[start:source.index(STAGE_FIVE) + len(STAGE_FIVE)]

        self.assertEqual(prefix((self.output / BODY).read_text()),
                         rename(prefix((PARENT / BODY).read_text())))

    def test_token_arithmetic_and_public_ownership_are_preserved(self):
        self.assertEqual((self.output / 'source/output_native.h').read_text(),
                         rename((JOIN / 'source/output_native.h').read_text()))
        for name in ('source/descriptor.h', 'source/contract.h', 'source/binding.cpp',
                     'body/source/routed_int4/routed_int4_front.h',
                     'body/source/routed_int4/routed_int4_dispatch.h',
                     'body/source/routed_int4/routed_int4_active_list.h'):
            self.assertEqual((self.output / name).read_text(),
                             rename((PARENT / name).read_text()), name)
        actual = (self.output / 'source/kernel.cpp').read_text().split('extern "C"', 1)[1]
        original = (PARENT / 'source/kernel.cpp').read_text().split('extern "C"', 1)[1]
        self.assertEqual(actual, rename(original))
        self.assertEqual((self.output / 'check.py').read_text(),
            (PARENT / 'check.py').read_text().replace(
                'torch.ops.tilert_ffn_active_descriptor',
                'torch.ops.tilert_ffn_active_token_join'))

    def test_final_publication_and_profile_guard(self):
        body = (self.output / BODY).read_text()
        kernel = (self.output / 'source/kernel.cpp').read_text()
        self.assertNotIn('timing.Barrier(5);', body)
        self.assertEqual(body.count('timing.Barrier('), 4)
        self.assertIn('#ifdef ROUTED_ENABLE_PROFILE\n#error', body)
        self.assertEqual(kernel.count('ffn_bridge::device::EndPhase(pipe);'), 1)
        self.assertNotIn('ffn_bridge::device::Epilogue(', kernel)
        self.assertEqual(kernel.count('SetSysWorkspace('), 2)
        self.assertEqual(self.report['public_scratch_count'], 33)

    def test_real_descriptor_addresses_and_combined_ub_budget(self):
        probe = rename((PARENT / 'source/descriptor_probe.cpp').read_text())
        probe = probe.replace('#include <descriptor.h>',
            '#include <descriptor.h>\n#include <output_contract.h>\n'
            'static_assert(ffn_active_token_join::kOutputResidentBytes == 116480);\n'
            'static_assert(ffn_active_token_join::kOutputResidentBytes < 196608);')
        path = Path(self.directory.name) / 'probe.cpp'
        binary = path.with_suffix('')
        path.write_text(probe)
        subprocess.run([
            'g++', '-std=c++17', '-O2', '-DROUTED_ROW_BALANCED=1',
            '-DROUTED_SHARED_PIPELINE=1', '-DROUTED_INPUT_FRONT=1',
            '-DROUTED_ACTIVE_LIST=1', '-DSHARED_W8_GROUPED_QUANT=1',
            '-I', str(self.output / 'source'), str(path), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
