"""Load the actual paired libraries and run caller contracts without an NPU."""

import argparse
import json
import os
from pathlib import Path
import unittest

import torch
import torch_npu

import bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--control-library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate = args.build / 'shared_attention_descriptor.so'
    assert bench.digest(candidate) == bench.CANDIDATE
    assert bench.digest(args.control_library) == bench.CONTROL
    assert not torch.npu.is_initialized()
    torch.ops.load_library(str(args.control_library))
    torch.ops.load_library(str(candidate))
    schemas = {}
    for name in ('shared_full_attention_prepare', 'shared_full_attention_out'):
        control = str(getattr(torch.ops.tilert, name).default._schema)
        candidate_schema = str(getattr(torch.ops.tilert_shared_descriptor, name)
                               .default._schema)
        assert candidate_schema.replace('tilert_shared_descriptor::', 'tilert::') == control
        schemas[name] = {'control': control, 'candidate': candidate_schema}
    os.environ['SHARED_DESCRIPTOR_BUILD'] = str(args.build)
    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).parent), pattern='test_bench.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    assert not torch.npu.is_initialized()
    report = {
        'pass': True, 'cpu_tests': result.testsRun,
        'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
        'npu_initialized': False, 'schemas': schemas,
        'library_sha256': {'control': bench.CONTROL, 'candidate': bench.CANDIDATE},
        'bench_sha256': bench.digest(Path(bench.__file__)),
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
