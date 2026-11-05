"""Read actual A3 resources and execute the native Carver hint path."""
import argparse
import hashlib
import json
import logging
logging.basicConfig(level=logging.DEBUG)
from pathlib import Path

import torch
import torch_npu
from tilelang import carver
from tilelang.carver.arch.ascend import Ascend
from tilelang.carver.roller.policy.ascend import AscendDefaultPolicy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.npu.set_device(0)

for name in ('tilelang.carver.utils', 'tilelang.carver.roller.policy.ascend', 'tilelang.carver.roller.policy.default'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
arch = Ascend()
policy = AscendDefaultPolicy(arch)
report = {
    'device': str(torch.npu.get_device_properties(0)),
    'chip_name': arch.chip_name,
    'capacities': {name: getattr(arch, name) for name in (
        'l0a_cap', 'l0b_cap', 'l0c_cap', 'l1_cap', 'ub_cap',
        'compute_max_core', 'l2_cache_size_bytes')},
    'sdk': getattr(arch, 'sdk_resources', None),
    'tile_256_256_64_fp32_allowed': policy._check_l0_usage(
        256 * 64 * 2, 64 * 256 * 2, 256 * 256 * 4, 2),
    'tile_128_256_64_fp32_allowed': policy._check_l0_usage(
        128 * 64 * 2, 64 * 256 * 2, 128 * 256 * 4, 2),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + '\n')
template = carver.MatmulTemplate(
    M=1024, N=1024, K=1024, trans_B=False,
    in_dtype='float16', accum_dtype='float32', out_dtype='float16',
).with_arch(arch)
hints = template.recommend_hints(topk=20)
report['hints'] = [
    {'block': list(hint.block), 'rstep': list(hint.rstep)} for hint in hints
]
report['hint_path_executed'] = True
report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
