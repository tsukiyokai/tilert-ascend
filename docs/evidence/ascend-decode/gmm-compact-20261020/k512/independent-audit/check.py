"""Audit GMM1 K512 integer and buffer contracts on CPU only."""

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path('/workspace/experiments/gmm-compact-20261020/kloop-audit')
SDK = Path('/opt/ascend/cann-9.2.0/aarch64-linux/asc')
OUT = Path('/workspace/offline/gmm1-k512-adversarial-20261020')
K = 6144


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grouped_dot(values, weights, base_k, step):
    accumulator = 0
    covered = []
    init_count = 0
    final_count = 0
    for outer in range(K // (base_k * step)):
        for inner in range(step):
            begin = (outer * step + inner) * base_k
            end = begin + base_k
            init_count += begin == 0
            final_count += end == K
            accumulator += sum(a * b for a, b in
                               zip(values[begin:end], weights[begin:end]))
            assert abs(accumulator) <= 64 * K < 2**31
            covered.extend(range(begin, end))
    assert covered == list(range(K))
    assert init_count == final_count == 1
    return accumulator


def main():
    activations = [[-128] * K, [127] * K,
                   [-128 if i % 2 else 127 for i in range(K)],
                   [(37 * i) % 256 - 128 for i in range(K)], [0] * K]
    for index in (255, 256, 511, 512, 1023, 1024, K - 1):
        activations[-1][index] = -128 if index % 2 else 127
    weights = [[-8] * K, [7] * K,
               [-8 if i % 2 else 7 for i in range(K)],
               [(11 * i) % 16 - 8 for i in range(K)]]
    cases = []
    for a_index, values in enumerate(activations):
        high = [value // 16 for value in values]
        low = [(value & 15) - 8 for value in values]
        assert all(value == 16 * h + l + 8
                   for value, h, l in zip(values, high, low))
        for w_index, weight in enumerate(weights):
            parts = []
            for base_k, step in ((256, 4), (512, 2)):
                h = grouped_dot(high, weight, base_k, step)
                l = grouped_dot(low, weight, base_k, step)
                assert 16 * h + l + 8 * sum(weight) == sum(
                    a * b for a, b in zip(values, weight))
                parts.append((h, l))
            assert parts[0] == parts[1]
            cases.append({'activation': a_index, 'weight': w_index,
                          'high_dot': parts[0][0], 'low_dot': parts[0][1]})
    resources = {}
    for name, n, k, depth in (('N256-K256', 256, 256, 8),
                              ('N128-K256', 128, 256, 8),
                              ('N128-K512', 128, 512, 4)):
        resource = {'l0a_double_bytes': 2 * 128 * k // 2,
                    'l0b_double_bytes': 2 * n * k // 2,
                    'l0c_bytes': 128 * n * 4,
                    'l1a_bytes': depth * 128 * k // 2,
                    'l1b_bytes': depth * n * k // 2}
        assert resource['l0a_double_bytes'] <= 65536
        assert resource['l0b_double_bytes'] <= 65536
        assert resource['l0c_bytes'] <= 131072
        resources[name] = resource
    assert resources['N128-K256']['l1a_bytes'] == resources['N128-K512']['l1a_bytes']
    assert resources['N128-K256']['l1b_bytes'] == resources['N128-K512']['l1b_bytes']
    assert 2 * 256 * 512 // 2 > 65536  # N256/K512/dbB2 is outside the bound.
    api = subprocess.run([str(ROOT / 'tiling_nsplit'), '128', '512', '128'],
                         text=True, check=True, capture_output=True)
    assert 'rc=0\n' in api.stdout and 'baseK=512\n' in api.stdout
    sources = [ROOT / 'paired_nsplit_k512.cpp',
               ROOT / 'nsplit-k512-three-arm.patch', ROOT / 'tiling_nsplit.cpp']
    sources += [SDK / path for path in (
        'include/adv_api/matmul/tiling.h',
        'impl/adv_api/detail/matmul/utils/matmul_utils.h',
        'impl/adv_api/detail/matmul/scheduler/base/scheduler_mdl.h',
        'impl/adv_api/detail/matmul/scheduler/base/scheduler_mdl_common.h',
        'impl/adv_api/detail/matmul/scheduler/base/scheduler_mdl_base.h',
        'impl/adv_api/detail/matmul/scheduler/base/scheduler_base.h',
        'impl/adv_api/detail/matmul/scheduler/iterator/k_loop/k_loop_mdl_base.h',
        'impl/adv_api/detail/matmul/resource/l0_buffer/tbuf_pool_l0_base.h')]
    result = {
        'scope': 'CPU integer/address/resource model and actual CPU tiling API; '
                 'no Fixpipe numerical simulation or NPU execution',
        'npu_launched': False, 'arithmetic_cases': cases, 'all_passed': True,
        'int32_absolute_bound': 64 * K, 'resources': resources,
        'actual_cpu_api_stdout': api.stdout,
        'manual_candidate': {'baseM': 128, 'baseN': 128, 'baseK': 512,
                             'stepKa': 2, 'stepKb': 2, 'depthA1': 4,
                             'depthB1': 4, 'iterateOrder': 1,
                             'dbL0A': 2, 'dbL0B': 2},
        'cpu_api_does_not_validate_manual_step_depth_order': True,
        'source_sha256': {str(path): sha(path) for path in sources},
        'script_sha256': sha(Path(__file__)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'all_passed': True, 'arithmetic_cases': len(cases),
                      'int32_absolute_bound': 64 * K,
                      'resources': resources, 'source_sha256':
                      result['source_sha256'][str(sources[0])]}, indent=2))


if __name__ == '__main__':
    main()
