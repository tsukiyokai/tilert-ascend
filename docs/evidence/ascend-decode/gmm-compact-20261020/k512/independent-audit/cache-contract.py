"""Mirror the B114 MDL double-buffer slot formulas on CPU."""

import hashlib
import json
from pathlib import Path


root = Path('/opt/ascend/cann-9.2.0/aarch64-linux/asc/impl/adv_api/detail/matmul')
cases = []
for base_k, step, depth in ((256, 4, 8), (512, 2, 4)):
    step_size = step * 1  # GetTotalCacheNum: stepKa*stepM or stepKb*stepN.
    factor = (depth // step_size - 1) & 1
    assert depth % step_size == 0 and depth // step_size == 2
    assert factor == 1
    matrix_bytes = 128 * base_k * 4 // 8
    bank_bytes = matrix_bytes * step_size
    assert bank_bytes == 65536
    blocks = 6144 // base_k
    assert blocks % step == 0
    outer_count = 6144 // (base_k * step)
    assert outer_count == 6
    assert blocks > step and step * base_k == 1024
    intervals = []
    states = []
    cached = [False, False]
    for outer in range(outer_count):
        inner_start = outer * step
        buffer_pos = inner_start // step
        slot = buffer_pos & factor
        assert buffer_pos == outer and not cached[slot]
        cached[slot] = True
        offsets = [inner * base_k for inner in range(step)]
        assert max(offsets) + base_k == 1024
        for offset in offsets:
            intervals.extend(range(outer * 1024 + offset,
                                   outer * 1024 + offset + base_k))
        # ClearL1BufferCache -> ClearLoadData -> FreeTensor at each outer edge.
        cached[slot] = False
        states.append({'outer': outer, 'slot': slot, 'inner_offsets': offsets})
    assert intervals == list(range(6144)) and cached == [False, False]
    cases.append({'baseK': base_k, 'step': step, 'depth': depth,
                  'cacheFactor': factor, 'queueDepth': 2,
                  'bankBytes': bank_bytes, 'states': states})
paths = [root / path for path in (
    'resource/cube_in_buffer/cube_in_buffer_double_buffer.h',
    'stage/copy_cube_in/base/copy_cube_in_base.h',
    'stage/copy_cube_in/base/copy_cube_in_params_base.h',
    'stage/copy_cube_in/base/copy_cube_in_mdl_base.h',
    'scheduler/iterator/k_loop/k_loop_mdl_base.h',
    'scheduler/base/scheduler_mdl_base.h')]
result = {'scope': 'CPU scalar model of source slot formulas; no device validation',
          'all_passed': True, 'npu_launched': False, 'cases': cases,
          'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in paths}}
out = Path('/workspace/offline/gmm1-k512-adversarial-20261020/cache-contract.json')
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'all_passed': True, 'path': str(out), 'cases': len(cases)}))
