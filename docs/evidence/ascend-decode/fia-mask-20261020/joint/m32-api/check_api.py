"""Two-call M32 API check: production SFA and the actual shared-FIA binder."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from types import SimpleNamespace

import torch


ROOT = Path('/workspace/offline/fia-m32-api-20261020-codex')
CONTRACT = Path('/workspace/offline/fia-m32-contract-20261020-codex')
QUOTA = 256 * 2**20
WIDTH, CAPACITY, DEVICE = 32, 9344, 2
ROOT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
outputs = {}
report = {'scope': 'Two eager attention calls, M32 layer0/stateA; no timing or graph',
          'utc_started': datetime.now(timezone.utc).isoformat(),
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'quota_bytes': QUOTA, 'device': DEVICE, 'attention_calls_attempted': 0,
          'npu_launched': False, 'checks': []}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def difference(actual, reference):
    delta = actual.double() - reference.double()
    return {'bit_equal': torch.equal(actual, reference),
            'max_abs': float(delta.abs().max()),
            'nrmse': float(delta.norm() / reference.double().norm())}


def unexpected_fallback(*args, **kwargs):
    raise AssertionError('Matching M32 helper width must not fall back')


try:
    pre = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    (ROOT / 'npu-smi-pre.txt').write_text(pre)
    match = re.search(r'^\|\s*0\s+2\s*\|\s*\S+\s*\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|', pre, re.M)
    assert match, 'Physical die2 resource row must be identifiable'
    report['preflight'] = {'aicore_percent': int(match[1]), 'hbm_mib': int(match[2]),
                           'no_process_npu1': 'No running processes found in NPU 1 ' in pre}
    assert report['preflight']['aicore_percent'] == 0
    assert report['preflight']['hbm_mib'] < 8192
    assert report['preflight']['no_process_npu1']
    report['fixture_sha256'] = sha(CONTRACT / 'fixture.pt')
    report['helper_sha256'] = sha(CONTRACT / 'cann_attention.py')
    assert report['fixture_sha256'] == '8aa188e2d5b12d2a3daa12dcba45aec7934e31fd2f0098dd3ed3d2ddb0250a2a'
    assert report['helper_sha256'] == '47503e3cd1b730cfcc3b21f27f8d1709f2bcf003cb72d3537d383d65df3417e6'
    (ROOT / 'cann_attention.py').write_bytes((CONTRACT / 'cann_attention.py').read_bytes())
    spec = importlib.util.spec_from_file_location('actual_cann_attention', ROOT / 'cann_attention.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    case = torch.load(CONTRACT / 'fixture.pt', map_location='cpu', weights_only=True)
    layer = case['layers'][0]
    assert all(bool(torch.isfinite(value).all()) for value in [layer['q'], layer['qr'], *layer['kv']])
    assert all(int(value[8224:].count_nonzero()) == 0 for value in layer['kv'])
    assert bool((case['ids_A'] < case['thresholds_A'].view(WIDTH, 1)).all())
    row = WIDTH - 1
    selected = case['ids_A'][row].long()
    key = layer['kv'][0][selected].double()
    key_rope = layer['kv'][1][selected].double()
    scores = (layer['q'][row].double() @ key.T + layer['qr'][row].double() @ key_rope.T) / 16
    fp64 = scores.softmax(-1) @ key
    outputs['fp64_query31_selected'] = fp64
    report['custom_opp_path'] = os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
    assert '/sfa-c1-' not in report['custom_opp_path'] and '/bounds/' not in report['custom_opp_path']
    report['visible_devices'] = {name: os.environ.get(name) for name in
                                 ['ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES']}
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    if visible:
        assert visible.split(',')[DEVICE].strip() == str(DEVICE), 'Logical die2 mapping changed'
    import torch_npu
    from vllm_ascend.utils import enable_custom_op

    torch_npu.npu.set_device(DEVICE)
    total = torch_npu.npu.get_device_properties(DEVICE).total_memory
    torch_npu.npu.set_per_process_memory_fraction(QUOTA / total)
    assert enable_custom_op()
    report['schemas'] = {'sfa': str(torch.ops._C_ascend.npu_sparse_flash_attention.default._schema),
                         'fia': str(torch.ops.npu.npu_fused_infer_attention_score_v2.default._schema)}
    (ROOT / 'schema.json').write_text(json.dumps(report['schemas'], indent=2) + '\n')
    device = torch.device('npu', DEVICE)
    q = layer['q'].to(device)
    qr = layer['qr'].to(device)
    kv = tuple(value.to(device) for value in layer['kv'])
    indices = case['ids_A'].to(device).view(WIDTH, 1, 2048)
    slots = (case['thresholds_A'] - 1).to(device)
    query_lengths = torch.tensor([WIDTH], dtype=torch.int32, device=device)
    key_lengths = torch.tensor([8224], dtype=torch.int32, device=device)
    table = torch.arange(CAPACITY // 128, dtype=torch.int32, device=device).view(1, -1)
    binder = helper.SharedKVAttention(WIDTH, CAPACITY, device)
    impl = SimpleNamespace(skip_topk=False, scale=1 / 16,
                           _execute_sparse_flash_attention_process=unexpected_fallback)
    binder.bind(impl)
    with torch.inference_mode():
        report['npu_launched'] = True
        report['attention_calls_attempted'] += 1
        report['last_started'] = 'production_sfa'
        result = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=q, key=kv[0].view(-1, 128, 1, 512), value=kv[0].view(-1, 128, 1, 512),
            query_rope=qr, key_rope=kv[1].view(-1, 128, 1, 64), sparse_indices=indices,
            block_table=table, actual_seq_lengths_query=query_lengths,
            actual_seq_lengths_kv=key_lengths, scale_value=1 / 16, sparse_block_size=1,
            layout_query='TND', layout_kv='PA_BSND', sparse_mode=3,
            attention_mode=2, return_softmax_lse=False)[0]
        outputs['sfa'] = result.cpu().clone()
        assert bool(torch.isfinite(outputs['sfa']).all()), 'SFA output nonfinite'
        report['checks'].append({'arm': 'sfa', 'finite': True})
        report['attention_calls_attempted'] += 1
        report['last_started'] = 'actual_binder_fia'
        result = impl._execute_sparse_flash_attention_process(
            q, qr, kv, indices, SimpleNamespace(slot_mapping=slots),
            query_lengths, key_lengths, block_table=table)
        outputs['shared_fia'] = result.cpu().clone()
        assert bool(torch.isfinite(outputs['shared_fia']).all()), 'Shared FIA output nonfinite'
        report['checks'].append({'arm': 'actual_binder_fia', 'finite': True})
        independent_rows = []
        for row_ids, threshold in zip(case['ids_A'], case['thresholds_A']):
            selected_set = set(row_ids.tolist())
            independent_rows.append([token not in selected_set or token >= int(threshold)
                                     for token in range(CAPACITY)])
        independent_mask = torch.tensor(independent_rows, dtype=torch.bool)
        assert torch.equal(binder.mask.cpu(), independent_mask)
        report['actual_binder_mask_matches_cpu_contract'] = True
        report['cross_kernel_diagnostic'] = difference(outputs['shared_fia'], outputs['sfa'])
        report['query31_fp64_diagnostics'] = {arm: difference(value[row], fp64)
                                             for arm, value in outputs.items() if arm != 'fp64_query31_selected'}
        assert report['attention_calls_attempted'] == 2
        report['status'] = 'two_call_api_finite_passed'
except Exception:
    report['status'] = 'first_error_stopped'
    report['exception'] = traceback.format_exc()
finally:
    torch.save(outputs, ROOT / 'outputs.pt')
    report['outputs_sha256'] = sha(ROOT / 'outputs.pt')
    if report['npu_launched']:
        report['peak_allocated_bytes'] = torch_npu.npu.max_memory_allocated(DEVICE)
        report['peak_reserved_bytes'] = torch_npu.npu.max_memory_reserved(DEVICE)
    paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if any(part in line for part in ['optiling', 'opmaster', 'opapi'])})
    report['loaded_libraries'] = {path: sha(Path(path)) for path in paths if Path(path).is_file()}
    report['utc_finished'] = datetime.now(timezone.utc).isoformat()
    (ROOT / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
if report['status'] == 'first_error_stopped':
    sys.exit(1)
