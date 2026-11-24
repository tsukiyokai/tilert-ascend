"""One production-vendor BSND joint-M MLA/full-bitmap API check."""
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import torch

ROOT = Path('/workspace/offline/fia-joint-m-contract-20261020-codex/device-187-api')
ROOT.mkdir(parents=True, exist_ok=False)
INPUT = Path('/workspace/offline/sfa-c1-bounds-20261019-codex/cpu-fixture/input.pt')
SFA = Path('/workspace/offline/sfa-c1-bounds-20261019-codex/correctness-v1/outputs.pt')
torch.set_num_threads(1)
report = {'scope': 'Exactly one eager BSND joint-M MLA API call, no timing',
          'npu_launched': False, 'quota_bytes': 256 * 2**20, 'device': 0,
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'original_input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
          'saved_sfa_outputs_sha256': hashlib.sha256(SFA.read_bytes()).hexdigest()}
outputs = {}


def diff(value, reference):
    delta = value.double() - reference.double()
    return {'bit_equal': torch.equal(value, reference),
            'finite': bool(torch.isfinite(value).all()),
            'max_abs': delta.abs().max().item(),
            'nrmse': (delta.norm() / reference.double().norm()).item()}


try:
    case = torch.load(INPUT, map_location='cpu', weights_only=True)
    query, query_rope = case['query'], case['query_rope']
    indices = case['sparse_indices'][:, 0].long()
    width, heads, _ = query.shape
    capacity = 9344
    written = int(case['actual_seq_lengths_kv'][0])
    table = case['block_table'][0].long()
    latent = torch.zeros(capacity, 512, dtype=torch.bfloat16)
    rope = torch.zeros(capacity, 64, dtype=torch.bfloat16)
    latent[:written].copy_(case['key'][table].reshape(-1, 512)[:written])
    rope[:written].copy_(case['key_rope'][table].reshape(-1, 64)[:written])
    assert torch.isfinite(latent).all() and torch.isfinite(rope).all()
    reference = []
    for row in range(width):
        assert indices[row].unique().numel() == 2048
        assert bool(((indices[row] >= 0) & (indices[row] < written - width + row + 1)).all())
        key, key_rope = latent[indices[row]].double(), rope[indices[row]].double()
        scores = (query[row].double() @ key.T + query_rope[row].double() @ key_rope.T) / 16
        reference.append(scores.softmax(-1) @ key)
    outputs['cpu_filtered_fp64'] = torch.stack(reference)
    outputs['saved_sfa'] = torch.load(SFA, map_location='cpu', weights_only=True)['a_eager_1']
    payload = {'query': query, 'query_rope': query_rope,
               'latent': latent, 'rope': rope, 'indices': indices}
    torch.save(payload, ROOT / 'input.pt')
    report['input_sha256'] = hashlib.sha256((ROOT / 'input.pt').read_bytes()).hexdigest()
    report['shape'] = {'M': width, 'heads': heads, 'capacity': capacity,
                       'query': [1, width, heads, 512],
                       'key': [1, capacity, 1, 512],
                       'mask': [1, 1, width, capacity]}
    report['custom_opp_path'] = os.environ.get('ASCEND_CUSTOM_OPP_PATH')
    assert '/sfa-c1-' not in report['custom_opp_path']
    assert '/bounds/' not in report['custom_opp_path']
    import torch_npu
    torch_npu.npu.set_device(0)
    total = torch_npu.npu.get_device_properties(0).total_memory
    torch_npu.npu.set_per_process_memory_fraction(256 * 2**20 / total)
    report['schema'] = str(torch.ops.npu.npu_fused_infer_attention_score_v2.default._schema)
    q, qr = query.to('npu:0'), query_rope.to('npu:0')
    key, key_rope = latent.to('npu:0'), rope.to('npu:0')
    selected = indices.to('npu:0')
    mask = torch.ones((width, capacity), dtype=torch.bool, device='npu:0')
    mask.scatter_(1, selected, False)
    torch_npu.npu.synchronize()
    report['npu_launched'] = True
    report['last_started'] = 'joint_BSND_fia'
    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        q.view(1, width, heads, 512), key.view(1, capacity, 1, 512),
        key.view(1, capacity, 1, 512),
        query_rope=qr.view(1, width, heads, 64),
        key_rope=key_rope.view(1, capacity, 1, 64),
        atten_mask=mask.view(1, 1, width, capacity),
        num_query_heads=heads, num_key_value_heads=1,
        softmax_scale=1 / 16, input_layout='BSND', sparse_mode=0,
        block_table=None, block_size=0,
        actual_seq_qlen=[width], actual_seq_kvlen=[capacity],
    )
    torch_npu.npu.synchronize()
    outputs['joint_fia'] = output.view(width, heads, 512).cpu().clone()
    assert torch.isfinite(outputs['joint_fia']).all()
    report['comparisons'] = {
        'joint_vs_saved_sfa': diff(outputs['joint_fia'], outputs['saved_sfa']),
        'joint_vs_filtered_fp64': diff(outputs['joint_fia'], outputs['cpu_filtered_fp64']),
    }
    report['status'] = 'one_joint_M_api_call_completed'
except Exception:
    report['status'] = 'first_error_stopped'
    report['exception'] = traceback.format_exc()
finally:
    torch.save(outputs, ROOT / 'outputs.pt')
    report['outputs_sha256'] = hashlib.sha256((ROOT / 'outputs.pt').read_bytes()).hexdigest()
    maps = Path('/proc/self/maps').read_text().splitlines()
    libraries = sorted({line.split()[-1] for line in maps
                        if any(x in line for x in ['optiling', 'opmaster', 'opapi'])})
    report['loaded_libraries'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                  for p in libraries if Path(p).is_file()}
    if report['npu_launched']:
        report['peak_allocated_bytes'] = torch_npu.npu.max_memory_allocated()
        report['peak_reserved_bytes'] = torch_npu.npu.max_memory_reserved()
    (ROOT / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
if report['status'] == 'first_error_stopped':
    sys.exit(1)
