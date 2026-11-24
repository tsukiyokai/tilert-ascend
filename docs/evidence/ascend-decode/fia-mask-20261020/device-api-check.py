"""One bounded production-vendor MLA mask API and causality check."""
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import torch

ROOT = Path('/workspace/offline/fia-mask-contract-20261019-codex/device-187-api-check')
INPUT = Path('/workspace/offline/sfa-c1-bounds-20261019-codex/cpu-fixture/input.pt')
VENDOR = '/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer'
ROOT.mkdir(exist_ok=False)
torch.set_num_threads(1)
report = {'scope': 'Single M8 eager API and causal counterfactual, no timing',
          'npu_launched': False, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'original_input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
          'custom_opp_path': os.environ.get('ASCEND_CUSTOM_OPP_PATH'),
          'quota_bytes': 256 * 2**20, 'device': 0}
outputs = {}


def diff(a, b):
    d = a.double() - b.double()
    return {'bit_equal': torch.equal(a, b), 'finite': bool(torch.isfinite(a).all()),
            'max_abs': d.abs().max().item(),
            'nrmse': (d.norm() / b.double().norm()).item()}


try:
    assert '/sfa-c1-' not in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
    assert '/bounds/' not in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
    import torch_npu
    from vllm_ascend.utils import enable_custom_op
    torch_npu.npu.set_device(0)
    total = torch_npu.npu.get_device_properties(0).total_memory
    torch_npu.npu.set_per_process_memory_fraction(256 * 2**20 / total)
    assert enable_custom_op()
    report['schema'] = str(torch.ops.npu.npu_fused_infer_attention_score_v2.default._schema)
    case = torch.load(INPUT, map_location='cpu', weights_only=True)
    original_indices = case['sparse_indices'].clone()
    logical_end = int(case['actual_seq_lengths_kv'][0])
    case['sparse_indices'][:, 0, 0] = logical_end
    case['sparse_indices'][:, 0, 1] = logical_end + 1
    torch.save(case, ROOT / 'input.pt')
    report['input_sha256'] = hashlib.sha256((ROOT / 'input.pt').read_bytes()).hexdigest()
    width = case['query'].shape[0]
    table = case['block_table'][0].long()
    latent_cpu = case['key'][table].reshape(-1, 512).double()
    rope_cpu = case['key_rope'][table].reshape(-1, 64).double()
    thresholds_cpu = (case['actual_seq_lengths_kv'][0].long()
                      - case['actual_seq_lengths_query'][0].long()
                      + torch.arange(width) + 1)
    reference = []
    for row in range(width):
        ids = case['sparse_indices'][row, 0].long()
        ids = ids[(ids >= 0) & (ids < thresholds_cpu[row])]
        k, kr = latent_cpu[ids], rope_cpu[ids]
        scores = (case['query'][row].double() @ k.T
                  + case['query_rope'][row].double() @ kr.T) / 16
        reference.append(scores.softmax(-1) @ k)
    reference = torch.stack(reference)
    outputs['cpu_filtered_fp64'] = reference
    args = {key: value.to('npu:0') for key, value in case.items()}
    row_offsets = torch.arange(width, dtype=torch.int32, device='npu:0') + 1
    torch_npu.npu.synchronize()

    def sfa():
        return torch.ops._C_ascend.npu_sparse_flash_attention(
            **args, value=args['key'], scale_value=1 / 16,
            sparse_block_size=1, layout_query='TND', layout_kv='PA_BSND',
            sparse_mode=3, attention_mode=2, return_softmax_lse=False)[0]

    def gathered_fia(atten_mask):
        ids = args['sparse_indices'][:, 0]
        thresholds = (args['actual_seq_lengths_kv'][0]
                      - args['actual_seq_lengths_query'][0] + row_offsets)
        invalid = (ids < 0) | (ids >= thresholds[:, None])
        safe = ids.masked_fill(invalid, 0).reshape(-1).long()
        pages = safe // 128
        physical = args['block_table'][0].long().index_select(0, pages)
        physical = physical * 128 + safe.remainder(128)
        key = args['key'].view(-1, 512).index_select(0, physical)
        rope = args['key_rope'].view(-1, 64).index_select(0, physical)
        key = key.view(width, 1, 2048, 512)
        return torch_npu.npu_fused_infer_attention_score_v2(
            args['query'].view(width, 4, 1, 512), key, key,
            query_rope=args['query_rope'].view(width, 4, 1, 64),
            key_rope=rope.view(width, 1, 2048, 64),
            atten_mask=atten_mask,
            num_query_heads=4, num_key_value_heads=1,
            softmax_scale=1 / 16, input_layout='BNSD', sparse_mode=0,
            block_table=None, block_size=0,
            actual_seq_qlen=[1] * width, actual_seq_kvlen=[2048] * width,
        )[0].view(width, 4, 512)

    report['comparisons'] = {}
    args['sparse_indices'].copy_(original_indices)
    report['npu_launched'] = True
    report['last_started'] = 'all_causal_fia_without_mask'
    value = gathered_fia(None)
    torch_npu.npu.synchronize()
    outputs['all_causal_fia_without_mask'] = value.cpu().clone()
    assert torch.isfinite(outputs['all_causal_fia_without_mask']).all()
    report['last_started'] = 'all_causal_fia_all_false_mask'
    all_false = torch.zeros((width, 1, 1, 2048), dtype=torch.bool, device='npu:0')
    value = gathered_fia(all_false)
    torch_npu.npu.synchronize()
    outputs['all_causal_fia_all_false_mask'] = value.cpu().clone()
    assert torch.isfinite(outputs['all_causal_fia_all_false_mask']).all()
    report['all_false_vs_none'] = diff(
        outputs['all_causal_fia_all_false_mask'],
        outputs['all_causal_fia_without_mask'])
    args['sparse_indices'].copy_(case['sparse_indices'])
    device_thresholds = (args['actual_seq_lengths_kv'][0]
                         - args['actual_seq_lengths_query'][0] + row_offsets)
    ids = args['sparse_indices'][:, 0]
    invalid = ((ids < 0) | (ids >= device_thresholds[:, None])).view(width, 1, 1, 2048)
    for label, poison in [('a', False), ('b', True), ('reset_a', False)]:
        for name in ['key', 'key_rope']:
            value = case[name].clone()
            if poison:
                for logical in [logical_end, logical_end + 1]:
                    page = int(table[logical // 128])
                    value[page, logical % 128].fill_(1000)
            args[name].copy_(value)
        torch_npu.npu.synchronize()
        report['last_started'] = label + '_sfa'
        report['npu_launched'] = True
        value = sfa()
        torch_npu.npu.synchronize()
        outputs[label + '_sfa'] = value.cpu().clone()
        assert torch.isfinite(outputs[label + '_sfa']).all()
        report['last_started'] = label + '_masked_fia'
        value = gathered_fia(invalid)
        torch_npu.npu.synchronize()
        outputs[label + '_masked_fia'] = value.cpu().clone()
        assert torch.isfinite(outputs[label + '_masked_fia']).all()
        report['comparisons'][label] = {
            'fia_vs_sfa': diff(outputs[label + '_masked_fia'], outputs[label + '_sfa']),
            'fia_vs_filtered_fp64': diff(outputs[label + '_masked_fia'], reference),
            'sfa_vs_filtered_fp64': diff(outputs[label + '_sfa'], reference),
        }
        if label != 'a':
            for name in ['sfa', 'masked_fia']:
                equal = torch.equal(outputs['a_' + name], outputs[label + '_' + name])
                report['comparisons'][label][name + '_future_invariant'] = equal
                assert equal, name + ' changed on invalid future payload'
    report['status'] = 'api_and_future_invariance_completed'
except Exception:
    report['status'] = 'first_error_stopped'
    report['exception'] = traceback.format_exc()
finally:
    torch.save(outputs, ROOT / 'outputs.pt')
    report['outputs_sha256'] = hashlib.sha256((ROOT / 'outputs.pt').read_bytes()).hexdigest()
    mappings = Path('/proc/self/maps').read_text().splitlines()
    libraries = sorted({line.split()[-1] for line in mappings
                        if any(s in line for s in ['opmaster', 'optiling', 'opapi', 'op_plugin'])})
    report['loaded_libraries'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                  for p in libraries if Path(p).is_file()}
    if report['npu_launched']:
        report['peak_allocated_bytes'] = torch_npu.npu.max_memory_allocated()
        report['peak_reserved_bytes'] = torch_npu.npu.max_memory_reserved()
    (ROOT / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
if report['status'] == 'first_error_stopped':
    sys.exit(1)
