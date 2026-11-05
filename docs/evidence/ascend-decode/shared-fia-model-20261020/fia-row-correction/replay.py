"""Replay saved local-attention Q/KV with only FIA inner_precise changed."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch

ROOT = Path('/workspace/experiments/m32-full-teacher-20261020/fia-row-correction')
OLD = Path('/workspace/experiments/layer-major-local-attention-20261020')
SOURCE = ROOT.parent / 'source'
PREFIX = Path('/workspace/offline/prefix8k-allocation-20261017/capture/case-00/main-prefix/rank-0.pt')
LAYERS = range(66, 74)
W, C, P = 32, 9344, 8192


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


legacy = module('saved_local_probe', OLD / 'probe.py')
helper = module('current_attention', SOURCE / 'tilert/models/glm_5_2_ascend/cann_attention.py')


def bits(value):
    value = value.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def metric(actual, reference):
    result = legacy.difference(actual, reference)
    a, b = actual.double(), reference.double()
    result['different_values'] = int((a != b).sum())
    result['different_bytes'] = int((actual.contiguous().view(torch.uint8)
                                     != reference.contiguous().view(torch.uint8)).sum())
    norm = float(a.norm() * b.norm())
    result['cosine'] = float((a * b).sum()) / norm if norm else None
    changed = (a != b).nonzero()
    if len(changed):
        index = tuple(changed[0].tolist())
        result['first_difference'] = {'index': list(index),
                                     'actual': float(a[index]),
                                     'reference': float(b[index])}
    return result


def fixture(prefix, layer):
    path = OLD / 'device-01' / f'layer-{layer:02d}.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    values = {'q': saved['q'].contiguous(), 'qr': saved['qr'].contiguous(),
              'ids': saved['indices'].reshape(W, 1, 2048).contiguous()}
    for name, width in (('latent', 512), ('rope', 64)):
        value = torch.zeros((C, width), dtype=torch.bfloat16)
        value[:P].copy_(prefix['layers'][layer][name].reshape(P, width))
        value[P:P + W].copy_(saved['new_cache_rows'][name].reshape(W, width))
        values[name] = value
    causal = torch.arange(C)[None] > torch.arange(P, P + W)[:, None]
    values['mask'] = helper.selection_mask(values['ids'], causal)
    assert values['q'].shape == (W, 4, 512)
    assert values['qr'].shape == (W, 4, 64)
    assert values['mask'].dtype == torch.bool
    assert torch.equal((~values['mask']).sum(1), torch.full((W,), 2048))
    ids = values['ids'].reshape(W, 2048)
    assert bool((ids.sort(1).values.diff() > 0).all())
    assert bool(((ids >= 0) & (ids <= torch.arange(P, P + W)[:, None])).all())
    for name in ('q', 'qr', 'latent', 'rope'):
        assert values[name].dtype == torch.bfloat16
        assert bool(torch.isfinite(values[name]).all())
    return values, saved, path


def prepare():
    original = json.loads((OLD / 'device-01/result.json').read_text())
    assert legacy.digest(PREFIX) == original['prefix_sha256']
    prefix = torch.load(PREFIX, map_location='cpu', weights_only=True, mmap=True)
    report = {'scope': 'Saved rank0 local-attention Q derived from frozen M1 inputs at layers66..73',
              'prefix_sha256': original['prefix_sha256'],
              'original_report_sha256': original['original_report_sha256'],
              'script_sha256': legacy.digest(Path(__file__)),
              'source_probe_sha256': legacy.digest(OLD / 'probe.py'),
              'helper_sha256': legacy.digest(SOURCE / 'tilert/models/glm_5_2_ascend/cann_attention.py'),
              'layers': []}
    for layer in LAYERS:
        values, saved, path = fixture(prefix, layer)
        report['layers'].append({'layer': layer, 'path': str(path),
                                 'saved_sha256': legacy.digest(path),
                                 'inputs': {name: bits(value) for name, value in values.items()}})
    report['torch_npu_imported'] = 'torch_npu' in sys.modules
    assert not report['torch_npu_imported']
    report['threads'] = torch.get_num_threads()
    report['interop_threads'] = torch.get_num_interop_threads()
    (ROOT / 'cpu-ready.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'cpu_ready': True, 'layers': len(report['layers']),
                      'torch_npu_imported': report['torch_npu_imported']}))


def device(output, die):
    frozen = json.loads((ROOT / 'cpu-ready.json').read_text())
    assert die == 15
    assert frozen['script_sha256'] == legacy.digest(Path(__file__))
    for key in ('ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES'):
        if os.environ.get(key):
            assert os.environ[key].split(',') == [str(i) for i in range(16)]
    import torch_npu
    from vllm_ascend.utils import enable_custom_op
    assert enable_custom_op()
    torch.npu.set_device(die)
    total = torch.npu.get_device_properties(die).total_memory
    torch.npu.set_per_process_memory_fraction(512 * 2**20 / total, die)
    output.mkdir(exist_ok=False)
    prefix = torch.load(PREFIX, map_location='cpu', weights_only=True, mmap=True)
    result = {'scope': frozen['scope'], 'die': die, 'quota_bytes': 512 * 2**20,
              'only_fia_attribute_changed': 'inner_precise 0 versus 2',
              'timing_measured': False, 'layers': [],
              'script_sha256': legacy.digest(Path(__file__)),
              'cpu_ready_sha256': legacy.digest(ROOT / 'cpu-ready.json'),
              'fia_schema': str(torch.ops.npu.npu_fused_infer_attention_score_v2.default._schema)}
    table = torch.arange(C // 128, dtype=torch.int32, device=f'npu:{die}')[None]
    qlens = torch.tensor([W], dtype=torch.int32, device=f'npu:{die}')
    klens = torch.tensor([P + W], dtype=torch.int32, device=f'npu:{die}')
    for expected in frozen['layers']:
        layer = expected['layer']
        cpu, saved, path = fixture(prefix, layer)
        assert legacy.digest(path) == expected['saved_sha256']
        assert {name: bits(value) for name, value in cpu.items()} == expected['inputs']
        values = {name: value.to(f'npu:{die}') for name, value in cpu.items()}
        q, qr = values['q'], values['qr']
        latent, rope = values['latent'], values['rope']
        results = {}
        with torch.inference_mode():
            results['sfa'] = torch.ops._C_ascend.npu_sparse_flash_attention(
                query=q, key=latent.view(-1, 128, 1, 512),
                value=latent.view(-1, 128, 1, 512),
                sparse_indices=values['ids'], scale_value=1 / 16,
                block_table=table, actual_seq_lengths_query=qlens,
                actual_seq_lengths_kv=klens, query_rope=qr,
                key_rope=rope.view(-1, 128, 1, 64), sparse_block_size=1,
                layout_query='TND', layout_kv='PA_BSND', sparse_mode=3,
                attention_mode=2, return_softmax_lse=False)[0].cpu()
            for precision in (0, 2):
                result_fia, _ = torch_npu.npu_fused_infer_attention_score_v2(
                    q.view(1, W, 4, 512), latent.view(1, C, 1, 512),
                    latent.view(1, C, 1, 512),
                    query_rope=qr.view(1, W, 4, 64),
                    key_rope=rope.view(1, C, 1, 64),
                    atten_mask=values['mask'].view(1, 1, W, C),
                    num_query_heads=4, num_key_value_heads=1,
                    softmax_scale=1 / 16, input_layout='BSND', sparse_mode=0,
                    block_table=None, block_size=0, actual_seq_qlen=[W],
                    actual_seq_kvlen=[C], inner_precise=precision)
                results[f'fia{precision}'] = result_fia.view(W, 4, 512).cpu()
                assert {name: bits(value) for name, value in values.items()} == expected['inputs']
        for value in results.values():
            assert value.shape == (W, 4, 512) and value.dtype == torch.bfloat16
            assert bool(torch.isfinite(value).all())
        comparisons = {}
        for a, b in (('fia0', 'sfa'), ('fia2', 'sfa'), ('fia2', 'fia0')):
            comparisons[f'{a}_vs_{b}'] = {
                'all': metric(results[a], results[b]),
                'per_query': [metric(results[a][i], results[b][i]) for i in range(W)]}
        comparisons['saved_sfa'] = metric(results['sfa'], saved['attention']['sfa'])
        comparisons['saved_fia0'] = metric(results['fia0'], saved['attention']['shared_fia'])
        torch.save(results, output / f'layer-{layer:02d}.pt')
        controls = {name: comparisons[name]['bitexact']
                    for name in ('saved_sfa', 'saved_fia0')}
        result['layers'].append({'layer': layer, 'inputs_unchanged': True,
                                 'baseline_controls': controls,
                                 'comparisons': comparisons})
        print(json.dumps({'layer': layer, **{k: v['all'] for k, v in comparisons.items() if 'all' in v}}), flush=True)
        del values, q, qr, latent, rope, result_fia
        gc.collect()
        torch.npu.empty_cache()
    result['peak_allocated'] = torch.npu.max_memory_allocated(die)
    result['peak_reserved'] = torch.npu.max_memory_reserved(die)
    result['process_group_initialized'] = torch.distributed.is_initialized()
    assert not result['process_group_initialized']
    result['baseline_controls_passed'] = all(
        all(layer['baseline_controls'].values()) for layer in result['layers'])
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    assert result['baseline_controls_passed'], 'Saved SFA/FIA0 baseline drift; flag attribution blocked'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'device'))
    parser.add_argument('--die', type=int, default=15)
    parser.add_argument('--output', type=Path, default=ROOT / 'device-01')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.mode == 'prepare':
        prepare()
    else:
        device(args.output, args.die)
