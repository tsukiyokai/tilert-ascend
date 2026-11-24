"""Teacher routes reconstructed from frozen M1 residual with current M1 kernels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
from weights import ModelSlimCheckpoint
from estimate_traffic import components

OLD = Path('/workspace/experiments/layer-major-m1-m32-20261020/old/device-78x32-sfa-01')
EXTRA = Path('/workspace/experiments/layer-major-local-attention-20261020/raw/rank-0')
CHECKPOINT = Path('/workspace/models/GLM-5.2-w4a8c8')
TEACHER = Path('/workspace/experiments/shared-fia-20261020/teacher-32.json')
QUOTA = 256 * 1024 ** 2
LAYERS = range(3, 78)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def raw_path(layer):
    name = f'layer-{layer:02d}.pt'
    return next((p / name for p in (ROOT / 'raw/rank-0', OLD / 'rank-0', EXTRA)
                 if (p / name).is_file()), None)


def residual(layer, report, teacher):
    path = raw_path(layer)
    assert path is not None, f'Missing rank0 layer {layer}'
    bindings = [sha for name, sha in report['artifact_sha256'].items()
                if name.endswith(f'/rank-0/layer-{layer:02d}.pt')]
    assert len(bindings) == 1 and digest(path) == bindings[0]
    data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    assert data['layer'] == layer and data['rank'] == 0
    assert data['positions'] == list(range(8192, 8224))
    assert data['teacher_tokens'] == teacher['teacher_tokens']
    assert data['checkpoint_sha256'] == report['checkpoint_sha256']
    value = data['residual'].clone().contiguous()
    assert value.shape == (32, 6144) and value.dtype == torch.bfloat16
    assert torch.isfinite(value).all()
    return value


def small_weights(checkpoint, layer):
    prefix = f'model.layers.{layer}.'
    names = [prefix + suffix for suffix in ('post_attention_layernorm.weight',
             'mlp.gate.weight', 'mlp.gate.e_score_correction_bias')]
    norm_bias_name = prefix + 'post_attention_layernorm.bias'
    if norm_bias_name in checkpoint.weight_map:
        names.append(norm_bias_name)
    tensors = dict(checkpoint.tensors(names))
    gamma, gate, bias = [tensors[name] for name in names[:3]]
    assert gamma.shape == (6144,) and gamma.dtype == torch.float32
    assert gate.shape == (256, 6144) and gate.dtype == torch.bfloat16
    assert bias.shape == (256,) and bias.dtype == torch.float32
    assert all(torch.isfinite(t).all() for t in tensors.values())
    norm_bias = tensors.get(norm_bias_name, torch.zeros(6144))
    return gamma.to(torch.bfloat16), norm_bias.to(torch.bfloat16), gate, bias


def fixture(value):
    x = torch.tensor([1.0, 1.5], dtype=torch.bfloat16).repeat(3072).reshape(1, 6144)
    r = torch.tensor([0.00389, -0.00389], dtype=torch.bfloat16).repeat(3072).reshape(1, 6144)
    cases = [('rounding_boundary', x, r)]
    x = torch.tensor([256, 1, -256, -1], dtype=torch.bfloat16).repeat(1536).reshape(1, 6144)
    r = torch.tensor([-256, -0.99609375, 256, 0.99609375], dtype=torch.bfloat16).repeat(1536).reshape(1, 6144)
    cases.append(('cancellation', x, r))
    x = torch.tensor([0.0, -0.0], dtype=torch.bfloat16).repeat(3072).reshape(1, 6144)
    cases.append(('signed_zero', x, -torch.zeros_like(x)))
    for row in (0, 11, 31):
        half = (value[row:row + 1].float() * 0.5).to(torch.bfloat16)
        assert torch.equal((half.float() + half.float()).to(torch.bfloat16), value[row:row + 1])
        cases.append((f'teacher_row_{row}', half, half.clone()))
    return cases


def native_identity():
    paths = {Path(row.split()[-1]) for row in Path('/proc/self/maps').read_text().splitlines()
             if '/' in row and any(s in row for s in ('opapi', 'optiling', 'opmaster', 'vllm_ascend_C'))}
    return {'host_library_sha256': {str(p): digest(p) for p in paths if p.is_file()},
            'opp_search_path': os.environ.get('ASCEND_CUSTOM_OPP_PATH'),
            'actual_device_elf': None, 'actual_tiling_key': None,
            'source_predicted_add_rms_norm_bias_key': 33,
            'historical_native_identity_verified': False}


def norm_proof(device, gamma, beta, cases):
    gamma, beta = gamma.to(device), beta.to(device)
    checks, outputs = [], {}
    for name, x, r in cases:
        expected_merge = (x.float() + r.float()).to(torch.bfloat16)
        zero_cpu = torch.copysign(torch.zeros_like(expected_merge), expected_merge)
        x, r = x.to(device), r.to(device)
        y, _, merged = torch.ops._C_ascend.npu_add_rms_norm_bias(x, r, gamma, beta, 1e-5)
        zero = zero_cpu.to(device)
        recovered, _, merged_again = torch.ops._C_ascend.npu_add_rms_norm_bias(merged, zero, gamma, beta, 1e-5)
        actual, expected = recovered.cpu(), y.cpu()
        same = torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        checks.append({'case': name, 'bit_equal': same, 'finite': bool(torch.isfinite(actual).all())})
        outputs[name] = {'y': expected, 'recovered': actual, 'residual': merged.cpu()}
        assert same and checks[-1]['finite'], checks[-1]
        assert torch.equal(merged.cpu().view(torch.uint8), expected_merge.view(torch.uint8))
        assert torch.equal(merged.cpu().view(torch.uint8), merged_again.cpu().view(torch.uint8))
    torch.save(outputs, ROOT / 'norm-outputs.pt')
    return checks


def collect_routes(device, checkpoint, report, teacher):
    from vllm_ascend.device.device_op import DeviceOperator

    routes, logits = [], []
    for layer in LAYERS:
        cpu_value = residual(layer, report, teacher)
        signed_zero = torch.copysign(torch.zeros_like(cpu_value), cpu_value).to(device)
        value = cpu_value.to(device)
        gamma, beta, gate, bias = [t.to(device) for t in small_weights(checkpoint, layer)]
        ids, scores = [], []
        for row in range(32):
            current = value[row:row + 1]
            zero = signed_zero[row:row + 1]
            norm, _, _ = torch.ops._C_ascend.npu_add_rms_norm_bias(current, zero, gamma, beta, 1e-5)
            score = torch.mm(norm, gate.t(), out_dtype=torch.float32)
            _, selected, _ = DeviceOperator.moe_gating_top_k(
                score, k=8, k_group=1, group_count=1, group_select_mode=1,
                renorm=1, norm_type=1, out_flag=False,
                routed_scaling_factor=2.5, eps=1e-20, bias_opt=bias)
            ids.append(selected)
            scores.append(score)
        selected, score = torch.cat(ids).cpu(), torch.cat(scores).cpu()
        assert selected.shape == (32, 8) and selected.dtype == torch.int32
        assert ((selected >= 0) & (selected < 256)).all() and torch.isfinite(score).all()
        assert (selected.sort(dim=1).values.diff(dim=1) > 0).all()
        routes.append(selected)
        logits.append(score)
        del value, gamma, beta, gate, bias, norm, current, zero, signed_zero, ids, scores
    raw = torch.stack(routes)
    torch.save({'scope': 'Teacher routes reconstructed from frozen rank0 M1 residual with current M1 kernels',
                'layers': list(LAYERS), 'teacher_tokens': teacher['teacher_tokens'],
                'routes': raw, 'router_logits': torch.stack(logits)}, ROOT / 'teacher-routes.pt')
    unions = {}
    for width in (8, 16, 32):
        groups = []
        for index, layer in enumerate(LAYERS):
            for start in range(0, 32, width):
                counts = torch.bincount(raw[index, start:start + width].flatten().long(), minlength=256)
                union = int((counts > 0).sum())
                groups.append({'layer': layer, 'teacher_start': start, 'union': union,
                               'max_expert_rows': int(counts.max()), 'expert_row_counts': counts.tolist(),
                               'logical_w4_weight_bytes_per_rank': union * (3 * 6144 * 128 // 2),
                               'existing_estimator_expert_bytes_per_rank': union * components()['one_expert']})
        unions[str(width)] = groups
    save('teacher-unions.json', {'scope': 'Contiguous teacher windows only; not rejected lookup suffixes or measured HBM traffic',
                                'existing_estimator_one_expert_bytes': components()['one_expert'], 'widths': unions})


def main(args):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.set_num_threads(1)
    report = json.loads((OLD / 'rank-0.json').read_text())
    teacher = json.loads(TEACHER.read_text())
    checkpoint = ModelSlimCheckpoint(CHECKPOINT)
    config = json.loads((CHECKPOINT / 'config.json').read_text())
    assert report['rank'] == 0 and report['layers'] == 78 and report['steps'] == 32
    assert report.get('query_width', 1) == 1 and len(teacher['teacher_tokens']) == 32
    assert (config['n_routed_experts'], config['num_experts_per_tok'], config['n_group'], config['topk_group']) == (256, 8, 1, 1)
    assert config['scoring_func'] == 'sigmoid' and config['norm_topk_prob'] is True
    assert config['rms_norm_eps'] == 1e-5 and config['routed_scaling_factor'] == 2.5
    assert any('norm.bias' in name for name in checkpoint.quantization)
    for name, sha in report['checkpoint_sha256'].items():
        assert digest(CHECKPOINT / name) == sha
    value = residual(3, report, teacher)
    gamma, beta, _, _ = small_weights(checkpoint, 3)
    assert 'model.layers.3.post_attention_layernorm.bias' in report['layers_report'][3]['coverage']['synthesized_zero_norm_bias']
    cases = fixture(value)
    binding = {'script_sha256': digest(Path(__file__)), 'teacher_sha256': digest(TEACHER),
               'checkpoint_sha256': report['checkpoint_sha256'], 'old_report_sha256': digest(OLD / 'rank-0.json'),
               'weight_reader_sha256': digest(ROOT / 'source/weights.py'),
               'byte_estimator_sha256': digest(ROOT / 'source/estimate_traffic.py')}
    result = {'status': 'prepared', 'phase': args.phase, 'binding': binding,
              'present_layers': [n for n in LAYERS if raw_path(n) is not None],
              'norm_cases': [name for name, _, _ in cases], 'quota_bytes': QUOTA,
              'norm_gamma_dtype': str(gamma.dtype), 'norm_bias_dtype': str(beta.dtype),
              'norm_bias_synthesized_zero': bool((beta == 0).all()),
              'historical_native_identity_verified': False,
              'scope': 'Frozen residual reconstructed with current same-shape M1 kernel; not original saved routes or actual lookup routes'}
    if args.phase != 'prepare':
        import torch_npu
        device = torch.device(f'npu:{args.die}')
        torch.npu.set_device(device)
        torch.npu.set_per_process_memory_fraction(QUOTA / torch.npu.get_device_properties(device).total_memory, device)
        import vllm_ascend.vllm_ascend_C
        from vllm_ascend.utils import enable_custom_op

        assert enable_custom_op()
        torch.npu.reset_peak_memory_stats(device)
        with torch.inference_mode():
            if args.phase == 'norm':
                result['checks'] = norm_proof(device, gamma, beta, cases)
                result['norm_output_sha256'] = digest(ROOT / 'norm-outputs.pt')
            else:
                receipt = json.loads((ROOT / 'norm-result.json').read_text())
                assert receipt['status'] == 'pass' and receipt['binding'] == binding
                assert result['present_layers'] == list(LAYERS), 'All 75 layer files required'
                collect_routes(device, checkpoint, report, teacher)
            torch.npu.synchronize(device)
        result.update(status='pass', native_identity=native_identity(),
                      peak_allocated_bytes=torch.npu.max_memory_allocated(device),
                      peak_reserved_bytes=torch.npu.max_memory_reserved(device))
        assert result['peak_allocated_bytes'] <= QUOTA
    save(f'{args.phase}-result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'binding'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'norm', 'routes'), default='prepare')
    parser.add_argument('--die', type=int, default=0)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        save(f'{args.phase}-failure.json', {'status': 'fail', 'error': f'{type(error).__name__}: {error}'})
        raise
