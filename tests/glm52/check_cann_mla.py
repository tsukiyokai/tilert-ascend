"""Compare one TP16 shard of CANN sparse MLA with frozen Prolog inputs.

Two teacher positions share one 128-row cache page. CPU compressed and HF
expanded references use the same observed query, latent and RoPE values.
The unabsorbed HF query is reconstructed from observed INT8 qA and scale;
it is not an additional device observation. No model or collective loads.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


HEADS = 4
LATENT = 512
NOPE = 192
ROPE = 64
VALUE = 256
PAGE = 128
TOP_K = 2048
SCALE = 1 / 16
ALLOCATOR_LIMIT = 1536 * 1024 ** 2
ASCEND = Path('/workspace/src/vllm-ascend-unified')
VENDOR = ASCEND / 'vllm_ascend/_cann_ops_custom/vendors/custom_transformer'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_inputs(paths):
    import torch

    frozen, provenance = [], []
    expected = {
        'query_nope': ((1, HEADS, LATENT), torch.bfloat16),
        'query_rope': ((1, HEADS, ROPE), torch.bfloat16),
        'latent': ((1, LATENT), torch.bfloat16),
        'rope': ((1, ROPE), torch.bfloat16),
        'query_norm': ((1, 2048), torch.int8),
        'query_norm_scale': ((1, 1), torch.float32),
    }
    for position, path in enumerate(paths):
        report = json.loads(path.with_suffix('.json').read_text())
        require(report['device_execution'] is True,
                'Inputs must contain a real device observation')
        require(report['artifact_sha256'] == digest(path),
                'Frozen tensor file differs from its device report')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        identity = json.loads(json.dumps(saved['identity']))
        require(identity == report['identity'], 'Input identity mismatch')
        require((identity['position'], identity['layer'],
                 identity['logical_rank'], identity['tp_size'])
                == (position, 0, 0, 16), 'Expected layer0/rank0/TP16 pos0/1')
        for name, (shape, dtype) in expected.items():
            value = saved['actual'][name]
            require(value.shape == shape and value.dtype == dtype
                    and value.device.type == 'cpu'
                    and bool(torch.isfinite(value).all()),
                    f'Invalid frozen tensor: {name}')
        frozen.append(saved)
        provenance.append({
            'path': str(path), 'sha256': digest(path),
            'report_sha256': digest(path.with_suffix('.json')),
            'input_report_status': report['status'], 'identity': identity,
            'source_sha256': report['source_sha256'],
        })
    require(frozen[0]['identity']['checkpoint_sha256']
            == frozen[1]['identity']['checkpoint_sha256'],
            'The two positions use different checkpoints')
    require(frozen[0]['identity']['manifest_sha256']
            == frozen[1]['identity']['manifest_sha256'],
            'The two positions use different teacher manifests')
    for name in ('weight', 'scale'):
        require(torch.equal(frozen[0]['weights']['q_b'][name],
                            frozen[1]['weights']['q_b'][name]),
                'Q-B weights differ between positions')
    weight = frozen[0]['weights']['kv_b']
    require(weight.shape == (HEADS * (NOPE + VALUE), LATENT)
            and weight.dtype == torch.bfloat16
            and torch.equal(weight, frozen[1]['weights']['kv_b']),
            'KV-B rank0 weights differ or have the wrong layout')
    return frozen, provenance, weight


def cpu_references(frozen, weight, checks):
    import torch
    from check_mla import compare, query_reference
    from check_mla import value_reference

    latent = torch.cat([item['actual']['latent'] for item in frozen])
    rope = torch.cat([item['actual']['rope'] for item in frozen])
    references = []
    for position, saved in enumerate(frozen):
        actual = saved['actual']
        qb = saved['weights']['q_b']
        integer = (actual['query_norm'].int() @ qb['weight'].int().T)
        query = (integer.float() * actual['query_norm_scale']) * qb['scale']
        q_nope = query.reshape(1, HEADS, NOPE + ROPE)[..., :NOPE].to(
            torch.bfloat16
        )
        absorbed = query_reference(q_nope, weight).to(torch.bfloat16)
        compare_heads(checks, f'pos{position}.reconstructed_query', absorbed,
                      actual['query_nope'], 'cpu_input_link', LATENT)
        indices = torch.arange(position, -1, -1, dtype=torch.int64)
        selected = latent[indices].float()
        scores = actual['query_nope'][0].float() @ selected.T
        scores += actual['query_rope'][0].float() @ rope[indices].float().T
        probability = torch.softmax(scores * SCALE, dim=-1)
        context = (probability @ selected).unsqueeze(0).to(torch.bfloat16)
        output = value_reference(context, weight).to(torch.bfloat16)
        expanded = hf_eager_reference(
            weight, q_nope, actual['query_rope'], latent, rope, indices, SCALE,
        )
        references.append({
            'indices': indices, 'q_nope_reconstructed': q_nope,
            'scores_fp32': scores, 'probability_fp32': probability,
            'latent_output': context, 'output': output,
            'hf_expanded': expanded,
        })
    compare(checks, 'pos0.single_key_identity',
            references[0]['latent_output'],
            latent[0].reshape(1, 1, LATENT).expand(1, HEADS, LATENT),
            'cpu_identity', exact=True)
    return latent, rope, references


def hf_eager_reference(weight, q_nope, q_rope, latent, rope, indices, scale):
    """Call installed HF eager attention on BF16 expanded checkpoint KV."""
    import torch
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        eager_attention_forward,
    )

    expanded = (latent[indices].float() @ weight.float().T).to(
        torch.bfloat16
    ).reshape(-1, HEADS, NOPE + VALUE)
    key, value = expanded.split((NOPE, VALUE), dim=-1)
    key_rope = rope[indices, None].expand(-1, HEADS, -1)
    key = torch.cat((key, key_rope), -1).transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)
    query = torch.cat((q_nope, q_rope), -1).transpose(0, 1).unsqueeze(0)
    module = SimpleNamespace(num_key_value_groups=1, training=False)
    output, _ = eager_attention_forward(
        module, query, key, value, None, scaling=scale, dropout=0.0,
    )
    return output.reshape(1, HEADS * VALUE)


def compare_heads(checks, name, actual, expected, category, width):
    from check_mla import compare

    actual = actual.reshape(1, HEADS, width)
    expected = expected.reshape(1, HEADS, width)
    for head in range(HEADS):
        compare(checks, f'{name}.head{head}', actual[:, head],
                expected[:, head], category)


def run_device(device, frozen, weight, latent, rope, references, checks):
    import torch
    import torch_npu
    from check_mla import compare, value_reference
    from tilert.models.glm_5_2_ascend.ops.mla import (
        _sparse_attention_operator,
    )
    from vllm_ascend.utils import bootstrap_custom_op_env

    torch.npu.set_device(device)
    properties = torch.npu.get_device_properties(device)
    torch.npu.set_per_process_memory_fraction(
        ALLOCATOR_LIMIT / properties.total_memory, device,
    )
    bootstrap_custom_op_env()
    require(str(VENDOR) in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
            .split(':'), 'The vLLM custom SFA vendor must precede built-ins')
    operation = _sparse_attention_operator()
    torch.npu.reset_peak_memory_stats(device)
    value_weight = weight.reshape(HEADS, NOPE + VALUE, LATENT)[:, NOPE:]
    value_weight = value_weight.transpose(1, 2).contiguous().to(device)
    cache = torch.zeros(1, PAGE, 1, LATENT, dtype=torch.bfloat16,
                        device=device)
    rope_cache = torch.zeros(1, PAGE, 1, ROPE, dtype=torch.bfloat16,
                             device=device)
    cache[0, :2, 0].copy_(latent.to(device))
    rope_cache[0, :2, 0].copy_(rope.to(device))
    block_table = torch.zeros(1, 1, dtype=torch.int32, device=device)
    query_length = torch.ones(1, dtype=torch.int32, device=device)
    observed, layouts = [], []
    for position, (saved, reference) in enumerate(zip(frozen, references)):
        query = saved['actual']['query_nope'].to(device)
        q_rope = saved['actual']['query_rope'].to(device)
        indices = torch.full((1, 1, TOP_K), -1, dtype=torch.int32,
                             device=device)
        count = position + 1
        indices[0, 0, :count].copy_(reference['indices'].to(device))
        key_length = torch.tensor([count], dtype=torch.int32, device=device)
        context, _, _ = operation(
            query=query, key=cache, value=cache, sparse_indices=indices,
            block_table=block_table, actual_seq_lengths_query=query_length,
            actual_seq_lengths_kv=key_length, query_rope=q_rope,
            key_rope=rope_cache, scale_value=SCALE, sparse_block_size=1,
            layout_query='TND', layout_kv='PA_BSND', sparse_mode=3,
            attention_mode=2, return_softmax_lse=False,
        )
        output = torch_npu.npu_transpose_batchmatmul(
            context, value_weight, perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2), perm_y=(1, 0, 2),
        ).reshape(1, HEADS * VALUE)
        torch.npu.synchronize(device)
        actual = {'latent_output': context.cpu().clone(),
                  'output': output.cpu().clone()}
        observed.append(actual)
        compare_heads(checks, f'pos{position}.sfa', actual['latent_output'],
                      reference['latent_output'], 'compressed_fp32', LATENT)
        local_value = value_reference(actual['latent_output'], weight).to(
            torch.bfloat16
        )
        compare_heads(checks, f'pos{position}.value_same_input',
                      actual['output'], local_value, 'value_projection', VALUE)
        compare_heads(checks, f'pos{position}.compressed_chain',
                      actual['output'], reference['output'],
                      'compressed_fp32', VALUE)
        compare_heads(checks, f'pos{position}.hf_expanded', actual['output'],
                      reference['hf_expanded'], 'hf_expanded', VALUE)
        expected_indices = torch.full((1, 1, TOP_K), -1, dtype=torch.int32)
        expected_indices[0, 0, :count] = reference['indices'].to(torch.int32)
        expected_cache = torch.zeros(1, PAGE, 1, LATENT, dtype=torch.bfloat16)
        expected_rope = torch.zeros(1, PAGE, 1, ROPE, dtype=torch.bfloat16)
        expected_cache[0, :2, 0] = latent
        expected_rope[0, :2, 0] = rope
        preserved = {
            'query': (query, saved['actual']['query_nope']),
            'query_rope': (q_rope, saved['actual']['query_rope']),
            'indices': (indices, expected_indices),
            'latent_cache': (cache, expected_cache),
            'rope_cache': (rope_cache, expected_rope),
            'query_length': (query_length, torch.tensor([1],
                                                       dtype=torch.int32)),
            'key_length': (key_length, torch.tensor([count],
                                                   dtype=torch.int32)),
            'block_table': (block_table, torch.zeros(1, 1, dtype=torch.int32)),
        }
        for name, (value, expected) in preserved.items():
            compare(checks, f'pos{position}.preserved.{name}', value, expected,
                    'exact_inputs_and_cache', exact=True)
        layouts.append({name: {
            'shape': list(value.shape), 'dtype': str(value.dtype),
            'stride': list(value.stride()),
            'format': torch_npu.get_npu_format(value),
        } for name, (value, _) in preserved.items()})
    memory = {
        'allocator_limit_bytes': ALLOCATOR_LIMIT,
        'peak_allocated_bytes': torch.npu.max_memory_allocated(device),
        'peak_reserved_bytes': torch.npu.max_memory_reserved(device),
        'scope': 'Torch allocator; non-Torch runtime allocations excluded',
    }
    return observed, layouts, memory, str(operation.default._schema)


def source_hashes():
    import check_mla
    from tilert.models.glm_5_2_ascend.ops import mla
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        eager_attention_forward,
    )

    paths = [Path(__file__), Path(check_mla.__file__), Path(mla.__file__),
             Path(inspect.getsourcefile(eager_attention_forward)),
             ASCEND / 'vllm_ascend/attention/sfa_v1.py',
             ASCEND / 'vllm_ascend/utils.py',
             ASCEND / 'csrc/attention/sparse_flash_attention/op_host/op_api/'
             'aclnn_sparse_flash_attention.cpp',
             VENDOR / 'op_api/lib/libcust_opapi.so']
    return {str(path): digest(path) for path in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--position-zero', type=Path, required=True)
    parser.add_argument('--position-one', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cpu-only', action='store_true')
    mode.add_argument('--device', help='One explicitly assigned npu:N')
    args = parser.parse_args(argv)
    import torch
    import torch_npu
    from check_mla import metrics

    torch.set_num_threads(2)
    require(not torch_npu.npu._initialized, 'Start in a fresh process')
    frozen, provenance, weight = load_inputs(
        (args.position_zero, args.position_one),
    )
    checks = []
    latent, rope, references = cpu_references(frozen, weight, checks)
    require(all(item['passed'] for item in checks),
            'CPU input reconstruction or single-key identity failed')
    observed, layouts, memory, schema = [], [], None, None
    if args.device:
        require(torch.device(args.device).type == 'npu',
                'The device branch requires npu:N')
        observed, layouts, memory, schema = run_device(
            args.device, frozen, weight, latent, rope, references, checks,
        )
    else:
        require(not torch_npu.npu._initialized,
                'CPU preflight initialized the NPU')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = args.output.with_suffix('.pt')
    torch.save({'references': references, 'actual': observed,
                'latent': latent, 'rope': rope, 'kv_b_weight': weight,
                'frozen_queries': [{
                    'nope': item['actual']['query_nope'],
                    'rope': item['actual']['query_rope'],
                } for item in frozen]}, artifact)
    report = {
        'status': 'pass' if all(item['passed'] for item in checks) else 'fail',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'device_execution': bool(args.device),
        'precision_verified': False, 'performance_measured': False,
        'scope': 'Layer0, TP16 rank0, four heads, cold teacher positions0/1; '
        'SFA core and value projection only; no timing or HCCL',
        'command': sys.argv, 'device': args.device,
        'npu_initialized_after': bool(torch_npu.npu._initialized),
        'frozen_inputs': provenance, 'checks': checks,
        'source_sha256': source_hashes(),
        'artifact': str(artifact), 'artifact_sha256': digest(artifact),
        'schema': schema, 'input_layouts': layouts, 'memory': memory,
        'packages': {name: version(name)
                     for name in ('torch', 'torch-npu', 'transformers')},
        'custom_opp_path': os.environ.get('ASCEND_CUSTOM_OPP_PATH'),
        'cpu_compressed_vs_hf_expanded': [metrics(
            item['output'], item['hf_expanded']
        ) for item in references],
        'reference_contract': {
            'sfa': 'Actual BF16 absorbed query and cache; FP32 QK, RoPE '
            'score, scale and softmax; FP32 latent PV -> BF16',
            'value': 'FP32 product with raw BF16 V weight -> BF16',
            'hf': 'Installed HF eager_attention_forward; reconstructed '
            'pre-absorption query, actual RoPE and BF16 expanded KV',
            'threshold': 'Per-head NRMSE<=0.01 and cosine>=0.999; '
            'indices, cache and inputs exact',
            'not_claimed': 'CPU FP32 softmax is not a bitwise emulation '
            'of CANN FlashAttention tiling or normalization',
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks),
                      'device_execution': bool(args.device),
                      'report': str(args.output)}))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
