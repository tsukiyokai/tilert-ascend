"""Execute real runner scheduling on CPU; CANN math and runtime are substitutes."""
import argparse
import ast
from contextlib import contextmanager, nullcontext, redirect_stderr
import hashlib
import inspect
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch
import weakref
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
torch.set_num_threads(1)

def extract(file, names):
    tree = ast.parse((SOURCE / file).read_text())
    nodes = [n for n in tree.body if getattr(n, 'name', None) in names]
    exec(compile(ast.Module(nodes, []), str(SOURCE / file), 'exec'), globals())
    return tree

def substitute(path, **values):
    for end in range(1, len(path.split('.')) + 1):
        name = '.'.join(path.split('.')[:end])
        sys.modules.setdefault(name, ModuleType(name))
    vars(sys.modules[path]).update(values)

class Placeholder:
    _registered_moe_layers = []

def noop(*args, **kwargs):
    return {}

# Assignments have no name; evaluate these two exact source expressions.
layout = ast.parse((SOURCE / 'ascend_layout.py').read_text())
for node in layout.body:
    if isinstance(node, ast.Assign) and node.targets[0].id in ('FULL_INDEX_LAYERS', 'INT8_INDEX_LAYERS'):
        exec(compile(ast.Module([node], []), '<layout>', 'exec'), globals())
substitute('vllm.model_executor.model_loader.utils', process_weights_after_loading=noop)
substitute('vllm.model_executor.models.utils', PPMissingLayer=Placeholder)
substitute('vllm_ascend.ascend_forward_context', set_ascend_forward_context=lambda *a, **k: nullcontext())
substitute('vllm_ascend.eplb.adaptor.vllm_adaptor', VllmEplbAdaptor=Placeholder)
substitute('vllm_ascend.quantization.methods.kv_cache.kv_c8', AscendSFAQuantAttentionMethod=Placeholder)
substitute('tilert.pd_vllm.ascend_layout', INT8_INDEX_LAYERS=INT8_INDEX_LAYERS)
extract('cann_attention.py', ['selection_mask', 'shared_kv_fia', 'SharedKVAttention'])
extract('run_cann_streamed.py', ['selections'])
runner = extract('run_cann_layer_major.py', ['arguments', 'run_layer', 'final_logits'])
checked = lambda name, action: action()
bound_rope_tables = make_skeleton = load_scope = noop
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
check_prefix = lambda engine, prefix, checks, label: checks.append(
    {'name': label + '.prefix', 'passed': all(torch.equal(engine.cache[0][k][:8192], v)
                                             for k, v in prefix[0].items())})
torch.npu = NS(empty_cache=lambda: None, synchronize=lambda *a: None)
torch.ops._C_ascend.npu_add_rms_norm_bias = lambda h, r, *a: (None, None, h + r)

class Norm(torch.nn.Module):
    def forward(self, hidden, residual=None):
        return hidden if residual is None else (hidden, hidden + residual)

def original(ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
             actual_seq_lengths_query, actual_seq_lengths_key, block_table=None):
    log.append(('sfa', int(ql_nope[0, 0, 0]), len(ql_nope)))
    return ql_nope

def shared_kv_fia(q, qr, cache, mask, scale):
    number = int(q[0, 0, 0])
    if number in FULL_INDEX_LAYERS:
        assert mask is not owner[1]
        owner[0], owner[1] = number, mask
    assert not bool(mask.gather(1, engine.topk[:len(q)].long()).any())
    assert bool(mask[torch.arange(9344)[None, :] > engine.positions[:, None]].all())
    assert mask is owner[1] and owner[0] == max(n for n in FULL_INDEX_LAYERS if n <= number)
    log.append(('fia', number, len(q), id(mask), owner[0]))
    return q

class Layer(torch.nn.Module):
    def __init__(self, number):
        super().__init__()
        self.number, self.input_layernorm = number, Norm()
        impl = NS(skip_topk=number not in FULL_INDEX_LAYERS, scale=1/16,
                  _execute_sparse_flash_attention_process=original)
        self.self_attn = NS(rotary_emb=Placeholder(), topk_indices_buffer=engine.topk,
            mla_attn=NS(skip_topk=impl.skip_topk, mla_attn=NS(impl=impl,
                indexer=NS(q_rot=None, k_rot=None))))
    def forward(self, positions, hidden, residual):
        self.input_layernorm(hidden) if residual is None else self.input_layernorm(hidden, residual)
        n, w = self.number, len(hidden)
        engine.write_slots.extend(positions.tolist())
        assert torch.equal(engine.input_ids, positions.to(torch.int32) - 8192 + 100)
        impl = self.self_attn.mla_attn.mla_attn.impl
        if not impl.skip_topk:
            ids = torch.cat((torch.arange(n*37, n*37+2047).expand(w, -1), positions[:, None]), 1).int()
            engine.topk[:w].copy_(ids)
        else:
            assert torch.equal(engine.topk[:w], engine.expected[positions - 8192])
        q = torch.full((w, 4, 512), n)
        impl._execute_sparse_flash_attention_process(q, None, (), engine.topk[:w], engine.metadata, None, None)
        for key, value in engine.cache[0].items():
            value[positions] = ((positions - 8192 + n*7) % 127 + 1).to(value.dtype).reshape(-1, *([1]*(value.ndim-1)))
        engine._tp_sum_original(hidden.float())
        engine._tp_sum_original(hidden.float())
        return hidden + n + 1, torch.full_like(hidden, n + 1) if residual is None else residual + n + 1

def bind_cache(e, number, prefix):
    e.cache = [{k: torch.zeros((9344, *v.shape[1:]), dtype=v.dtype) for k, v in prefix.items()}]
    for k, v in prefix.items():
        e.cache[0][k][:8192].copy_(v)
    e._shared_attention.bind(e.model.model.layers[number].self_attn.mla_attn.mla_attn.impl)
    return {}

def metadata(width):
    engine.positions = torch.arange(engine.position, engine.position + width)
    engine.input_ids = torch.empty(width, dtype=torch.int32)
    engine.metadata = NS(slot_mapping=engine.positions.int())
    return engine.metadata

results, endpoints = [], []
model_tree = ast.parse((SOURCE / 'cann_model.py').read_text())
cls = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == 'CannModel')
methods = [n for n in cls.body if getattr(n, 'name', None) in ('_install_tp_sum', '_tp_reduce', '_restore_tp_sum', 'shutdown')]
exec(compile(ast.fix_missing_locations(ast.Module([ast.ClassDef('Engine', [], [], methods, [])], [])), '<actual methods>', 'exec'), globals())
substitute('vllm.distributed', get_tp_group=lambda: NS(_all_reduce_out_place=lambda value: value))
execute = next(n for n in runner.body if getattr(n, 'name', None) == 'execute')
init = [n for n in ast.walk(execute) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Tuple) and any(isinstance(x, ast.Attribute) and x.attr in ('onesided_endpoint', 'graph_width', '_tp_sum_group') for x in t.elts) for t in n.targets)]
for width in (1, 32):
    engine, log, owner = Engine(), [], [None, None]
    exec(compile(ast.Module(init, []), '<actual init assignments>', 'exec'), globals())
    assert engine.onesided_endpoint is None and engine._onesided is None and engine.graph_width == 32
    engine._install_tp_sum()
    engine.device, engine.topk = torch.device('cpu'), torch.empty(32, 2048, dtype=torch.int32)
    engine.config = NS(compilation_config=NS(static_forward_context={}, static_all_moe_layers=[]), model_config=None)
    engine.model = NS(model=NS(layers={}, norm=lambda h, r: (h+r, None)))
    engine.source_metadata, engine._metadata = lambda: {}, metadata
    engine._shared_attention, engine.norm_units = SharedKVAttention(32, 9344, 'cpu'), torch.ones(6144)
    args, teacher, embedding_calls = NS(query_width=width, fp32_head=True), list(range(100,132)), []
    engine.model.embed_input_ids = lambda ids: (embedding_calls.append(len(ids)) or (ids-100).to(torch.bfloat16)[:, None].expand(-1, 6144))
    embedding_loop = next(n for n in ast.walk(execute) if isinstance(n, ast.For) and 'embed_input_ids' in ast.unparse(n))
    embeddings = []
    exec(compile(ast.Module([embedding_loop], []), '<actual embedding loop>', 'exec'), globals())
    state = (torch.cat(embeddings), None, None)
    assert embedding_calls == [width] * (32//width)
    for number in range(8):
        fields = {'latent': (512, torch.bfloat16), 'rope': (64, torch.bfloat16)}
        if number in FULL_INDEX_LAYERS: fields['index_keys'] = (128, torch.int8 if number == 6 else torch.bfloat16)
        if number == 6: fields['index_scales'] = (1, torch.float16)
        prefix = {k: torch.zeros((8192, d), dtype=t) for k, (d, t) in fields.items()}
        engine.expected, engine.write_slots = state[2], []
        engine.topk.fill_(-1)
        state, data, checks, *_ = run_layer(engine, lambda prefix: Layer(number), Placeholder(), number, prefix, state, list(range(100, 132)), width)
        assert all(c['passed'] for c in checks) and data['hidden'].shape == (32, 6144)
        assert engine.write_slots == list(range(8192, 8224))
        assert all(v.shape[0] == 32 and bool(v.reshape(32, -1).count_nonzero(dim=1).all()) for v in data['cache_rows'].values())
    head_shapes = []
    engine.model.compute_logits = lambda x: (head_shapes.append(tuple(x.shape)) or x[:, :1].expand(-1, 154880).float())
    normalized, logits = final_logits(engine, state[0], state[1], width)
    assert head_shapes == [(width, 6144)] * (32//width) and logits.shape == (32, 154880)
    assert torch.equal(normalized[:, 0], torch.arange(32) + 72)
    gate_names = ('logits_shape_and_dtype', 'final_normalized_shape_and_dtype', 'final_outputs_finite')
    gates = [n for n in ast.walk(execute) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'checks.append' and any(name in ast.unparse(n) for name in gate_names)]
    dtype = torch.float32
    exec(compile(ast.Module(gates, []), '<actual final gates>', 'exec'), globals())
    assert len(gates) == 3 and all(c['passed'] for c in checks[-3:])
    logits[0, 0] = float('nan')
    exec(compile(ast.Module(gates, []), '<actual NaN final gates>', 'exec'), globals())
    assert not checks[-1]['passed']
    assert sum(v[0] == 'fia' for v in log) == (8 if width == 32 else 0)
    results.append({'width': width, 'layer_calls': len(log), 'head_calls': head_shapes, 'embedding_calls': embedding_calls, 'new_KV_rows_per_layer': 32, 'actual_finite_gate_rejects_NaN': True})
    engine.worker = NS(shutdown=lambda: None)
    engine.shutdown()
for extra in (['--query-width','32','--steps','31'], ['--query-width','32','--attention-kernel','gather_fia']):
    argv = ['check'] + [x for n in ('checkpoint','capture-case','teacher-json','prompt-json','output') for x in ('--'+n,'/unused')] + extra
    with patch.object(sys, 'argv', argv), redirect_stderr(io.StringIO()):
        try: arguments()
        except SystemExit as error: assert error.code == 2
        else: raise AssertionError('Invalid actual argument accepted')
argv = ['check'] + [x for n in ('checkpoint','capture-case','teacher-json','prompt-json','output') for x in ('--'+n,'/unused')] + ['--query-width','32','--attention-kernel','shared_fia']
with patch.object(sys, 'argv', argv):
    assert arguments().query_width == 32
report = {'status':'pass','torch_npu_imported':'torch_npu' in sys.modules,'results':results,'argument_rejections':2,
          'stub_scope':'Layer arithmetic, CANN/HCCL/cache allocation/metadata runtime substituted; actual run_layer, selections, SharedKVAttention, final_logits, argument rejection, init and TP lifecycle AST executed',
          'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.iterdir()}}
assert not report['torch_npu_imported']
(ROOT / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
