"""CPU witnesses for current layer-major constructor and chunk-mask contracts."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent
source = ROOT / 'source'
torch.set_num_threads(1)
model_tree = ast.parse((source / 'cann_model.py').read_text())
model = next(n for n in model_tree.body if isinstance(n, ast.ClassDef)
             and n.name == 'CannModel')
methods = [n for n in model.body if isinstance(n, ast.FunctionDef)
           and n.name in ('_install_tp_sum', 'shutdown')]
unit = ast.Module(body=[ast.ClassDef(name='AuditedMethods', bases=[], keywords=[],
                                  body=methods, decorator_list=[])], type_ignores=[])
namespace = {}
exec(compile(ast.fix_missing_locations(unit), str(source / 'cann_model.py'), 'exec'), namespace)
group = SimpleNamespace(_all_reduce_out_place=lambda value: value)
vllm = ModuleType('vllm')
distributed = ModuleType('vllm.distributed')
distributed.get_tp_group = lambda: group
sys.modules['vllm'] = vllm
sys.modules['vllm.distributed'] = distributed
engine = namespace['AuditedMethods']()
engine._tp_reduce = lambda value: value
engine.graph = None
engine._restore_tp_sum = lambda: None
engine.worker = SimpleNamespace(shutdown=lambda: None)
errors = {}
for name in ('_install_tp_sum', 'shutdown'):
    try:
        getattr(engine, name)()
    except AttributeError as error:
        errors[name] = str(error)
assert 'onesided_endpoint' in errors['_install_tp_sum']
assert '_onesided' in errors['shutdown']
engine.onesided_endpoint, engine._onesided = None, None
engine._install_tp_sum()
engine.shutdown()

spec = importlib.util.spec_from_file_location('audited_attention', source / 'cann_attention.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.shared_kv_fia = lambda q, qr, kv, mask, scale: mask.clone()
width, capacity = 16, 9344
binder = helper.SharedKVAttention(width, capacity, torch.device('cpu'))
producer = SimpleNamespace(skip_topk=False, scale=1/16,
                          _execute_sparse_flash_attention_process=lambda *a, **kw: None)
consumer = SimpleNamespace(skip_topk=True, scale=1/16,
                          _execute_sparse_flash_attention_process=lambda *a, **kw: None)
binder.bind(producer)
binder.bind(consumer)
query = torch.zeros(width, 4, 512)
banks, calls = [], []
for chunk in range(2):
    slots = torch.arange(8192 + chunk*width, 8192 + (chunk+1)*width, dtype=torch.int32)
    first = torch.arange(chunk*100, chunk*100+2047, dtype=torch.int32).expand(width, -1)
    indices = torch.cat((first, slots.view(width, 1)), dim=1)
    metadata = SimpleNamespace(slot_mapping=slots)
    args = (query, None, None, indices, metadata, None, None)
    banks.append(producer._execute_sparse_flash_attention_process(*args))
    calls.append(args)
stale = consumer._execute_sparse_flash_attention_process(*calls[0])
assert not torch.equal(stale, banks[0])
assert torch.equal(stale, banks[1])
for index, args in enumerate(calls):
    binder.mask = banks[index]
    assert torch.equal(consumer._execute_sparse_flash_attention_process(*args), banks[index])
config_fn = next(n for n in model_tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'create_config')
engine_args = next(n for n in ast.walk(config_fn) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == 'EngineArgs')
token_expr = next(k.value for k in engine_args.keywords
                  if k.arg == 'max_num_batched_tokens')
config_sizes = {width: eval(compile(ast.Expression(token_expr), '<source formula>', 'eval'),
                            {'max': max, 'graph_width': width})
                for width in (4, 16, 32)}
assert config_sizes == {4: 16, 16: 16, 32: 32}
result = {
    'status': 'pass', 'npu_imported': 'torch_npu' in sys.modules,
    'constructor_missing_attribute_witness': errors,
    'explicit_None_fields_pass_actual_AST_methods': True,
    'M16_two_chunk_last_mask_counterexample': True,
    'mask_bank_restore_exact_actual_binder': True,
    'stale_mask_different_rows': int((stale != banks[0]).any(dim=1).count_nonzero()),
    'mask_bank_bytes': sum(v.numel() * v.element_size() for v in banks),
    'actual_config_formula_by_graph_width': config_sizes,
    'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in source.iterdir() if p.is_file()},
}
assert not result['npu_imported']
(ROOT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
