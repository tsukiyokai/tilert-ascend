"""CPU fullgraph contract for actual Ascend opaque MLA and shared binder."""

import ast
import __future__
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
import torch._dynamo


ROOT = Path('/workspace/offline/shared-fia-compile-contract-20261020-codex')
HELPER = Path('/workspace/offline/shared-fia-integration-20261020/tree/tilert/models/glm_5_2_ascend/cann_attention.py')
MLA = Path('/workspace/src/vllm-ascend-unified/vllm_ascend/ops/mla.py')
FIXTURE = Path('/workspace/offline/fia-four-layer-contract-20261020-codex/fixture.pt')
CONFIG = Path('/workspace/models/GLM-5.2-w4a8c8/config.json')
ROOT.mkdir(exist_ok=True, parents=True)
torch.set_num_threads(1)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
helper_spec = importlib.util.spec_from_file_location('actual_shared_helper', HELPER)
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == '47503e3cd1b730cfcc3b21f27f8d1709f2bcf003cb72d3537d383d65df3417e6'
fixture = torch.load(FIXTURE, map_location='cpu', weights_only=True)
config = json.loads(CONFIG.read_text())
assert config['num_hidden_layers'] == 78 and config['index_topk_freq'] == 4
offset = config.get('index_skip_topk_offset', 2)
roles = [max(layer - offset + 1, 0) % config['index_topk_freq'] != 0
         for layer in range(78)]
assert [i for i, skip in enumerate(roles) if not skip] == [0, 1, 2, *range(6, 78, 4)]
indices = fixture['ids_A'].clone()
metadata = SimpleNamespace(slot_mapping=fixture['thresholds_A'].clone() - 1)
binder = helper.SharedKVAttention(8, 9344, 'cpu')
context = SimpleNamespace(no_compile_layers={}, attn_metadata={})
order = []
probe = next(token for token in fixture['ids_B'][0, :-1].tolist()
             if token not in set(fixture['ids_A'][0].tolist()))
fake_npu = ModuleType('torch_npu')
fake_npu.__spec__ = importlib.util.spec_from_loader('torch_npu', loader=None)


def fia_mask_observer(query, key, value, **kwargs):
    # Observable mask-state output only; this is not attention arithmetic.
    selected = (~kwargs['atten_mask'][0, 0, :, probe]).to(query.dtype)
    return query + selected.view(1, 8, 1, 1), None


fake_npu.npu_fused_infer_attention_score_v2 = fia_mask_observer
sys.modules['torch_npu'] = fake_npu
tree = ast.parse(MLA.read_text())
outer_class = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef)
                   and node.name == 'AscendMultiHeadLatentAttention')
outer = next(node for node in outer_class.body
             if isinstance(node, ast.FunctionDef) and node.name == 'forward')
kernel = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == 'mla_forward')
fake = next(node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == 'mla_forward_fake')
# FULL mode executes the decorator's wrapped kernel directly; preserve its body.
kernel.decorator_list = []
namespace = {'torch': torch, 'get_forward_context': lambda: context}
for node in [outer, kernel, fake]:
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(MLA), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
outer_forward = namespace['forward']
library = torch.library.Library('vllm', 'FRAGMENT')
library.define('mla_forward(Tensor hidden_states, Tensor(a!) output, str layer_name) -> ()')
library.impl('mla_forward', namespace['mla_forward'], 'CPU')
torch.library.register_fake('vllm::mla_forward', namespace['mla_forward_fake'])
kv = fixture['layers'][0]['kv']
query_rope = fixture['layers'][0]['qr']
length = torch.tensor([8], dtype=torch.int32)
wrappers = []
for layer, skip in enumerate(roles):
    impl = SimpleNamespace(skip_topk=skip, scale=1 / 16,
                           _execute_sparse_flash_attention_process=lambda *args, **kwargs: None)
    binder.bind(impl)

    def runtime_forward(name, hidden, cache, meta, output, impl=impl, layer=layer):
        order.append(layer)
        result = impl._execute_sparse_flash_attention_process(
            hidden.view(8, 4, 512), query_rope, cache, indices, meta, length, length)
        output.copy_(result.reshape_as(output))

    impl.forward = runtime_forward
    name = f'model.layers.{layer}.self_attn'
    wrapper = SimpleNamespace(hidden_size=2048, output_token_shard_size=1,
                              prefix=name, mla_attn=SimpleNamespace(
                                  layer_name=name, kv_cache=kv, impl=impl))
    wrappers.append(wrapper)
    context.no_compile_layers[name] = wrapper
    context.attn_metadata[name] = metadata


def backbone(hidden, positions):
    for wrapper in wrappers:
        hidden = outer_forward(wrapper, positions, hidden)
    return hidden


graphs = []


def backend(graph, example_inputs):
    graphs.append(graph)
    (ROOT / f'graph-{len(graphs)}.py').write_text(graph.code)
    return graph.forward


compiled = torch.compile(backbone, backend=backend, fullgraph=True, dynamic=False)
hidden = torch.zeros((8, 2048), dtype=torch.bfloat16)
positions = torch.arange(8)
checks, values = [], {}
with torch.inference_mode():
    for state in ['A', 'B', 'reset']:
        suffix = 'B' if state == 'B' else 'A'
        indices.copy_(fixture['ids_' + suffix])
        metadata.slot_mapping.copy_(fixture['thresholds_' + suffix] - 1)
        order.clear()
        expected = backbone(hidden, positions)
        assert order == list(range(78))
        order.clear()
        actual = compiled(hidden, positions)
        assert order == list(range(78))
        assert torch.equal(actual, expected)
        checks.append({'state': state, 'bit_equal_cpu_mask_observer': True,
                       'runtime_order_0_to_77': True, 'compiled_graphs_so_far': len(graphs)})
        values[state] = actual.clone()
assert torch.equal(values['A'], values['reset'])
assert not torch.equal(values['A'], values['B'])
counts = [sum(node.op == 'call_function'
              and str(node.target) == 'vllm.mla_forward' for node in graph.graph.nodes)
          for graph in graphs]
assert len(graphs) == 1 and counts == [78]
report = {'scope': 'Server CPU Dynamo/FX opaque-op contract; no CANN backend codegen',
          'npu_launched': False, 'torch_version': torch.__version__,
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'helper_sha256': hashlib.sha256(HELPER.read_bytes()).hexdigest(),
          'actual_mla_source_sha256': hashlib.sha256(MLA.read_bytes()).hexdigest(),
          'checkpoint_config_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
          'index_skip_topk_offset': offset,
          'producers': [i for i, skip in enumerate(roles) if not skip],
          'fullgraph': True, 'compiled_graph_count': len(graphs),
          'opaque_mla_nodes_per_graph': counts, 'checks': checks,
          'state_B_changes_output': True, 'reset_restores_A': True,
          'limit': ('CPU registration preserves the actual opaque MLA schema/forward/kernel '
                    'body and real binder. The FIA primitive is a mask observer, '
                    'not CANN arithmetic. NPU Inductor/static-kernel compilation '
                    'and real 78-layer resource use remain device acceptance.')}
(ROOT / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
