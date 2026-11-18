"""CPU compilation probe; never loads or impersonates NPU implementations."""

import ast
import hashlib
import json
from pathlib import Path

import torch
from torch._dynamo.backends.common import aot_autograd
from torch.fx.node import has_side_effect


SOURCE = Path('/Users/shanshan/repo/ascend/msprobe/python/msprobe/pytorch/'
              'aclgraph_dump/__init__.py')
CPP = Path('/Users/shanshan/repo/ascend/msprobe/ccsrc/aclgraph_dump/'
           'aclgraph_dump.cpp')
RESULT = Path('/tmp/tilert-graph-observer-traceability-20261028.json')
calls = []
library = torch.library.Library('my_ns', 'DEF')
library.define('acl_save(Tensor x, str path) -> Tensor')


def cpu_implementation(x, path):
    # CPU probe implementation, not the real ACL callback/C++ implementation.
    calls.append({'path': path, 'values': x.detach().clone().tolist()})
    return x.clone()


library.impl('acl_save', cpu_implementation, 'CPU')


@torch.library.register_fake('my_ns::acl_save')
def fake_implementation(x, path):
    # This is the actual Python fake implementation's shape/device contract.
    return torch.empty_strided(x.size(), x.stride(), dtype=x.dtype,
                              device='meta')


has_side_effect(torch.ops.my_ns.acl_save.default)
tree = ast.parse(SOURCE.read_text())
function = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == 'acl_save')
namespace = {'torch': torch}
module_ast = ast.Module(body=[function], type_ignores=[])
exec(compile(module_ast, str(SOURCE), 'exec'), namespace)
acl_save = namespace['acl_save']


class Network(torch.nn.Module):
    def __init__(self, hook):
        super().__init__()
        self.layer = torch.nn.Identity()
        self.hook = hook
        if hook:
            self.layer.register_forward_hook(self.observe)

    @staticmethod
    def observe(module, inputs, output):
        acl_save(output, '/tmp/probe-hook.pt')

    def forward(self, x):
        x = self.layer(x * 2)
        if not self.hook:
            acl_save(x, '/tmp/probe-inline.pt')
        return x + 1


def nodes(gm):
    return [{'name': node.name, 'op': node.op, 'target': str(node.target),
             'impure': node.is_impure()} for node in gm.graph.nodes]


def run(aot, hook):
    torch._dynamo.reset()
    calls.clear()
    stages = []

    def compiled(gm, example_inputs):
        stages.append({'stage': 'aot' if aot else 'dynamo',
                       'before_dce': nodes(gm)})
        gm.graph.eliminate_dead_code()
        gm.recompile()
        stages[-1]['after_dce'] = nodes(gm)
        return gm.forward

    aot_backend = aot_autograd(fw_compiler=compiled,
                              inference_compiler=compiled,
                              keep_inference_input_mutations=True)

    def backend(gm, example_inputs):
        if aot:
            stages.append({'stage': 'dynamo', 'before_dce': nodes(gm)})
            return aot_backend(gm, example_inputs)
        return compiled(gm, example_inputs)

    net = torch.compile(Network(hook), backend=backend, fullgraph=True)
    with torch.inference_mode():
        first = net(torch.tensor([1., 2.]))
        second = net(torch.tensor([3., 4.]))
    return {'aot': aot, 'hook': hook, 'stages': stages, 'calls': list(calls),
            'outputs': [first.tolist(), second.tolist()],
            'passed': len(calls) == 2 and calls[0]['values'] == [2., 4.]
            and calls[1]['values'] == [6., 8.]
            and first.tolist() == [3., 5.] and second.tolist() == [7., 9.]}


def main():
    torch.set_num_threads(1)
    result = {'torch': torch.__version__, 'device': 'cpu',
              'scope': 'Dynamo/AOT node retention with real wrapper/schema; '
              'CPU probe implementation only; NPU/SK untested',
              'source_sha256': {
                  str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (Path(__file__), SOURCE, CPP)}, 'cases': []}
    for aot in (False, True):
        for hook in (False, True):
            try:
                case = run(aot, hook)
            except Exception as error:
                case = {'aot': aot, 'hook': hook, 'passed': False,
                        'error': f'{type(error).__name__}: {error}'}
            result['cases'].append(case)
    result['passed'] = all(case['passed'] for case in result['cases'])
    RESULT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(RESULT), 'passed': result['passed'],
                      'cases': [{key: value for key, value in case.items()
                                 if key not in ('stages', 'calls')}
                                for case in result['cases']]}))


if __name__ == '__main__':
    main()
