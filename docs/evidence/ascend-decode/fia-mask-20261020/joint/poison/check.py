import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

torch.set_num_threads(1)
source = Path(__file__).with_name('run_cann_mtp_rounds.py')
tree = ast.parse(source.read_text())
loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
            and isinstance(n.target, ast.Tuple)
            and [getattr(t, 'id', None) for t in n.target.elts]
            == ['layers', 'begin', 'dense_mla'])
code = compile(ast.Module(body=[loop], type_ignores=[]), str(source), 'exec')
checks = []
for mode in ('sfa', 'gather_fia', 'shared_fia'):
    def cache():
        return {key: torch.zeros(10, 4, dtype=dtype) for key, dtype in
                (('latent', torch.bfloat16), ('rope', torch.bfloat16),
                 ('index_keys', torch.int8), ('index_scales', torch.float16))}
    target = SimpleNamespace(cache=[cache(), cache()], attention_kernel=mode)
    draft = SimpleNamespace(cache=cache())
    exec(code, {'target': target, 'draft': draft, 'initial': 2,
                'args': SimpleNamespace(max_tokens=2)})
    for kind, layers, begin in [('main', target.cache, 2),
                                ('draft', [draft.cache], 1)]:
        for layer, fields in enumerate(layers):
            for field, value in fields.items():
                tail = value[begin:8]
                assert torch.count_nonzero(value[:begin]) == 0
                assert torch.count_nonzero(value[8:]) == 0
                if not value.is_floating_point():
                    good = bool((tail == -128).all())
                elif mode == 'shared_fia' and kind == 'main' and field in ('latent', 'rope'):
                    good = bool((tail == 7.0).all())
                else:
                    good = bool(torch.isnan(tail).all())
                assert good, (mode, kind, layer, field)
                checks.append([mode, kind, layer, field])
report_checks = 0
for node in ast.walk(tree):
    if not isinstance(node, ast.Dict):
        continue
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and key.value == 'serial_speculative_consistency':
            expression = compile(ast.Expression(value), str(source), 'eval')
            for runs, passed, expected in [({}, True, None),
                                           ({'serial': {}}, True, True),
                                           ({'serial': {}}, False, False)]:
                actual = eval(expression, {'runs': runs, 'passed': passed, 'success': passed})
                assert actual is expected
                report_checks += 1
assert report_checks == 6
result = {'report_checks': report_checks, 'status': 'pass', 'checks': len(checks), 'source_sha256':
          hashlib.sha256(source.read_bytes()).hexdigest(),
          'scope': 'Server CPU execution of actual poison loop; prefix and suffix preserved'}
Path(__file__).with_name('result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
