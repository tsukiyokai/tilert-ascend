import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch

base = Path(__file__).resolve().parent
source = base / 'tree/tests/glm52/run_cann_mtp_rounds.py'
tree = ast.parse(source.read_text())
branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
              and ast.unparse(node.test) == "label == 'graph'")
code = compile(ast.Module(body=branch.body[:2], type_ignores=[]), str(source), 'exec')
torch.set_num_threads(1)
checks = 0
for width in (8, 32):
    end = 2 + 4 + width - 2
    for mode in ('sfa', 'shared_fia'):
        def cache():
            return {name: torch.zeros(end + 1, 4, dtype=dtype)
                    for name, dtype in [('latent', torch.bfloat16),
                                        ('rope', torch.bfloat16),
                                        ('index_keys', torch.int8),
                                        ('index_scales', torch.float16)]}
        target = SimpleNamespace(cache=[cache()], attention_kernel=mode)
        draft = SimpleNamespace(cache=cache())
        exec(code, dict(target=target, draft=draft, initial=2,
                        args=SimpleNamespace(width=width, max_tokens=4)))
        for kind, fields, begin in [('main', target.cache[0], 2),
                                    ('draft', draft.cache, 1)]:
            for field, value in fields.items():
                assert torch.count_nonzero(value[:begin]) == 0
                assert torch.count_nonzero(value[end:]) == 0
                tail = value[begin:end]
                if not value.is_floating_point():
                    assert (tail == -128).all()
                elif mode == 'shared_fia' and kind == 'main' and field in ('latent', 'rope'):
                    assert (tail == 7).all()
                else:
                    assert torch.isnan(tail).all()
                checks += 1
assert 'torch_npu' not in sys.modules
result = dict(status='pass', field_range_checks=checks, widths=[8, 32],
              npu_imported=False, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
(base / 'poison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
