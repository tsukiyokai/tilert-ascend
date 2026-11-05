"""Check PR 1955's collection boundary without importing Torch or executing NPU calls."""

import ast
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace


ROOT = Path('/tmp/trt-carver-ci-1955-20261029')
BEFORE = Path('/tmp/trt-carver-1955-before-test.py')
AFTER = ROOT / 'testing/python/carver/test_ascend_specs.py'

COLLECT = r'''
import importlib.abc
import sys
import types
import pytest

# Skip unrelated native TileLang initialization; reject the Carver import itself.
package = types.ModuleType("tilelang")
package.__path__ = []
sys.modules["tilelang"] = package

class RejectCarver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("tilelang.carver"):
            raise RuntimeError("CARVER_IMPORTED_DURING_COLLECTION: " + fullname)

sys.meta_path.insert(0, RejectCarver())
raise SystemExit(pytest.main([
    "-c", "/dev/null", "-o", "addopts=", "--collect-only", "-q", sys.argv[1]
]))
'''


def collect(path):
    return subprocess.run(
        [sys.executable, '-B', '-c', COLLECT, str(path)],
        capture_output=True, text=True, check=False,
    )


with tempfile.TemporaryDirectory(prefix='trt-carver-1955-collect-') as temporary:
    before = Path(temporary) / 'test_ascend_specs.py'
    before.write_bytes(BEFORE.read_bytes())
    old = collect(before)
    new = collect(AFTER)
    assert old.returncode == 2 and 'CARVER_IMPORTED_DURING_COLLECTION' in old.stdout
    assert new.returncode == 0 and '6 tests collected' in new.stdout
    print('BEFORE collection: rejected Carver import, exit=2')
    print('AFTER collection: no Carver import, exit=0, 6 tests collected')


class RemoveImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None


assert ast.dump(RemoveImports().visit(ast.parse(BEFORE.read_text()))) == ast.dump(
    RemoveImports().visit(ast.parse(AFTER.read_text()))
)
print('Test/fixture expressions, assertions and parametrization unchanged')

# Execute the actual class definition to demonstrate import-time default evaluation.
# Dependencies are CPU placeholders; the called architecture function is the real source.
arch_path = ROOT / 'tilelang/carver/arch/__init__.py'
arch_tree = ast.parse(arch_path.read_text())
auto = next(node for node in arch_tree.body if isinstance(node, ast.FunctionDef)
            and node.name == 'auto_infer_current_arch')
events = []


def get_arch(target):
    events.append(target)
    return object()


namespace = {
    'TileDevice': object, 'get_arch': get_arch,
    'torch': SimpleNamespace(version=SimpleNamespace(hip=None),
                             npu=SimpleNamespace(is_available=lambda: True)),
}
exec(compile(ast.Module(body=[auto], type_ignores=[]), str(arch_path), 'exec'), namespace)
base_path = ROOT / 'tilelang/carver/template/base.py'
base_tree = ast.parse(base_path.read_text())
base = next(node for node in base_tree.body if isinstance(node, ast.ClassDef)
            and node.name == 'BaseTemplate')
namespace.update(ABC=object, dataclass=dataclass, field=field,
                 abstractmethod=lambda value: value, PrimFunc=object,
                 OutputNode=object, Hint=object, List=list)
exec(compile(ast.Module(body=[base], type_ignores=[]), str(base_path), 'exec'), namespace)
assert events == ['ascend']
print('Actual BaseTemplate class + auto_infer_current_arch source: get_arch(ascend) at definition')
