"""Single-core CPU comparison of frozen checkpoint reads and scoped shard handles."""
from contextlib import closing, contextmanager
import ast
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
from types import ModuleType, SimpleNamespace

import safetensors
import torch

ROOT = Path('/tmp/layer-major-m32-20261020')
SOURCE = ROOT / 'loader-actual-weights.py'
CHECKPOINT = Path('/workspace/models/GLM-5.2-w4a8c8')
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location('frozen_weights', SOURCE)
weights = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = weights
spec.loader.exec_module(weights)
checkpoint = weights.ModelSlimCheckpoint(CHECKPOINT)
metadata = json.dumps([checkpoint.weight_map, checkpoint.quantization], sort_keys=True)
names = sorted(name for name in checkpoint.weight_map
               if any(name.startswith(f'model.layers.3.mlp.experts.{i}.') for i in (0, 1, 2)))
shards = sorted({checkpoint.weight_map[name] for name in names})
assert len(names) == 36
counts = {'opened': 0, 'closed': 0, 'active': 0}


@contextmanager
def counted(*args, **kwargs):
    with safetensors.safe_open(*args, **kwargs) as handle:
        counts['opened'] += 1
        counts['active'] += 1
        try:
            yield handle
        finally:
            counts['closed'] += 1
            counts['active'] -= 1


def fd_count():
    return len(list(Path('/proc/self/fd').iterdir()))


def maps_count():
    return sum(any(shard in row for shard in shards)
               for row in Path('/proc/self/maps').read_text().splitlines())


def read_batch(arm):
    if arm == 'A':
        with patch.object(weights, 'safe_open', counted):
            return [checkpoint.tensor(name) for name in names]
    with patch.object(weights, 'safe_open', counted):
        with closing(checkpoint.tensors(names)) as iterator:
            pairs = list(iterator)
    assert [name for name, _ in pairs] == names
    return [value for _, value in pairs]


reference = read_batch('A')
records = []
for arm in ('A', 'B', 'B', 'A'):
    gc.collect()
    before_fd, before_maps = fd_count(), maps_count()
    counts.update(opened=0, closed=0, active=0)
    start_cpu, start_wall = time.process_time_ns(), time.perf_counter_ns()
    values = read_batch(arm)
    wall_ms = (time.perf_counter_ns() - start_wall) / 1e6
    cpu_ms = (time.process_time_ns() - start_cpu) / 1e6
    equal = [a.shape == b.shape and a.dtype == b.dtype
             and a.stride() == b.stride() and a.is_contiguous() and a._base is None
             and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
             for a, b in zip(values, reference)]
    assert len(equal) == len(names) and all(equal)
    assert len({v.untyped_storage().data_ptr() for v in values}) == len(values)
    assert counts['active'] == 0 and counts['opened'] == counts['closed']
    assert counts['opened'] == (len(names) if arm == 'A' else len(shards))
    assert fd_count() == before_fd and maps_count() == before_maps
    assert metadata == json.dumps([checkpoint.weight_map, checkpoint.quantization], sort_keys=True)
    records.append({'arm': arm, 'wall_ms': wall_ms, 'cpu_ms': cpu_ms,
                    **counts, 'all_bit_equal': True, 'metadata_equal': True,
                    'owned_storage': True, 'name_order_equal': True,
                    'fd_before_after': [before_fd, fd_count()],
                    'shard_maps_before_after': [before_maps, maps_count()]})
    del values

cleanup = []
before_fd, before_maps = fd_count(), maps_count()
counts.update(opened=0, closed=0, active=0)
with patch.object(weights, 'safe_open', counted):
    iterator = checkpoint.tensors(names)
    first_name, first_value = next(iterator)
    assert counts['active'] == 1 and iterator.gi_frame is not None
    iterator.close()
    assert iterator.gi_frame is None and counts['active'] == 0
    assert counts['opened'] == counts['closed'] == 1
    assert fd_count() == before_fd and maps_count() == before_maps
    assert first_name == names[0] and first_value._base is None
    assert torch.equal(first_value.view(torch.uint8), reference[0].view(torch.uint8))
cleanup.append({'case': 'early iterator.close with clone and iterator retained',
                'passed': True, **counts, 'fd_before_after': [before_fd, fd_count()],
                'shard_maps_before_after': [before_maps, maps_count()]})

tree = ast.parse((ROOT / 'loader-actual-streamed.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_scope')
namespace = {'closing': closing}
exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual-load-scope', 'exec'), namespace)


class DeliberateFailure(Exception):
    pass


class FailingModel:
    def __init__(self):
        self.model = SimpleNamespace()
        self.iterator = None

    def named_parameters(self):
        return [(name, None) for name in names]

    def load_weights(self, iterator):
        self.iterator = iterator
        self.first = next(iterator)
        raise DeliberateFailure('consumer stops after one actual tensor')


modules = {name: ModuleType(name) for name in (
    'vllm', 'vllm.model_executor', 'vllm.model_executor.models',
    'vllm_ascend', 'vllm_ascend.ops', 'vllm_ascend.ops.layernorm')}
modules['vllm.model_executor.models'].utils = SimpleNamespace(
    _model_to_pp_missing_layer_names={})
modules['vllm_ascend.ops.layernorm'].AscendRMSNorm = type('AscendRMSNorm', (), {})
model = FailingModel()
before_fd, before_maps = fd_count(), maps_count()
counts.update(opened=0, closed=0, active=0)
with patch.dict(sys.modules, modules), patch.object(weights, 'safe_open', counted):
    try:
        namespace['load_scope'](SimpleNamespace(model=model), checkpoint,
                                tuple(f'model.layers.3.mlp.experts.{i}.' for i in (0, 1, 2)))
        raise AssertionError('consumer did not raise')
    except DeliberateFailure:
        pass
assert model.iterator is not None and model.iterator.gi_frame is None
assert counts['active'] == 0 and counts['opened'] == counts['closed'] == 1
assert fd_count() == before_fd and maps_count() == before_maps
assert model.first[0] == names[0] and model.first[1]._base is None
assert torch.equal(model.first[1].view(torch.uint8), reference[0].view(torch.uint8))
cleanup.append({'case': 'actual load_scope AST, consumer raises; iterator retained',
                'passed': True, **counts, 'generator_frame': None,
                'fd_before_after': [before_fd, fd_count()],
                'shard_maps_before_after': [before_maps, maps_count()]})
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
    SOURCE, ROOT / 'loader-actual-bench.py', ROOT / 'loader-actual-streamed.py',
    CHECKPOINT / 'quant_model_weights.safetensors.index.json',
    CHECKPOINT / 'quant_model_description.json')}
result = {'scope': 'CPU loader only; cached pages; no model/NPU/decode timing',
          'source_sha256': hashes, 'names': names, 'shards': shards,
          'tensor_metadata': [{'name': n, 'shape': list(t.shape), 'dtype': str(t.dtype),
                               'bytes': t.numel() * t.element_size()}
                              for n, t in zip(names, reference)],
          'torch_threads': torch.get_num_threads(), 'affinity': sorted(os.sched_getaffinity(0)),
          'records': records, 'cleanup_checks': cleanup}
(ROOT / 'loader-actual-result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'tensors': len(names), 'bytes': sum(t.numel() * t.element_size() for t in reference),
                  'shards': len(shards), 'records': records, 'cleanup_checks': cleanup}))
