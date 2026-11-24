import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[3]
PATCH = ROOT / 'tilert/models/glm_5_2_ascend/moe_combine.py'
NAMES = ('_update_indexcache_topk_indices', '_get_indexcache_topk_indices')


def methods(path, owner):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if getattr(node, 'name', None) == owner)
    return [node for node in node.body
            if isinstance(node, ast.FunctionDef) and node.name in NAMES]


def bind(nodes, path, namespace):
    for node in nodes:
        module = ast.Module(body=[node], type_ignores=[])
        exec(compile(module, str(path), 'exec'), namespace)
    return [namespace[name] for name in NAMES]


def context_proxy(path, v2):
    context = SimpleNamespace(additional_kwargs={})
    namespace = {
        'Any': Any,
        'envs_vllm': SimpleNamespace(VLLM_USE_V2_MODEL_RUNNER=v2),
        'get_forward_context': lambda: context,
    }
    node = next(node for node in ast.parse(path.read_text()).body
                if getattr(node, 'name', None) == '_ExtraForwardContextProxy')
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['_ExtraForwardContextProxy']()


def check(enabled, dimensions, v2, originals, nodes, context_path, legacy=False):
    context = context_proxy(context_path, v2)
    namespace = {
        'skip_topk_copy': enabled,
        'last_topk_by_buf': {},
        'orig_update_topk': originals[0],
        'orig_get_topk': originals[1],
        '_EXTRA_CTX': context,
    }
    nodes = copy.deepcopy(nodes)
    if legacy:
        for node in nodes:
            for expression in ast.walk(node):
                if isinstance(expression, ast.BoolOp):
                    expression.values = [
                        value for value in expression.values
                        if ast.unparse(value) != 'not _EXTRA_CTX.is_draft_model'
                    ]
    update, get = bind(nodes, PATCH, namespace)
    main = SimpleNamespace(topk_indices_buffer=torch.full(
        (32, 2048), -1, dtype=torch.int32,
    ))
    draft = SimpleNamespace(topk_indices_buffer=torch.full(
        (32, 2048), -1, dtype=torch.int32,
    ))
    fresh = torch.arange(8192, dtype=torch.int32).reshape(4, 2048)
    tensor = fresh.unsqueeze(1) if dimensions == 3 else fresh
    context.is_draft_model = False
    update(main, tensor)
    main_matches = torch.equal(get(main, 4).squeeze(1), fresh)
    main_cached = main.topk_indices_buffer.data_ptr() in namespace['last_topk_by_buf']
    main_uncopied = bool(torch.all(main.topk_indices_buffer == -1))
    context.is_draft_model = True
    update(draft, tensor)
    exported = draft.topk_indices_buffer[:4].clone()
    exported_matches = torch.equal(exported, fresh)
    selections = []
    for row in (0, 2):
        wanted = fresh[row:row + 1].clone()
        draft.topk_indices_buffer[:1].copy_(wanted)
        consumed = get(draft, 1).squeeze(1)
        selections.append({
            'row': row,
            'matches': torch.equal(consumed, wanted),
            'expected_first_index': int(wanted[0, 0]),
            'consumed_first_index': int(consumed[0, 0]),
        })
    context.is_draft_model = False
    main_return_matches = torch.equal(get(main, 4).squeeze(1), fresh)
    assert main_matches and main_return_matches
    assert main_cached == enabled
    assert main_uncopied == enabled
    assert exported_matches == (not (legacy and enabled))
    assert selections[0]['matches']
    assert selections[1]['matches'] == (not (legacy and enabled))
    return {
        'skip_topk_copy': enabled,
        'index_dimensions': dimensions,
        'v2_context': v2,
        'legacy_guard_removed_in_memory': legacy,
        'main_read_matches': main_matches,
        'main_cached': main_cached,
        'main_buffer_uncopied': main_uncopied,
        'cann_mtp_exported_indices_match': exported_matches,
        'explicit_selection': selections,
        'return_to_main_matches': main_return_matches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    vendor = runtime / 'attention/sfa_v1.py'
    context = runtime / 'ascend_forward_context.py'
    originals = bind(methods(vendor, 'AscendSFAImpl'), vendor, {'torch': torch})
    nodes = methods(PATCH, '_install_sfa_superkernel_fusion')
    report = {
        'status': 'draft_fix_and_main_preservation_passed',
        'scope': 'Current source methods on CPU tensors; no NPU or model forward',
        'runtime_source': {
            'root': str(runtime),
            'context_field': 'is_draft_model',
        },
        'torch': torch.__version__,
        'sources': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (PATCH, vendor, context, Path(__file__).resolve())},
        'cases': [check(enabled, dimensions, v2, originals, nodes, context)
                  for enabled in (True, False) for dimensions in (2, 3)
                  for v2 in (False, True)],
        'negative_controls': [
            check(True, dimensions, False, originals, nodes, context, legacy=True)
            for dimensions in (2, 3)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({
        'status': report['status'],
        'cases': len(report['cases']),
        'negative_controls': len(report['negative_controls']),
        'output': str(args.output),
    }))


if __name__ == '__main__':
    main()
