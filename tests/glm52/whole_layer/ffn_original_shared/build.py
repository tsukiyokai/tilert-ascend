"""Build the original shared activation with the frozen FFN commands."""

import argparse
import json
from pathlib import Path
import subprocess

from prepare import PARENT_LIBRARY, digest, prepare


def build(parent, output):
    if digest(parent / 'ffn_active_token_join.so') != PARENT_LIBRARY:
        raise ValueError('The active descriptor parent library changed')
    report = prepare(parent, output)
    commands = [json.loads(line) for line in
                (parent / 'commands.jsonl').read_text().splitlines()]
    kernel = next(command for command in commands if command['step'] == 'kernel')['argv']
    original = str(Path(kernel[kernel.index('-o') + 1]).parent)
    for command in commands:
        argv = [arg.replace(original, str(output)).replace(
            '/ffn_active_token_join.so', '/ffn_original_shared.so').replace(
            'SHARED_W8_GROUPED_QUANT', 'SHARED_W8_DYNAMIC_PERFORMANCE_QUANT')
            for arg in command['argv']]
        if any('ROUTED_ENABLE_PROFILE' in arg for arg in argv):
            raise ValueError('The token-output tail requires an uninstrumented build')
        with (output / 'commands.jsonl').open('a') as log:
            log.write(json.dumps({'step': command['step'], 'argv': argv}) + '\n')
        with (output / (command['step'] + '.log')).open('w') as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    for name, expected in report['sources'].items():
        if digest(output / name) != expected:
            raise RuntimeError(f'Source changed during compilation: {name}')
    import torch
    import torch_npu

    torch.ops.load_library(str(parent / 'ffn_active_token_join.so'))
    torch.ops.load_library(str(output / 'ffn_original_shared.so'))
    schemas = {}
    for name in ('prepare', 'forward_out'):
        old = str(getattr(torch.ops.tilert_ffn_active_token_join, name).default._schema)
        new = str(getattr(torch.ops.tilert_ffn_original_shared, name).default._schema)
        if new != old.replace('ffn_active_token_join', 'ffn_original_shared'):
            raise RuntimeError(f'The public schema changed: {name}')
        schemas[name] = new
    if torch.npu.is_initialized():
        raise RuntimeError('The CPU build initialized NPU')
    report.update({
        'namespace': 'tilert_ffn_original_shared',
        'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
        'parent_co_load': True, 'npu_initialized': False, 'schemas': schemas,
        'artifacts': {path.name: digest(path) for path in output.iterdir()
                      if path.suffix in ('.so', '.o')},
        'device_validation': 'not run',
    })
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'pass': True, 'library_sha256':
                     report['artifacts']['ffn_original_shared.so']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.parent.resolve(), args.output.resolve())
