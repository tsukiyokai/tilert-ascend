"""Compose the verified token-output tail with the active descriptor FFN."""

import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = 'f7621e49a818a77d7377246f74c7feaa9a5b8cb9a7b81feacc4cf195d1bd1696'
JOIN_LIBRARY = '4417fd158e48a2da4d59e61fec48ebd0dd6e113022f13cbeb4f01d596eead757'
TRANSFORM = '41f2edfe75f5bb6aa24eded9fc4277450ac8310551cb4997f77f5426b4e2781b'
CHECKER = '605674f60505600c189e7ce7c07da39daca2aa83540016a361855f491c1c99b8'
BODY = 'body/source/routed_int4/routed_body_native.h'
STAGE_FIVE = '    timing.Barrier(4);\n    timing.Begin(5);\n'
BODY_MARKER = 'template <CubeFormat weight_format, bool from_dispatch>\n'
NAMESPACE = 'ffn_active_token_join'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f'Unexpected frozen seam: {before!r}')
    return source.replace(before, after, 1)


def rename(source):
    return (source.replace('ffn_active_descriptor', NAMESPACE)
            .replace('FFN_ACTIVE_DESCRIPTOR', 'FFN_ACTIVE_TOKEN_JOIN')
            .replace('ffn_token_join', NAMESPACE)
            .replace('FFN_TOKEN_JOIN', 'FFN_ACTIVE_TOKEN_JOIN'))


def verify(build, library_name, identity):
    manifest = json.loads((build / 'manifest.json').read_text())
    if manifest['artifacts'][library_name] != identity:
        raise ValueError(f'Unexpected frozen parent: {library_name}')
    for name, expected in manifest['sources'].items():
        if digest(build / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    return manifest


def bind_body(source, join_source, transform):
    source = replace_once(source, '#include "routed_int4_kernel.h"',
                          '#include "routed_int4_kernel.h"\n'
                          '#include "../../../source/output_contract.h"')
    source = replace_once(source, '}  // namespace\n\n' + BODY_MARKER,
                          '#include "../../../source/output_native.h"\n\n'
                          '}  // namespace\n\n' + BODY_MARKER)
    source = replace_once(source,
        '    TPipe& pipe, const ffn_active_descriptor::BodyArguments& arguments)',
        '    TPipe& pipe, const ffn_active_descriptor::BodyArguments& arguments,\n'
        '    __gm__ bfloat16_t* scaled_output, __gm__ bfloat16_t* combined_output)')
    join_start = join_source.index(STAGE_FIVE) + len(STAGE_FIVE)
    join_end = join_source.index('\n}\n\n}  // namespace routed_body', join_start)
    tail = join_source[join_start:join_end]
    tail = transform.TOKEN.sub(
        lambda match: transform.BINDINGS.get(match[0], match[0]), tail)
    start = source.index(STAGE_FIVE) + len(STAGE_FIVE)
    end = source.index('\n}\n\n}  // namespace routed_body', start)
    source = source[:start] + tail + source[end:]
    return replace_once(source, '#define TILERT_ROUTED_BODY_NATIVE_H\n\n',
        '#define TILERT_ROUTED_BODY_NATIVE_H\n\n'
        '#ifdef ROUTED_ENABLE_PROFILE\n'
        '#error "Token tail join requires the uninstrumented NoTrace body"\n'
        '#endif\n\n')


def bind_kernel(source):
    source = replace_once(source,
        '    routed_body::Forward<format, true>(pipe, arguments);',
        '    routed_body::Forward<format, true>(pipe, arguments,\n'
        '        (__gm__ bfloat16_t*)bridge.values[ffn_bridge::kScaledRouted],\n'
        '        (__gm__ bfloat16_t*)bridge.values[ffn_bridge::kCombined]);')
    start = source.index('    ffn_bridge::device::Epilogue(pipe,')
    end = source.index('\n}\n}  // namespace ffn_active_descriptor_device', start)
    if source[start:end].count('ffn_bridge::device::EndPhase(pipe);') != 1:
        raise ValueError('Expected the epilogue and its final join')
    return source[:start] + source[end:]


def prepare(parent, join, output):
    manifest = verify(parent, 'ffn_active_descriptor.so', PARENT_LIBRARY)
    verify(join, 'ffn_token_join.so', JOIN_LIBRARY)
    if digest(parent / 'check.py') != CHECKER:
        raise ValueError('The active-table numerical checker changed')
    transform_path = parent / 'source/descriptor_transform.py'
    if digest(transform_path) != TRANSFORM:
        raise ValueError('The verified argument substitution changed')
    spec = importlib.util.spec_from_file_location('frozen_descriptor', transform_path)
    transform = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transform)
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest['sources']:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    before = {name: (parent / name).read_text() for name in (BODY, 'source/kernel.cpp')}
    (output / BODY).write_text(bind_body(before[BODY], (join / BODY).read_text(), transform))
    (output / 'source/kernel.cpp').write_text(bind_kernel(before['source/kernel.cpp']))
    for name in ('output_native.h', 'output_contract.h'):
        text = (join / 'source' / name).read_text()
        if name == 'output_contract.h':
            text = text.replace('112352 + kOutputMetadataBytes',
                                '112352 + 1056 + kOutputMetadataBytes')
            text = text.replace('kOutputResidentBytes == 115424',
                                'kOutputResidentBytes == 116480')
        (output / 'source' / name).write_text(text)
    for path in output.rglob('*'):
        if path.suffix in ('.h', '.cpp'):
            path.write_text(rename(path.read_text()))
    (output / 'check.py').write_text((parent / 'check.py').read_text().replace(
        'torch.ops.tilert_ffn_active_descriptor', 'torch.ops.tilert_' + NAMESPACE))
    patch = []
    for name, source in before.items():
        patch.extend(difflib.unified_diff(
            source.splitlines(keepends=True),
            (output / name).read_text().splitlines(keepends=True),
            fromfile='parent/' + name, tofile='candidate/' + name))
    (output / 'token-tail.patch').write_text(''.join(patch))
    tools = output / 'candidate_tools'
    tools.mkdir()
    for name in ('prepare.py', 'build.py', 'test_prepare.py', 'README.md'):
        shutil.copy2(Path(__file__).with_name(name), tools / name)
    shutil.copy2(parent / 'manifest.json', output / 'parent-manifest.json')
    shutil.copy2(join / 'manifest.json', output / 'token-join-manifest.json')
    report = {
        'parent_library_sha256': PARENT_LIBRARY,
        'token_join_library_sha256': JOIN_LIBRARY,
        'parent_checker_sha256': CHECKER,
        'descriptor_bytes': 408, 'public_scratch_count': 33,
        'additional_gm_bytes_vs_parent': 0,
        'additional_ub_bytes_vs_parent': 3072,
        'maximum_resident_ub_bytes': 116480,
        'body_barriers': 5, 'outer_end_phases': 1,
        'profile_supported': False,
        'sources': {str(path.relative_to(output)): digest(path)
                    for path in sorted(output.rglob('*')) if path.is_file()},
    }
    (output / 'source-transforms.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('join', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.join.resolve(), args.output.resolve())
