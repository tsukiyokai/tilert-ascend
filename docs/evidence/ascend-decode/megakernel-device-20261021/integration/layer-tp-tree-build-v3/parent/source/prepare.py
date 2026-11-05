"""Freeze the local layer and insert the two validated SHMEM reduction phases."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
from textwrap import dedent


LOCAL_LIBRARY_SHA = "609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f"
COMMUNICATION_SOURCES = {
    "communication.h": "be3b248287e3cb51c288b68f3f466fcb4d4750979f36eee2b7ab9378ff0efff8",
    "communication_device.h": "cf9e9a843c82ff77a4a4c2d6e4ff65024107cbd3c45044280c6502d65be8bde7",
    "communication_host.cpp": "0c1cc5abde6fec3812b844833b7631958a4ae965f2b13fcc7fef1f93167e4701",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError(f"The frozen layer anchor changed: {old[:80]}")
    return source.replace(old, new)


def compose(source):
    source = source.replace("layer_local_chain", "layer_tp_chain")
    source = source.replace("layer_local_indexed_launch", "layer_tp_indexed_launch")
    source = source.replace("layer_local_shared_launch", "layer_tp_shared_launch")
    source = source.replace('"../attention/', '"../local/attention/')
    source = source.replace('"../ffn/', '"../local/ffn/')
    source = replace(source, '#include "contract.h"',
        '#include "contract.h"\n#include "../communication/communication_device.h"')
    helper = dedent('''\
        __aicore__ inline void ReducePhase(TPipe& pipe,
            const Communication& communication, uint64_t partial,
            uint64_t output, int64_t rows, int64_t blocks)
        {
            whole_layer::CommunicationBuffers buffers;
            if ASCEND_IS_AIV {
                buffers.Init(pipe);
            }
            whole_layer::AllReduceBf16(buffers,
                (__gm__ uint8_t*)communication.window,
                (__gm__ bfloat16_t*)partial, (__gm__ bfloat16_t*)output,
                rows, communication.capacity, communication.rank,
                communication.ranks, GetBlockIdx(), 2 * blocks);
        }

        ''')
    source = replace(source, 'template <bool indexed>\n',
                     helper + 'template <bool indexed>\n')
    source = replace(source,
        'void Kernel(uint64_t sync, Config config, Addresses addresses)',
        'void Kernel(uint64_t sync, Config config, Addresses addresses,\n'
        '            Communication communication)')
    source = replace(source,
        '    // The attention callable publishes its GM outputs and resets the pipe.\n',
        '    ReducePhase(pipe, communication, addresses.attention.output,\n'
        '                communication.attention_sum, rows, blocks);\n'
        '    whole_layer_attention::EndPhase(pipe);\n\n')
    source = replace(source, 'ffn_bridge::Forward(pipe, attention.output,',
                     'ffn_bridge::Forward(pipe, (GM_ADDR)communication.attention_sum,')
    source = replace(source,
        '    // The next consumer is a separate task on the same stream.\n',
        '    // FFN now has an in-kernel consumer: publish every producer and\n'
        '    // reset the pipe before allocating the communication buffers.\n'
        '    whole_layer_attention::EndPhase(pipe);\n'
        '    ReducePhase(pipe, communication, addresses.bridge[ffn_bridge::kCombined],\n'
        '                communication.output, rows, blocks);\n')
    source = replace(source,
        '    const layer_tp_chain::Addresses* addresses)\n',
        '    const layer_tp_chain::Addresses* addresses,\n'
        '    const layer_tp_chain::Communication* communication)\n', count=2)
    source = replace(source,
        'if (config == nullptr || addresses == nullptr ||',
        'if (config == nullptr || addresses == nullptr || communication == nullptr ||',
        count=2)
    source = replace(source,
        '!layer_tp_chain::Valid(*config, *addresses)',
        '!layer_tp_chain::Valid(*config, *addresses, *communication)', count=2)
    source = replace(source, 'sync, *config, *addresses);',
                     'sync, *config, *addresses, *communication);', count=2)
    return source


def prepare(local, output):
    build = json.loads((local / 'results/build.json').read_text())
    if (build['library_sha256'] != LOCAL_LIBRARY_SHA
            or digest(local / 'results/layer_local_chain.so') != LOCAL_LIBRARY_SHA):
        raise ValueError('Use the frozen 609c973e local layer library')
    manifest = json.loads((local / 'manifest.json').read_text())
    if digest(local / 'manifest.json') != build['parent_manifest_sha256']:
        raise ValueError('The local layer source manifest changed')
    for name, expected in manifest['sources'].items():
        if digest(local / name) != expected:
            raise ValueError(f'The local layer source changed: {name}')
    directory = Path(__file__).resolve().parent
    for name, expected in COMMUNICATION_SOURCES.items():
        if digest(directory.parent / name) != expected:
            raise ValueError(f'The accepted communication source changed: {name}')
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest['sources']:
        target = output / 'local' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local / name, target)
    shutil.copy2(local / 'manifest.json', output / 'local-manifest.json')
    communication = output / 'communication'
    communication.mkdir()
    for name in COMMUNICATION_SOURCES:
        shutil.copy2(directory.parent / name, communication / name)
    source = output / 'source'
    source.mkdir()
    for name in ('contract.h', 'contract_test.cpp', 'prepare.py', 'build.py',
                 'host.cpp'):
        shutil.copy2(directory / name, source / name)
    original = (local / 'source/kernel.cpp').read_text()
    combined = compose(original)
    (source / 'kernel.cpp').write_text(combined)
    (output / 'communication.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), combined.splitlines(keepends=True),
        fromfile='local/kernel.cpp', tofile='tp/kernel.cpp')))
    report = {
        'scope': 'One layer launch with input norm, attention, AR1, complete FFN and AR2',
        'local_library_sha256': LOCAL_LIBRARY_SHA,
        'communication_sources': COMMUNICATION_SOURCES,
        'communication_order': 'Ascending rank FP32 accumulation and one BF16 rounding',
        'communication_ub_bytes': 20480,
        'kernel_argument_bytes': 1648,
        'device_execution': False,
        'sources': {str(path.relative_to(output)): digest(path)
                    for path in sorted(output.rglob('*')) if path.is_file()},
    }
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'sources': len(report['sources']), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('local', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.local.resolve(), args.output.resolve())
