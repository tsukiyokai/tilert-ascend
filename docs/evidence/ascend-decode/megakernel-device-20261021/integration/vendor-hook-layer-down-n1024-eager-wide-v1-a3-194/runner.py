"""Run an immutable local-layer fixture after a fresh idle-die admission."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
TASK = '/home/shanshan/tilert-megakernel-20261021'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('eager', 'graph'))
    args = parser.parse_args()
    handoff = json.loads((ROOT / 'layer-down-n1024-handoff.json').read_text())
    package = handoff['packages'][args.stage]
    local = Path(package['local'])
    remote = package['remote']
    name = 'vendor-hook-layer-down-n1024-' + args.stage + '-wide-v1'

    spec = importlib.util.spec_from_file_location('idle_gate', ROOT / 'two-die-idle-gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.SSH[-1] = 'a3-194'
    gate.DEVICES = (4,)
    gate.LOG = ROOT / (name + '-admission.jsonl')

    assert digest(local / 'manifest.json') == package['manifest_sha256']
    frozen = {name: sha for sha, name in
              (line.split('  ', 1) for line in
               (local / 'frozen-inputs.sha256').read_text().splitlines())}
    assert len(frozen) == package['files_verified']
    for filename, expected in frozen.items():
        assert digest(local / filename) == expected, filename
    assert frozen['complete_ffn.so'] == handoff['ffn_library_sha256']
    assert frozen['layer_local_chain.so'] == handoff['chain_library_sha256']
    assert frozen['local_chain_host.py'] == '70bfd7393e7ceb2a6921100be6057f0cea3f922c52d6e3df2e02ea398c66e6f9'

    cpu_path = Path(package['cpu_report'])
    assert digest(cpu_path) == package['cpu_report_sha256']
    cpu = json.loads(cpu_path.read_text())
    assert cpu['pass'] and not cpu['device_execution']
    assert not cpu['npu_initialized_before'] and not cpu['npu_initialized_after']
    assert cpu['mapped_davinci_devices'] == []
    assert cpu['torch'] == '2.12.0+cu130' and cpu['torch_npu'] == '2.12.0'
    assert cpu['contracts_run'] == package['contracts_run']
    assert len(cpu['cli_controls']) == package['cli_controls']
    assert len(cpu['libraries']) == 4

    verify = (
        'from pathlib import Path; import hashlib,json; '
        f'p=Path({remote!r}); identities=json.loads({json.dumps(frozen)!r}); '
        'bad=[n for n,h in identities.items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h]; '
        'assert not bad,bad; print("Verified frozen fixture files",len(identities))')
    subprocess.run(gate.SSH + ['python3 -c ' + shlex.quote(verify)], check=True)
    oracle = json.loads(Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-bridge-build-v2/oracle-package-v4/manifest.json').read_text())
    verify = (
        'import hashlib,json; from pathlib import Path; '
        f'p=Path({(TASK + "/tests/glm52/whole_layer/ffn_bridge/oracle-package-v4/install")!r}); '
        f'expected=json.loads({json.dumps(oracle["installed_sha256"])!r}); '
        'bad=[n for n,h in expected.items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h]; '
        'assert not bad,bad; print("Verified oracle files",len(expected))')
    subprocess.run(gate.SSH + ['python3 -c ' + shlex.quote(verify)], check=True)
    for filename, expected in {
        'whole_layer_indexed_attention.so': '8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f',
        'manifest.json': 'db2f5af406d2bfd728f2e6f4550b336e0b8a3d1aee9682fe06b2815f6298ece9',
    }.items():
        path = TASK + '/logs/attention-wk-m1-build-v1/' + filename
        actual = subprocess.check_output(gate.SSH + ['sha256sum', path], text=True).split()[0]
        assert actual == expected, filename
    memory = subprocess.check_output(gate.SSH + ['cat', '/proc/meminfo'], text=True)
    available = int(next(line.split()[1] for line in memory.splitlines()
                         if line.startswith('MemAvailable:')))
    assert available > 96 * 1024 * 1024, available
    (ROOT / (name + '.host-memory.json')).write_text(json.dumps({
        'utc_epoch': time.time(), 'available_kib': available,
        'container_limit_gib': 64}, indent=2) + '\n')
    assert gate.vacant()
    time.sleep(5)
    assert gate.vacant()
    command = package['wide_argv'].copy()
    command[3] = name
    command += ['--mode', 'int8', '--graph-residual-modes', 'provided']
    (ROOT / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
    print('Starting 609 plus TopK copy and N1024 ' + args.stage + ' on physical die4', flush=True)
    with (ROOT / (name + '.launcher.log')).open('w') as log:
        result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=log,
                                stderr=subprocess.STDOUT)
    print('Local layer ' + args.stage + ' device exit', result.returncode, flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
