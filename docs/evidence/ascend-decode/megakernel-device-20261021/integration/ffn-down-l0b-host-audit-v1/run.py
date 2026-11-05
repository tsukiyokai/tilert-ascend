"""Inspect five public host-tiler requests in a container without devices."""

import hashlib
import json
from pathlib import Path
import shlex
import subprocess


ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
LOCAL = ROOT / 'ffn-down-l0b-host-audit-v1'
REMOTE = '/home/shanshan/tilert-megakernel-20261021/ffn-down-l0b-host-audit-v1'
SDK = '/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'a3-78']
IMAGE = 'sha256:7448df539bcb40397063c4b7e83a6a1590ba20bcc16639c0aa83f1c60f1a0449'


def main():
    source = LOCAL / 'tiling_probe.cpp'
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    subprocess.run(SSH + ['mkdir', '-p', shlex.quote(REMOTE)], check=True)
    subprocess.run(['scp', '-q', str(source), 'a3-78:' + REMOTE + '/'],
                   check=True)
    verify = (
        'from pathlib import Path; import hashlib; '
        f'p=Path({REMOTE!r})/"tiling_probe.cpp"; '
        f'assert hashlib.sha256(p.read_bytes()).hexdigest()=={source_sha!r}')
    subprocess.run(SSH + ['python3 -c ' + shlex.quote(verify)], check=True)
    script = r'''set -eo pipefail
source /opt/ascend/cann-9.2.0/set_env.sh
set -u
python3 - <<'CHECK'
from pathlib import Path
import hashlib,json
devices=[str(p) for p in Path('/dev').iterdir()
         if p.name.startswith(('davinci','devmm','hisi'))]
assert not devices,devices
base=Path('/opt/ascend/cann-9.2.0')
files=['aarch64-linux/asc/include/adv_api/matmul/matmul_tiling_base.h',
       'aarch64-linux/lib64/libtiling_api.a']
report={'devices':devices,'host_only':True,'source_sha256':
        hashlib.sha256(Path('/evidence/tiling_probe.cpp').read_bytes()).hexdigest(),
        'files':{name:{'resolved':str((base/name).resolve()),
                       'sha256':hashlib.sha256((base/name).read_bytes()).hexdigest()}
                 for name in files}}
Path('/evidence/environment.json').write_text(json.dumps(report,indent=2)+'\n')
CHECK
g++ -std=c++17 -O2 -I /opt/ascend/cann-9.2.0/include -I /opt/ascend/cann-9.2.0/aarch64-linux/tikcpp/tikcfw -I /opt/ascend/cann-9.2.0/aarch64-linux/tikcpp/tikcfw/lib -I /opt/ascend/cann-9.2.0/aarch64-linux/tikcpp/tikcfw/impl -I /opt/ascend/cann-9.2.0/aarch64-linux/tikcpp/tikcfw/interface /evidence/tiling_probe.cpp -o /evidence/tiling-probe -L /opt/ascend/cann-9.2.0/aarch64-linux/lib64 -L /opt/ascend/cann-9.2.0/lib64 -ltiling_api -lplatform -lunified_dlog -lc_sec -lregister > /evidence/build.log 2>&1
/evidence/tiling-probe > /evidence/tilings.jsonl 2> /evidence/run.stderr
'''
    command = ['docker', 'run', '--rm', '--pull=never', '--read-only',
               '--network', 'none', '--name', 'shanshan-tilert-down-l0b-host-v1',
               '--cpus', '1', '--memory', '4g', '--pids-limit', '64',
               '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=512m',
               '-v', REMOTE + ':/evidence:rw',
               '-v', SDK + ':/opt/ascend/cann-9.2.0:ro',
               '--entrypoint', '/bin/bash', IMAGE, '-lc', script]
    (LOCAL / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (LOCAL / 'launcher.log').open('w') as log:
        result = subprocess.run(SSH + [shlex.join(command)], stdout=log,
                                stderr=subprocess.STDOUT)
    for name in ('environment.json', 'build.log', 'run.stderr', 'tilings.jsonl'):
        subprocess.run(['scp', '-q', 'a3-78:' + REMOTE + '/' + name,
                        str(LOCAL / name)], check=False)
    print(json.dumps({'exit_code': result.returncode, 'output': str(LOCAL)}),
          flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
