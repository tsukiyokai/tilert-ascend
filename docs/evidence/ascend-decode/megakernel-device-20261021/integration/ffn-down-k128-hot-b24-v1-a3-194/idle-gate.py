"""Record admission for two idle A3 dies before an isolated device run."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess


SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', 'a3-194']
ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
LOG = ROOT / 'two-die-admission.jsonl'
DEVICES = (6, 7)


def run(command):
    result = subprocess.run(SSH + [shlex.join(command)], capture_output=True,
                            text=True, timeout=30)
    record = {'utc': datetime.now(timezone.utc).isoformat(), 'host': SSH[-1],
              'command': command, 'returncode': result.returncode,
              'stdout': result.stdout, 'stderr': result.stderr}
    with LOG.open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    return result


def mapping_may_use_devices(config):
    if any(config[key] for key in ('Privileged', 'DeviceCgroupRules', 'DeviceRequests')):
        return True
    targets = {f'/dev/davinci{device}' for device in DEVICES}
    if any(device.get('PathOnHost') in targets for device in config['Devices'] or []):
        return True
    sources = [binding.split(':', 1)[0] for binding in config['Binds'] or []]
    sources.extend(mount.get('Source', '') for mount in config['Mounts'] or [])
    return any(source in ('/', '/dev') or source.startswith('/dev/') for source in sources)


def telemetry_is_idle(output):
    process_sections = output.split('Process id', 1)
    if len(process_sections) != 2:
        return False
    for physical in DEVICES:
        board, chip = divmod(physical, 2)
        row = re.search(r'^\|\s*' + str(chip) + r'\s+' + str(physical)
                        + r'\s*\|\s*[0-9A-Fa-f:]+\.\d+\s*\|([^\n]+)',
                        process_sections[0], re.MULTILINE)
        if row is None:
            return False
        previous = output[:row.start()].rstrip().splitlines()[-1]
        if not re.search(r'\|\s*' + str(board) + r'\s+Ascend\S*\s*\|\s*OK\s*\|', previous):
            return False
        if not re.match(r'\s*0\s+', row.group(1)):
            return False
        memory = [(int(used), int(total)) for used, total in
                  re.findall(r'(\d+)\s*/\s*(\d+)', row.group(1)) if int(total) >= 8192]
        if len(memory) != 1 or memory[0][0] / memory[0][1] > .08:
            return False
        if re.search(r'^\|\s*' + str(board) + r'\s+' + str(chip) + r'\s*\|\s*\d+',
                     process_sections[1], re.MULTILINE):
            return False
    return True


def vacant():
    telemetry = run(['npu-smi', 'info'])
    if telemetry.returncode or not telemetry_is_idle(telemetry.stdout):
        return False
    containers = run(['docker', 'ps', '--format', '{{.Names}}|{{.Status}}'])
    if containers.returncode:
        return False
    fields = ('Privileged', 'Devices', 'DeviceCgroupRules', 'DeviceRequests', 'Binds', 'Mounts')
    template = '\n'.join('{{json .HostConfig.' + field + '}}' for field in fields)
    for line in containers.stdout.splitlines():
        if '|' not in line:
            return False
        name, status = line.split('|', 1)
        if re.search(r'^Up (\d+ seconds?|About a minute|[12] minutes?|Less than a second)', status):
            inspected = run(['docker', 'inspect', '--format', template, name])
            if inspected.returncode:
                return False
            try:
                values = [json.loads(line) for line in inspected.stdout.splitlines()]
            except (TypeError, ValueError):
                return False
            if len(values) != len(fields) or mapping_may_use_devices(dict(zip(fields, values))):
                return False
    holders = run(['fuser', *(f'/dev/davinci{physical}' for physical in DEVICES)])
    return holders.returncode == 1 and not holders.stdout.strip() and not holders.stderr.strip()


if __name__ == '__main__':
    raise SystemExit(0 if vacant() else 2)
