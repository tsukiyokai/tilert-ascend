"""Run exactly two AB/BA pairs of the bounded single-die SFA screen."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def snapshot(output, prefix):
    smi = subprocess.run(['npu-smi', 'info'], capture_output=True,
                         text=True, check=True).stdout
    (output / (prefix + '-npu-smi.txt')).write_text(smi)
    (output / (prefix + '-host-memory.txt')).write_text(
        Path('/proc/meminfo').read_text())
    (output / (prefix + '-host-load.txt')).write_text(
        Path('/proc/loadavg').read_text())
    for name in ('memory.current', 'memory.max'):
        source = Path('/sys/fs/cgroup') / name
        if source.exists():
            (output / (prefix + '-cgroup-' + name)).write_text(
                source.read_text())
    match = re.search(r'\b(\d+)\s*/\s*(65536)\b', smi)
    if not match:
        raise RuntimeError('No die0 HBM reading')
    free_mib = int(match[2]) - int(match[1])
    if free_mib < 2048:
        raise RuntimeError(f'Die0 free HBM {free_mib}MiB is below 2GiB')
    return free_mib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    manifest = json.loads((root / 'manifest.json').read_text())
    for arm in ('original', 'bounds', 'v1'):
        report = json.loads((root / f'correctness-{arm}/result.json').read_text())
        if report['status'] != 'device_reproducible':
            raise RuntimeError('Ordinary-index correctness did not pass')
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for label, arm in (('A1', 'v1'), ('B1', 'bounds'),
                       ('B2', 'bounds'), ('A2', 'v1')):
        free_mib = snapshot(args.output, label + '-before')
        output = args.output / label
        started = time.perf_counter()
        with (args.output / (label + '-driver.log')).open('w') as log:
            process = subprocess.run(
                ['bash', str(root / 'run-screen.sh'), '0', str(output), arm],
                stdout=log, stderr=subprocess.STDOUT)
        snapshot(args.output, label + '-after')
        if process.returncode:
            failure = {'status': 'aborted_after_failure', 'label': label,
                       'returncode': process.returncode,
                       'no_remaining_arms_launched': True}
            (args.output / 'summary.json').write_text(
                json.dumps(failure, indent=2) + '\n')
            raise SystemExit(process.returncode)
        report = json.loads((output / 'result.json').read_text())
        expected_key = 1158 if arm == 'v1' else 134
        assert report['observed_tiling_keys'] == [expected_key]
        assert report['selected_device_elf_sha256'] == manifest[
            'compiled_variants']['bounds']['elf_sha256']
        assert report['input_sha256'] == manifest['frozen_input']['input_sha256']
        assert report['indices_storage'] == 'ordinary'
        assert len(report['samples']) == 30
        assert report['maximum_host_operator_calls'] == 320
        host = (root / arm / 'vendors/custom_transformer/op_impl/ai_core/tbe'
                '/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so')
        assert report['host_library_sha256'][str(host)] == manifest[
            'performance_host_sha256']['v1' if arm == 'v1' else 'c1']
        report['driver_elapsed_seconds'] = time.perf_counter() - started
        report['before_free_hbm_mib'] = free_mib
        reports[label] = report
        print(json.dumps({'label': label, 'arm': arm,
                          'key': report['observed_tiling_keys'],
                          'event_us_median': report['event_us_per_call_median'],
                          'wall_us_median': report['wall_us_per_call_median'],
                          'peak_reserved_bytes': report['peak_reserved_bytes']}),
              flush=True)
    pairs = []
    for number in (1, 2):
        a, b = reports[f'A{number}'], reports[f'B{number}']
        a_us, b_us = (x['event_us_per_call_median'] for x in (a, b))
        pairs.append({'pair': number, 'order': 'AB' if number == 1 else 'BA',
                      'v1_event_us': a_us, 'c1_event_us': b_us,
                      'c1_minus_v1_us': b_us - a_us,
                      'c1_over_v1': b_us / a_us})
    signs = [x['c1_minus_v1_us'] > 0 for x in pairs]
    summary = {'status': 'two_pairs_completed', 'pairs': pairs,
               'directions_agree': signs[0] == signs[1],
               'maximum_operator_calls_total': 1280,
               'samples_per_arm': 30, 'calls_per_graph': 10,
               'scope': 'Hot synthetic M8 inputs, single die under shared load.',
               'model_tpot_measured': False,
               'additional_pairs_launched': False,
               'arm_event_us_medians': {
                   key: value['event_us_per_call_median']
                   for key, value in reports.items()}}
    (args.output / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
