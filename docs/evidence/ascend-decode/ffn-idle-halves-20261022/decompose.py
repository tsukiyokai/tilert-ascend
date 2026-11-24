"""Decompose the routed body's window into cube work, vector work and barrier.

The stage trace records three marks a stage a core: the start, the moment the
core's own work is done, and the moment every core has finished. So the window
splits into work and waiting, and the kind field splits the work into halves.
"""

import argparse
import json
import statistics

TICKS_A_MICROSECOND = 50.0
STAGES = ('restore_dispatch', 'metadata_and_split', 'gate_up_matmul',
          'activate', 'down_matmul', 'store_routes', 'combine')


def summarise(case):
    cores = case['per_core_summary']
    fields = sorted(cores[0]['stages'][0])
    held = {}
    for core in cores:
        for stage in core['stages']:
            held.setdefault((core['kind'], stage['stage']), []).append(stage)
    return fields, held


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile')
    parser.add_argument('--width', type=int, default=12)
    args = parser.parse_args()
    report = json.loads(open(args.profile).read())

    for case in report['cases']:
        if case['width'] != args.width:
            continue
        fields, held = summarise(case)
        print('stage fields:', fields)
        windows = [core['median_window_ticks'] / TICKS_A_MICROSECOND
                   for core in case['per_core_summary']]
        print(f'window us  median {statistics.median(windows):.2f}'
              f'  max {max(windows):.2f}')
        for field in fields:
            if not isinstance(held[('AIC', STAGES[0])][0].get(field), (int, float)):
                continue
            print(f'\n--- {field}')
            for kind in ('AIC', 'AIV'):
                total = 0.0
                parts = []
                for stage in STAGES:
                    values = [entry[field] for entry in held[(kind, stage)]]
                    median = statistics.median(values) / TICKS_A_MICROSECOND
                    total += median
                    parts.append(f'{stage}={median:.2f}')
                print(f'  {kind} total {total:7.2f}  ' + ' '.join(parts))


if __name__ == '__main__':
    main()
