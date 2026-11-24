"""Split the MLA tail's window into cube work, vector work and barrier."""

import argparse
import json
import statistics

TICKS_A_MICROSECOND = 50.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile')
    parser.add_argument('--rows', type=int, default=12)
    args = parser.parse_args()
    report = json.loads(open(args.profile).read())

    for case in report['cases']:
        if case['rows'] != args.rows:
            continue
        cores = case['per_core_summary']
        stages = [entry['stage'] for entry in cores[0]['stages']]
        fields = [name for name, value in cores[0]['stages'][0].items()
                  if isinstance(value, (int, float)) and name != 'stage']
        print(f"rows {case['rows']}  cores {len(cores)}  stages {stages}")
        print(f'fields {fields}')
        for field in fields:
            print(f'\n--- {field}')
            for kind in ('AIC', 'AIV'):
                held = [core for core in cores if core['kind'] == kind]
                if not held:
                    continue
                total, parts = 0.0, []
                for index, stage in enumerate(stages):
                    values = [core['stages'][index][field] for core in held]
                    median = statistics.median(values) / TICKS_A_MICROSECOND
                    total += median
                    parts.append(f'{stage}={median:.2f}')
                print(f'  {kind} total {total:7.2f}  ' + ' '.join(parts))


if __name__ == '__main__':
    main()
