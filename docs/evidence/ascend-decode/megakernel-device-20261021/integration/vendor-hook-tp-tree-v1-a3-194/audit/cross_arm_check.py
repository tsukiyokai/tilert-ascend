"""Compare the eager and graph arms case by case.

Capture and replay must not change the answer. Every case holds the same row
count and iteration in both arms, so their retained output digests must match.
"""

import argparse
import json
from pathlib import Path

RANKS = (0, 1, 2)
BOUNDARIES = ('consumer.output', 'consumer.retained')
ARMS = {'eager-v2': 'tp-chain.json', 'graph-v1': 'tp-chain-graph.json'}


def report(raw, arm, rank):
    return json.loads((raw / arm / f'device-rank{rank}' / ARMS[arm]).read_text())


def compare(raw):
    matched = 0
    for rank in RANKS:
        eager, graph = (report(raw, arm, rank) for arm in ARMS)
        assert len(eager['cases']) == len(graph['cases'])
        for left, right in zip(eager['cases'], graph['cases']):
            assert (left['rows'], left['iteration']) == (right['rows'], right['iteration'])
            assert left['pass'] and right['pass']
            for boundary in BOUNDARIES:
                assert left['outputs'][boundary]['sha256'] == right['outputs'][boundary]['sha256']
                matched += 1
    return {'pass': True, 'ranks': len(RANKS), 'cases_per_rank': len(eager['cases']),
            'identical_digests': matched, 'boundaries': list(BOUNDARIES)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    print(json.dumps(compare(parser.parse_args().raw)))
