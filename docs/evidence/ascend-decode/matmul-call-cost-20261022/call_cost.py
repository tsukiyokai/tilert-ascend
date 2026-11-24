"""The fixed cost of a matmul call, at two weight sizes and several splits.

A layer's projections share activations, so a single launch can concatenate
them along the output axis. That only pays if the cost is per call. Holding the
bytes and the products fixed and varying only the number of calls isolates it.

One total fits the 192 MiB L2 and one exceeds it, because an L2-resident weight
would otherwise let cache reuse stand in for the effect.
"""

import argparse
import json

import torch
import torch_npu

INNER = 6144
TOTALS = {'l2_resident': 2816, 'exceeds_l2': 24576}
SPLITS = (1, 2, 3, 4, 6, 8)
ROWS = 12


def timed(pieces, activation, iterations):
    for _ in range(5):
        for piece in pieces:
            torch.matmul(activation, piece)
    torch.npu.synchronize()
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        for piece in pieces:
            torch.matmul(activation, piece)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1e3 / iterations


def sweep(total, iterations, repeats, device):
    activation = torch.randn(ROWS, INNER, dtype=torch.bfloat16, device=device)
    weight = torch.randn(INNER, total, dtype=torch.bfloat16, device=device)
    megabytes = INNER * total * 2 / 1e6
    points = []
    for splits in SPLITS:
        if total % splits:
            continue
        pieces = [piece.contiguous()
                  for piece in weight.split(total // splits, dim=1)]
        samples = [timed(pieces, activation, iterations) for _ in range(repeats)]
        median = sorted(samples)[len(samples) // 2]
        points.append({'calls': splits, 'median_us': round(median, 2),
                       'samples_us': [round(value, 2) for value in samples]})
        del pieces
    del weight
    return {'total_outer': total, 'weight_mb': round(megabytes, 2),
            'points': points}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(20261022)
    print(json.dumps({
        'inner': INNER, 'rows': ROWS, 'iterations': args.iterations,
        'repeats': args.repeats,
        'sweeps': {name: sweep(total, args.iterations, args.repeats, args.device)
                   for name, total in TOTALS.items()},
    }, indent=2))
