"""Is the output projection occupancy-bound in its narrow output axis?

At TP16 the output projection is [12, 16384] x [16384, 384]. The K axis is
large and the N axis is 384, which gives the cube only a few tiles to spread
over its cores. If the cost is occupancy rather than bytes, a wider N costs the
same wall time while moving several times the bytes.

One matmul an arm, so no invocation cost enters the comparison.
"""

import argparse
import json

import torch
import torch_npu

INNER = 16384
OUTERS = (128, 384, 768, 1536, 3072, 6144, 12288)
ROWS = 12


def timed(activation, weight, iterations):
    for _ in range(5):
        torch.matmul(activation, weight)
    torch.npu.synchronize()
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        torch.matmul(activation, weight)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1e3 / iterations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(20261022)

    activation = torch.randn(ROWS, INNER, dtype=torch.bfloat16,
                             device=args.device)
    points = []
    for outer in OUTERS:
        weight = torch.randn(INNER, outer, dtype=torch.bfloat16,
                             device=args.device)
        megabytes = INNER * outer * 2 / 1e6
        samples = [timed(activation, weight, args.iterations)
                   for _ in range(args.repeats)]
        median = sorted(samples)[len(samples) // 2]
        points.append({'outer': outer, 'weight_mb': round(megabytes, 2),
                       'median_us': round(median, 2),
                       'gb_s': round(megabytes * 1e3 / median)})
        del weight
    print(json.dumps({'inner': INNER, 'rows': ROWS,
                      'iterations': args.iterations, 'points': points}, indent=2))
