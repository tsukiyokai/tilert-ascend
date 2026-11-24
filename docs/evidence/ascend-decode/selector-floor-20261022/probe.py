"""What does selecting 2048 of 8192 a row cost at twelve rows?

The indexed attention's selector stage measures 54.84 microseconds and is pure
vector work, which cannot hide under the 98 microseconds of cube the phase has
against 199 of vector. So its floor decides whether it is worth replacing.
"""

import argparse
import json

import torch
import torch_npu

POSITIONS = 8192
SELECTED = 2048
ROWS = 12


def timed(call, iterations):
    for _ in range(5):
        call()
    torch.npu.synchronize()
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1e3 / iterations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--rows', type=int, default=ROWS)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(20261022)

    scores = torch.randn(args.rows, POSITIONS, dtype=torch.float,
                         device=args.device)
    arms = {
        'topk_sorted': lambda: torch.topk(scores, SELECTED, dim=1, sorted=True),
        'topk_unsorted': lambda: torch.topk(scores, SELECTED, dim=1,
                                            sorted=False),
        'full_sort': lambda: torch.sort(scores, dim=1, descending=True),
        'threshold_only': lambda: (scores > 0.0).sum(dim=1),
    }
    median = {}
    for name, call in arms.items():
        samples = [timed(call, args.iterations) for _ in range(args.repeats)]
        median[name] = round(sorted(samples)[len(samples) // 2], 2)
    print(json.dumps({
        'positions': POSITIONS, 'selected': SELECTED, 'rows': args.rows,
        'score_bytes': args.rows * POSITIONS * 4,
        'iterations': args.iterations, 'repeats': args.repeats,
        'median_us': median,
    }, indent=2))
