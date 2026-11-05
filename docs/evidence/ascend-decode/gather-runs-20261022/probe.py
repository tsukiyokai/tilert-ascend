"""Does the KV gather cost follow the rows gathered or the runs they form?

Sparse attention selects 2048 of 8192 positions a row. If the cost follows the
rows, run structure is irrelevant and nothing about a copying workload helps.
If it follows the runs, selections that fall in contiguous spans -- which is
what copied input produces -- can be coalesced, and the stage is addressable.

Each arm gathers the same 2048 positions a row, differing only in how long the
contiguous runs are.
"""

import argparse
import json

import torch
import torch_npu

POSITIONS = 8192
COLUMNS = 576          # 512 latent and 64 rope a position
SELECTED = 2048
RUNS = (1, 2, 8, 64, 256, 2048)


def indices(run_length, rows, device):
    """`SELECTED` positions a row, in runs of `run_length`."""
    count = SELECTED // run_length
    starts = torch.randint(0, POSITIONS - run_length, (rows, count),
                           device=device)
    offsets = torch.arange(run_length, device=device)
    return (starts.unsqueeze(-1) + offsets).reshape(rows, SELECTED)


def timed(cache, picks, iterations):
    for _ in range(3):
        torch.index_select(cache, 0, picks)
    torch.npu.synchronize()
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        torch.index_select(cache, 0, picks)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1e3 / iterations


def arms(rows, iterations, device):
    torch.manual_seed(20261022)
    cache = torch.randn(POSITIONS, COLUMNS, dtype=torch.bfloat16, device=device)
    return {str(run): timed(cache, indices(run, rows, device).reshape(-1), iterations)
            for run in RUNS}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--rows', type=int, default=12)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    torch.npu.set_device(args.device)

    samples = [arms(args.rows, args.iterations, args.device)
               for _ in range(args.repeats)]
    median = {run: sorted(sample[run] for sample in samples)[len(samples) // 2]
              for run in samples[0]}
    gathered_mb = args.rows * SELECTED * COLUMNS * 2 / 1e6
    print(json.dumps({
        'positions': POSITIONS, 'columns': COLUMNS, 'selected_a_row': SELECTED,
        'rows': args.rows, 'gathered_mb': round(gathered_mb, 2),
        'iterations': args.iterations, 'repeats': args.repeats,
        'median_us_by_run_length': {run: round(value, 2)
                                    for run, value in median.items()},
        'gb_s_by_run_length': {run: round(gathered_mb * 1e3 / value)
                               for run, value in median.items()},
    }, indent=2))
