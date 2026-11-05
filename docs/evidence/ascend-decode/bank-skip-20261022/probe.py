"""Does the vendor grouped matmul pay for an expert whose group is empty?

Three arms share one routing total. `spread` gives every one of the sixteen
experts a nonzero count. `concentrated` gives the same total rows to five
experts and leaves eleven at zero, keeping the full sixteen-expert weight
tensor. `narrowed` subsets that tensor to the five and renumbers the groups.

If the kernel skips empty groups, concentrated costs what narrowed costs. If it
reads the whole bank, concentrated costs what spread costs.
"""

import argparse
import json

import torch
import torch_npu

EXPERTS = 16
HIDDEN = 6144
INTERMEDIATE = 2048
TOUCHED = 5
ROWS = 96


def counts(touched):
    """Spread ROWS over `touched` experts, the rest left empty."""
    base, extra = divmod(ROWS, touched)
    return [base + (slot < extra) for slot in range(touched)] + [0] * (EXPERTS - touched)


def grouped(rows, weight, group_list):
    return torch.ops.npu.npu_grouped_matmul(
        [rows], [weight], group_list=group_list, split_item=3, group_type=0,
        group_list_type=1)[0]


def timed(rows, weight, group_list, iterations):
    for _ in range(3):
        grouped(rows, weight, group_list)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        grouped(rows, weight, group_list)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1e3 / iterations


def arms(device, iterations):
    torch.manual_seed(20261022)
    rows = torch.randn(ROWS, HIDDEN, dtype=torch.bfloat16, device=device)
    weight = torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16,
                         device=device)
    spread = torch.tensor(counts(EXPERTS), dtype=torch.int64, device=device)
    concentrated = torch.tensor(counts(TOUCHED), dtype=torch.int64, device=device)
    narrow_weight = weight[:TOUCHED].contiguous()
    narrowed = torch.tensor(counts(TOUCHED)[:TOUCHED], dtype=torch.int64, device=device)

    return {
        'spread': timed(rows, weight, spread, iterations),
        'concentrated': timed(rows, weight, concentrated, iterations),
        'narrowed': timed(rows, narrow_weight, narrowed, iterations),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    torch.npu.set_device(args.device)

    samples = [arms(args.device, args.iterations) for _ in range(args.repeats)]
    median = {name: sorted(sample[name] for sample in samples)[len(samples) // 2]
              for name in samples[0]}
    print(json.dumps({
        'experts': EXPERTS, 'touched': TOUCHED, 'rows': ROWS,
        'hidden': HIDDEN, 'intermediate': INTERMEDIATE,
        'iterations': args.iterations, 'repeats': args.repeats,
        'median_us': {name: round(value, 2) for name, value in median.items()},
        'samples_us': [{name: round(value, 2) for name, value in sample.items()}
                       for sample in samples],
    }, indent=2))
