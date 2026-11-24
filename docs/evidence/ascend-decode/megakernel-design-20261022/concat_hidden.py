"""Concatenate the projections that read the normalized hidden.

At K=6144 a layer projects the same normalized hidden four ways: the query
latent, the key-value latent with its rope, the indexer's key, and the router's
gate. The N axis is spread across cores inside one invocation, so each extra
invocation pays the K depth again. Concatenating them along N pays it once.

Both arms compute the same products from the same activation and move the same
bytes. Only the number of invocations differs.
"""

import argparse
import json

import torch
import torch_npu

INNER = 6144
# Per-rank output widths that all read the normalized hidden.
PARTS = {'query_latent': 2048, 'kv_latent_rope': 576,
         'indexer_key': 160, 'router_gate': 256}
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
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(20261022)

    total = sum(PARTS.values())
    activation = torch.randn(ROWS, INNER, dtype=torch.bfloat16,
                             device=args.device)
    joined = torch.randn(INNER, total, dtype=torch.bfloat16, device=args.device)
    offsets, cursor = {}, 0
    for name, width in PARTS.items():
        offsets[name] = (cursor, width)
        cursor += width
    separate = [joined[:, start:start + width].contiguous()
                for start, width in offsets.values()]

    samples = [{
        'one_invocation': timed(lambda: torch.matmul(activation, joined),
                                args.iterations),
        'four_invocations': timed(
            lambda: [torch.matmul(activation, piece) for piece in separate],
            args.iterations),
    } for _ in range(args.repeats)]
    median = {name: sorted(sample[name] for sample in samples)[len(samples) // 2]
              for name in samples[0]}
    print(json.dumps({
        'inner': INNER, 'rows': ROWS, 'parts': PARTS, 'total_outer': total,
        'weight_mb': round(INNER * total * 2 / 1e6, 2),
        'iterations': args.iterations, 'repeats': args.repeats,
        'median_us': {name: round(value, 2) for name, value in median.items()},
        'saving_us': round(median['four_invocations'] - median['one_invocation'], 2),
    }, indent=2))
