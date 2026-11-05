"""Is the per-call cost on the device, or is it eager host dispatch?

The eager sweep charges 18.5 microseconds for each extra matmul call. The
engine replays a captured graph, where host dispatch is gone. If the cost
survives capture it belongs to the device and a megakernel can remove it; if it
collapses, the eager figure was the host and the lever is not there.

Both arms hold the bytes, the products and the activation fixed.
"""

import argparse
import json

import torch
import torch_npu

INNER = 6144
TOTAL = 2816
SPLITS = (1, 2, 4, 8)
ROWS = 12


def pieces_for(weight, splits):
    return [piece.contiguous() for piece in weight.split(TOTAL // splits, dim=1)]


def eager(pieces, activation, iterations):
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


def captured(pieces, activation, iterations):
    """One graph holding `iterations` repetitions, replayed and timed."""
    for _ in range(5):
        for piece in pieces:
            torch.matmul(activation, piece)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(iterations):
            for piece in pieces:
                torch.matmul(activation, piece)
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    graph.replay()
    end.record()
    torch.npu.synchronize()
    elapsed = start.elapsed_time(end) * 1e3 / iterations
    graph.reset()
    return elapsed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(20261022)

    activation = torch.randn(ROWS, INNER, dtype=torch.bfloat16,
                             device=args.device)
    weight = torch.randn(INNER, TOTAL, dtype=torch.bfloat16, device=args.device)

    points = []
    for splits in SPLITS:
        pieces = pieces_for(weight, splits)
        samples = {
            'eager': [eager(pieces, activation, args.iterations)
                      for _ in range(args.repeats)],
            'graph': [captured(pieces, activation, args.iterations)
                      for _ in range(args.repeats)],
        }
        points.append({'calls': splits, **{
            mode: round(sorted(values)[len(values) // 2], 2)
            for mode, values in samples.items()}})

    print(json.dumps({'inner': INNER, 'total_outer': TOTAL, 'rows': ROWS,
                      'weight_mb': round(INNER * TOTAL * 2 / 1e6, 2),
                      'iterations': args.iterations, 'repeats': args.repeats,
                      'points': points}, indent=2))
