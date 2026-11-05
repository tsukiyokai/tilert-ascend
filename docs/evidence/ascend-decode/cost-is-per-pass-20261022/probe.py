"""Does a vector operator's fixed cost survive graph capture?

Any per-row reduction over twelve by 8192 floats costs about 18 microseconds
whatever it computes, which looks like a per-invocation cost. The engine replays
a captured graph, so if the cost is host dispatch it does not pay it and the
figure prices nothing. If it survives capture it belongs to the device, and a
layer's dozen or so operator calls carry it.
"""

import argparse
import json

import torch
import torch_npu

POSITIONS = 8192
SELECTED = 2048
ROWS = 12


def eager(call, iterations):
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


def captured(call, iterations):
    for _ in range(5):
        call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(iterations):
            call()
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

    scores = torch.randn(ROWS, POSITIONS, dtype=torch.float,
                         device=args.device)
    # One call against several, so the per-call increment is visible under
    # capture as well as eager.
    arms = {
        'one_sum': lambda: scores.sum(dim=1),
        'four_sums': lambda: [scores.sum(dim=1) for _ in range(4)],
        'one_topk': lambda: torch.topk(scores, SELECTED, dim=1, sorted=False),
        'four_topk': lambda: [torch.topk(scores, SELECTED, dim=1, sorted=False)
                              for _ in range(4)],
    }
    report = {}
    for name, call in arms.items():
        report[name] = {}
        for mode, runner in (('eager', eager), ('graph', captured)):
            samples = [runner(call, args.iterations)
                       for _ in range(args.repeats)]
            report[name][mode] = round(sorted(samples)[len(samples) // 2], 2)
    print(json.dumps({
        'positions': POSITIONS, 'selected': SELECTED, 'rows': ROWS,
        'iterations': args.iterations, 'repeats': args.repeats,
        'median_us': report,
    }, indent=2))
