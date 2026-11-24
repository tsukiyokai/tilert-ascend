"""Compare the fused and split arms of one dependency level, then time them.

Both arms compute the same products into the same concatenated layout and
differ only in how many matmul invocations each core issues: one over the whole
output, or one a projection. So they must agree byte for byte, and the fused arm
must be faster by the invocations it does not make.

Run with `-I` and pass the library as an argument.
"""

import argparse
import ctypes
import json
from pathlib import Path

import torch
import torch_npu

SYSTEM_BYTES = 16 << 20

# Each level's shared activation width and the projections that read it, per
# rank at TP16.
# Each level's shared activation and the projections that read it, per rank at
# TP16. `prolog` is the grouping the engine can actually use: Prolog's two
# down-projections both read the int8 `token_x`, while the indexer's key reads
# the bf16 hidden and cannot join them, and the router's gate belongs after
# attention. `hidden` keeps all four to show what the mechanism does when it
# removes three invocations instead of one.
LEVELS = {
    'prolog': {
        'reduction': 6144,
        'parts': {'query_latent': 2048, 'kv_latent_rope': 576},
    },
    'hidden': {
        'reduction': 6144,
        'parts': {'query_latent': 2048, 'kv_latent_rope': 576,
                  'indexer_key': 160, 'router_gate': 256},
    },
    'query_norm': {
        'reduction': 2048,
        'parts': {'query_up': 1024, 'indexer_query': 4096},
    },
}

# Prolog's columns against a shrinking reduction, to separate a matmul call's
# fixed cost from the cost of walking its reduction.
for _depth in (256, 512, 1024, 2048, 4096, 6144):
    LEVELS[f'depth{_depth}'] = {
        'reduction': _depth,
        'parts': {'query_latent': 2048, 'kv_latent_rope': 576},
    }


def load(library):
    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    return torch.ops.megakernel_front


DTYPES = {1: (torch.int8, torch.int32), 2: (torch.bfloat16, torch.float)}


def sample(shape, dtype, device, generator):
    if dtype is torch.int8:
        return torch.randint(-127, 128, shape, dtype=dtype, device=device,
                             generator=generator)
    return torch.randn(shape, dtype=dtype, device=device, generator=generator)


def operands(level, rows, stride, element, device, generator):
    return {
        'activation': sample((rows, level['reduction']), element, device,
                             generator),
        'weight': sample((level['reduction'], stride), element, device,
                         generator),
    }


def submit(front, held, parts, blocks, sync, tiling, system, witness, product,
           vector_passes, barriers, skew_steps, vector_ops, fused_ops):
    front.forward(held['activation'], held['weight'], product, tiling, system,
                  witness, blocks, parts, vector_passes, barriers,
                  skew_steps, vector_ops, fused_ops, sync)


def timed(submit_once, iterations):
    for _ in range(5):
        submit_once()
    torch.npu.synchronize()
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        submit_once()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1e3 / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library', type=Path)
    parser.add_argument('--level', choices=sorted(LEVELS), default='hidden')
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--rows', type=int, default=12)
    parser.add_argument('--blocks', type=int, default=24)
    parser.add_argument('--inputs', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--stride', type=int, default=0)
    parser.add_argument('--vector-passes', type=int, default=0)
    parser.add_argument('--barriers', type=int, default=0)
    parser.add_argument('--skew-steps', type=int, default=0)
    parser.add_argument('--vector-ops', type=int, default=1)
    parser.add_argument('--fused-ops', type=int, default=1)
    parser.add_argument('--base-m', type=int, default=-1)
    parser.add_argument('--base-n', type=int, default=-1)
    parser.add_argument('--base-k', type=int, default=-1)
    args = parser.parse_args()
    torch.npu.set_device(args.device)

    front = load(args.library)
    level = LEVELS[args.level]
    parts = list(level['parts'].values())
    columns = sum(parts)
    element, accumulator = DTYPES[front.element_bytes()]
    stride = args.stride or front.aligned_width(columns)
    sync = front.sync_base()
    tiling = front.build_tiling(args.rows, args.blocks, level['reduction'],
                                parts, stride, args.base_m, args.base_n,
                                args.base_k).to(args.device)
    system = torch.zeros(SYSTEM_BYTES, dtype=torch.uint8, device=args.device)
    witness = torch.zeros(front.witness_floats(), dtype=torch.float,
                          device=args.device)

    generator = torch.Generator(device=args.device).manual_seed(20261022)
    cases = []
    for index in range(args.inputs):
        held = operands(level, args.rows, stride, element, args.device,
                        generator)
        product = torch.zeros(args.rows, stride, dtype=accumulator,
                              device=args.device)
        witness.zero_()
        submit(front, held, parts, args.blocks, sync, tiling, system, witness,
               product, args.vector_passes, args.barriers,
               args.skew_steps, args.vector_ops, args.fused_ops)
        torch.npu.synchronize()
        computed = product[:, :columns]
        reference = (held['activation'].float()
                     @ held['weight'][:, :columns].float())
        if accumulator is torch.int32:
            reference = reference.round()
        cases.append({
            'input': index,
            'product_sha256': torch.sum(
                computed.cpu().contiguous().view(torch.int32).long()).item(),
            'max_absolute_difference':
                (computed.float() - reference).abs().max().item(),
            'reference_magnitude': reference.abs().max().item(),
            'finite': bool(torch.isfinite(computed).all()),
        })

    # Each core sums kVectorChunks chunks of 1024 copies of `pass + 1`, over
    # `vector_passes` passes, so the published total is fixed arithmetic. A
    # mismatch means the vector stage did not run.
    chunks = front.vector_chunks()
    expected = sum(1024.0 * (step + 1) for step in range(args.vector_passes))
    expected *= chunks
    witness_stride = front.witness_stride()
    published = witness.cpu()[:args.blocks * witness_stride:witness_stride]

    held = operands(level, args.rows, stride, element, args.device, generator)
    product = torch.zeros(args.rows, stride, dtype=accumulator,
                          device=args.device)
    samples = [timed(lambda: submit(front, held, parts, args.blocks, sync,
                                    tiling, system, witness, product,
                                    args.vector_passes,
                                    args.barriers,
                                    args.skew_steps, args.vector_ops,
                                    args.fused_ops), args.iterations)
               for _ in range(args.repeats)]

    print(json.dumps({
        'library': str(args.library), 'level': args.level,
        'reduction': level['reduction'], 'parts': level['parts'],
        'columns': columns, 'weight_stride': stride,
        'element_bytes': front.element_bytes(),
        'vector_passes': args.vector_passes,
        'barriers': args.barriers,
        'skew_steps': args.skew_steps,
        'vector_ops': args.vector_ops,
        'fused_ops': args.fused_ops,
        'vector_expected': expected,
        'vector_published_min': published.min().item(),
        'vector_published_max': published.max().item(),
        'vector_ran': bool(args.vector_passes == 0
                           or (published == expected).all()),
        'rows': args.rows, 'blocks': args.blocks,
        'base': [args.base_m, args.base_n, args.base_k],
        'cases': cases,
        'median_us': round(sorted(samples)[len(samples) // 2], 2),
        'samples_us': [round(value, 2) for value in samples],
    }, indent=2))


if __name__ == '__main__':
    main()
