"""Time one installed GMM2 ELF with the fixed rotating-bank workload."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch
import torch_npu
from validate import build_bank
from bench_retile import stats, CALLS, BANKS


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(15)
    torch.npu.config.allow_internal_format = True
    total = torch.npu.get_device_properties('npu:15').total_memory
    torch.npu.set_per_process_memory_fraction(3584 * 2**20 / total)
    banks = [build_bank('npu:15')[:3] for _ in range(BANKS)]
    generator = torch.Generator().manual_seed(20261020)
    report = {'scope': 'Whole GMM2 package; one shared die, not TPOT',
              'banks': BANKS, 'calls_per_graph': CALLS, 'warmup': 10,
              'samples': 30, 'profiler': False, 'clocks_locked': False,
              'outliers': 'none discarded', 'includes_input_restore': True,
              'cases': []}
    for width, distribution in ((1, 'random'), (8, 'random'),
                                (32, 'random'), (32, 'shared8'),
                                (32, 'full256')):
        rows = width * 8
        frozen = torch.randint(-127, 128, (rows, 128), dtype=torch.int8,
                               generator=generator).to('npu:15')
        x = frozen.clone()
        scale = (torch.rand(rows, generator=generator) * .016 + .004).to('npu:15')
        counts_cpu = []
        for _ in range(CALLS):
            selected = (torch.randperm(256, generator=generator)[:8].repeat(width)
                        if distribution == 'shared8' else
                        torch.cat([torch.randperm(256, generator=generator)[:8]
                                   for _ in range(width)]))
            counts_cpu.append(torch.ones(256, dtype=torch.int64)
                              if distribution == 'full256' else
                              torch.bincount(selected, minlength=256))
        counts = [value.to('npu:15') for value in counts_cpu]

        def call(index):
            w, s, b = banks[index % BANKS]
            x.copy_(frozen)
            return torch_npu.npu_grouped_matmul(
                x=[x], weight=w, scale=s, bias=b, per_token_scale=[scale],
                group_list=counts[index], split_item=2, group_type=0,
                group_list_type=1, output_dtype=torch.bfloat16)[0]

        call(0)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = [call(i) for i in range(CALLS)]
        for _ in range(10):
            graph.replay()
        torch.npu.synchronize()
        samples = []
        for _ in range(30):
            begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            begin.record()
            graph.replay()
            end.record()
            torch.npu.synchronize()
            samples.append(begin.elapsed_time(end) * 1000 / CALLS)
        digest = hashlib.sha256()
        for output in outputs:
            digest.update(output.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        result = {'width': width, 'distribution': distribution,
                  'us': stats(samples), 'raw_us': samples,
                  'output_sha256': digest.hexdigest(),
                  'route_sha256': hashlib.sha256(torch.stack(counts_cpu).numpy().tobytes()).hexdigest()}
        report['cases'].append(result)
        report['peak_reserved_bytes'] = torch.npu.max_memory_reserved()
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'raw_us'}), flush=True)
        del graph, outputs, output, counts, x
    report['status'] = 'pass'
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
