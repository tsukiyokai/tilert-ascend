"""Screen V1 versus C1 with a fixed synthetic M8 attention graph."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

import torch


SOURCE = Path(__file__).with_name('compare-sfa-c1-standalone.py')
spec = importlib.util.spec_from_file_location('sfa_replay', SOURCE)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(case, device_id):
    import torch_npu
    from vllm_ascend.utils import enable_custom_op

    torch_npu.npu.set_device(device_id)
    total = torch_npu.npu.get_device_properties(device_id).total_memory
    torch_npu.npu.set_per_process_memory_fraction(256 * 2**20 / total)
    if not enable_custom_op():
        raise RuntimeError('The vLLM custom operator is unavailable')
    device = torch.device('npu', device_id)
    arguments = {name: value.to(device) for name, value in case.items()}

    def operation():
        return torch.ops._C_ascend.npu_sparse_flash_attention(
            **arguments, value=arguments['key'], scale_value=1 / 16,
            sparse_block_size=1, layout_query='TND', layout_kv='PA_BSND',
            sparse_mode=3, attention_mode=2, return_softmax_lse=False,
        )[0]

    stream = torch_npu.npu.Stream()
    samples = []
    with torch.inference_mode(), torch_npu.npu.stream(stream):
        for _ in range(10):
            warm_output = operation()
        stream.synchronize()
        reference = warm_output.cpu().clone()
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=stream):
            for _ in range(10):
                graph_output = operation()
        stream.synchronize()
        start = torch_npu.npu.Event(enable_timing=True)
        end = torch_npu.npu.Event(enable_timing=True)
        for batch in range(30):
            wall_start = time.perf_counter_ns()
            start.record(stream)
            enqueue_start = time.perf_counter_ns()
            graph.replay()
            enqueue_end = time.perf_counter_ns()
            end.record(stream)
            stream.synchronize()
            wall_end = time.perf_counter_ns()
            elapsed_ms = start.elapsed_time(end)
            samples.append({
                'batch': batch, 'calls': 10,
                'event_total_ms': elapsed_ms,
                'event_us_per_call': elapsed_ms * 1000 / 10,
                'wall_total_ms': (wall_end - wall_start) / 1e6,
                'wall_us_per_call': (wall_end - wall_start) / 1e3 / 10,
                'host_graph_enqueue_us': (enqueue_end - enqueue_start) / 1e3,
            })
        actual = graph_output.cpu().clone()
        assert torch.equal(actual, reference)
        assert torch.isfinite(actual).all() and actual.count_nonzero() > 0
        graph.reset()
    mappings = Path('/proc/self/maps').read_text().splitlines()
    libraries = sorted({line.split()[-1] for line in mappings
                        if 'opmaster' in line or 'optiling' in line})
    return {
        'samples': samples,
        'event_us_per_call_median': statistics.median(
            x['event_us_per_call'] for x in samples),
        'wall_us_per_call_median': statistics.median(
            x['wall_us_per_call'] for x in samples),
        'peak_allocated_bytes': torch_npu.npu.max_memory_allocated(),
        'peak_reserved_bytes': torch_npu.npu.max_memory_reserved(),
        'host_libraries': libraries,
        'host_library_sha256': {p: sha256(p) for p in libraries},
        'graph_output_equal_eager': True,
        'output_nonzero': int(actual.count_nonzero()),
        'output_sha256': hashlib.sha256(
            actual.view(torch.int16).numpy().tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cpu-only', action='store_true')
    mode.add_argument('--device', type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    case = (torch.load(args.input, map_location='cpu', weights_only=True)
            if args.input else replay.synthetic_case(8, 8192, 20261019))
    rows = replay.selection_audit(case)
    assert case['query'].shape == (8, 4, 512)
    assert case['query_rope'].shape == (8, 4, 64)
    assert all(x['valid_tokens'] == 2048 and x['future_tokens'] == 0
               and x['sentinels'] == 0 for x in rows)
    torch.save(case, args.output / 'input.pt')
    report = {
        'status': 'cpu_fixture_verified',
        'input_kind': 'synthetic_production_shape_not_model_capture',
        'input_sha256': sha256(args.input or args.output / 'input.pt'),
        'source_sha256': sha256(__file__),
        'replay_source_sha256': sha256(SOURCE),
        'shape': {'M': 8, 'query_heads': 4, 'top_k': 2048,
                  'head_dim': 512, 'rope_dim': 64, 'prefix_length': 8192},
        'rows': rows, 'indices_storage': 'ordinary',
        'input_tensor_bytes': sum(x.numel() * x.element_size()
                                  for x in case.values()),
        'torch_allocator_quota_bytes': 256 * 2**20,
        'quota_scope': 'Torch allocator; excludes NPU runtime/context',
        'warmup_eager_calls': 10, 'capture_calls': 10,
        'measured_batches': 30, 'calls_per_graph': 10,
        'maximum_host_operator_calls': 320,
        'timing_scope': ('NPU events surround one replay of a ten-call graph; '
                         'wall time includes event enqueue and stream sync. '
                         'Stable addresses and repeated synthetic payloads '
                         'give a hot input screening workload under shared load.'),
        'decoder_precision_verified': False,
        'model_tpot_measured': False, 'npu_launched': False,
        'custom_opp_path': os.environ.get('ASCEND_CUSTOM_OPP_PATH'),
    }
    if args.device is not None:
        report.update(execute(case, args.device), status='screen_completed',
                      npu_launched=True)
    (args.output / 'result.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
