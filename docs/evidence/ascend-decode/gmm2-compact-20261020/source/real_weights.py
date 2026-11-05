"""Replay frozen layer-four down projection with real checkpoint weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
sys.path.insert(0, "/workspace/experiments/gmm-compact-20261020")
import torch
import torch_npu
import gmm1_inputs
from vllm.config import set_current_vllm_config
from tilert.models.glm_5_2_ascend.reference_weights import ReferenceWeights


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:15")
    args = parser.parse_args()
    torch.set_num_threads(1)
    trace_path = Path('/workspace/experiments/gmm-compact-20261020/frozen/rank-0/step-08192.pt')
    metadata = json.loads((trace_path.parent.parent / 'rank-0.json').read_text())
    trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    assert trace_sha == (metadata['trace_sha256'] | metadata['reset_trace_sha256'])[trace_path.name]
    trace = torch.load(trace_path, map_location='cpu', weights_only=True, mmap=True)
    stages = trace['mlp_stages']
    counts = stages['gmm2_group_counts']
    weights = ReferenceWeights('/workspace/models/GLM-5.2-w4a8c8', expert_cache_size=1)
    assert weights.contract['metadata_sha256'] == metadata['checkpoint_sha256']
    shapes = {'w13_weight': (256, 128, 6144), 'w2_weight': (256, 3072, 128),
              'w13_weight_scale': (256, 256, 1), 'w2_weight_scale': (256, 6144, 1),
              'w13_scale_bias': (256, 256, 1), 'w2_scale_bias': (256, 6144, 1)}
    bank = torch.nn.Module()
    for name, shape in shapes.items():
        dtype = torch.int8 if name.endswith('_weight') else torch.float32
        bank.register_parameter(name, torch.nn.Parameter(torch.zeros(shape, dtype=dtype),
                                                        requires_grad=False))
    expected, hashes, scale_low_bits = [], {}, []
    active = counts.nonzero().flatten().tolist()
    assert len(active) == 8 and counts.sum() == 8
    for row, expert in enumerate(active):
        weight = weights._integer_weight(f'model.layers.{trace["mlp_layer"]}.mlp.experts.{expert}.down_proj')
        bank.w2_weight[expert].copy_(weight.values[:, :128])
        bank.w2_weight_scale[expert].copy_(weight.scales[:, None])
        bank.w2_scale_bias[expert].copy_(weight.correction[:, :1])
        expected.append(weights._integer_product(
            weight, stages['fused_a8'][row:row+1], stages['fused_scale'][row:row+1],
            (0, 6144), (0, 128)))
        hashes[str(expert)] = [digest(getattr(bank, name)[expert])
                              for name in ('w2_weight', 'w2_weight_scale', 'w2_scale_bias')]
        scale_low_bits.append(int(((weight.scales.view(torch.int32) & 8191) != 0).sum()))
    expected = torch.cat(expected)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    capacity = torch.npu.get_device_properties(args.device).total_memory
    torch.npu.set_per_process_memory_fraction(1536 * 2**20 / capacity)
    config, method = gmm1_inputs.quant_method()
    bank.to(args.device)
    with set_current_vllm_config(config):
        method.process_weights_after_loading(bank)
        _, _, w2, scale = method._get_mlp_weights(bank)
        _, bias = method._get_mlp_scale_bias(bank)
    frozen = stages['fused_a8'].to(args.device)
    x = frozen.clone()
    x_scale, groups = stages['fused_scale'].to(args.device), counts.to(args.device)

    def call():
        x.copy_(frozen)
        return torch_npu.npu_grouped_matmul(
            x=[x], weight=w2, scale=scale, bias=bias, per_token_scale=[x_scale],
            group_list=groups, split_item=2, group_type=0, group_list_type=1,
            output_dtype=torch.bfloat16)[0]

    actual = [call().cpu() for _ in range(2)]
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = call()
    for _ in range(2):
        graph.replay()
        torch.npu.synchronize()
        actual.append(output.cpu())
    report = {'layer': trace['mlp_layer'], 'position': trace['position'], 'rank': 0,
              'trace_sha256': trace_sha, 'checkpoint_sha256': metadata['checkpoint_sha256'],
              'active_payload_sha256': hashes, 'nonzero_scale_low_bits': scale_low_bits,
              'expected_sha256': digest(expected), 'output_sha256': [digest(t) for t in actual],
              'exact': [torch.equal(t, expected) for t in actual],
              'frozen_down_exact': [torch.equal(t, stages['down']) for t in actual],
              'peak_reserved_bytes': torch.npu.max_memory_reserved(),
              'opp_path': os.environ.get('ASCEND_CUSTOM_OPP_PATH')}
    report['status'] = 'pass' if all(report['exact']) else 'fail'
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    assert report['status'] == 'pass'


if __name__ == '__main__':
    main()
