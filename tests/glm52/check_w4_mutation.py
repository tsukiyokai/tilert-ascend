"""Record CANN A8W4 input mutation using one real TP16 gate projection."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu

from tilert.models.glm_5_2_ascend.ops.quantized import NpuQuantizedLinear
from tilert.models.glm_5_2_ascend.weights import ModelSlimCheckpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    source = ModelSlimCheckpoint(args.checkpoint).expert(3, 0, 0).gate
    operation = NpuQuantizedLinear.from_checkpoint(source)
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(1, source.in_features, generator=generator).to(torch.bfloat16).npu()
    activation, scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
    before = activation.cpu()
    expected = source.reference(before, scale.cpu()).to(torch.bfloat16)
    actual = torch_npu.npu_grouped_matmul(
        x=[activation], weight=[operation.weight], scale=[operation.scale],
        bias=[operation.scale_bias], per_token_scale=[scale],
        group_list=torch.tensor([1], dtype=torch.int64, device="npu:0"),
        split_item=2, group_type=0, group_list_type=0,
        output_dtype=torch.bfloat16,
    )[0].cpu()
    after = activation.cpu()
    changed = (before != after).sum().item()
    nrmse = (actual.float() - expected.float()).norm().item() / expected.float().norm().item()
    report = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "shape": list(activation.shape),
        "seed": 1234,
        "input_modified": changed != 0,
        "modified_elements": changed,
        "before_first32": before.flatten()[:32].tolist(),
        "after_first32": after.flatten()[:32].tolist(),
        "projection_nrmse_against_original_input": nrmse,
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
