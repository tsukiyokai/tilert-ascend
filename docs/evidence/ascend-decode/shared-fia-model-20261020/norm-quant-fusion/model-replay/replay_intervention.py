"""Propagate only the three dense norm/quant substitutions through the model.

The existing layer-resident runner owns weights, KV, TP and teacher checks.
Both arms pass prepared INT8 input into the unchanged gate/up projection.
This is an eager numerical intervention, not a compiled request benchmark.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


SOURCE = Path("/workspace/experiments/m32-full-teacher-20261020"
              "/two-rounds/source")
sys.path[:0] = [str(SOURCE / "tests/glm52"), str(SOURCE)]

import run_cann_layer_major as runner


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--norm-fusion", choices=("on", "off"), required=True)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = runner.arguments()
    rank = int(os.environ["RANK"])
    original = runner.run_layer
    receipts = []

    def run_layer(engine, factory, checkpoint, number, *tail, **kwargs):
        if number >= 3:
            return original(engine, factory, checkpoint, number, *tail,
                            **kwargs)
        import torch
        import torch_npu
        from vllm_ascend.quantization.prepared_linear_input import (
            PreparedLinearInput,
        )
        from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
            AscendW8A8DynamicLinearMethod,
        )

        snapshots = []

        def intercepted_factory(*factory_args, **factory_kwargs):
            layer = factory(*factory_args, **factory_kwargs)
            gate = layer.mlp.gate_up_proj
            scheme = gate.quant_method.quant_method
            if (type(scheme) is not AscendW8A8DynamicLinearMethod
                    or gate.custom_op is not None):
                raise TypeError("Expected the deployed dense W8A8 projection")
            norm = layer.post_attention_layernorm

            def norm_forward(x, residual):
                if options.norm_fusion == "on":
                    q, _, retained, scale, _ = (
                        torch_npu.npu_add_rms_norm_dynamic_quant(
                            x, residual, norm.weight, beta=norm.bias,
                            epsilon=norm.variance_epsilon,
                            output_mask=[True, False],
                        )
                    )
                else:
                    value, _, retained = (
                        torch.ops._C_ascend.npu_add_rms_norm_bias(
                            x, residual, norm.weight, norm.bias,
                            norm.variance_epsilon,
                        )
                    )
                    q, scale = torch_npu.npu_dynamic_quant(
                        value, dst_type=torch.int8,
                    )
                snapshots.append({
                    name: value.detach().cpu().clone()
                    for name, value in {
                        "x": x, "residual": residual,
                        "gamma": norm.weight, "beta": norm.bias,
                        "q": q, "scale": scale, "retained": retained,
                    }.items()
                })
                return PreparedLinearInput(q, scale, x.dtype), retained

            norm.forward = norm_forward
            return layer

        result = original(engine, intercepted_factory, checkpoint, number,
                          *tail, **kwargs)
        if len(snapshots) != 2:
            raise ValueError("Expected exactly two dense norm calls per layer")
        path = args.output / f"rank-{rank}/norm-layer-{number:02d}.pt"
        torch.save(snapshots, path)
        receipts.append({"layer": number, "calls": len(snapshots),
                         "path": str(path), "sha256": digest(path)})
        return result

    with patch.object(runner, "run_layer", run_layer):
        status = runner.main()
    result = {
        "arm": options.norm_fusion, "returncode": status,
        "scope": "Only dense post-attention norm/quant arithmetic changes; "
                 "eager layer-major execution, no timing",
        "script_sha256": digest(__file__),
        "runner_sha256": digest(inspect.getfile(original)),
        "records": receipts,
    }
    path = args.output / f"norm-intervention-rank-{rank}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
