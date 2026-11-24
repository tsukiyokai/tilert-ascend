"""Compare fused and separate norm/quant on one frozen post-attention row.

Preparation uses CPU only. Device execution requires the run subcommand
and an explicit NPU. M8 repeats the saved M1 operands eight times.
"""

import argparse
import ast
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch


ARTIFACT_SHA = (
    "451d95d43dcc1a18a3ddcc5c9463c23f829eda08860391b75b50c9090f097ebe"
)
STAGES_SHA = (
    "9b9fa90d36d785853e7b61f16f186a6e7ad6192797e3d19e37d5cab21faa3bd3"
)
NORM = "model.layers.0.post_attention_layernorm"
EPSILON = 1e-5
WIDTHS = (1, 8)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_digest(value):
    raw = value.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def compiled_contract(path):
    if digest(path) != ARTIFACT_SHA:
        raise ValueError("Historical compiled artifact SHA differs")
    source = path.read_text()
    lines = source.splitlines()
    calls = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call)]
    names = Counter(ast.unparse(node.func) for node in calls)
    fused_name = "torch.ops.npu.npu_add_rms_norm_dynamic_quant.default"
    sites = []
    for node in calls:
        if ast.unparse(node.func) != fused_name:
            continue
        keywords = {item.arg: ast.unparse(item.value)
                    for item in node.keywords}
        if (keywords.get("epsilon") != "1e-05"
                or keywords.get("output_mask") != "[True, False]"
                or set(keywords) != {"beta", "epsilon", "output_mask"}):
            raise ValueError("Historical fused norm arguments differ")
        sites.append({
            "line": node.lineno,
            "arguments": [ast.unparse(value) for value in node.args],
            "keywords": keywords,
            "fragment_first_line": node.lineno - 3,
            "fragment": "\n".join(
                lines[node.lineno - 4:node.lineno + 10],
            ),
        })
    if len(sites) != 3:
        raise ValueError("Expected exactly three dense post-attention sites")
    return {
        "artifact_sha256": ARTIFACT_SHA,
        "sites": sorted(sites, key=lambda value: value["line"]),
        "dense_swiglu_calls": names["torch.ops.npu.npu_swiglu.default"],
        "dynamic_quant_calls": names[
            "torch.ops.npu.npu_dynamic_quant.default"
        ],
        "fused_dequant_swiglu_calls": sum(
            count for name, count in names.items()
            if "dequant_swiglu" in name
        ),
        "native_compiled_elf_identity": "unknown",
    }


def checkpoint_tensor(checkpoint, weights, name):
    from safetensors import safe_open

    with safe_open(checkpoint / weights[name], framework="pt",
                   device="cpu") as stream:
        return stream.get_tensor(name).clone().contiguous()


def prepare(args):
    directory = args.directory
    report_path = directory / "frozen-report.json"
    stages_path = directory / "frozen-stages.pt"
    report = json.loads(report_path.read_text())
    if (report["status"] != "pass" or digest(stages_path) != STAGES_SHA
            or report["output"]["sha256"] != STAGES_SHA):
        raise ValueError("Frozen post-attention operand binding differs")
    metadata_hashes = report["checkpoint_sha256"]
    for name, expected in metadata_hashes.items():
        if digest(args.checkpoint / name) != expected:
            raise ValueError("Checkpoint metadata differs: " + name)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        frozen = torch.load(stages_path, map_location="cpu",
                            weights_only=True)
    stages = frozen["stages"]
    inputs = {"x": stages["attention.output"].reshape(1, 6144).clone(),
              "residual": stages["h0"].reshape(1, 6144).clone()}
    index_path = args.checkpoint / "quant_model_weights.safetensors.index.json"
    weights = json.loads(index_path.read_text())["weight_map"]
    gamma = checkpoint_tensor(args.checkpoint, weights, NORM + ".weight")
    gamma = gamma.to(torch.bfloat16)
    if not torch.equal(gamma, stages["postnorm.gamma"].reshape(6144)):
        raise ValueError("Frozen gamma differs from deployed checkpoint gamma")
    bias_name = NORM + ".bias"
    if bias_name in weights:
        beta = checkpoint_tensor(args.checkpoint, weights, bias_name)
        beta = beta.to(torch.bfloat16)
        beta_source = "checkpoint tensor"
    else:
        description_path = args.checkpoint / "quant_model_description.json"
        description = json.loads(description_path.read_text())
        if not any("norm.bias" in name for name in description):
            raise ValueError("Installed norm's zero-bias creation is unbound")
        beta = torch.zeros_like(gamma)
        beta_source = "missing tensor; installed norm creates BF16 zero bias"
    config = json.loads((args.checkpoint / "config.json").read_text())
    if config["rms_norm_eps"] != EPSILON:
        raise ValueError("Checkpoint epsilon differs from compiled site")
    inputs.update(gamma=gamma, beta=beta)
    if any(value.dtype != torch.bfloat16 or not torch.isfinite(value).all()
           for value in inputs.values()):
        raise ValueError("Operands must be finite deployed BF16 tensors")
    output = directory / "inputs.pt"
    if output.exists():
        raise FileExistsError(output)
    torch.save(inputs, output)
    contract = compiled_contract(directory / "historical-compiled.py")
    write_json(directory / "compiled-contract.json", contract)
    receipt = {
        "status": "prepared", "device_execution": False,
        "torch_npu_imported": "torch_npu" in sys.modules,
        "scope": "Actual frozen HF layer0 post-attention M1 operands; "
                 "M8 repeats this same row, not historical M8 operands",
        "frozen_stages_sha256": STAGES_SHA,
        "frozen_report_sha256": digest(report_path),
        "checkpoint_metadata_sha256": metadata_hashes,
        "gamma_matches_current_checkpoint": True,
        "beta_source": beta_source, "epsilon": EPSILON,
        "smooth_scale1": None, "smooth_scale2": None,
        "output_mask": [True, False], "widths": list(WIDTHS),
        "inputs_sha256": digest(output),
        "input_tensors": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "sha256": tensor_digest(value)}
            for name, value in inputs.items()
        },
        "historical_artifact_sha256": ARTIFACT_SHA,
        "script_sha256": digest(__file__),
    }
    write_json(directory / "prepared.json", receipt)
    print(json.dumps(receipt, indent=2))


def difference(left, right):
    left, right = left.detach().cpu(), right.detach().cpu()
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Compared tensor contracts differ")
    different = left != right
    delta = left.double() - right.double()
    return {
        "bitexact": torch.equal(left.contiguous().view(torch.uint8),
                                right.contiguous().view(torch.uint8)),
        "different": int(different.sum()),
        "elements": left.numel(), "maxabs": float(delta.abs().max()),
        "first_indices": different.nonzero()[:8].tolist(),
    }


def run(args):
    import torch_npu
    import vllm_ascend.vllm_ascend_C
    from vllm_ascend.utils import enable_custom_op

    directory = args.directory
    prepared = json.loads((directory / "prepared.json").read_text())
    if (digest(directory / "inputs.pt") != prepared["inputs_sha256"]
            or prepared["historical_artifact_sha256"] != ARTIFACT_SHA):
        raise ValueError("Prepared operands or historical artifact differ")
    device = torch.device(args.device)
    if device.type != "npu" or device.index is None:
        raise ValueError("Specify one explicit NPU")
    torch.npu.set_device(device)
    torch.npu.set_per_process_memory_fraction(
        256 * 1024 ** 2 / torch.npu.get_device_properties(device).total_memory,
        device,
    )
    if not enable_custom_op():
        raise RuntimeError("Production custom norm is unavailable")
    frozen = torch.load(directory / "inputs.pt", map_location="cpu",
                        weights_only=True)
    outputs, cases = {}, []
    with torch.inference_mode():
        for width in WIDTHS:
            x = frozen["x"].repeat(width, 1).to(device)
            residual = frozen["residual"].repeat(width, 1).to(device)
            gamma, beta = (frozen[name].to(device)
                           for name in ("gamma", "beta"))
            normal, rstd, separate_residual = (
                torch.ops._C_ascend.npu_add_rms_norm_bias(
                    x.clone(), residual.clone(), gamma, beta, EPSILON,
                )
            )
            separate_q, separate_scale = torch_npu.npu_dynamic_quant(
                normal, dst_type=torch.int8,
            )
            fused_q, _, fused_residual, fused_scale, _ = (
                torch.ops.npu.npu_add_rms_norm_dynamic_quant.default(
                    x.clone(), residual.clone(), gamma, beta=beta,
                    epsilon=EPSILON, output_mask=[True, False],
                )
            )
            torch.npu.synchronize(device)
            values = {
                "x": x, "residual_input": residual,
                "separate_norm_bf16": normal, "separate_rstd": rstd,
                "separate_q": separate_q, "separate_scale": separate_scale,
                "separate_residual": separate_residual,
                "fused_q": fused_q, "fused_scale": fused_scale,
                "fused_residual": fused_residual,
            }
            values = {name: value.cpu().clone()
                      for name, value in values.items()}
            total = frozen["x"].float() + frozen["residual"].float()
            fp32_norm = total * torch.rsqrt(
                total.square().mean(-1, keepdim=True) + EPSILON,
            ) * frozen["gamma"].float() + frozen["beta"].float()
            values["cpu_fp32_norm_diagnostic"] = fp32_norm.repeat(width, 1)
            values["cpu_fp32_norm_cast_bf16"] = (
                fp32_norm.bfloat16().repeat(width, 1)
            )
            outputs[str(width)] = values
            cases.append({
                "width": width,
                "input_kind": "actual M1 pair" if width == 1
                              else "eight repeats of actual M1 pair",
                "q": difference(fused_q, separate_q),
                "scale": difference(fused_scale, separate_scale),
                "residual": difference(fused_residual, separate_residual),
            })
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output / "outputs.pt"
    torch.save({"outputs": outputs, "inputs": frozen}, raw)
    report = {
        "status": "completed", "device_execution": True,
        "scope": prepared["scope"], "device": str(device),
        "cases": cases, "timing_measured": False,
        "fused_normalized_tensor": "not exposed by fused operator",
        "cpu_formula": "IEEE FP32 diagnostic; native reduction/Div "
                       "equivalence is not asserted",
        "prepared_sha256": digest(directory / "prepared.json"),
        "outputs_sha256": digest(raw), "script_sha256": digest(__file__),
        "hostname": platform.node(), "command": sys.argv,
        "packages": {name: version(name) for name in
                     ("torch", "torch-npu", "vllm-ascend")},
        "peak_allocated": torch.npu.max_memory_allocated(device),
        "native_compiled_historical_elf_identity": "unknown",
    }
    write_json(args.output / "result.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path(__file__).resolve().parent)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--checkpoint", type=Path, required=True)
    device = commands.add_parser("run")
    device.add_argument("--device", required=True)
    device.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
