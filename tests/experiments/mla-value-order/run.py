"""Isolate BF16 Value reassociation with one frozen HF probability row."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def digest(path):
    hashed = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 2**20):
            hashed.update(block)
    return hashed.hexdigest()


@torch.backends.mkldnn.flags(
    enabled=False, deterministic=None, allow_tf32=None, fp32_precision=None
)
def prepare(args):
    from check_mla import metrics

    from tilert.models.glm_5_2_ascend.reference_weights import ReferenceWeights

    reference = args.root / "hub/trt-a3-reference-20261028-2205/complete"
    hf_path = reference / "step-08192.pt"
    prefix_path = (
        args.root
        / "assets/checkpoint-owned/prefix8k-v3/case-00/main-prefix/rank-0.pt"
    )
    manifest = json.loads((reference / "manifest.json").read_text())
    assert digest(hf_path) == manifest["trace_sha256"][hf_path.name]
    assert digest(prefix_path) == manifest["prefix_files_sha256"]["rank-0.pt"]
    hf = torch.load(hf_path, map_location="cpu", weights_only=True)
    prefix = torch.load(prefix_path, map_location="cpu", weights_only=True)
    cached = prefix["layers"][0]
    current = hf["cache_rows"][0]
    latent = torch.cat((cached["latent"], current["latent"].reshape(1, 512)))
    rope = torch.cat((cached["rope"], current["rope"].reshape(1, 64)))
    probability = hf["attention_weights"][0][:4].unsqueeze(1).contiguous()
    assert probability.shape == (4, 1, 8193)
    assert probability.dtype == latent.dtype == torch.bfloat16
    assert latent.shape == (8193, 512) and rope.shape == (8193, 64)

    weights = ReferenceWeights(Path("/models/glm52"))
    name = "model.layers.0.self_attn.kv_b_proj"
    assert weights.quantization.get(name + ".weight", "FLOAT") == "FLOAT"
    weight = weights.tensor(
        name + ".weight", rows=(0, 1792), dtype=torch.bfloat16
    )
    assert weight.shape == (1792, 512)
    expanded_prefix = weights._floating_product(
        name, cached["latent"], rows=(0, 1792)
    )
    expanded_current = weights._floating_product(
        name, current["latent"].reshape(1, 512), rows=(0, 1792)
    )
    expanded = torch.cat((expanded_prefix, expanded_current)).reshape(
        8193, 4, 448
    )
    value = expanded[:, :, 192:].permute(1, 0, 2).contiguous()
    value_weight = (
        weight.reshape(4, 448, 512)[:, 192:].transpose(1, 2).contiguous()
    )
    cpu_expanded = torch.matmul(probability, value)
    cpu_latent = torch.matmul(torch.matmul(probability, latent), value_weight)
    values = {
        "probability": probability,
        "latent": latent,
        "rope": rope,
        "value": value,
        "value_weight": value_weight,
        "cpu_expanded": cpu_expanded,
        "cpu_latent": cpu_latent,
    }
    for tensor in values.values():
        assert tensor.dtype == torch.bfloat16 and bool(
            torch.isfinite(tensor).all()
        )
    identity = {
        "hf_sha256": digest(hf_path),
        "prefix_sha256": digest(prefix_path),
        "prepare_script_sha256": digest(__file__),
        "rank": 0,
        "layer": 0,
        "weight_tensor_sha256": hashlib.sha256(
            weight.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "position": 8192,
        "probabilities_recomputed": False,
        "hf_pv_output_directly_observed": False,
        "cpu_mkldnn_enabled": torch.backends.mkldnn.enabled,
        "prefix_and_current_projected_separately": True,
        "cpu_latent_vs_expanded": metrics(cpu_latent, cpu_expanded),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"identity": identity, "tensors": values}, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )
    print(json.dumps(identity))


def replay(args):
    import torch_npu
    from check_mla import compare

    torch.npu.set_device(0)
    fixture = torch.load(args.input, map_location="cpu", weights_only=True)
    cpu = fixture["tensors"]
    data = {
        key: cpu[key].npu()
        for key in ("probability", "latent", "value", "value_weight")
    }
    checks, outputs = [], {}
    with torch.inference_mode():
        for arm in ("expanded", "latent"):
            values = []
            for _ in range(32):
                if arm == "expanded":
                    result = torch.matmul(data["probability"], data["value"])
                else:
                    hidden = torch.matmul(data["probability"], data["latent"])
                    result = torch.matmul(hidden, data["value_weight"])
                torch.npu.synchronize()
                values.append(result.cpu().contiguous())
            outputs[arm] = values
            compare(
                checks,
                arm + "_vs_cpu_expanded",
                values[0],
                cpu["cpu_expanded"],
                "fixed_probability_value",
            )
            compare(
                checks,
                arm + "_vs_own_cpu",
                values[0],
                cpu["cpu_" + arm],
                "fixed_probability_value",
            )
            checks.append(
                {
                    "name": arm + "_repeat_raw_bytes",
                    "passed": all(
                        torch.equal(
                            value.view(torch.uint8),
                            values[0].view(torch.uint8),
                        )
                        for value in values[1:]
                    ),
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, args.output / "outputs.pt")
    report = {
        "scope": "Rank0/layer0 frozen HF probability, Value-only BF16 reassociation; not a full attention replay",
        "input_sha256": digest(args.input),
        "script_sha256": digest(__file__),
        "identity": fixture["identity"],
        "checks": checks,
        "device": str(torch.npu.get_device_properties(0)),
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "torch_npu_extension_sha256": digest(torch_npu._C.__file__),
        "npu_executed": True,
        "performance_measured": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("prepare", "replay"))
parser.add_argument("--root", type=Path, default=Path("/work"))
parser.add_argument("--source", type=Path, default=Path("/work/source-v12"))
parser.add_argument("--input", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path[:0] = [str(args.source), str(args.source / "tests/glm52")]
if args.mode == "replay" and args.input is None:
    parser.error("replay requires --input")
prepare(args) if args.mode == "prepare" else replay(args)
