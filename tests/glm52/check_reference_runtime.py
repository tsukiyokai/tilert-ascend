"""Run the real 78-layer HF collector with a small synthetic checkpoint."""

import argparse
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import traceback

import torch
from safetensors.torch import save_file
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

from run_hf_reference import digest, validate_trace
from tilert.models.glm_5_2_ascend.reference import HfReference
from tilert.models.glm_5_2_ascend.ops.indexer import CANN_ARCH22_SOURCE_SHA256


def make_checkpoint(root):
    indexers = ["full" if i < 3 or (i - 2) % 4 == 0 else "shared"
                for i in range(78)]
    config = GlmMoeDsaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        moe_intermediate_size=32, num_hidden_layers=78,
        num_attention_heads=64, num_key_value_heads=64,
        n_routed_experts=16, num_experts_per_tok=8, n_shared_experts=1,
        kv_lora_rank=16, q_lora_rank=16, qk_rope_head_dim=64,
        qk_nope_head_dim=192, v_head_dim=8, index_n_heads=32,
        index_head_dim=128, indexer_types=indexers,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    with torch.device("meta"):
        model = GlmMoeDsaForCausalLM(config).to(torch.bfloat16)
    tensors = {}
    for name, template in model.state_dict().items():
        if ".experts.gate_up_proj" in name:
            prefix = name.removesuffix(".gate_up_proj")
            for expert in range(16):
                for projection in ("gate_proj", "up_proj"):
                    tensors[f"{prefix}.{expert}.{projection}.weight"] = (
                        torch.zeros(32, 16, dtype=torch.bfloat16)
                    )
        elif ".experts.down_proj" in name:
            prefix = name.removesuffix(".down_proj")
            for expert in range(16):
                tensors[f"{prefix}.{expert}.down_proj.weight"] = torch.zeros(
                    16, 32, dtype=torch.bfloat16
                )
        else:
            tensor = torch.zeros(template.shape, dtype=torch.bfloat16)
            if name.endswith(("layernorm.weight", ".k_norm.weight")) or (
                name == "model.norm.weight"
            ):
                tensor.fill_(1)
            tensors[name] = tensor
    embedding = tensors["model.embed_tokens.weight"]
    embedding[1].fill_(1)
    embedding[2].fill_(1)
    embedding[2, 0] = -1
    head = tensors["lm_head.weight"]
    head[:, :2] = -1
    head[0, :2] = torch.tensor([1, 0], dtype=torch.bfloat16)
    head[17, :2] = torch.tensor([1, 2**-8], dtype=torch.bfloat16)
    for number, kind in enumerate(indexers):
        prefix = f"model.layers.{number}.self_attn"
        tensors[prefix + ".q_a_proj.weight"].copy_(torch.eye(16))
        tensors[prefix + ".kv_a_proj_with_mqa.weight"][:16].copy_(
            torch.eye(16)
        )
        tensors[prefix + ".kv_a_proj_with_mqa.weight"][16:, 0] = 1
        if number >= 3:
            mlp = f"model.layers.{number}.mlp"
            tensors[mlp + ".gate.weight"][:, 0] = (
                torch.arange(16, dtype=torch.float32) / 8 - 1
            )
            tensors[mlp + ".gate.e_score_correction_bias"][1::2] = 2**-4
        if kind == "full":
            for projection in ("wq_b", "wk"):
                tensor = tensors[prefix + f".indexer.{projection}.weight"]
                tensor[0::2, 0] = 1
                tensor[1::2, 1] = 1
            tensors[prefix + ".indexer.wk.weight"][1::2, 1] = -1
            tensors[prefix + ".indexer.weights_proj.weight"][:, 1] = 2**-5
    recipe = {name: "FLOAT" for name in tensors}
    recipe["group_size"] = 0
    for number in range(6, 67, 4):
        recipe[f"model.layers.{number}.self_attn.indexer.quant_type"] = (
            "INT8_DYNAMIC"
        )
    (root / "config.json").write_text(config.to_json_string())
    (root / "quant_model_description.json").write_text(json.dumps(recipe))
    (root / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {
            name: "weights.safetensors" for name in tensors
        }})
    )
    save_file(tensors, root / "weights.safetensors")


def check(checks, name, passed, **values):
    checks.append({"name": name, "passed": bool(passed), **values})


def check_headers(checkpoint, source, checks):
    with TemporaryDirectory() as directory:
        changed = Path(directory)
        for name in CANN_ARCH22_SOURCE_SHA256:
            shutil.copyfile(source / name, changed / name)
        path = changed / next(iter(CANN_ARCH22_SOURCE_SHA256))
        path.write_bytes(path.read_bytes() + b"\n")
        try:
            HfReference(checkpoint, c8_source_directory=changed)
        except ValueError as error:
            rejected = "source hashes differ" in str(error)
        else:
            rejected = False
        check(checks, "changed_c8_source_rejected", rejected)
        try:
            HfReference(checkpoint, c8_source_directory=changed / "absent")
        except FileNotFoundError:
            rejected = True
        else:
            rejected = False
        check(checks, "missing_c8_source_rejected", rejected)


def check_trace(trace, precise, checks):
    validate_trace(trace, fp32_head=precise)
    token = trace["token"]
    hidden = torch.ones(1, 16, dtype=torch.bfloat16)
    if token == 2:
        hidden[0, 0] = -1
    first = float(hidden[0, 0])
    expected = torch.full((1, 32), -first - 1, dtype=torch.float32)
    expected[0, 0] = first
    expected[0, 17] = first + 2**-8
    if not precise:
        expected = expected.bfloat16()
    prefix = f"{'fp32' if precise else 'bf16'}.step{trace['position']}"
    check(checks, prefix + ".hidden_oracle", all(
        torch.equal(value, hidden) for value in trace["hidden_states"]
    ))
    check(checks, prefix + ".all_78_head_oracles", all(
        value.dtype == expected.dtype and torch.equal(value, expected)
        for value in (trace["logits"], *trace["layer_logits"])
    ))
    choices = sorted(range(16), key=lambda expert: (
        -(1 / (1 + math.exp(-first * (expert / 8 - 1)))
          + (2**-4 if expert % 2 else 0)), expert,
    ))[:8]
    check(checks, prefix + ".all_75_route_oracles", all(
        value.tolist() == choices for value in trace["routes"][3:]
    ), expected=choices)
    check(checks, prefix + ".all_78_latent_oracles", all(
        torch.equal(row["latent"], hidden[0]) for row in trace["cache_rows"]
    ))
    check(checks, prefix + ".all_78_uniform_attention_oracles", all(
        torch.equal(value, torch.full_like(value, 1 / (
            trace["position"] + 1
        ))) for value in trace["attention_weights"]
    ))
    check(checks, prefix + ".nonzero_rope_rows", all(
        bool(torch.isfinite(row["rope"]).all())
        and bool(row["rope"].count_nonzero()) for row in trace["cache_rows"]
    ))
    if token == 1:
        check(checks, prefix + ".nonzero_c8_keys_and_scales", all(
            row["index"].dtype == torch.int8
            and bool(row["index"].count_nonzero())
            and bool(row["index_scale"].gt(0).all())
            for row in trace["cache_rows"] if row["index_scale"] is not None
        ))


def run(args, report):
    torch.set_num_threads(1)
    project = Path(__file__).resolve().parents[2]
    checkpoint = args.output / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    make_checkpoint(checkpoint)
    report["checkpoint_sha256"] = {
        path.name: digest(path) for path in checkpoint.iterdir()
    }
    report["c8_source_sha256"] = {
        name: digest(args.c8_source_directory / name)
        for name in CANN_ARCH22_SOURCE_SHA256
    }
    checks = report["checks"]
    check_headers(checkpoint, args.c8_source_directory, checks)
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTHONPATH": os.pathsep.join((str(project),
                                                  str(Path(__file__).parent)))}
    for precise in (False, True):
        name = "fp32" if precise else "bf16"
        output = args.output / name
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_hf_reference.py")),
            "--checkpoint", str(checkpoint), "--output", str(output),
            "--tokens", "1", "2", "1", "--generate", "1", "--capacity", "3",
            "--source-commit", report["git_head"],
            "--c8-source-directory", str(args.c8_source_directory),
        ]
        if precise:
            command.append("--fp32-head")
        process = subprocess.run(command, env=environment, text=True,
                                 capture_output=True)
        manifest = json.loads((output / "manifest.json").read_text())
        check(checks, name + ".collector", process.returncode == 0
              and manifest["status"] == "pass", command=command,
              returncode=process.returncode,
              checks_total=manifest["checks_total"],
              checks_failed=manifest["checks_failed"],
              manifest_sha256=digest(output / "manifest.json"))
        if process.returncode:
            raise RuntimeError(manifest.get("error", process.stderr))
        check(checks, name + ".canonical_c8_source", manifest[
            "weight_arithmetic"]["c8_index_scoring"]["source_directory"]
              == str(args.c8_source_directory.resolve()))
        check(checks, name + ".greedy_head_difference",
              manifest["generated_tokens"] == [17 if precise else 0])
        for path in sorted(output.glob("step-*.pt")):
            trace = torch.load(path, map_location="cpu", weights_only=True)
            check_trace(trace, precise, checks)
    report["status"] = "pass" if all(c["passed"] for c in checks) else "fail"


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c8-source-directory", type=Path, default=(
        project / "docs/evidence/ascend-decode/c8-arch22"
    ))
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.c8_source_directory = args.c8_source_directory.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running", "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=project
        ).strip(), "command": [sys.executable, *sys.argv],
        "hostname": platform.node(), "python": platform.python_version(),
        "torch": str(torch.__version__), "scope": (
            "Synthetic 78-layer native HF execution; hidden/latent rank 16, "
            "vocab 32, floating weights, C8 full/shared indexer schedule, "
            "zero attention/FFN outputs, nonzero caches and changing routing; "
            "two teacher inputs over three positions, reset and disk readback"
        ),
        "precision_verified": False, "formal_precision_gate_passed": False,
        "performance_measured": False,
        "source_sha256": {
            str(path.relative_to(project)): digest(path) for path in (
                Path(__file__).resolve(),
                Path(inspect.getsourcefile(HfReference)).resolve(),
            )
        },
        "checks": [],
    }
    try:
        run(args, report)
    except Exception:
        report.update(status="error", error=traceback.format_exc())
    report["checks_total"] = len(report["checks"])
    report["checks_failed"] = sum(not c["passed"] for c in report["checks"])
    path = args.output / "runtime.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(path)}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
