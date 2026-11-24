"""Capture one TP16 MTP call and replay; optionally compare an oracle trace.

Run with torchrun --nproc-per-node=16. --input is a torch.save dictionary
with tokens, normalized_parent_hidden (BF16 [M,6144]), parent_kind
('target_final_norm' or 'mtp_recycle_norm'), position, and optional
external_indices ([M,2048]). For position > 0, --prefix contains rank-N.pt
bundles with metadata_sha256 and layer (latent/rope/BF16 index_keys).

--reference contains manifest.json and rank-N.pt snapshots. The manifest
must identify a passing independent implementation and its source hashes,
the input SHA256, checkpoint metadata SHA256 values, and parent-position
semantics. rank_sha256 maps each rank-N.pt filename to its SHA256.
Without this artifact, status is capture_replay, not parity.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback

from prefix_io import mtp_prefix_layer


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=9216)
    parser.add_argument("--expert-bank", action="store_true")
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def snapshot(mtp, result):
    def cpu(value):
        return value.detach().cpu().clone()

    start = result.start_position
    end = mtp.position
    return {
        "position": start, "committed_position": end,
        "logits": cpu(result.logits),
        "pre_norm_hidden": cpu(result.pre_norm_hidden),
        "recycle_hidden": cpu(result.recycle_hidden),
        "indices": cpu(result.indices), "routes": cpu(result.routes),
        "stages": {name: cpu(value) for name, value in result.stages.items()},
        "cache": {
            "latent": cpu(mtp.layer.mla.latent_cache.view(-1, 512)[start:end]),
            "rope": cpu(mtp.layer.mla.rope_cache.view(-1, 64)[start:end]),
            "index_keys": cpu(mtp.layer.indexer.keys.view(-1, 128)[start:end]),
        },
    }


def tensor_checks(actual, expected, checks, exact=False, prefix=""):
    import torch
    import torch.nn.functional as functional

    for key, left in actual.items():
        name = prefix + key
        if key not in expected:
            checks.append({"name": name, "passed": False,
                           "error": "missing reference field"})
            continue
        right = expected[key]
        if isinstance(left, dict):
            tensor_checks(left, right, checks, exact, name + ".")
        elif not isinstance(left, torch.Tensor):
            checks.append({"name": name, "passed": left == right})
        elif exact:
            checks.append({"name": name, "passed": torch.equal(left, right)})
        elif left.shape != right.shape or left.dtype != right.dtype:
            checks.append({"name": name, "passed": False,
                           "error": "shape or dtype mismatch"})
        elif not left.is_floating_point():
            if key in ("indices", "routes"):
                left = left.sort(dim=-1).values
                right = right.sort(dim=-1).values
            checks.append({"name": name, "passed": torch.equal(left, right)})
        else:
            a, b = left.float(), right.float()
            difference = a - b
            nrmse = difference.norm().item() / max(b.norm().item(), 1e-12)
            cosine = (1.0 if torch.equal(a, b) else
                      functional.cosine_similarity(
                          a.flatten(), b.flatten(), dim=0
                      ).item())
            maximum = difference.abs().max().item()
            passed = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            passed = passed and nrmse <= 1e-2 and cosine >= 0.999
            if key == "logits":
                passed = (passed and maximum <= 0.1
                          and torch.equal(a.argmax(-1), b.argmax(-1)))
            checks.append({"name": name, "passed": passed,
                           "nrmse": nrmse, "cosine": cosine,
                           "max_abs": maximum})


def validate_reference(
    path, input_hash, checkpoint_hashes, rank, prefix_hash=None,
):
    from tilert.models.glm_5_2_ascend.ops.indexer import (
        cann_arch22_scoring_contract,
    )

    manifest = json.loads((path / "manifest.json").read_text())
    if (manifest.get("status") != "pass"
            or manifest.get("implementation")
            != "transformers.GlmMoeDsaDecoderLayer.MTP"
            or not manifest.get("source_sha256")):
        raise ValueError("Reference must identify a passing oracle and hashes")
    if manifest.get("numerical_execution") is not True:
        raise ValueError("MTP reference requires numerical execution")
    scorer = "tilert/models/glm_5_2_ascend/ops/indexer.py"
    expected = cann_arch22_scoring_contract()["reference_source_sha256"]
    if manifest["source_sha256"].get(scorer) != expected:
        raise ValueError("MTP reference needs its current scorer dependency")
    if manifest.get("input_sha256") != input_hash:
        raise ValueError("Reference uses a different MTP input")
    if manifest.get("checkpoint_sha256") != checkpoint_hashes:
        raise ValueError("Reference uses different checkpoint metadata")
    if manifest.get("position_convention") != "parent":
        raise ValueError("Reference must use parent RoPE/cache positions")
    filename = f"rank-{rank}.pt"
    if manifest.get("prefix_files_sha256", {}).get(filename) != prefix_hash:
        raise ValueError("Reference uses a different MTP cache prefix")
    hashes = manifest.get("rank_sha256", {})
    if hashes.get(filename) != digest(path / filename):
        raise ValueError(f"Reference tensor SHA256 mismatch: {filename}")
    return manifest


def execute(args, report):
    import torch
    import torch.distributed as distributed
    import torch_npu

    from tilert.models.glm_5_2_ascend.native_engine import NativeDecodeEngine
    from tilert.models.glm_5_2_ascend.native_mtp import NativeMtp

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    report["packages"] = {
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
    }
    report["world_size"] = int(os.environ["WORLD_SIZE"])
    report["local_rank"] = local_rank
    if int(os.environ["WORLD_SIZE"]) != 16:
        raise ValueError("MTP runner requires sixteen ranks")
    torch.npu.set_device(local_rank)
    distributed.init_process_group("hccl")
    device = torch.device(f"npu:{local_rank}")
    report["device"] = str(device)
    inputs = torch.load(args.input, map_location="cpu", weights_only=True)
    parent_kind = inputs["parent_kind"]
    if parent_kind not in ("target_final_norm", "mtp_recycle_norm"):
        raise ValueError("Input must declare the normalized parent's role")
    position = inputs["position"]
    if (isinstance(position, bool) or not isinstance(position, int)
            or position < 0):
        raise ValueError("Input must use a nonnegative parent position")
    if not isinstance(inputs["tokens"], (tuple, list)):
        raise ValueError("Input tokens must be a sequence")
    if position and args.prefix is None:
        raise ValueError("Nonzero position requires an MTP prefix")
    parent = inputs["normalized_parent_hidden"].to(device)
    external = inputs.get("external_indices")
    if external is not None:
        external = external.to(device)
    base = NativeDecodeEngine(
        args.checkpoint, device, args.capacity,
        use_expert_bank=args.expert_bank,
    )
    mtp = NativeMtp(base)
    prefix = None
    if args.prefix is not None:
        path = args.prefix / f"rank-{rank}.pt"
        report["prefix_sha256"] = digest(path)
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        prefix = {
            name: value.to(device) for name, value in mtp_prefix_layer(
                bundle, report["checkpoint_sha256"]
            ).items()
        }

    def initialize():
        mtp.reset()
        if prefix is not None:
            mtp.inject(prefix)
        if mtp.position != position:
            raise ValueError("Prefix length differs from parent position")

    initialize()
    result = mtp.forward(inputs["tokens"], parent, trace=True,
                         external_indices=external)
    actual = snapshot(mtp, result)
    initialize()
    replay = snapshot(mtp, mtp.forward(
        inputs["tokens"], parent, trace=True, external_indices=external
    ))
    checks = []
    tensor_checks(actual, replay, checks, exact=True, prefix="replay.")
    checks.append({"name": "backbone_position_unchanged",
                   "passed": base.position == 0})
    initialize()
    greedy = mtp.forward(inputs["tokens"], parent, full_logits=False,
                         external_indices=external)
    checks.append({"name": "distributed_greedy",
                   "passed": torch.equal(greedy.token_ids.cpu(),
                                         actual["logits"].argmax(-1))})
    if args.reference is not None:
        expected = torch.load(args.reference / f"rank-{rank}.pt",
                              map_location="cpu", weights_only=True)
        tensor_checks(actual, expected, checks, prefix="oracle.")
    destination = args.output / f"rank-{rank}.pt"
    torch.save(actual, destination)
    passed = all(item["passed"] for item in checks)
    status = "parity" if args.reference is not None else "capture_replay"
    report.update(
        status=status if passed else "fail", passed=passed, checks=checks,
        contract=mtp.contract, parent_kind=parent_kind,
        position=position, width=len(inputs["tokens"]),
        external_indices=external is not None,
        trace_sha256=digest(destination),
    )


def main():
    args = parse_arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    root = Path(__file__).resolve().parents[2]
    model = root / "tilert/models/glm_5_2_ascend"
    source_paths = [
        model / name for name in (
            "native_mtp.py", "native_engine.py", "config.py", "engine.py",
            "weights.py", "moe_plan.py",
        )
    ]
    source_paths.extend(sorted((model / "ops").glob("*.py")))
    source_paths.append(Path(__file__).resolve())
    source_paths.append(Path(__file__).with_name("prefix_io.py"))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "python": sys.version, "rank": rank,
        "source_commit": args.source_commit,
        "command": [sys.executable, *sys.argv], "status": "error",
        "passed": False, "performance_measured": False,
        "expert_backend": "bank" if args.expert_bank else "per_expert",
        "input_sha256": digest(args.input),
        "checkpoint_sha256": {
            name: digest(args.checkpoint / name) for name in (
                "config.json", "quant_model_description.json",
                "quant_model_weights.safetensors.index.json",
            )
        },
        "source_sha256": {
            str(path.relative_to(root)): digest(path) for path in source_paths
        },
    }
    try:
        report["prefix_sha256"] = (
            digest(args.prefix / f"rank-{rank}.pt")
            if args.prefix is not None else None
        )
        if args.reference is not None:
            report["oracle"] = validate_reference(
                args.reference, report["input_sha256"],
                report["checkpoint_sha256"], rank,
                prefix_hash=report["prefix_sha256"],
            )
            report["oracle_manifest_sha256"] = digest(
                args.reference / "manifest.json"
            )
        execute(args, report)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        (args.output / f"rank-{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        distributed = sys.modules.get("torch.distributed")
        if distributed is not None and distributed.is_initialized():
            distributed.destroy_process_group()
    print(json.dumps({"rank": rank, "status": report["status"],
                      "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
