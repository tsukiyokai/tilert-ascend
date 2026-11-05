"""Freeze three MTP call inputs from one real TP16 width-four round.

--capture-case contains the original main-prefix, mtp-prefix and capture
manifest.json. Each output stage (seed, chain, verify) contains input.pt
and prefix/rank-N.pt for run_mtp_reference.py and run_native_mtp.py.
The seed prefix is copied byte for byte. This captures inputs; it does
not establish numerical parity or measure performance.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback

from prefix_io import mtp_prefix_layer, prefix_prompt_identity, prompt_identity


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.pd_vllm.ascend_cache import (  # noqa: E402
    AscendRankEngine, RankPrefix, checkpoint_fingerprint,
)
from tilert.pd_vllm.ascend_layout import (  # noqa: E402
    FULL_INDEX_LAYERS, INT8_INDEX_LAYERS,
)
from tilert.pd_vllm.decode_timing import atomic_json  # noqa: E402
from tilert.pd_vllm.prefix_capture import (  # noqa: E402
    copy_to_cpu, digest, save_bundle,
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=9216)
    parser.add_argument("--expert-bank", action="store_true")
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args(argv)
    if args.capacity < 5:
        parser.error("Capacity must hold a prompt and four verification rows")
    return args


def gather(value):
    import torch.distributed as distributed

    values = [None] * 16
    distributed.all_gather_object(values, value)
    return values


def checked(label, operation):
    """Rendezvous on local failures before any subsequent model call."""
    value, error = None, None
    try:
        value = operation()
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    errors = gather(error)
    failed = {rank: error for rank, error in enumerate(errors) if error}
    if failed:
        raise RuntimeError(f"{label} failed: {failed}")
    return value


def require_equal(values, fields, label):
    expected = {name: values[0][name] for name in fields}
    if any({name: value[name] for name in fields} != expected
           for value in values):
        raise ValueError(f"Sixteen ranks disagree on {label}")


def read_capture(args, rank, checkpoint):
    manifest_path = args.capture_case / "manifest.json"
    saved = json.loads(manifest_path.read_text())
    identity = prompt_identity(saved.get("prompt_token_ids"))
    length, first = identity["prompt_length"], saved.get("first_token_id")
    if (saved.get("status") != "capture_complete"
            or saved.get("seq_len") != length
            or saved.get("prompt_sha256") != identity["prompt_sha256"]
            or type(first) is not int or not 0 <= first < 154880
            or saved.get("delivered_tokens") != [first]):
        raise ValueError("Capture needs a completed prompt/first-token pair")
    if args.capacity < length + 4:
        raise ValueError("Capacity must retain a full width-four round")
    if saved.get("runtime", {}).get("checkpoint_sha256") != checkpoint:
        raise ValueError("Capture checkpoint differs from decode")
    receipts = saved.get("ranks", [])
    expected = {
        "status": "saved", "request_id": saved["request_id"],
        "generation": saved["generation"], "seq_len": length,
        "mtp_seq_len": length - 1,
        "prompt_sha256": identity["prompt_sha256"],
        "checkpoint_sha256": checkpoint,
    }
    if (len(receipts) != 16
            or any(receipt.get("rank") != number
                   or any(receipt.get(key) != value
                          for key, value in expected.items())
                   for number, receipt in enumerate(receipts))):
        raise ValueError("Capture requires sixteen matching rank receipts")
    paths = {role: args.capture_case / f"{role}-prefix" / f"rank-{rank}.pt"
             for role in ("main", "mtp")}
    hashes = {role: digest(path) for role, path in paths.items()}
    if any(hashes[role] != receipts[rank]["files"][role]["sha256"]
           for role in paths):
        raise ValueError("Prefix file SHA256 differs from its capture receipt")
    return saved, identity, paths, hashes, digest(manifest_path)


def validate_layer(layer, length, number):
    import torch

    expected = {"latent": ((length, 512), torch.bfloat16),
                "rope": ((length, 64), torch.bfloat16)}
    if number in FULL_INDEX_LAYERS:
        dtype = torch.int8 if number in INT8_INDEX_LAYERS else torch.bfloat16
        expected["index_keys"] = ((length, 128), dtype)
    if number in INT8_INDEX_LAYERS:
        expected["index_scales"] = ((length,), torch.float16)
    if set(layer) != set(expected):
        raise ValueError(f"Layer {number} has different cache regions")
    for name, (shape, dtype) in expected.items():
        if layer[name].shape != shape or layer[name].dtype != dtype:
            raise ValueError(f"Layer {number} {name} shape or dtype differs")


def load_prefix(paths, capture, identity, checkpoint, device):
    import torch

    main = torch.load(paths["main"], map_location="cpu", weights_only=True)
    saved = torch.load(paths["mtp"], map_location="cpu", weights_only=True)
    length = identity["prompt_length"]
    if main.get("metadata_sha256") != checkpoint:
        raise ValueError("Main prefix checkpoint differs from decode")
    if (prefix_prompt_identity(main, length) != identity
            or main["prompt_token_ids"] != capture["prompt_token_ids"]):
        raise ValueError("Main prefix prompt differs from the captured prompt")
    layers, draft = main["layers"], mtp_prefix_layer(saved, checkpoint)
    if len(layers) != 78:
        raise ValueError("Main prefix requires all 78 layers")
    for number, layer in enumerate(layers):
        validate_layer(layer, length, number)
    validate_layer(draft, length - 1, 78)
    prefix = RankPrefix(
        [{name: value[:length - 1].to(device)
          for name, value in layer.items()} for layer in layers],
        {name: value.to(device) for name, value in draft.items()},
        identity["last_prompt_token"], length,
    )
    return prefix, draft


def content_digest(tensors, metadata=None):
    """Hash logical tensor bytes, dtype and shape, not Torch archive IDs."""
    import torch

    result = hashlib.sha256(json.dumps(
        metadata, sort_keys=True, separators=(",", ":"),
    ).encode("ascii"))
    for name, tensor in sorted(tensors.items()):
        result.update(json.dumps(
            [name, str(tensor.dtype), list(tensor.shape)],
            separators=(",", ":"),
        ).encode("ascii"))
        result.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


class CaptureDraft:
    """Observe forward boundaries while retaining the real round executor."""

    def __init__(self, draft, args, rank, checkpoint, identity, seed_prefix,
                 seed_path, seed_sha256):
        self.draft = draft
        self.args = args
        self.rank = rank
        self.checkpoint = checkpoint
        self.length = identity["prompt_length"]
        self.seed_prefix = seed_prefix
        self.seed_path = seed_path
        self.seed_sha256 = seed_sha256
        self.calls = 0
        self.stages = {}

    def __getattr__(self, name):
        return getattr(self.draft, name)

    def freeze(self, stage, tokens, parent, external, parent_kind):
        metadata = {"tokens": list(tokens), "position": self.position,
                    "parent_kind": parent_kind}
        tensors = {"normalized_parent_hidden": copy_to_cpu(parent)}
        if external is not None:
            tensors["external_indices"] = copy_to_cpu(external)
        inputs = {**metadata, **tensors}
        layer = self.draft.layer
        prefix = {
            "latent": copy_to_cpu(
                layer.mla.latent_cache.view(-1, 512)[:self.position]),
            "rope": copy_to_cpu(
                layer.mla.rope_cache.view(-1, 64)[:self.position]),
            "index_keys": copy_to_cpu(
                layer.indexer.keys.view(-1, 128)[:self.position]),
        }
        if (layer.mla.length != self.position
                or layer.indexer.length != self.position):
            raise ValueError("MTP valid cache lengths differ at capture")
        prefix_hash = content_digest(prefix)
        directory = self.args.output / stage
        path = directory / "prefix" / f"rank-{self.rank}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if stage == "seed":
            if prefix_hash != content_digest(self.seed_prefix):
                raise ValueError("Seed cache differs from the original prefix")
            shutil.copyfile(self.seed_path, path)
            if digest(path) != self.seed_sha256:
                raise ValueError("Seed prefix byte copy changed SHA256")
        else:
            save_bundle({"metadata_sha256": self.checkpoint,
                         "layer": prefix}, path)
        receipt = {
            "rank": self.rank, "stage": stage, **metadata,
            "width": len(tokens), "external_indices": external is not None,
            "input_content_sha256": content_digest(tensors, metadata),
            "prefix_content_sha256": prefix_hash,
            "prefix_sha256": digest(path),
        }
        return inputs, receipt

    def capture(self, stage, tokens, parent, external, parent_kind):
        inputs, receipt = checked(
            f"{stage} freeze",
            lambda: self.freeze(stage, tokens, parent, external, parent_kind),
        )
        receipts = gather(receipt)
        require_equal(receipts, (
            "stage", "tokens", "position", "parent_kind", "width",
            "external_indices", "input_content_sha256",
            "prefix_content_sha256",
        ), stage + " inputs and prefix contents")
        path = self.args.output / stage / "input.pt"
        checked(stage + " input write", lambda: (
            save_bundle(inputs, path) if self.rank == 0 else None
        ))
        self.stages[stage] = {
            **{key: value for key, value in receipt.items()
               if key not in ("rank", "prefix_sha256")},
            "input": f"{stage}/input.pt", "prefix": f"{stage}/prefix",
            "input_sha256": checked(stage + " input hash",
                                    lambda: digest(path)),
            "prefix_files_sha256": {
                f"rank-{item['rank']}.pt": item["prefix_sha256"]
                for item in receipts
            },
        }

    def forward(self, tokens, parent, **kwargs):
        sequence = (
            ("seed", 1, self.length - 1, False, "target_final_norm"),
            ("chain", 1, self.length, True, "mtp_recycle_norm"),
            (None, 1, self.length + 1, True, "mtp_recycle_norm"),
            ("verify", 4, self.length, False, "target_final_norm"),
        )
        external = kwargs.get("external_indices")

        def boundary():
            if self.calls >= len(sequence):
                raise ValueError("Capture permits exactly one M4 round")
            stage, width, position, shared, parent_kind = sequence[self.calls]
            if (len(tokens) != width or self.position != position
                    or (external is not None) != shared
                    or kwargs.get("full_logits") is not False):
                raise ValueError("MTP call differs from the M4 stage schedule")
            return stage, parent_kind

        stage, parent_kind = checked("MTP call boundary", boundary)
        if stage is not None:
            self.capture(stage, tokens, parent, external, parent_kind)
        self.calls += 1
        return self.draft.forward(tokens, parent, **kwargs)


def run(args, report):
    import torch
    import torch.distributed as distributed
    import torch_npu

    from tilert.models.glm_5_2_ascend.native_engine import NativeDecodeEngine
    from tilert.models.glm_5_2_ascend.native_mtp import NativeMtp

    torch.set_num_threads(1)
    device = torch.device(f"npu:{int(os.environ['LOCAL_RANK'])}")
    torch.npu.set_device(device)
    distributed.init_process_group("hccl")
    rank = distributed.get_rank()
    if distributed.get_world_size() != 16 or rank != device.index:
        raise ValueError("Capture requires one local TP16 group")
    checkpoint = checked("checkpoint metadata", lambda: (
        checkpoint_fingerprint(args.checkpoint)
    ))
    capture, identity, paths, hashes, capture_hash = checked(
        "capture input", lambda: read_capture(args, rank, checkpoint),
    )
    require_equal(gather({
        "manifest": capture_hash, "checkpoint": checkpoint,
        "sources": report["source_sha256"],
    }), ("manifest", "checkpoint", "sources"), "capture/checkpoint/source")
    checked("output directory", lambda: (
        args.output.mkdir(parents=True, exist_ok=False) if rank == 0 else None
    ))
    report.update(
        checkpoint_sha256=checkpoint, capture_manifest_sha256=capture_hash,
        capture_case=str(args.capture_case), request_id=capture["request_id"],
        prompt_token_ids=capture["prompt_token_ids"], **identity,
        first_token_id=capture["first_token_id"],
        packages={"torch": str(torch.__version__),
                  "torch_npu": str(torch_npu.__version__)},
    )
    prefix, original_mtp = checked("prefix load", lambda: load_prefix(
        paths, capture, identity, checkpoint, device,
    ))
    target = checked("main weights", lambda: NativeDecodeEngine(
        args.checkpoint, device, args.capacity,
        use_expert_bank=args.expert_bank,
    ))
    draft = checked("MTP weights", lambda: NativeMtp(target))
    observer = CaptureDraft(draft, args, rank, checkpoint, identity,
                            original_mtp, paths["mtp"], hashes["mtp"])
    engine = AscendRankEngine(target, observer)
    with torch.inference_mode():
        engine.inject(prefix)
        first_tokens = gather(engine.first_token_id)
        if first_tokens != [capture["first_token_id"]] * 16:
            raise ValueError("Prompt replay differs from the P first token")
        engine.start_mtp(capture["first_token_id"], 5, 4, ())
        executed = engine.advance_mtp()
        if observer.calls != 4 or set(observer.stages) != {"seed", "chain",
                                                         "verify"}:
            raise ValueError("The M4 round did not capture all three inputs")
        report["round"] = {
            "candidates": executed.candidates,
            "predictions": executed.target_predictions,
            "accepted_tokens": executed.commit.accepted_tokens,
            "main_position": target.position, "mtp_position": draft.position,
        }
        require_equal(gather(report["round"]), tuple(report["round"]),
                      "completed round")
        engine.finish_mtp(target.position)
        engine.reset()
    report.update(status="inputs_captured", numerical_execution=True,
                  stages=observer.stages, rank_input_contents_equal=True,
                  seed_prefix_bytes_preserved=True)
    checked("capture manifest", lambda: (
        atomic_json(args.output / "manifest.json", report)
        if rank == 0 else None
    ))
    distributed.destroy_process_group()


def main(argv=None):
    args = parse_arguments(argv)
    model = ROOT / "tilert/models/glm_5_2_ascend"
    sources = [Path(__file__), Path(__file__).with_name("prefix_io.py"),
               *(model / name for name in (
                   "native_engine.py", "native_mtp.py", "mtp_round.py",
                   "mtp_state.py", "config.py", "engine.py", "weights.py",
                   "moe_plan.py",
               )), *(model / "ops").glob("*.py"),
               *(ROOT / "tilert/pd_vllm" / name for name in (
                   "ascend_cache.py", "ascend_layout.py", "prefix_capture.py",
                   "decode_timing.py",
               ))]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "python": platform.python_version(),
        "command": [sys.executable, *sys.argv],
        "source_commit": args.source_commit,
        "source_sha256": {str(path.relative_to(ROOT)): digest(path)
                          for path in sources},
        "scope": "Actual MTP inputs and prefixes; numerical parity pending",
        "status": "error", "world_size": 16, "mtp_width": 4,
        "position_convention": "parent", "ignore_eos": True,
        "expert_backend": "bank" if args.expert_bank else "per_expert",
        "numerical_execution": False, "numerical_parity": False,
        "performance_measured": False,
    }
    try:
        run(args, report)
    except Exception:
        report["error"] = traceback.format_exc()
        print(json.dumps(report), file=sys.stderr)
        return 1
    if int(os.environ["RANK"]) == 0:
        print(json.dumps({"status": report["status"],
                          "manifest": str(args.output / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
