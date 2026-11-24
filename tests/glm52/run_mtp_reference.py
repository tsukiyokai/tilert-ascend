"""Generate independent CPU HF MTP snapshots for the TP16 native runner.

The input and optional rank-N.pt prefixes use run_native_mtp.py's format.
--metadata-only reads checkpoint headers and copied HF sources without Torch.
Numerical execution produces all sixteen replicated traces after reset replay.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback

from prefix_io import mtp_prefix_layer


IMPLEMENTATION = "transformers.GlmMoeDsaDecoderLayer.MTP"
STAGES = (
    "masked_embedding", "rotated_parent", "enorm", "hnorm", "eh_proj",
    "attention", "post_attention", "ffn",
)
HF_FILES = (
    "modeling_glm_moe_dsa.py", "configuration_glm_moe_dsa.py",
    "cache_utils.py",
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=9216)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--hf-source", type=Path)
    args = parser.parse_args()
    if args.capacity < 1:
        parser.error("Capacity must be positive")
    if not args.metadata_only and args.input is None:
        parser.error("Numerical execution requires --input")
    if args.metadata_only and args.hf_source is None:
        parser.error("Metadata checks require --hf-source")
    return args


def audit_hf_sources(paths):
    """Check the exact HF return/cache APIs used by this adapter."""
    trees = {name: ast.parse(path.read_text()) for name, path in paths.items()}
    model = trees["modeling_glm_moe_dsa.py"]
    classes = {node.name: node for node in model.body
               if isinstance(node, ast.ClassDef)}
    checks = []
    for name, returns in (
        ("GlmMoeDsaDecoderLayer", ("hidden_states", "topk_indices")),
        ("GlmMoeDsaAttention",
         ("attn_output", "attn_weights", "topk_indices")),
    ):
        forward = next(node for node in classes[name].body
                       if isinstance(node, ast.FunctionDef)
                       and node.name == "forward")
        result = forward.body[-1]
        if (not isinstance(result, ast.Return)
                or not isinstance(result.value, ast.Tuple)
                or not all(isinstance(node, ast.Name)
                           for node in result.value.elts)):
            raise ValueError(f"HF {name}.forward return contract changed")
        actual = tuple(node.id for node in result.value.elts)
        if actual != returns:
            raise ValueError(f"HF {name}.forward return contract changed")
        checks.append({"name": name + ".forward_return", "passed": True,
                       "line": result.lineno, "values": actual})
    cache = trees["cache_utils.py"]
    indexed = next(node for node in cache.body
                   if isinstance(node, ast.ClassDef)
                   and node.name == "DynamicIndexedLayer")
    if not any(isinstance(node, ast.FunctionDef)
               and node.name == "update_indexer" for node in indexed.body):
        raise ValueError("HF DynamicIndexedLayer lacks update_indexer")
    checks.append({"name": "DynamicIndexedLayer.update_indexer",
                   "passed": True})
    return checks


def load_prefix(args, report):
    import torch

    if args.prefix is None:
        return None
    reference = None
    for rank in range(16):
        path = args.prefix / f"rank-{rank}.pt"
        saved = mtp_prefix_layer(
            torch.load(path, map_location="cpu", weights_only=True),
            report["checkpoint_sha256"],
        )
        if reference is None:
            reference = saved
        elif any(reference[name].dtype != saved[name].dtype
                 or not torch.equal(reference[name], saved[name])
                 for name in reference):
            raise ValueError("Replicated MTP prefixes differ between ranks")
        report["prefix_files_sha256"][path.name] = digest(path)
    return reference


def exact_equal(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and torch.equal(left, right))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(exact_equal(left[key], right[key]) for key in left))
    return left == right


def validate_snapshot(snapshot, position, rows):
    import torch

    if (snapshot["position"] != position
            or snapshot["committed_position"] != position + rows):
        raise ValueError("MTP trace has incorrect parent positions")
    expected = {
        "logits": ((rows, 154880), torch.bfloat16),
        "pre_norm_hidden": ((rows, 6144), torch.bfloat16),
        "recycle_hidden": ((rows, 6144), torch.bfloat16),
        "indices": ((rows, 2048), torch.int32),
        "routes": ((rows, 8), torch.int64),
    }
    regions = [(name, snapshot[name], shape, dtype)
               for name, (shape, dtype) in expected.items()]
    if set(snapshot["stages"]) != set(STAGES):
        raise ValueError("MTP trace must contain all eight stages")
    regions.extend((name, snapshot["stages"][name], (rows, 6144),
                    torch.bfloat16) for name in STAGES)
    if set(snapshot["cache"]) != {"latent", "rope", "index_keys"}:
        raise ValueError("MTP trace has incorrect cache regions")
    regions.extend((name, snapshot["cache"][name], (rows, width),
                    torch.bfloat16) for name, width in (
                        ("latent", 512), ("rope", 64), ("index_keys", 128)
                    ))
    for name, value, shape, dtype in regions:
        if (tuple(value.shape) != shape or value.dtype != dtype
                or value.device.type != "cpu"):
            raise ValueError(f"Invalid CPU MTP trace region: {name}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite MTP trace region: {name}")


def run(args, report):
    # Disable optional backend autoload before the first Torch import.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch
    import transformers
    from transformers import cache_utils
    from transformers.models.glm_moe_dsa import configuration_glm_moe_dsa
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaDecoderLayer,
    )

    from tilert.models.glm_5_2_ascend.mtp_reference import MtpReference

    if transformers.__version__ != "5.14.1":
        raise ValueError("This oracle requires Transformers 5.14.1")
    torch.set_num_threads(8)
    report["versions"].update(torch=torch.__version__,
                              transformers=transformers.__version__)
    paths = {
        "modeling_glm_moe_dsa.py": Path(inspect.getsourcefile(
            GlmMoeDsaDecoderLayer)),
        "configuration_glm_moe_dsa.py": Path(
            configuration_glm_moe_dsa.__file__
        ),
        "cache_utils.py": Path(cache_utils.__file__),
    }
    report["checks"].extend(audit_hf_sources(paths))
    report["installed_hf_source_sha256"] = {
        str(path): digest(path) for path in paths.values()
    }
    inputs = torch.load(args.input, map_location="cpu", weights_only=True)
    parent_kind = inputs["parent_kind"]
    if parent_kind not in ("target_final_norm", "mtp_recycle_norm"):
        raise ValueError("Input must declare the normalized parent's role")
    position = inputs["position"]
    if type(position) is not int or position < 0:
        raise ValueError("Parent position must be a nonnegative integer")
    if position and args.prefix is None:
        raise ValueError("Nonzero parent position requires a prefix")
    parent = inputs["normalized_parent_hidden"]
    external = inputs.get("external_indices")
    original_parent = parent.clone()
    original_external = None if external is None else external.clone()
    prefix = load_prefix(args, report)
    engine = MtpReference(args.checkpoint, args.capacity)
    report["weight_arithmetic"] = engine.weights.contract

    def initialize():
        engine.reset()
        engine.check_lengths(0)
        if prefix is not None:
            engine.inject(prefix)
        engine.check_lengths(position)

    initialize()
    actual = engine.forward(inputs["tokens"], parent, external)
    validate_snapshot(actual, position, len(inputs["tokens"]))
    report["checks"].append({"name": "snapshot_contract", "passed": True})
    initialize()
    replay = engine.forward(inputs["tokens"], parent, external)
    validate_snapshot(replay, position, len(inputs["tokens"]))
    checks = {
        "reset_replay": exact_equal(actual, replay),
        "parent_input_preserved": exact_equal(parent, original_parent),
        "external_indices_preserved": exact_equal(external, original_external),
    }
    report["checks"].extend({"name": name, "passed": passed}
                             for name, passed in checks.items())
    if not all(checks.values()):
        raise ValueError("MTP reference reset or input preservation failed")
    first = args.output / "rank-0.pt"
    torch.save(actual, first)
    for rank in range(16):
        path = args.output / f"rank-{rank}.pt"
        if rank:
            shutil.copyfile(first, path)
        report["rank_sha256"][path.name] = digest(path)
    report.update(status="pass", numerical_execution=True,
                  parent_kind=parent_kind,
                  position=position, width=len(inputs["tokens"]),
                  external_indices=external is not None,
                  committed_position=engine.position)


def main():
    args = parse_arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    model = root / "tilert/models/glm_5_2_ascend"
    paths = [Path(__file__), *(model / name for name in (
        "mtp_reference.py", "reference.py", "reference_weights.py",
        "ops/indexer.py",
    ))]
    paths.append(Path(__file__).with_name("prefix_io.py"))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": IMPLEMENTATION, "status": "running",
        "source_commit": args.source_commit, "hostname": platform.node(),
        "versions": {"python": platform.python_version()},
        "command": [sys.executable, *sys.argv], "tp_size": 16,
        "position_convention": "parent", "checks": [],
        "rank_sha256": {}, "prefix_files_sha256": {},
        "scope": "CPU HF oracle and reset replay; native parity pending",
        "timed": False, "profiled": False,
        "execution": "M=1 causal rows for each M=1/2/4 input",
        "rank_layout": "replicated logits, stages, indices and compressed KV",
        "rounding": (
            "HF expanded BF16 KV and eager attention; FP32 rank-order TP sum; "
            "CANN absorbed-MLA and HCCL rounding are not reproduced"
        ),
    }
    try:
        from tilert.models.glm_5_2_ascend.mtp_reference import (
            checkpoint_contract,
        )

        report["source_sha256"] = {
            str(path.relative_to(root)): digest(path) for path in paths
        }
        report["checkpoint_sha256"] = {
            name: digest(args.checkpoint / name) for name in (
                "config.json", "quant_model_description.json",
                "quant_model_weights.safetensors.index.json",
            )
        }
        report["input_sha256"] = (
            None if args.input is None else digest(args.input)
        )
        report["contract"] = checkpoint_contract(args.checkpoint)
        if args.metadata_only:
            sources = {name: args.hf_source / name for name in HF_FILES}
            report["checks"].extend(audit_hf_sources(sources))
            report["copied_hf_source_sha256"] = {
                str(path): digest(path) for path in sources.values()
            }
            if "torch" in sys.modules:
                raise RuntimeError("Metadata mode unexpectedly imported Torch")
            report.update(status="metadata_only", torch_imported=False,
                          numerical_execution=False)
        else:
            run(args, report)
    except Exception:
        report.update(status="error", error=traceback.format_exc())
    report["checks_total"] = len(report["checks"])
    report["checks_failed"] = sum(not item["passed"]
                                  for item in report["checks"])
    destination = args.output / "manifest.json"
    destination.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": report["status"], "report": str(destination)}))
    return int(report["status"] not in ("pass", "metadata_only"))


if __name__ == "__main__":
    raise SystemExit(main())
