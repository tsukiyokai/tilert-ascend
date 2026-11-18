"""Prepare frozen BF16 prefix Values with the unchanged HF CPU projection."""

import argparse
import gc
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from time import perf_counter

FIXTURE_SHA = "10c8abc1b891f4de0903cec67d33aa5b31e85829a8b44e57c75d4a8829b36f71"
REFERENCE_SHA = "38e13cb59fc87b96e4e7be8ce1d1a0846bb4cd0c43ea776ad2b44904537dfa8e"
PREFIX, HEADS, LATENT, HEAD_WIDTH, VALUE = 8192, 4, 512, 448, 256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def tensor_sha(value):
    import torch

    raw = value.detach().contiguous().view(torch.uint8).reshape(-1).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def bf16(value, shape, name):
    import torch

    require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
            and value.dtype == torch.bfloat16 and tuple(value.shape) == shape
            and bool(torch.isfinite(value).all()), "Invalid CPU BF16 " + name)


def project(weights, latent, layer, rank):
    import torch

    name = f"model.layers.{layer}.self_attn.kv_b_proj"
    require(weights.quantization.get(name + ".weight", "FLOAT") == "FLOAT",
            "KV-b projection must retain the frozen FLOAT recipe")
    bf16(latent, (PREFIX, LATENT), "latent prefix")
    rows = (rank * HEADS * HEAD_WIDTH, (rank + 1) * HEADS * HEAD_WIDTH)
    weight = weights.tensor(name + ".weight", rows=rows, dtype=torch.bfloat16)
    bf16(weight, (HEADS * HEAD_WIDTH, LATENT), "KV-b weight slice")
    # Keep the original 1792-column GEMM, including the discarded Key columns.
    expanded = weights._floating_product(name, latent, rows=rows)
    bf16(expanded, (PREFIX, HEADS * HEAD_WIDTH), "expanded KV prefix")
    value = expanded.reshape(PREFIX, HEADS, HEAD_WIDTH)[:, :, 192:]
    value = value.permute(1, 0, 2).contiguous()
    bf16(value, (HEADS, PREFIX, VALUE), "Value prefix")
    return value, weight, name, rows


def positive_control(args, weights, load_capture):
    import torch

    started = perf_counter()
    require(digest(args.fixture) == FIXTURE_SHA, "First-difference fixture bytes differ")
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    identity = fixture["identity"]
    require(identity["rank"] == 2 and identity["cpu_threads"] == 8
            and identity["cpu_mkldnn_enabled"] is False
            and identity["cpu_torch"] == str(torch.__version__)
            and identity["reference_weights_sha256"] == REFERENCE_SHA
            and identity["checkpoint_metadata_sha256"]
            == weights.contract["metadata_sha256"], "Fixture CPU/source/checkpoint differs")
    prefixes, _, inputs = load_capture(args.capture_case, 2,
                                      weights.contract["metadata_sha256"])
    require(inputs["initial_position"] == PREFIX
            and inputs["prefix_sha256"]
            == fixture["capture_inputs_identity"]["prefix_sha256"],
            "Positive control uses different prefix bytes")
    actual = prefixes[0]["latent"]
    table = fixture["inputs"]["block_table"]
    cache = fixture["inputs"]["key_cache"]
    require(table.dtype == torch.int32 and tuple(table.shape) == (1, 73),
            "Positive-control page table differs")
    bf16(actual, (PREFIX, LATENT), "captured prefix")
    require(cache.dtype == torch.bfloat16
            and tuple(cache.shape) == (73, 128, 1, LATENT), "Fixture KV layout differs")
    pages = table[0, :64].long()
    require(pages.unique().numel() == 64
            and bool(torch.all((pages >= 0) & (pages < 73))), "Invalid prefix pages")
    positions = torch.arange(PREFIX)
    physical = table[0].long()[positions // 128] * 128 + positions % 128
    logical = cache.reshape(-1, LATENT).index_select(0, physical)
    require(torch.equal(actual, logical), "Fixture and captured BF16 prefix differ")
    expected = fixture["cpu"]["value_prefix"]
    bf16(expected, (HEADS, PREFIX, VALUE), "fixture Value prefix")
    require(tensor_sha(expected) == identity["tensor_sha256"]["cpu"]["value_prefix"],
            "Fixture expected Value digest differs")
    value, weight, _, rows = project(weights, actual, 0, 2)
    uv = weight.reshape(HEADS, HEAD_WIDTH, LATENT)[:, 192:].transpose(1, 2).contiguous()
    require(torch.equal(uv, fixture["weights"]["uv"]), "Fixture UV weight differs")
    exact = torch.equal(value, expected)
    result = {"status": "pass" if exact else "fail", "rank": 2, "layer": 0,
              "fixture_sha256": FIXTURE_SHA, "prefix_sha256": inputs["prefix_sha256"],
              "latent_sha256": tensor_sha(actual), "kv_b_rows": list(rows),
              "weight_sha256": tensor_sha(weight), "actual_sha256": tensor_sha(value),
              "expected_sha256": tensor_sha(expected), "exact": exact,
              "seconds": perf_counter() - started}
    write_json(args.output / "positive-control.json", result)
    print(json.dumps({"stage": "positive_control", **result}), flush=True)
    require(exact, "Rank2 Value prefix is not bitwise equal; full preparation refused")
    return result


def run(args, report):
    require(os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0",
            "Disable device backend autoload before importing Torch")
    import torch

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    sys.path[:0] = [str(args.source_root), str(args.source_root / "tests/glm52")]
    from run_cann_model import load_capture

    from tilert.models.glm_5_2_ascend.reference_weights import ReferenceWeights

    source = Path(inspect.getfile(ReferenceWeights)).resolve()
    require(source == args.source_root / "tilert/models/glm_5_2_ascend/reference_weights.py"
            and digest(source) == REFERENCE_SHA, "Frozen ReferenceWeights source differs")
    weights = ReferenceWeights(args.checkpoint, expert_cache_size=0)
    verified = json.loads(args.checkpoint_receipt.read_text())
    files = verified["files"]
    require(verified["pass"] is True and len(files) == 109
            and len({item["name"] for item in files}) == 109
            and all(item["pass"] is True and item["sha256"] == item["expected_sha256"]
                    and (args.checkpoint / item["name"]).stat().st_size == item["bytes"]
                    for item in files), "Prior full checkpoint verification is incomplete")
    complete_hashes = {item["name"]: item["sha256"] for item in files}
    require(all(complete_hashes[name] == value
                for name, value in weights.contract["metadata_sha256"].items()),
            "Current checkpoint metadata differs from full verification")
    sources = {str(Path(__file__).resolve()): digest(__file__)}
    for loaded in list(sys.modules.values()):
        path = getattr(loaded, "__file__", None)
        if path:
            path = Path(path).resolve()
            if path.is_relative_to(args.source_root) and path.suffix == ".py":
                sources[str(path)] = digest(path)
    report.update(
        source_sha256=sources,
        checkpoint={"metadata_sha256": weights.contract["metadata_sha256"],
                    "prior_full_verification_sha256": digest(args.checkpoint_receipt),
                    "prior_full_file_sha256": complete_hashes,
                    "full_checkpoint_rehashed_in_this_run": False},
        capture_manifest_sha256=digest(args.capture_case / "manifest.json"),
        cpu={"torch": str(torch.__version__), "threads": torch.get_num_threads(),
             "interop_threads": torch.get_num_interop_threads(), "mkldnn_enabled": False,
             "torch_config": torch.__config__.show(), "image_sha256": args.image_sha256},
    )
    report["positive_control"] = positive_control(args, weights, load_capture)
    write_json(args.output / "summary.json", report)
    for rank in args.ranks:
        started = perf_counter()
        prefixes, _, inputs = load_capture(args.capture_case, rank,
                                          weights.contract["metadata_sha256"])
        require(len(prefixes) == 78 and inputs["initial_position"] == PREFIX,
                "A full 78-layer 8K prefix is required")
        bank = torch.empty((len(args.layers), HEADS, PREFIX, VALUE), dtype=torch.bfloat16)
        layers = []
        for slot, layer in enumerate(args.layers):
            layer_started = perf_counter()
            latent = prefixes[layer]["latent"]
            value, weight, name, rows = project(weights, latent, layer, rank)
            bank[slot].copy_(value)
            shard = weights.weight_map[name + ".weight"]
            layers.append({"layer": layer, "kv_b_rows": list(rows),
                           "prefix_latent_sha256": tensor_sha(latent),
                           "weight_tensor_sha256": tensor_sha(weight),
                           "checkpoint_shard": shard,
                           "checkpoint_shard_prior_sha256": complete_hashes[shard],
                           "value_tensor_sha256": tensor_sha(value),
                           "seconds": perf_counter() - layer_started})
            print(json.dumps({"stage": "projection", "rank": rank,
                              "layer": layer, "seconds": layers[-1]["seconds"]}), flush=True)
        path = args.output / f"rank-{rank}.pt"
        with path.open("xb") as stream:
            torch.save(bank, stream)
        receipt = {"rank": rank, "layers": layers, "shape": list(bank.shape),
                   "dtype": str(bank.dtype), "prefix": inputs,
                   "tensor_sha256": tensor_sha(bank), "file_sha256": digest(path),
                   "file_bytes": path.stat().st_size, "seconds": perf_counter() - started,
                   "source_sha256": sources, "checkpoint": report["checkpoint"],
                   "capture_manifest_sha256": report["capture_manifest_sha256"],
                   "positive_control": report["positive_control"],
                   "npu_executed": False, "performance_measured": False}
        receipt_path = args.output / f"rank-{rank}.json"
        write_json(receipt_path, receipt)
        report["ranks"].append({"rank": rank, "file_sha256": receipt["file_sha256"],
                                "tensor_sha256": receipt["tensor_sha256"],
                                "receipt_sha256": digest(receipt_path),
                                "seconds": receipt["seconds"]})
        write_json(args.output / "summary.json", report)
        print(json.dumps({"stage": "rank_complete", **report["ranks"][-1]}), flush=True)
        del bank, prefixes, value, weight
        gc.collect()
    require("torch_npu" not in sys.modules
            and not any(name.startswith("transformers.models.") for name in sys.modules),
            "CPU preparation imported a device backend or HF model")
    report["status"] = "prepared"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "checkpoint", "capture-case", "fixture",
                 "checkpoint-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--layers", type=int, nargs="+", choices=range(78), default=list(range(78)))
    parser.add_argument("--ranks", type=int, nargs="+", choices=range(16), default=list(range(16)))
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    require(len(set(args.layers)) == len(args.layers)
            and len(set(args.ranks)) == len(args.ranks), "Duplicate layer or rank")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "cpu-prefix-values-v1", "status": "preparing", "ranks": [],
              "scope": "CPU frozen-prefix preparation; not NPU or decode latency",
              "layers": args.layers, "requested_ranks": args.ranks,
              "formula": "BF16(FP32(C[8192,512]) @ FP32(W_rank[1792,512]).T); "
                         "reshape[8192,4,448][:,:,192:].permute[1,0,2]",
              "command": [sys.executable, *sys.argv], "npu_executed": False,
              "precision_verified": False, "formal_precision_gate_passed": False,
              "performance_measured": False}
    started = perf_counter()
    try:
        run(args, report)
    except Exception as error:
        import traceback

        report.update(status="fail", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report["seconds"] = perf_counter() - started
        write_json(args.output / "summary.json", report)
    print(json.dumps({"status": report["status"], "ranks": len(report["ranks"]),
                      "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
