"""Compare speculative CANN rounds with serial AR and MTP teacher forcing.

This is an execution-consistency check, not independent HF numerical parity.
The optional main prefix is the usual 78-layer rank-N.pt bundle. The MTP
prefix bundle has metadata_sha256 and layer (latent/rope/index_keys).
With prefixes, --tokens contains only the last prompt token; it is replayed
at P-1 to recover the normalized parent and first generated token.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import traceback

from prefix_io import mtp_prefix_layer
from run_native_decode import compare_tensor, digest


def parse_arguments(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", nargs="+", type=int, required=True)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--mtp-prefix", type=Path)
    parser.add_argument("--width", type=int, choices=(2, 4), default=4)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=9344)
    parser.add_argument("--expert-bank", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if (args.prefix is None) != (args.mtp_prefix is None):
        parser.error("Main and MTP prefixes must be supplied together")
    if args.prefix is not None and len(args.tokens) != 1:
        parser.error("With a prefix, --tokens is the last prompt token")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    return args


def cache_rows(main, mtp, start, draft_start=None):
    if draft_start is None:
        draft_start = start - 1

    def capture(layer, begin, end):
        result = {
            "latent": layer.mla.latent_cache.view(-1, 512)[begin:end],
            "rope": layer.mla.rope_cache.view(-1, 64)[begin:end],
        }
        if layer.indexer is not None:
            index = layer.indexer
            result["index"] = index.keys.view(-1, 128)[begin:end]
            if index.scales is not None:
                result["index_scale"] = index.scales.view(-1)[begin:end]
        return {name: value.detach().cpu().clone()
                for name, value in result.items()}

    return {
        "main": [capture(layer, start, main.position)
                 for layer in main.layers],
        "mtp": [capture(mtp.layer, draft_start, mtp.position)],
    }


def run(args, report):
    import torch
    import torch.distributed as distributed
    import torch_npu

    from tilert.models.glm_5_2_ascend.mtp_round import MtpRoundExecutor
    from tilert.models.glm_5_2_ascend.native_engine import NativeDecodeEngine
    from tilert.models.glm_5_2_ascend.native_mtp import NativeMtp

    torch.set_num_threads(1)
    device = f"npu:{int(os.environ['LOCAL_RANK'])}"
    torch.npu.set_device(device)
    distributed.init_process_group("hccl")
    rank = distributed.get_rank()
    main = NativeDecodeEngine(
        args.checkpoint, device, args.capacity,
        use_expert_bank=args.expert_bank,
    )
    mtp = NativeMtp(main)
    report.update(rank=rank, device=device, torch=torch.__version__,
                  torch_npu=torch_npu.__version__)
    config = json.loads((args.checkpoint / "config.json").read_text())
    stops = config["eos_token_id"]
    stops = (stops,) if isinstance(stops, int) else tuple(stops)
    if args.ignore_eos:
        stops = ()
    main_prefix, mtp_prefix = None, None

    def load_bundle(directory):
        path = directory / f"rank-{rank}.pt"
        report.setdefault("prefix_sha256", {})[str(path)] = digest(path)
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if bundle["metadata_sha256"] != report["metadata_sha256"]:
            raise ValueError("Prefix checkpoint differs from decode")
        return bundle

    if args.prefix is not None:
        bundle = load_bundle(args.prefix)
        main_prefix = [{name: value.to(device) for name, value in item.items()}
                       for item in bundle["layers"]]
        bundle = load_bundle(args.mtp_prefix)
        mtp_prefix = {
            name: value.to(device) for name, value in mtp_prefix_layer(
                bundle, report["metadata_sha256"]
            ).items()
        }

    def initialize():
        main.reset()
        mtp.reset()
        if main_prefix is not None:
            main.inject(main_prefix)
            mtp.inject(mtp_prefix)
            if main.position < 1:
                raise ValueError("Injected prompt must contain a final token")
            main.truncate(main.position - 1)
        result = None
        for offset, token in enumerate(args.tokens):
            result = main.verify((token,), full_logits=False)
            if offset + 1 < len(args.tokens):
                mtp.forward((args.tokens[offset + 1],), result.last_hidden,
                            full_logits=False)
        first = int(result.token_ids[0].item())
        if mtp.position not in (main.position - 1, main.position):
            raise ValueError("MTP prefix does not align with the main prompt")
        return first, result.last_hidden

    with torch.inference_mode():
        first, parent = initialize()
        initial = main.position
        if initial + args.max_tokens - 1 > args.capacity:
            raise ValueError("Prompt and output exceed cache capacity")
        # Independent serial schedule: every MTP row uses a true target
        # parent, and no draft token changes the main-model input stream.
        if first not in stops and args.max_tokens > 1:
            mtp.truncate(initial - 1)
            mtp.forward((first,), parent, full_logits=False)
        serial = [first]
        while len(serial) < args.max_tokens and serial[-1] not in stops:
            result = main.verify((serial[-1],), full_logits=False)
            token = int(result.token_ids[0].item())
            serial.append(token)
            mtp.forward((token,), result.last_hidden, full_logits=False)
        expected = cache_rows(main, mtp, initial)
        serial_positions = main.position, mtp.position

        first_again, parent = initialize()
        executor = MtpRoundExecutor(main, mtp, args.width,
                                    args.max_tokens, stops)
        executor.start(first_again, parent)
        speculative = [first_again]
        rounds = []
        while executor.state.finish_reason is None:
            executed = executor.advance()
            speculative.extend(executed.commit.accepted_tokens)
            rounds.append({
                "width": executed.commit.width,
                "candidates": executed.candidates,
                "predictions": executed.target_predictions,
                "committed": executed.commit.committed,
                "matched": executed.commit.matched_drafts,
                "main_length": main.position, "mtp_length": mtp.position,
            })
        actual = cache_rows(main, mtp, initial)
    checks = report["checks"]
    checks.append({"name": "greedy_output",
                   "passed": serial == speculative})
    checks.append({"name": "committed_positions",
                   "passed": (main.position, mtp.position)
                   == serial_positions})
    for model in ("main", "mtp"):
        for number, (left, right) in enumerate(zip(actual[model],
                                                  expected[model])):
            for name in right:
                value = left[name]
                if value.numel() == 0 or right[name].numel() == 0:
                    checks.append({
                        "name": f"{model}.{number}.{name}",
                        "passed": value.shape == right[name].shape
                        and value.dtype == right[name].dtype,
                    })
                    continue
                compare_tensor(
                    f"{model}.{number}.{name}", value, right[name], checks,
                    exact=(not value.is_floating_point()
                           or name == "index_scale"),
                )
    report.update(serial_tokens=serial, speculative_tokens=speculative,
                  initial_position=initial, rounds=rounds,
                  statistics=executor.state.statistics())
    failed = torch.tensor([int(any(not item["passed"] for item in checks))],
                          dtype=torch.int32, device=device)
    distributed.all_reduce(failed, op=distributed.ReduceOp.MAX)
    report["status"] = "fail" if failed.item() else "capture_replay"
    distributed.barrier()
    distributed.destroy_process_group()


def main():
    args = parse_arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__),
               Path(__file__).with_name("run_native_decode.py"),
               Path(__file__).with_name("prefix_io.py")]
    sources.extend((root / "tilert/models/glm_5_2_ascend").rglob("*.py"))
    report = {
        "status": "error", "git_head": args.source_commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "command": [sys.executable, *sys.argv],
        "scope": "CANN serial/speculative consistency, not HF parity",
        "timed": False, "profiled": False, "checks": [],
        "expert_backend": "bank" if args.expert_bank else "per_expert",
        "ignore_eos": args.ignore_eos,
    }
    try:
        report["source_sha256"] = {
            str(path.relative_to(root)): digest(path) for path in sources
        }
        report["metadata_sha256"] = {
            name: digest(args.checkpoint / name) for name in (
                "config.json", "quant_model_description.json",
                "quant_model_weights.safetensors.index.json",
            )
        }
        run(args, report)
    except Exception:
        report["error"] = traceback.format_exc()
    rank = int(os.environ.get("RANK", "0"))
    output = args.output / f"rank-{rank}.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rank": rank, "status": report["status"],
                      "report": str(output)}))
    return int(report["status"] != "capture_replay")


if __name__ == "__main__":
    raise SystemExit(main())
