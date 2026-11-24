"""Compare the TP16 serving loop with serial CANN AR and MTP caches.

Use a cold prompt or paired main/MTP prefixes as in run_native_mtp_rounds.
Requests run normally, cancel before delivery, cancel after one/two tokens,
then repeat normally. This checks CANN execution consistency, not HF parity
or HIXL transfer. The production serving factory and TP command loop run.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import threading
import traceback

from prefix_io import mtp_prefix_layer
from run_native_decode import check_lengths, compare_tensor, digest
from run_native_mtp_rounds import cache_rows, parse_arguments


def replay_prefix(main, mtp, last_prompt_token):
    from tilert.pd_vllm.ascend_cache import RankPrefix

    length = main.position

    def layer_prefix(layer):
        result = {
            "latent": layer.mla.latent_cache.view(-1, 512)[:length - 1],
            "rope": layer.mla.rope_cache.view(-1, 64)[:length - 1],
        }
        if layer.indexer is not None:
            index = layer.indexer
            result["index_keys"] = index.keys.view(-1, 128)[:length - 1]
            if index.scales is not None:
                result["index_scales"] = index.scales.view(-1)[:length - 1]
        return {name: value.clone() for name, value in result.items()}

    return RankPrefix([layer_prefix(layer) for layer in main.layers],
                      layer_prefix(mtp.layer), last_prompt_token, length)


def compare_caches(actual, expected, positions, initial, label, checks):
    for model, end in zip(("main", "mtp"), positions):
        layer_count = 78 if model == "main" else 1
        checks.append({
            "name": f"{label}.{model}.layer_count",
            "passed": len(actual[model]) == len(expected[model])
            == layer_count,
        })
        for number, (left, right) in enumerate(zip(actual[model],
                                                  expected[model])):
            checks.append({
                "name": f"{label}.{model}.{number}.fields",
                "passed": set(left) == set(right)
                and {"latent", "rope"} <= set(left),
            })
            for name, value in left.items():
                key = f"{label}.{model}.{number}.{name}"
                if name not in right:
                    continue
                reference = right[name][:end]
                same_layout = (value.shape == reference.shape
                               and value.dtype == reference.dtype)
                checks.append({"name": key + ".layout",
                               "passed": same_layout})
                if not same_layout:
                    continue
                frozen = min(initial - 1, end)
                for section, rows in (("prefix", slice(0, frozen)),
                                      ("new", slice(frozen, end))):
                    if not value[rows].numel():
                        continue
                    exact = (section == "prefix"
                             or not value.is_floating_point()
                             or name == "index_scale")
                    compare_tensor(
                        key + "." + section, value[rows], reference[rows],
                        checks, exact=exact,
                    )


def run(args, report):
    import torch
    import torch.distributed as distributed
    import torch_npu

    from tilert.models.glm_5_2_ascend.native_engine import NativeDecodeEngine
    from tilert.models.glm_5_2_ascend.native_mtp import NativeMtp
    from tilert.models.glm_5_2_ascend.session import PrefixRequest
    from tilert.models.glm_5_2_ascend.tp_runtime import (
        CoordinatedBackend, RankExecutor, TorchControlChannel,
    )
    from tilert.pd_vllm.ascend_cache import AscendRankEngine
    from tilert.pd_vllm.ascend_server import (
        AscendServingEngine, create_session,
    )

    torch.set_num_threads(1)
    device = torch.device(f"npu:{int(os.environ['LOCAL_RANK'])}")
    torch.npu.set_device(device)
    distributed.init_process_group("hccl")
    rank = distributed.get_rank()
    if distributed.get_world_size() != 16:
        raise ValueError("The serving consistency runner requires TP16")
    main = NativeDecodeEngine(
        args.checkpoint, device, args.capacity,
        use_expert_bank=args.expert_bank,
    )
    mtp = NativeMtp(main)
    report.update(rank=rank, world_size=distributed.get_world_size(),
                  device=str(device), torch=torch.__version__,
                  torch_npu=torch_npu.__version__)
    config = json.loads((args.checkpoint / "config.json").read_text())
    stops = config["eos_token_id"]
    stops = (stops,) if isinstance(stops, int) else tuple(stops)
    if args.ignore_eos:
        stops = ()

    def load_bundle(directory):
        path = directory / f"rank-{rank}.pt"
        report.setdefault("prefix_sha256", {})[str(path)] = digest(path)
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if bundle["metadata_sha256"] != report["metadata_sha256"]:
            raise ValueError("Prefix checkpoint differs from decode")
        return bundle

    if args.prefix is not None:
        bundle = load_bundle(args.prefix)
        main.inject([
            {name: value.to(device) for name, value in item.items()}
            for item in bundle["layers"]
        ])
        bundle = load_bundle(args.mtp_prefix)
        mtp.inject({
            name: value.to(device) for name, value in mtp_prefix_layer(
                bundle, report["metadata_sha256"]
            ).items()
        })
        if main.position < 1:
            raise ValueError("Injected prompt must contain a final token")
        main.truncate(main.position - 1)
    with torch.inference_mode():
        for offset, token in enumerate(args.tokens):
            result = main.verify((token,), full_logits=False)
            if offset + 1 < len(args.tokens):
                mtp.forward((args.tokens[offset + 1],), result.last_hidden,
                            full_logits=False)
        first = int(result.token_ids[0].item())
        initial = main.position
        if mtp.position not in (initial - 1, initial):
            raise ValueError("Main and MTP prompt lengths differ")
        if mtp.position != initial - 1:
            mtp.truncate(initial - 1)
        if initial + args.max_tokens - 1 > args.capacity:
            raise ValueError("Prompt and output exceed cache capacity")
        prefix = replay_prefix(main, mtp, args.tokens[-1])
        serial = []
        token = first
        if token not in stops and args.max_tokens > 1:
            mtp.forward((token,), result.last_hidden, full_logits=False)
        while token not in stops and len(serial) < args.max_tokens:
            serial.append(token)
            if len(serial) == args.max_tokens:
                break
            result = main.verify((token,), full_logits=False)
            token = int(result.token_ids[0].item())
            mtp.forward((token,), result.last_hidden, full_logits=False)
        expected = cache_rows(main, mtp, 0, draft_start=0)
        serial_positions = (main.position, mtp.position)

        class CheckedRankEngine(AscendRankEngine):
            def __init__(self):
                super().__init__(main, mtp)
                self.label = None
                self.expected_positions = None
                self.rounds = []

            def advance_mtp(self):
                executed = super().advance_mtp()
                self.rounds.append({
                    "request_id": self.label,
                    "candidates": executed.candidates,
                    "predictions": executed.target_predictions,
                    "committed": executed.commit.committed,
                    "matched": executed.commit.matched_drafts,
                })
                return executed

            def reset(self):
                actual_positions = (main.position, mtp.position)
                report["checks"].append({
                    "name": self.label + ".positions",
                    "passed": actual_positions == self.expected_positions,
                    "actual": actual_positions,
                    "expected": self.expected_positions,
                })
                check_lengths(report, self.label + ".main", main,
                              self.expected_positions[0])
                draft_length = self.expected_positions[1]
                report["checks"].append({
                    "name": self.label + ".mtp.layer_lengths",
                    "passed": mtp.layer.mla.length == draft_length
                    and (mtp.layer.indexer is None
                         or mtp.layer.indexer.length == draft_length),
                })
                compare_caches(
                    cache_rows(main, mtp, 0, draft_start=0), expected,
                    self.expected_positions, initial, self.label,
                    report["checks"],
                )
                super().reset()

        rank_engine = CheckedRankEngine()

        def prefix_loader(request, local_rank):
            if local_rank != rank or request.seq_len != initial:
                raise ValueError("Request prefix belongs to another rank")
            rank_engine.label = request.request_id
            rank_engine.expected_positions = tuple(
                request.metadata["expected_positions"]
            )
            return prefix

        executor = RankExecutor(rank_engine, rank, prefix_loader)
        channel = TorchControlChannel()
        if rank == 0:
            backend = CoordinatedBackend(executor, channel)
            session = create_session(backend, args.width, stops)
            engine = AscendServingEngine(session, rank_engine, device)
            report["requests"] = []
            for number, cancel_after in enumerate((None, 0, 1, 2, None)):
                output = serial if cancel_after is None else serial[
                    :cancel_after
                ]
                positions = serial_positions
                if cancel_after is not None and len(serial) >= cancel_after:
                    retained = initial + max(0, len(output) - 1)
                    positions = (retained, initial - 1 if len(output) < 2
                                 else retained)
                request = PrefixRequest(
                    f"session-{number}", initial,
                    {"expected_positions": positions},
                )
                engine.inject(request)
                delivered, cancelled = [], threading.Event()
                if cancel_after == 0:
                    cancelled.set()

                def on_token(token):
                    delivered.append(token)
                    if len(delivered) == cancel_after:
                        cancelled.set()

                actual = engine.decode(
                    first, max_tokens=args.max_tokens,
                    sampling={"temperature": 0,
                              "ignore_eos": args.ignore_eos},
                    on_token=on_token,
                    cancel_event=cancelled,
                )
                report["checks"].append({
                    "name": request.request_id + ".output",
                    "passed": actual == delivered == output,
                    "actual": actual, "expected": output,
                })
                report["requests"].append({
                    "request_id": request.request_id,
                    "cancel_after": cancel_after, "tokens": actual,
                    "stats": dict(session.last_stats),
                })
                engine.reset()
            backend.close()
        else:
            executor.serve(channel)
        report["rounds"] = rank_engine.rounds
        report["serial_tokens"] = serial
        report["initial_position"] = initial
        state = torch.tensor([main.position, mtp.position, executor.serial],
                             dtype=torch.int64, device=device)
        states = [torch.empty_like(state) for _ in range(16)]
        distributed.all_gather(states, state)
        report["final_rank_states"] = [item.tolist() for item in states]
        report["checks"].append({
            "name": "all_rank_reset_and_serial",
            "passed": all(item[:2] == [0, 0]
                          and item[2] == executor.serial
                          for item in report["final_rank_states"]),
        })
        failed = torch.tensor([
            int(any(not item["passed"] for item in report["checks"]))
        ], dtype=torch.int32, device=device)
        distributed.all_reduce(failed, op=distributed.ReduceOp.MAX)
        report["status"] = "fail" if failed.item() else "capture_replay"
    distributed.barrier()
    distributed.destroy_process_group()


def main():
    args = parse_arguments(__doc__)
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    sources = [
        Path(__file__), Path(__file__).with_name("run_native_decode.py"),
        Path(__file__).with_name("run_native_mtp_rounds.py"),
        Path(__file__).with_name("prefix_io.py"),
        *(root / "tilert/models/glm_5_2_ascend").rglob("*.py"),
        root / "tilert/pd_vllm/ascend_cache.py",
        root / "tilert/pd_vllm/ascend_server.py",
    ]
    report = {
        "status": "error", "git_head": args.source_commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "command": [sys.executable, *sys.argv],
        "scope": "TP16 serving/serial CANN consistency, not HF parity",
        "hixl_transfer": "not executed", "numerical_parity": "not compared",
        "timed": False, "profiled": False, "checks": [],
        "expert_backend": "bank" if args.expert_bank else "per_expert",
        "mtp_width": args.width,
        "last_prompt_token": args.tokens[-1],
        "max_tokens": args.max_tokens,
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
