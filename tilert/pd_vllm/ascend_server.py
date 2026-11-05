"""Serve batch-one greedy GLM-5.2 decode on one sixteen-die A3 host."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


class AscendDeviceContext:
    """Bind FastAPI worker threads to the rank-zero NPU context."""

    def __init__(self, session, device):
        self.session = session
        self.device = device

    def __getattr__(self, name):
        return getattr(self.session, name)

    def inject(self, request):
        import torch

        with torch.npu.device(self.device):
            self.session.inject(request)

    def decode(self, first_token_id, **kwargs):
        import torch

        with torch.npu.device(self.device):
            return self.session.decode(first_token_id, **kwargs)

    def reset(self):
        import torch

        with torch.npu.device(self.device):
            self.session.reset()


class AscendServingEngine(AscendDeviceContext):
    """Check the prefill token against the injected target prefix."""

    def __init__(self, session, rank_engine, device):
        super().__init__(session, device)
        self.rank_engine = rank_engine

    def decode(self, first_token_id, **kwargs):
        if first_token_id != self.rank_engine.first_token_id:
            raise ValueError(
                "Prefill first token differs from the decode prefix: "
                f"{first_token_id} != {self.rank_engine.first_token_id}"
            )
        return super().decode(first_token_id, **kwargs)


def create_session(backend, mtp_width, stop_ids, rounds_per_command=1):
    from tilert.models.glm_5_2_ascend.mtp_session import AscendMtpPDEngine
    from tilert.models.glm_5_2_ascend.session import AscendPDEngine

    if mtp_width == 1:
        return AscendPDEngine(
            backend, stop_ids=stop_ids,
            select_token=lambda result: int(result.token_ids[0].item()),
        )
    return AscendMtpPDEngine(backend, width=mtp_width, stop_ids=stop_ids,
                             rounds_per_command=rounds_per_command)


def create_serving_engine(backend, rank_engine, device, *, mtp_width,
                          stop_ids, timing_dir=None, metadata=None):
    if timing_dir is not None:
        from tilert.pd_vllm.decode_timing import TimedBackend

        backend = TimedBackend(backend)
    session = create_session(backend, mtp_width, stop_ids)
    engine = AscendServingEngine(session, rank_engine, device)
    if timing_dir is not None:
        from tilert.pd_vllm.decode_timing import TimedEngine

        engine = TimedEngine(engine, backend, timing_dir, metadata=metadata)
    return engine


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hixl-host", required=True,
                        help="numeric business IP advertised by HIXL")
    parser.add_argument("--hixl-port-base", type=int, default=26100)
    parser.add_argument("--hixl-library", type=Path)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--ctrl-port", type=int, default=5556)
    parser.add_argument("--http-port", type=int, default=18080)
    parser.add_argument("--capacity", type=int, default=9344)
    parser.add_argument("--mtp-width", type=int, choices=(1, 2, 4), default=1,
                        help="1: serial decode; 2/4: greedy MTP verification")
    parser.add_argument("--expert-bank", action="store_true")
    parser.add_argument("--timing-dir", type=Path,
                        help="optional per-request callback/TP timing JSON")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args):
    import torch
    import torch.distributed as distributed
    import torch_npu
    import uvicorn

    from tilert.models.glm_5_2_ascend.native_engine import NativeDecodeEngine
    from tilert.models.glm_5_2_ascend.native_mtp import NativeMtp
    from tilert.models.glm_5_2_ascend.tp_runtime import (
        CoordinatedBackend, RankExecutor, TorchControlChannel,
    )
    from tilert.pd_vllm import decode_server
    from tilert.pd_vllm.ascend_cache import (
        AscendRankEngine, AscendReceiveBuffer, checkpoint_fingerprint,
        receive_device_id,
    )
    from tilert.pd_vllm.ascend_layout import NUM_RANKS, cache_layout
    from tilert.pd_vllm.ascend_receiver import AscendReceiveServer
    from tilert.pd_vllm.hixl_transport import HixlTransport

    if not 1 <= args.hixl_port_base <= 65536 - NUM_RANKS:
        raise ValueError("The HIXL port range must fit sixteen endpoints")
    capacity = cache_layout(args.capacity)["capacity_tokens"]
    config = json.loads((args.checkpoint / "config.json").read_text())
    stop_ids = config["eos_token_id"]
    if isinstance(stop_ids, int):
        stop_ids = [stop_ids]
    torch.set_num_threads(1)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"npu:{receive_device_id(local_rank)}")
    torch.npu.set_device(device)
    distributed.init_process_group("hccl")
    rank = distributed.get_rank()
    if distributed.get_world_size() != NUM_RANKS or rank != local_rank:
        raise ValueError("Ascend PD serving requires sixteen local ranks")
    target = NativeDecodeEngine(
        args.checkpoint, device, capacity, use_expert_bank=args.expert_bank,
    )
    draft = NativeMtp(target)
    preload_experts = args.expert_bank or args.timing_dir is not None
    if preload_experts:
        target.prepare_experts()
        draft.prepare_experts()
    transport = HixlTransport(
        library_path=args.hixl_library, port=args.hixl_port_base + rank,
    )
    receive_buffer = AscendReceiveBuffer(
        rank, device, capacity, transport, args.hixl_host,
    )
    endpoints = [None] * NUM_RANKS
    distributed.all_gather_object(endpoints, receive_buffer.endpoint())
    rank_engine = AscendRankEngine(target, draft)
    executor = RankExecutor(
        rank_engine, rank, lambda request, _local_rank:
        receive_buffer.prefix(request),
    )
    channel = TorchControlChannel()
    root = Path(__file__).resolve().parents[2]
    sources = [
        *(root / "tilert/models/glm_5_2_ascend").rglob("*.py"),
        *(root / "tilert/pd_vllm").rglob("*.py"),
    ]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "command": [sys.executable, *sys.argv],
        "rank": rank, "world_size": NUM_RANKS, "device": str(device),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "python": platform.python_version(),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest() for path in sources
        },
        "checkpoint_sha256": checkpoint_fingerprint(args.checkpoint),
        "layout": receive_buffer.layout, "endpoints": endpoints,
        "decode": "CANN greedy, serial" if args.mtp_width == 1
        else f"CANN greedy, MTP width {args.mtp_width}",
        "mtp_width": args.mtp_width,
        "expert_backend": "bank" if args.expert_bank else "per_expert",
        "expert_loading": "preloaded" if preload_experts else "on first use",
        "status": "initialized",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / f"rank-{rank}.json"
    report.write_text(json.dumps(metadata, indent=2) + "\n")
    if rank == 0:
        backend = CoordinatedBackend(executor, channel)
        engine = create_serving_engine(
            backend, rank_engine, device, mtp_width=args.mtp_width,
            stop_ids=stop_ids, timing_dir=args.timing_dir, metadata=metadata,
        )
        server = AscendReceiveServer(
            receive_buffer.layout, endpoints, metadata["checkpoint_sha256"],
            host=args.bind_host, ctrl_port=args.ctrl_port,
        )
        try:
            uvicorn.run(decode_server.build_app(server, engine),
                        host=args.bind_host, port=args.http_port)
        finally:
            server.close()
            backend.close()
    else:
        executor.serve(channel)
    distributed.barrier()
    transport.close()
    metadata.update(status="stopped", command_serial=executor.serial,
                    final_position=target.position)
    report.write_text(json.dumps(metadata, indent=2) + "\n")
    distributed.destroy_process_group()


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
