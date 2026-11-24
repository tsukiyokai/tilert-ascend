"""Serve CANN serial, MTP or lookup decode from full-P HIXL prefixes."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.pd_vllm.ascend_cache import (  # noqa: E402
    CannRankEngine, drafter_metadata,
)


def create_model(args, device, rank):
    from tilert.models.glm_5_2_ascend.cann_model import CannModel

    options = {}
    if args.lookup:
        options = {
            "compile_target": args.compile_target,
            "super_kernel": args.super_kernel,
            "fp32_head": args.fp32_head,
            "graph_width": args.mtp_width,
        }
    return CannModel(
        args.checkpoint, device, rank, capacity=args.capacity,
        gpu_memory_utilization=args.gpu_memory_utilization, **options,
    )


class ServingRank(CannRankEngine):
    """Warm and capture one target width before delivering request tokens."""

    def __init__(self, target, draft, *, width, target_graph, barrier):
        super().__init__(target, draft)
        self.width = width
        self.target_graph = target_graph
        self.barrier = barrier
        self.on_inject = None

    def inject(self, prefix):
        super().inject(prefix)
        if (self.target_graph and self.target.graph is None
                and self.position + self.width <= self.capacity):
            position = self.position
            for iteration in range(4):
                if iteration == 3:
                    self.barrier()
                    self.target.enable_graph(width=self.width)
                result = self.target.verify(
                    [self.first_token_id] * self.width, full_logits=False,
                )
                self.target.host_tokens(result.token_ids)
                self.target.truncate(position)
            self.barrier()
        if self.on_inject is not None:
            self.on_inject()


def create_rank_engine(args, model, barrier):
    if args.mtp_width == 1:
        return model, None
    draft = None
    if not args.lookup:
        from tilert.models.glm_5_2_ascend.cann_mtp import CannMtp

        draft = CannMtp(model)
    return ServingRank(model, draft, width=args.mtp_width,
                       target_graph=args.target_graph, barrier=barrier), draft


class DeferredModel:
    """Keep the live prefix on device until the model-loading gate exists."""

    def __init__(self, args, device, rank):
        self.args = args
        self.device = device
        self.rank = rank
        self.capacity = args.capacity
        self.model = None
        self.on_load = None

    @property
    def position(self):
        return 0 if self.model is None else self.model.position

    def inject(self, prefix):
        if self.model is None:
            while not self.args.load_model_after.exists():
                time.sleep(0.1)
            self.model = create_model(self.args, self.device, self.rank)
            if self.on_load is not None:
                self.on_load(self.model)
        return self.model.inject(prefix)

    def step(self, token):
        return self.model.step(token)

    def reset(self):
        if self.model is not None:
            self.model.reset()

    def source_metadata(self):
        return {} if self.model is None else self.model.source_metadata()

    def shutdown(self):
        if self.model is not None:
            self.model.shutdown()
        else:
            import torch.distributed as distributed

            distributed.destroy_process_group()


def run(args):
    host_drafter = drafter_metadata("lookup") if args.lookup else None
    import torch
    import torch.distributed as distributed
    import torch_npu  # noqa: F401
    import uvicorn

    from tilert.models.glm_5_2_ascend.tp_runtime import (
        CoordinatedBackend, RankExecutor, TorchControlChannel,
    )
    from tilert.pd_vllm import decode_server
    from tilert.pd_vllm.ascend_cache import (
        AscendReceiveBuffer, checkpoint_fingerprint,
        receive_device_id,
    )
    from tilert.pd_vllm.ascend_receiver import AscendReceiveServer
    from tilert.pd_vllm.ascend_server import (
        AscendDeviceContext, AscendServingEngine, create_session,
    )
    from tilert.pd_vllm.decode_timing import (
        TimedBackend, TimedEngine, atomic_json,
    )
    from tilert.pd_vllm.hixl_transport import HixlTransport
    from tilert.pd_vllm.prefix_capture import digest

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank != local_rank:
        raise ValueError("CANN PD serving requires one local TP16 group")
    device = torch.device(f"npu:{receive_device_id(local_rank)}")
    torch.set_num_threads(1)
    if args.load_model_after is not None:
        torch.npu.set_device(device)
        distributed.init_process_group("hccl")
        if distributed.get_world_size() != 16:
            raise ValueError("Deferred CANN serving requires TP16")
        model = DeferredModel(args, device, rank)
    else:
        model = create_model(args, device, rank)
    channel = TorchControlChannel()
    rank_engine, draft = create_rank_engine(
        args, model, lambda: distributed.barrier(group=channel.group),
    )
    speculative = args.mtp_width != 1
    transport = HixlTransport(
        args.hixl_library, port=args.hixl_port_base + rank,
    )
    receive = AscendReceiveBuffer(
        rank, device, args.capacity, transport, args.hixl_host,
        parent_hidden=speculative,
    )
    endpoints = [None] * 16
    distributed.all_gather_object(endpoints, receive.endpoint())
    prefix_loader = (receive.main_prefix if not speculative
                     else lambda request: receive.mtp_prefix(
                         request, lookup=args.lookup))
    executor = RankExecutor(rank_engine, rank, lambda request, _rank:
                            prefix_loader(request))
    fingerprint = checkpoint_fingerprint(args.checkpoint)
    sources = [Path(__file__),
               *(ROOT / "tilert/models/glm_5_2_ascend").glob("*.py"),
               *(ROOT / "tilert/pd_vllm").glob("*.py")]
    if draft is not None:
        sources.append(Path(inspect.getfile(type(draft.model))))
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": "CannModel", "rank": rank, "device": str(device),
        "scope": "HTTP/HIXL continuation from full-P main KV",
        "mtp_width": args.mtp_width,
        "parent_hidden": speculative,
        "drafter": "lookup" if args.lookup else "mtp" if speculative else None,
        "host_drafter": host_drafter,
        "compile_target": args.compile_target,
        "super_kernel": args.super_kernel, "fp32_head": args.fp32_head,
        "target_graph_requested": args.target_graph,
        "execution": {},
        "packages": {name: version(name) for name in
                     ("torch", "torch-npu", "vllm", "vllm-ascend")},
        "runtime_env": {name: os.environ.get(name) for name in (
            "ASCEND_CUSTOM_OPP_PATH", "VLLM_CACHE_ROOT",
            "TORCHINDUCTOR_AUTOGRAD_CACHE", "HCCL_NPU_SOCKET_PORT_RANGE",
            "HCCL_OP_EXPANSION_MODE", "HCCL_DETERMINISTIC",
        )},
        "precision_verified": False, "performance_measured": False,
        "checkpoint_sha256": fingerprint,
        "source_sha256": {**model.source_metadata(),
                          **{str(path): digest(path) for path in sources},
                          **({} if host_drafter is None else
                             host_drafter["source_sha256"])},
        "hixl_library_sha256": digest(args.hixl_library),
        "world_size": distributed.get_world_size(),
        "capacity": model.capacity,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "load_model_after": (str(args.load_model_after)
                             if args.load_model_after is not None else None),
        "command": [sys.executable, *sys.argv], "status": "initialized",
    }
    report = args.output / f"rank-{rank}.json"
    if speculative:
        def record_execution():
            graph = model.graph
            metadata["execution"]["target_graph"] = (
                None if graph is None else {
                    "width": graph.width, "captures": graph.capture_count,
                    "replays_at_inject": graph.replay_count,
                }
            )
            atomic_json(report, metadata)

        rank_engine.on_inject = record_execution
    if isinstance(model, DeferredModel):
        def record_model_sources(loaded):
            metadata["source_sha256"].update(loaded.source_metadata())
            atomic_json(report, metadata)

        model.on_load = record_model_sources
    atomic_json(report, metadata)
    if rank == 0:
        config = json.loads((args.checkpoint / "config.json").read_text())
        stop_ids = config["eos_token_id"]
        if type(stop_ids) is int:
            stop_ids = [stop_ids]
        backend = CoordinatedBackend(executor, channel)
        if args.timing_dir is not None:
            backend = TimedBackend(backend)
        session = create_session(backend, args.mtp_width, stop_ids)
        if not speculative:
            engine = AscendDeviceContext(session, device)
        else:
            engine = AscendServingEngine(session, rank_engine, device)
        if args.timing_dir is not None:
            engine = TimedEngine(engine, backend, args.timing_dir,
                                 metadata=metadata)
        server = AscendReceiveServer(
            receive.layout, endpoints, fingerprint, host=args.bind_host,
            ctrl_port=args.ctrl_port, require_prompt_token_ids=args.lookup,
        )
        try:
            uvicorn.run(decode_server.build_app(server, engine),
                        host=args.bind_host, port=args.http_port)
        finally:
            server.close()
            with torch.npu.device(device):
                backend.close()
    else:
        executor.serve(channel)
    distributed.barrier()
    transport.close()
    metadata.update(status="stopped", command_serial=executor.serial,
                    final_position=model.position)
    atomic_json(report, metadata)
    model.shutdown()


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hixl-host", required=True)
    parser.add_argument("--hixl-library", type=Path, required=True)
    parser.add_argument("--hixl-port-base", type=int, default=26100)
    parser.add_argument("--capacity", type=int, default=9344)
    parser.add_argument("--mtp-width", type=int,
                        choices=(1, 2, 4, 6, 8, 12, 16, 24, 32), default=1)
    parser.add_argument("--lookup", action="store_true",
                        help="Draft from the transferred prompt IDs")
    parser.add_argument("--compile-target", action="store_true")
    parser.add_argument("--super-kernel", action="store_true")
    parser.add_argument("--fp32-head", action="store_true")
    parser.add_argument("--target-graph", action="store_true",
                        help="Warm and capture lookup width before delivery")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--ctrl-port", type=int, default=5556)
    parser.add_argument("--http-port", type=int, default=18080)
    parser.add_argument("--timing-dir", type=Path)
    parser.add_argument("--load-model-after", type=Path,
                        help="Load M1 weights after this gate file exists")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.lookup and args.mtp_width == 1:
        parser.error("--lookup requires --mtp-width greater than 1")
    if not args.lookup and args.mtp_width not in (1, 2, 4):
        parser.error("Wide verification requires --lookup")
    if (args.compile_target or args.super_kernel or args.fp32_head
            or args.target_graph) and not args.lookup:
        parser.error("Target performance options require --lookup")
    if args.super_kernel and not args.compile_target:
        parser.error("--super-kernel requires --compile-target")
    if args.target_graph and args.compile_target and args.mtp_width == 2:
        parser.error("Compiled target graph requires width 4 or greater")
    if args.load_model_after is not None and args.mtp_width != 1:
        parser.error("--load-model-after requires --mtp-width 1")
    return args


def main():
    run(parse_arguments())


if __name__ == "__main__":
    main()
