"""Capture actual expert IDs during the existing session diagnostic.

Captured copy_ operations update persistent buffers on ACL graph replay.
CPU snapshots perturb execution; neither clocks nor IDs measure HBM traffic.
"""

from contextlib import ExitStack
from functools import wraps
import inspect
import json
import os
from pathlib import Path
from unittest.mock import patch

import bench_cann_mtp as runner
from tilert.models.glm_5_2_ascend.cann_model import CannModel
from tilert.models.glm_5_2_ascend.cann_mtp import CannMtp


WIDTHS = (1, 2, 4, 6, 8, 12, 16, 24, 32)


def main():
    import torch

    args = runner.arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    path = args.output / f"routes-rank-{rank}.jsonl"
    collectors, counts, buffers = {}, {}, {}
    request, sequence, code = -1, 0, None
    sources = {str(Path(source).resolve()): runner.digest(Path(source))
               for source in (__file__, runner.__file__,
                              inspect.getfile(CannModel),
                              inspect.getfile(CannMtp))}
    atomic_json = runner.atomic_json

    def diagnostic_json(destination, value):
        saved = dict(value, performance_measured=False,
                     route_capture=True,
                     scope="Expert-ID capture; instrumented clocks are not "
                           "performance, and IDs are not HBM byte counts")
        if saved.get("status") == "measured":
            saved["status"] = "diagnostic"
        if "tpot_ms" in saved:
            saved["tpot_ms"] = None
        atomic_json(destination, saved)

    def route_copy(original, destination):
        @wraps(original)
        def capture(*values, **options):
            result = original(*values, **options)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("Final expert selection must return two items")
            ids = result[1]
            if (ids.dtype not in (torch.int32, torch.int64)
                    or ids.ndim != 2 or ids.shape[1] != 8
                    or ids.shape[0] not in WIDTHS
                    or ids.shape[0] > destination.shape[0]):
                raise ValueError("Expected integer expert IDs [M,8]")
            destination[:ids.shape[0]].copy_(ids)
            return result
        return capture

    def construct(original, kind):
        @wraps(original)
        def create(engine, *values, **options):
            original(engine, *values, **options)
            backbone = engine.model.model
            if kind == "target":
                layers = list(enumerate(backbone.layers))[3:]
                if len(layers) != 75:
                    raise ValueError("Expected 75 routed target layers")
            else:
                layers = [(78, backbone.layers["78"].mtp_block)]
            with torch.npu.device(engine.device), torch.inference_mode():
                buffer = torch.full((len(layers), args.width, 8), -1,
                                    dtype=torch.int32, device=engine.device)
            collectors[kind] = buffer, [number for number, _ in layers]
            buffers[kind] = {"shape": list(buffer.shape),
                             "dtype": str(buffer.dtype),
                             "device": str(buffer.device),
                             "address": buffer.data_ptr()}
            for slot, (_, layer) in enumerate(layers):
                router = layer.mlp.experts.routed_experts
                method = router._select_experts
                source = inspect.getfile(inspect.unwrap(method))
                sources[source] = runner.digest(Path(source))
                stack.enter_context(patch.object(
                    router, "_select_experts",
                    route_copy(method, buffer[slot]),
                ))
        return create

    def observe(original, kind):
        @wraps(original)
        def call(engine, *values, **options):
            nonlocal sequence
            buffer, layers = collectors[kind]
            start = engine.position
            tokens = values[0] if values else options["tokens"]
            input_ids = (tokens.detach().reshape(-1).cpu().tolist()
                         if isinstance(tokens, torch.Tensor) else list(tokens))
            with torch.npu.device(engine.device), torch.inference_mode():
                buffer.fill_(-1)
                result = original(engine, *values, **options)
                width = engine.position - start
                if (width not in WIDTHS or width != len(input_ids)
                        or width > buffer.shape[1]):
                    raise ValueError("Forward did not append M valid rows")
                ids = buffer[:, :width].cpu().transpose(0, 1).tolist()
            if any(len(set(row)) != 8 or min(row) < 0 or max(row) >= 256
                   for token in ids for row in token):
                raise ValueError("Missing, repeated or invalid expert IDs")
            external = options.get("external_indices") is not None
            stage = ("target" if kind == "target" else "chain" if external
                     else "conditioned" if width > 1 else "seed")
            record = {"call": sequence, "request": request, "kind": kind,
                      "stage": stage, "position": start, "width": width,
                      "position_after": engine.position,
                      "input_ids": input_ids,
                      "positions": list(range(start, engine.position)),
                      "external_indices": external,
                      "layout": "query,layer,topk",
                      "shape": [width, len(layers), 8],
                      "layers": layers, "expert_ids": ids}
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            sequence += 1
            counts[stage] = counts.get(stage, 0) + 1
            return result
        return call

    original_inject = CannModel.inject

    @wraps(original_inject)
    def inject(engine, *values, **options):
        nonlocal request
        result = original_inject(engine, *values, **options)
        request += 1
        return result

    try:
        with path.open("w", encoding="utf-8") as stream, ExitStack() as stack:
            for cls, name, replacement in (
                (CannModel, "__init__",
                 construct(CannModel.__init__, "target")),
                (CannMtp, "__init__", construct(CannMtp.__init__, "mtp")),
                (CannModel, "verify", observe(CannModel.verify, "target")),
                (CannMtp, "forward", observe(CannMtp.forward, "mtp")),
                (CannModel, "inject", inject),
                (runner, "atomic_json", diagnostic_json),
            ):
                stack.enter_context(patch.object(cls, name, replacement))
            code = runner.main()
    finally:
        required = (["target"] if args.lookup
                    else ["target", "seed", "conditioned"])
        if not args.lookup and args.width > 2:
            required.append("chain")
        complete = all(counts.get(stage, 0) for stage in required)
        report_path = args.output / f"rank-{rank}.json"
        atomic_json(args.output / f"routes-rank-{rank}.json", {
            "status": "captured" if code == 0 and complete else "fail",
            "rank": rank, "performance_measured": False,
            "hbm_traffic_measured": False, "calls": sequence,
            "buffers": buffers,
            "counts": counts, "injections": request + 1,
            "graph_prewarm_requests": int(
                args.target_graph or args.draft_graph),
            "source_sha256": sources, "records": str(path),
            "records_sha256": runner.digest(path),
            "runner_report_sha256": (runner.digest(report_path)
                                     if report_path.exists() else None),
        })
    return 0 if code == 0 and complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
