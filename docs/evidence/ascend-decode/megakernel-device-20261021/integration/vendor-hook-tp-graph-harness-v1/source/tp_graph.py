"""Compare two-rank layer launches with separate parents and real HCCL SUM."""

import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from local_chain import (
    BLOCKS, CHAIN_SHA, INDEXED_SHA, SHARED_SHA, Inputs, LayerCase,
    LayerFFNAdapter, Library, bind_caches, digest, load_ffn, nd, publish_topk,
    signature, snapshot, source_identity, tensor_witness,
)
from local_chain_views import poison, strict_checks
from tp_host import Library as TPLibrary, Prepared as TPPrepared, Window
from tp_parent import separate
from tp_graph_runtime import CapturedChain, prepare_graph_registry, reserve_capture_stream


TP_SHA = "c2b72e469c3ee80a5fe7d9df4a3cceedc0e65cca95e6f426aea762c213392462"


def ordered_sum(torch, partials):
    if (len(partials) != 2 or any(value.dtype != torch.bfloat16 for value in partials)
            or partials[0].shape != partials[1].shape):
        raise ValueError("The reduction oracle requires two BF16 rank partials")
    result = partials[0].float().clone()
    result.add_(partials[1].float())
    return result.bfloat16()


def check_rank_witnesses(records):
    if len(records) != 2 or [row["rank"] for row in records] != [0, 1]:
        raise RuntimeError("The two-rank witness order changed")
    left, right = records
    for name in ("input_hidden", "input_residual", "layer6.retained", "layer7.retained",
                 "layer6.ar1", "layer6.ar2", "layer7.ar1", "layer7.ar2"):
        if left[name] != right[name]:
            raise RuntimeError(f"Replicated or reduced values differ between ranks: {name}")
    for name in ("layer6.attention_partial", "layer6.ffn_partial",
                 "layer7.attention_partial", "layer7.ffn_partial"):
        if left[name] == right[name]:
            raise RuntimeError(f"The two rank partials are indistinguishable: {name}")


class TPCase:
    def __init__(self, torch, torch_npu, local, window):
        self.local = local
        self.attention_sum = nd(torch_npu, torch.empty_like(local.partial))
        self.output = nd(torch_npu, torch.empty_like(local.partial))
        self.prepared = TPPrepared(local.prepared, window, self.attention_sum, self.output)

    def separate(self, reduce):
        separate(self.local, self.attention_sum, self.output, reduce)

    def combined(self):
        if not self.local.parent_validated:
            raise RuntimeError("Run the real parent validators before the TP C ABI")
        self.prepared.launch()

    def values(self):
        return {**self.local.values, "reduction.attention": self.attention_sum,
                "reduction.output": self.output}


def run_tp_graph(torch, layers, config, state, args):
    import torch_npu
    from attention_check import metric

    rank = args.device_rank
    if (rank not in (0, 1) or args.mode != "int8" or len(layers) != 2
            or args.forward_sequence != 8192 or args.graph_residual_modes != ["provided"]
            or torch.distributed.get_world_size(args.oracle_hccl) != 2):
        raise ValueError("Use two C8 shards, sequence8192 and a provided BF16 residual")
    for path, expected in ((args.indexed_library, INDEXED_SHA),
                           (args.shared_library, SHARED_SHA),
                           (args.tp_library, TP_SHA)):
        if digest(path) != expected:
            raise RuntimeError("A frozen numerical parent or TP library changed")
    os.environ["TILERT_INDEXED_ATTENTION_LIBRARY"] = str(args.indexed_library)
    os.environ["TILERT_SHARED_ATTENTION_LIBRARY"] = str(args.shared_library)
    sys.path.insert(0, str(args.adapter.resolve().parents[3]))
    adapter = importlib.import_module("tilert.models.glm_5_2_ascend.attention_native")
    if Path(adapter.__file__).resolve() != args.adapter.resolve():
        raise RuntimeError("The frozen attention packing source changed")
    adapter.load()
    native_ffn = load_ffn(args.ffn_library)
    local_library = Library(args.chain_library, CHAIN_SHA)
    library = TPLibrary(args.tp_library, TP_SHA)
    topk = layers[0].self_attn.mla_attn.topk_indices_buffer
    if (topk.device.index != rank or tuple(topk.shape) != (32, 2048)
            or any(layer.self_attn.mla_attn.topk_indices_buffer is not topk
                   or layer.self_attn.mla_attn.mla_attn.impl.topk_indices_buffer is not topk
                   for layer in layers)):
        raise RuntimeError("The real layers lost their persistent rank-local top-k owner")
    caches, cache_owners = bind_caches(torch, torch_npu, layers, config, args.mode)
    generator = torch.Generator().manual_seed(42022)
    bases = {}
    for name, cache in caches.items():
        if cache.dtype == torch.int8:
            base = torch.randint(-127, 128, cache.shape, generator=generator, dtype=torch.int8)
        elif name.endswith("index_scale"):
            base = (torch.rand(cache.shape, generator=generator) / 128 + 1 / 64).to(cache.dtype)
        else:
            base = torch.randn(cache.shape, generator=generator).to(cache.dtype)
        bases[name] = nd(torch_npu, base.to(topk.device))
    ffns = [LayerFFNAdapter(torch, torch_npu, layer, config, native_ffn)
            for layer in layers]
    cache_signatures = {name: signature(value) for name, value in caches.items()}
    topk_signature = signature(topk)
    report = {
        "scope": "Official ACL graph of two TP16-shaped local shards with SHMEM2; eager separate parents plus HCCL2 oracle",
        "rank": rank, "ranks": 2, "constructor_tp_size": 16,
        "tp16_collectives_executed": 0, "original_decoder_forward_called": False,
        "original_hccl_entry": "torch.distributed.all_reduce(SUM, group=actual_hccl2)",
        "residual_contract": "Next layer consumes reduced FFN output and separately retained residual",
        "script": source_identity(run_tp_graph), "actual_decoder": source_identity(type(layers[0])),
        "parent_phases": source_identity(separate), "tp_host": source_identity(TPPrepared),
        "libraries": {str(path): digest(path) for path in (args.chain_library,
            args.tp_library, args.indexed_library, args.shared_library, args.ffn_library)},
        "cache_signatures": cache_signatures, "topk_signature": topk_signature,
        "widths": args.forward_widths, "iterations": args.forward_replays,
        "blocks": BLOCKS, "eager": False, "graph_capture": True, "timing": False,
        "captures": [], "cases": [], "pass": False,
    }
    destination = args.output / "tp-chain-graph.json"
    window = None
    initialized = False
    current_graph = None
    capture_owner = None

    def persist():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    def gather(value):
        peers = [None, None]
        torch.distributed.all_gather_object(peers, value, group=args.oracle_cpu)
        return peers

    persist()
    try:
        prepare_graph_registry(args.forward_widths)
        stream, capture_owner = reserve_capture_stream(torch)
        stream.wait_stream(torch.npu.current_stream(topk.device))
        with torch.npu.stream(stream), torch.inference_mode():
            library.initialize(rank, rank, 2, args.shmem_endpoint, 64 << 20)
            initialized = True
            window = Window(library, 32)
            if (window.rank, window.ranks, window.capacity, window.device, window.bytes) != (
                    rank, 2, 32, rank, 1573184):
                raise RuntimeError("The actual SHMEM window geometry differs")
            report["window"] = {"address": window.address, "bytes": window.bytes,
                                "capacity": window.capacity, "stream": stream.npu_stream,
                                "epoch_contract": "Four device increments per explicit two-layer replay; never reset by host",
                                "capture_manager_retained": True}
            persist()
            for rows in args.forward_widths:
                inputs = Inputs(torch, torch_npu, layers, rows, topk.device)
                first = TPCase(torch, torch_npu, LayerCase(
                    torch, torch_npu, layers[0], 6, inputs, inputs.hidden, inputs.residual,
                    topk, adapter, ffns[0], local_library), window)
                second = TPCase(torch, torch_npu, LayerCase(
                    torch, torch_npu, layers[1], 7, inputs, first.output,
                    first.local.bridge[1], topk, adapter, ffns[1], local_library), window)
                if first.local.mode != "int8" or second.local.mode != "shared":
                    raise RuntimeError("The actual full/shared layer selection changed")
                cases = (first, second)
                topk_copies = {f"layer{number}.persistent_topk": torch.empty_like(topk[:rows])
                               for number in (6, 7)}
                consumers = {"consumer.output": torch.empty_like(second.output),
                             "consumer.retained": torch.empty_like(second.local.bridge[1])}
                values = {**{f"layer{number}.{name}": value
                    for number, case in zip((6, 7), cases)
                    for name, value in case.values().items()},
                    **caches, **topk_copies, **consumers}
                if len(values) != 113:
                    raise RuntimeError("The C8 two-layer boundary set changed")
                owners = [case.local.owners() for case in cases]
                reduction_owners = [signature(value) for case in cases
                                    for value in (case.attention_sum, case.output)]
                inputs.lock_owners()

                def reset(iteration, integer):
                    for name, cache in caches.items():
                        cache.copy_(bases[name].roll(iteration, 0))
                    topk.fill_(-1)
                    for case in cases:
                        poison(case.values(), integer)
                        case.local.norm[5].fill_(float("nan"))
                        case.local.body[20].fill_(1)
                        case.local.body[20][0].zero_()
                    poison(topk_copies, integer)
                    poison(consumers, integer)

                def execute(combined, reductions):
                    for number, case in zip((6, 7), cases):
                        def reduce(partial, result, phase):
                            result.copy_(partial)
                            torch.distributed.all_reduce(
                                result, op=torch.distributed.ReduceOp.SUM,
                                group=args.oracle_hccl)
                            torch.npu.synchronize()
                            partials = gather(snapshot(torch, partial))
                            expected = ordered_sum(torch, partials)
                            check = metric(torch, snapshot(torch, result), expected)
                            if not check["pass"] or not check["bitwise_equal"]:
                                raise RuntimeError("HCCL2 differs from the ordered BF16-input oracle")
                            reductions.append({"layer": number, "phase": phase,
                                "check": check,
                                "partials": [tensor_witness(torch, value) for value in partials]})

                        if combined:
                            case.combined()
                        else:
                            case.separate(reduce)
                        if number == 6:
                            publish_topk(case.local.selected, topk, rows)
                        topk_copies[f"layer{number}.persistent_topk"].copy_(topk[:rows])
                    consumers["consumer.output"].copy_(second.output)
                    consumers["consumer.retained"].copy_(second.local.bridge[1])

                def collect():
                    torch.npu.synchronize()
                    return {name: snapshot(torch, value) for name, value in values.items()}

                current_graph = CapturedChain(
                    torch, config, layers, inputs, cases, window, stream,
                    execute, consumers, values)
                for iteration in range(args.forward_replays):
                    plan = inputs.update(iteration)
                    original_inputs = {name: snapshot(torch, value)
                                       for name, value in inputs.fields().items()}
                    reset(iteration, 1)
                    reductions = []
                    execute(False, reductions)
                    expected = collect()
                    reset(iteration, 2)
                    if iteration == 0:
                        report["captures"].append(current_graph.capture())
                        persist()
                    torch.npu.synchronize()
                    torch.distributed.barrier(group=args.oracle_cpu)
                    if rank == iteration % 2:
                        time.sleep(0.01)
                    current_graph.replay()
                    actual = collect()
                    inputs.check_owners()
                    if (owners != [case.local.owners() for case in cases]
                            or reduction_owners != [signature(value) for case in cases
                                for value in (case.attention_sum, case.output)]
                            or signature(topk) != topk_signature
                            or cache_signatures != {name: signature(value) for name, value in caches.items()}):
                        raise RuntimeError("A live tensor or communication consumer changed owner")
                    for name, value in original_inputs.items():
                        now = snapshot(torch, inputs.fields()[name])
                        if not torch.equal(value.view(torch.uint8), now.view(torch.uint8)):
                            raise RuntimeError("A layer overwrote a live input or metadata producer")
                    checks, passed = strict_checks(torch, actual, expected, metric)
                    witness = {"rank": rank,
                        "input_hidden": tensor_witness(torch, original_inputs["hidden"])["sha256"],
                        "input_residual": tensor_witness(torch, original_inputs["residual"])["sha256"]}
                    for number in (6, 7):
                        for name, source in (("retained", "bridge.retained"),
                                             ("attention_partial", "attention.partial"),
                                             ("ffn_partial", "bridge.combined"),
                                             ("ar1", "reduction.attention"),
                                             ("ar2", "reduction.output")):
                            witness[f"layer{number}.{name}"] = tensor_witness(
                                torch, actual[f"layer{number}.{source}"])["sha256"]
                    peers = gather(witness)
                    check_rank_witnesses(peers)
                    record = {"rows": rows, "iteration": iteration, "pass": passed,
                              "checks": checks, "hccl_ordered_checks": reductions,
                              "rank_witnesses": peers, "inputs": inputs.witness(),
                              "explicit_replay": current_graph.replays,
                              "python_calls_during_replay": 0,
                              "capture_stream": current_graph.capture_stream.npu_stream,
                              "outputs": {name: tensor_witness(torch, actual[name])
                                          for name in consumers}}
                    report["cases"].append(record)
                    persist()
                    statuses = gather(passed)
                    if not all(statuses):
                        torch.save({"plan": plan, "inputs": original_inputs,
                                    "expected": expected, "actual": actual},
                                   args.output / f"tp-chain-graph-m{rows}-r{iteration}.pt")
                        raise RuntimeError("The TP graph differs from eager parents and HCCL2")
                observed = report["cases"][-args.forward_replays:]
                for name in consumers:
                    if any(item["outputs"][name]["nonzero_elements"] == 0 for item in observed):
                        raise RuntimeError("A final output or residual witness is entirely zero")
                    if (args.forward_replays > 1 and
                            len({item["outputs"][name]["sha256"] for item in observed}) < 2):
                        raise RuntimeError("A final output or residual did not consume changing inputs")
                current_graph.close()
                current_graph = None
                del cases, first, second, values, consumers, topk_copies, inputs
            torch.npu.synchronize()
            torch.distributed.barrier(group=args.oracle_cpu)
            window.close()
            window = None
            library.finalize()
            initialized = False
        report["pass"] = True
        persist()
        return report
    except Exception:
        report["error"] = traceback.format_exc()
        persist()
        raise
    finally:
        if current_graph is not None:
            current_graph.close()
        if initialized:
            torch.npu.synchronize()
            torch.distributed.barrier(group=args.oracle_cpu)
            if window is not None:
                window.close()
            library.finalize()
