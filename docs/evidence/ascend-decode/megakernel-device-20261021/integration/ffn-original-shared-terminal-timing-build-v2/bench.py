"""Pair frozen FFN C ABIs using identical scratch in each of two owner arenas."""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import statistics
import sys
import traceback
from types import SimpleNamespace


class DispatchInput(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "rows", "scales", "inverse", "counts", "probabilities", "bf16_rows", "expert_ids")]


class SharedInput(ctypes.Structure):
    _fields_ = ([(name, ctypes.c_uint64) for name in (
        "input", "input_scale", "gate_weight", "gate_scale", "down_weight", "down_scale",
        "payload", "tiling", "output")]
        + [("capacity", ctypes.c_int64)]
        + [(name, ctypes.c_uint64) for name in
           ("partial_offset", "middle_offset", "scale_offset", "down_offset")])


def pointers(values):
    return (ctypes.c_void_p * len(values))(*(value.data_ptr() for value in values))


class Launchers:
    def __init__(self, fused, bridge, body, candidate, kind, active):
        self.libraries = [ctypes.CDLL(str(path.resolve())) for path in (fused, bridge, body)]
        self.libraries.append(ctypes.CDLL(str(active.resolve())))
        self.base = self.libraries[-1].ffn_original_shared_launch
        self.libraries.append(ctypes.CDLL(str(candidate.resolve())))
        self.fused = self.libraries[-1].ffn_original_shared_terminal_launch
        self.kind = kind
        self.extra_by_arena = {}
        array = ctypes.POINTER(ctypes.c_void_p)
        self.fused.argtypes = ([ctypes.c_void_p, ctypes.c_uint64]
            + [ctypes.c_void_p] * 5 + [array] * 3
            + [SharedInput, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_float])
        self.base.argtypes = list(self.fused.argtypes)
        self.base.restype = None
        if kind == "profile":
            self.fused.argtypes = [*self.fused.argtypes, ctypes.c_void_p]
        self.bridge = self.libraries[1].ffn_bridge_launch
        self.bridge.argtypes = ([ctypes.c_void_p, ctypes.c_uint64]
            + [ctypes.c_void_p] * 5 + [array, ctypes.c_int64, ctypes.c_int64, ctypes.c_float])
        self.body = self.libraries[2].routed_shared_body_launch
        self.body.argtypes = ([ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64]
            + [ctypes.c_void_p] * 23 + [ctypes.c_int64] * 3 + [DispatchInput, SharedInput])
        self.finalize = self.libraries[1].ffn_bridge_epilogue_launch
        self.finalize.argtypes = ([ctypes.c_void_p, ctypes.c_uint64]
            + [ctypes.c_void_p] * 2 + [array, ctypes.c_int64, ctypes.c_int64])
        for function in (self.fused, self.bridge, self.body, self.finalize):
            function.restype = None
        if ctypes.sizeof(DispatchInput) != 56 or ctypes.sizeof(SharedInput) != 112:
            raise RuntimeError("Frozen C ABI POD sizes differ")


class Calls:
    def __init__(self, torch, launchers, arena, bank, hidden, residual, rows, blocks, layout):
        self.torch = torch
        self.launchers = launchers
        self.arena = arena
        self.bank = bank
        self.hidden, self.residual = hidden, residual
        self.rows, self.blocks, self.layout = rows, blocks, layout
        b, r, self.sync = arena
        self.bridge_ptrs, self.body_ptrs = pointers(b), pointers(r)
        self.weight_ptrs = pointers(bank["routed"])
        key = b[8].data_ptr()
        if key not in launchers.extra_by_arena:
            shape = (264,) if launchers.kind == "active" else (3 * blocks, 32)
            dtype = torch.int32 if launchers.kind == "active" else torch.int64
            extra = torch.empty(shape, dtype=dtype, device=hidden.device)
            if extra.data_ptr() in {t.data_ptr() for t in (*b, *r)}:
                raise RuntimeError("Extra variant storage aliases the common arena")
            launchers.extra_by_arena[key] = extra
        self.extra = launchers.extra_by_arena[key]
        self.candidate_body_ptrs = pointers([*r, self.extra]) if launchers.kind == "active" else self.body_ptrs
        capacity = 16 if rows <= 16 else 32
        partial = capacity * 6144
        middle = partial + 6 * 256 * capacity * 4
        scale = middle + capacity * 128
        down = scale + capacity * 8 * 4
        self.shared = SharedInput(
            b[6].data_ptr(), b[7].data_ptr(),
            *(value.data_ptr() for value in bank["shared"]),
            r[17].data_ptr(), r[18].data_ptr(), r[19].data_ptr(), capacity,
            partial, middle, scale, down)
        self.dispatch = DispatchInput(0, 0, 0, 0, b[4].data_ptr(), b[0].data_ptr(), b[5].data_ptr())
        values = ([r[10], *bank["routed"], r[11], *r[12:17]]
                  + [r[index] for index in (0, 1, 2, 3, 4, 5, 9, 6, 7, 8)])
        self.body_arguments = [value.data_ptr() for value in values]
        if len(self.body_arguments) != 23:
            raise RuntimeError("Frozen body pointer ABI differs")

    def input_arguments(self):
        return [value.data_ptr() for value in (
            self.hidden, self.residual, self.bank["gamma"], self.bank["gate"], self.bank["bias"])]

    def fused(self):
        stream = self.torch.npu.current_stream().npu_stream
        self.launchers.fused(stream, self.sync, *self.input_arguments(), self.weight_ptrs,
            self.bridge_ptrs, self.candidate_body_ptrs, self.shared, self.rows, self.blocks, self.layout, 1e-5,
            *([self.extra.data_ptr()] if self.launchers.kind == "profile" else []))

    def separate(self):
        stream = self.torch.npu.current_stream().npu_stream
        self.launchers.base(stream, self.sync, *self.input_arguments(), self.weight_ptrs,
            self.bridge_ptrs, self.candidate_body_ptrs, self.shared, self.rows, self.blocks, self.layout, 1e-5)

    def owning_torch(self, owner):
        b, r, sync = self.arena
        torch, bank = self.torch, self.bank
        torch.ops.tilert_ffn_original_shared.forward_out(
            self.hidden, self.residual, bank["gamma"], bank["gate"], bank["bias"],
            bank["routed"], bank["shared"], b, [*r, self.extra],
            1e-5, self.blocks, self.layout, sync)


def prepare_arena(torch, launchers, hidden, blocks):
    bridge, body, sync = torch.ops.tilert_ffn_original_shared.prepare(
        hidden, blocks, 1)
    launchers.extra_by_arena[bridge[8].data_ptr()] = body[20]
    return bridge, body[:20], sync


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_hash(torch, value):
    raw = value.detach().cpu().reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def poison_outputs(values):
    """Invalidate logical outputs; keep address-bearing integer metadata legal."""
    bounded = {"bridge.ids", "body.counts", "body.starts", "body.source_row",
               "body.source_slot"}
    for name, value in values.items():
        value.fill_(float("nan") if value.is_floating_point()
                    else 0 if name in bounded else 1)


def summary(values):
    ordered = sorted(values)

    def percentile(fraction):
        position = fraction * (len(ordered) - 1)
        left = int(position)
        right = min(left + 1, len(ordered) - 1)
        return ordered[left] + (ordered[right] - ordered[left]) * (position - left)

    return {"samples": len(values), "p50": percentile(.5), "p90": percentile(.9),
            "p99": percentile(.99), "std": statistics.pstdev(values),
            "min": min(values), "max": max(values)}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "bridge-library", "body-library", "checker",
                 "bridge-helper", "body-helper", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--candidate-kind", choices=("active",), required=True)
    parser.add_argument("--active-library", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--blocks", type=int, choices=(16, 24), default=24)
    parser.add_argument("--banks", type=int, default=16)
    parser.add_argument("--calls", type=int, default=16)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--timed-case", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--shared-scale-dtype", choices=("float32", "bfloat16"),
                        default="bfloat16")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if (args.rows not in range(1, 33) or args.banks < 1 or args.calls < 1
            or args.calls % args.banks or args.samples < 30 or args.samples % 2
            or args.iterations < 1 or args.replays < 8):
        parser.error("Require M1..32, complete bank cycles, even samples>=30, iterations>0, replays>=8")
    return args


def run(args):
    import torch
    import torch_npu

    torch.set_num_threads(1)
    checker = load_module("ffn_composition_decoder", args.checker)
    sys.path.insert(0, str(args.bridge_helper.resolve().parent))
    bridge = load_module("ffn_bridge_helpers", args.bridge_helper)
    fixture = load_module("ffn_routed_fixture", args.body_helper)
    previous = sys.modules.get("check")
    sys.modules["check"] = fixture
    try:
        pipeline = load_module("ffn_pipeline_fixture", args.body_helper.with_name("check_pipeline.py"))
    finally:
        if previous is None:
            del sys.modules["check"]
        else:
            sys.modules["check"] = previous
    libraries = (args.library, args.bridge_library, args.body_library)
    for path in (*libraries, args.active_library, args.candidate_library):
        torch.ops.load_library(str(path.resolve()))
    launchers = Launchers(*libraries, args.candidate_library,
                          args.candidate_kind, args.active_library)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Matched-address token-split terminal-completion FFN with original shared quantization versus its original-shared parent, TP excluded",
        "arm_names": {"fused": "original_shared_terminal", "separate": "original_shared_parent"},
        "candidate_library": {"path": str(args.candidate_library), "sha256": digest(args.candidate_library)},
        "scratch_matching": "All 33 tensors, including the INT32[264] active table, share exact addresses within each pair",
        "owner_validation": "Original-shared Torch validator checks all 33 tensors in both independently allocated arenas before CABI comparison",
        "utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "command": sys.argv,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "libraries": {str(path): digest(path) for path in (*libraries, args.active_library, args.candidate_library)},
        "sources": {str(path): digest(path) for path in (Path(__file__), args.checker,
            args.bridge_helper, args.bridge_helper.with_name("fixtures.py"),
            args.bridge_helper.with_name("gate_probe.py"), args.body_helper,
            args.body_helper.with_name("check_pipeline.py"))},
        "abi_bytes": {"DispatchInput": ctypes.sizeof(DispatchInput), "SharedInput": ctypes.sizeof(SharedInput)},
        "warmup": 3, "samples_per_arm_per_arena": args.samples,
        "graph_replays_per_event_sample": args.iterations, "timing_unit": "us per independent FFN call",
        "input_copy_and_clone_scope": "outside timing graphs and event windows",
        "fresh_write_gate": "Before each checked CABI replay and timed arm warmup: logical floating outputs become NaN, bounded integer metadata becomes zero, other integer outputs become one; padding/tilings/system/inputs stay unchanged",
        "sequence_inputs": "Independent calls reuse the same input; outputs do not feed the next call",
        "numerical_reference": "Original-shared complete FFN Torch entry, independently compared with the original DecoderLayer norm/MLP",
        "arenas": [], "pass": False,
    }

    def persist():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    if args.preflight_only:
        report.update(cpu_preflight=True, npu_initialized=torch.npu.is_initialized())
        persist()
        if report["npu_initialized"]:
            raise RuntimeError("CPU preflight initialized NPU")
        return
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    report.update(device=str(torch.npu.get_device_properties(args.device)),
                  cann=fixture.cann_versions(torch_npu))

    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    source = fixture.load_weights(torch, SimpleNamespace(fixture=None, checkpoint=None, seed=args.seed))
    base = {name: nd(value) for name, value in source.items()}
    shared = pipeline.shared_weights(torch, nd, torch.Generator().manual_seed(args.seed + 17),
                                     scale_dtype=args.shared_scale_dtype)
    h, residual_cpu, gamma_cpu, gate_cpu, bias_cpu = bridge.bridge_case(torch, args.rows, 0)
    hidden, residual = nd(h), nd(residual_cpu)
    banks, descriptions = [], []
    route_names = ("gate_up", "gu_bias", "gu_scale", "down", "down_bias", "down_scale")
    for index in range(args.banks):
        owners = {name: torch_npu.npu_format_cast(base[name].clone(), 29)
                  for name in ("gate_up", "down")}
        routed = {name: owners[name].view(torch.int32) if name in owners else base[name].clone()
                  for name in route_names}
        sw = [shared[name].clone() for name in ("gate_tiles", "gate_scale", "down_tiles", "down_scale")]
        bank = {"owners": owners, "routed": [routed[name] for name in route_names],
                "shared": sw, "gamma": nd(gamma_cpu), "gate": nd(gate_cpu), "bias": nd(bias_cpu)}
        banks.append(bank)
        logical = {**{name: owners.get(name, routed[name]) for name in route_names},
                   **{f"shared_{i}": value for i, value in enumerate(sw)},
                   **{name: bank[name] for name in ("gamma", "gate", "bias")}}
        description = {name: {"address": value.data_ptr(),
            "format": torch_npu.get_npu_format(value),
            "storage_bytes": value.untyped_storage().nbytes(),
            "logical_sha256": raw_hash(torch, value)} for name, value in logical.items()}
        if descriptions:
            for name, value in description.items():
                if value["logical_sha256"] != descriptions[0][name]["logical_sha256"]:
                    raise RuntimeError(f"Weight bank {index} differs: {name}")
                if value["address"] in {item[name]["address"] for item in descriptions}:
                    raise RuntimeError(f"Weight bank {index} aliases: {name}")
        descriptions.append(description)
    report["banks"] = descriptions
    report["bank_descriptors_scope"] = "Initial fixture case 0; routed/shared weights stay fixed, gamma/gate/bias update between correctness cases"
    report["logical_minimum_active_matrix_bytes"] = args.banks * (
        6144 * 256 * 2 + 8 * (6144 * 256 // 2 + 128 * 6144 // 2)
        + 6144 * 256 + 128 * 6144)
    report["logical_payload_interpretation"] = "Selected matrix sizes only; no HBM traffic or cache-miss measurement"

    def update(iteration):
        x, r, gamma, gate, bias = bridge.bridge_case(torch, args.rows, iteration)
        hidden.copy_(x)
        residual.copy_(r)
        for bank in banks:
            for name, value in (("gamma", gamma), ("gate", gate), ("bias", bias)):
                bank[name].copy_(value)
        torch.npu.synchronize()

    def capture(call):
        for _ in range(3):
            call()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            call()
        return graph

    def snapshot(arena):
        values = checker.all_values(torch, arena[0], arena[1], args.rows)
        return {key: bridge.snapshot(torch, value) for key, value in values.items()}

    def poison(arena):
        poison_outputs(checker.all_values(torch, arena[0], arena[1], args.rows))
        extra = launchers.extra_by_arena[arena[0][8].data_ptr()]
        if args.candidate_kind == "active":
            extra.fill_(1)
            extra[0].zero_()
        else:
            extra.zero_()
        torch.npu.synchronize()


    def check_extra(arena, expected):
        extra = launchers.extra_by_arena[arena[0][8].data_ptr()]
        actual = extra.cpu()
        if args.candidate_kind == "active":
            ids = (expected["body.counts"].cpu() != 0).nonzero().reshape(-1).to(torch.int32)
            reference = torch.zeros(264, dtype=torch.int32)
            reference[0] = ids.numel()
            reference[8:8 + ids.numel()] = ids
            comparison = bridge.compare(torch, actual, reference)
            if not comparison["bitwise_equal"]:
                raise RuntimeError("Active-expert table did not match reference counts")
            return comparison
        records = actual.tolist()
        for slot, record in enumerate(records):
            kind = int(slot >= args.blocks)
            index = slot - args.blocks if kind else slot
            if record[:5] != [1, kind, index, args.rows, args.blocks]:
                raise RuntimeError(f"Profile identity differs at worker {slot}")
            marks = record[8:21]
            if marks[0] <= 0 or any(b < a for a, b in zip(marks, marks[1:])):
                raise RuntimeError(f"Profile clock order differs at worker {slot}")
        return {"valid_worker_traces": len(records)}

    def checked(actual, expected, label):
        if not all(bool(torch.isfinite(value).all()) for value in expected.values()):
            raise RuntimeError(f"Non-finite numerical reference: {label}")
        values = {key: bridge.compare(torch, actual[key], value) for key, value in expected.items()}
        if not all(value["bitwise_equal"] for value in values.values()):
            torch.save({"actual": actual, "expected": expected}, args.output / (label + ".pt"))
            report["failed_comparison"] = {"name": label, "checks": values}
            persist()
            raise RuntimeError(f"FFN byte check failed: {label}")
        return values

    try:
        with torch.inference_mode():
            cb, cr, cs = prepare_arena(torch, launchers, hidden, args.blocks)
            sb, sr, ss = prepare_arena(torch, launchers, hidden, args.blocks)
            if cs != ss:
                raise RuntimeError("Owner arenas have different device sync bases")
            arenas = {"fused": (cb, cr, cs), "separate": (sb, sr, ss)}
            for slot in (8,):
                checked({"tiling": cb[slot]}, {"tiling": sb[slot]}, "bridge-tiling")
            for slot in (6, 7, 18):
                checked({"tiling": cr[slot]}, {"tiling": sr[slot]}, f"body-tiling-{slot}")
            for owner, arena in arenas.items():
                calls = [Calls(torch, launchers, arena, bank, hidden, residual,
                               args.rows, args.blocks, 1) for bank in banks]
                result = {"owner": owner, "sync_base": arena[2],
                    "bridge_addresses": [value.data_ptr() for value in arena[0]],
                    "body_addresses": [value.data_ptr() for value in arena[1]],
                    "scratch_bytes": sum(value.nbytes for value in (*arena[0], *arena[1])),
                    "prechecks": [], "samples": [], "postchecks": [], "pass": False}
                result["extra_checks"] = []
                result["extra_storage"] = {"address": calls[0].extra.data_ptr(),
                    "shape": list(calls[0].extra.shape), "dtype": str(calls[0].extra.dtype),
                    "bytes": calls[0].extra.nbytes}
                report["arenas"].append(result)
                update(0)
                # Owning Torch validators run before any direct CABI capture.
                for call in calls:
                    call.owning_torch(owner)
                torch.npu.synchronize()
                graphs = [{name: capture(getattr(call, name)) for name in ("fused", "separate")}
                          for call in calls]
                for iteration in range(args.replays):
                    update(iteration)
                    for index, call in enumerate(calls):
                        call.owning_torch(owner)
                        torch.npu.synchronize()
                        expected = snapshot(arena)
                        for arm, graph in graphs[index].items():
                            poison(arena)
                            graph.replay()
                            torch.npu.synchronize()
                            label = f"{owner}-bank{index}-r{iteration}-{arm}"
                            checks = checked(snapshot(arena), expected, label)
                            result["extra_checks"].append({"arm": arm, "name": label,
                                "checks": check_extra(arena, expected)})
                            result["prechecks"].append({"name": label, "checks": checks, "pass": True})
                        persist()

                update(args.timed_case)
                expected_banks = []
                for call in calls:
                    call.owning_torch(owner)
                    torch.npu.synchronize()
                    expected_banks.append(snapshot(arena))
                result["timed_route"] = {"case": args.timed_case,
                    "active_experts": int((expected_banks[0]["body.counts"] > 0).sum()),
                    "maximum_routes_per_expert": int(expected_banks[0]["body.counts"].max()),
                    "counts": expected_banks[0]["body.counts"].tolist(),
                    "hidden_sha256": raw_hash(torch, hidden),
                    "residual_sha256": raw_hash(torch, residual),
                    "gate_sha256": raw_hash(torch, banks[0]["gate"]),
                    "hidden_nonzero": int(torch.count_nonzero(hidden)),
                    "residual_nonzero": int(torch.count_nonzero(residual)),
                    "bank_router_sha256": [{name: raw_hash(torch, bank[name])
                                            for name in ("gamma", "gate", "bias")}
                                           for bank in banks]}
                if any(value != result["timed_route"]["bank_router_sha256"][0]
                       for value in result["timed_route"]["bank_router_sha256"]):
                    raise RuntimeError("Timed router values differ across weight banks")
                for index, expected in enumerate(expected_banks):
                    checked({"counts": expected["body.counts"]},
                            {"counts": expected_banks[0]["body.counts"]},
                            f"{owner}-bank{index}-timed-route")
                sequences = calls * (args.calls // args.banks)

                def sequence(arm):
                    for call in sequences:
                        getattr(call, arm)()

                timed = {arm: capture(lambda arm=arm: sequence(arm)) for arm in ("fused", "separate")}
                expected_tail = {key: expected_banks[-1][key] for key in
                                 ("bridge.combined", "bridge.retained", "body.counts")}
                for sample in range(args.samples):
                    order = ("fused", "separate") if sample % 2 == 0 else ("separate", "fused")
                    record = {"sample": sample, "order": order, "event_total_us": {}, "per_call_us": {}}
                    for arm in order:
                        poison(arena)
                        for _ in range(3):
                            timed[arm].replay()
                        torch.npu.synchronize()
                        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                        start.record()
                        for _ in range(args.iterations):
                            timed[arm].replay()
                        end.record()
                        end.synchronize()
                        total = start.elapsed_time(end) * 1000
                        record["event_total_us"][arm] = total
                        record["per_call_us"][arm] = total / (args.iterations * args.calls)
                        actual_tail = {"bridge.combined": arena[0][10], "bridge.retained": arena[0][1],
                                       "body.counts": arena[1][12]}
                        checked(actual_tail, expected_tail, f"{owner}-sample{sample}-{arm}-tail")
                        result["extra_checks"].append({"arm": arm, "name": f"sample{sample}-tail",
                            "checks": check_extra(arena, expected_tail)})
                    record["paired_saving_us"] = record["per_call_us"]["separate"] - record["per_call_us"]["fused"]
                    record["post_event_tail_byte_equal"] = True
                    result["samples"].append(record)
                    persist()
                for arm in ("fused", "separate"):
                    for index, call in enumerate(calls):
                        poison(arena)
                        graphs[index][arm].replay()
                        torch.npu.synchronize()
                        checks = checked(snapshot(arena), expected_banks[index], f"{owner}-{arm}-bank{index}-post")
                        result["extra_checks"].append({"arm": arm, "name": f"bank{index}-post",
                            "checks": check_extra(arena, expected_banks[index])})
                        result["postchecks"].append({"arm": arm, "bank": index, "checks": checks, "pass": True})
                        # Re-enter the owner validator after timing, then compare a
                        # new CPU snapshot; never compare aliased live outputs.
                        call.owning_torch(owner)
                        torch.npu.synchronize()
                        owner_checks = checked(snapshot(arena), expected_banks[index],
                                               f"{owner}-bank{index}-owner-post")
                        result["postchecks"].append({"arm": "owning_torch_after_" + arm,
                            "bank": index, "checks": owner_checks, "pass": True})
                result["summary_us"] = {arm: summary([item["per_call_us"][arm] for item in result["samples"]])
                                        for arm in ("fused", "separate")}
                result["paired_saving_us"] = summary([item["paired_saving_us"] for item in result["samples"]])
                result["paired_wins"] = sum(item["paired_saving_us"] > 0 for item in result["samples"])
                result["pass"] = True
                for group in graphs:
                    for graph in group.values():
                        graph.reset()
                for graph in timed.values():
                    graph.reset()
                persist()
        report["pass"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        persist()
    print(json.dumps({"pass": report["pass"], "paired": [
        {"owner": item["owner"], "saving_us": item["paired_saving_us"], "wins": item["paired_wins"]}
        for item in report["arenas"]]}))


if __name__ == "__main__":
    run(arguments())
