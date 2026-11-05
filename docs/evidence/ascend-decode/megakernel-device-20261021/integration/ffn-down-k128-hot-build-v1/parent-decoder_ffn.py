"""Compare a complete native FFN with the actual DecoderLayer norm and MLP."""

from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import traceback
from types import MethodType


LIBRARY_SHA = "0842cd0f3a946c39aefb3e926e01eb870a61d6d1ad8689aebcf52a1b3daa2b65"
SCALE_SOURCE_SHA = "efdcc610c05a44952c2548e0d8b0cd48ca7704c3e2cf8247f8f08f5e765b830b"


def unpack_scale(torch, packed, columns):
    """Invert the official FP32-bit-pattern-to-INT64 encoding on the CPU."""
    import numpy as np

    if packed.dtype != torch.int64 or tuple(packed.shape) not in (
            (256, columns), (256, 1, columns)):
        raise ValueError("Require the actual per-channel INT64 scale carrier")
    words = packed.detach().cpu().contiguous().reshape(256, columns)
    if not bool(((words >= 0) & (words <= 0xffffffff)).all()):
        raise ValueError("The carrier must contain zero-extended FP32 bits")
    bits = words.numpy().astype(np.uint32)
    values = torch.from_numpy(bits.view(np.float32).copy())
    if not bool(torch.isfinite(values).all()):
        raise ValueError("The decoded scale must be finite")
    if not np.array_equal(values.numpy().view(np.uint32), bits):
        raise RuntimeError("Scale decoding changed the original FP32 bits")
    return values


def shared_weights(torch, torch_npu, shared):
    gate, down = shared.layer.gate_up_proj, shared.layer.down_proj
    gu = torch_npu.npu_format_cast(gate.weight.detach(), 2)
    dw = torch_npu.npu_format_cast(down.weight.detach(), 2)
    if (gu.dtype != torch.int8 or dw.dtype != torch.int8
            or tuple(gu.shape) != (6144, 256) or tuple(dw.shape) != (128, 6144)):
        raise ValueError("Require actual TP16 shared W8A8 weights")
    gu = gu.reshape(6, 1024, 4, 64).permute(0, 2, 1, 3).contiguous()
    dw = dw.reshape(128, 24, 256).permute(1, 0, 2).contiguous()
    return [torch_npu.npu_format_cast(gu.reshape(24, 1024, 64), 2),
            gate.weight_scale_fp32.detach().contiguous(),
            torch_npu.npu_format_cast(dw, 2),
            down.weight_scale.detach().float().contiguous()]


def clone_output(torch, value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, (tuple, list)) and len(value) == 2 and value[1] is None:
        return value[0].clone()
    raise RuntimeError("The actual router returned an unexpected value")


@contextmanager
def observe_original(torch, layer, group):
    """Queue device snapshots while preserving the original FFN execution."""
    captured = {}
    original_reduce = group.all_reduce
    runner = layer.mlp.experts
    original_body = runner._forward_impl
    reductions = []

    def reduce(unused_self, value):
        if reductions:
            raise RuntimeError("The local FFN reached more than one TP boundary")
        reductions.append(value.clone())
        return value

    def body(unused_self, *args, **kwargs):
        result = original_body(*args, **kwargs)
        if len(result) != 2 or any(not isinstance(x, torch.Tensor) for x in result):
            raise RuntimeError("The actual MoE runner must return shared and routed outputs")
        captured["shared"], captured["routed"] = [value.clone() for value in result]
        return result

    def gate_hook(unused_module, unused_inputs, value):
        captured["logits"] = clone_output(torch, value)

    group.all_reduce = MethodType(reduce, group)
    runner._forward_impl = MethodType(body, runner)
    handle = layer.mlp.gate.register_forward_hook(gate_hook)
    try:
        yield captured
        if len(reductions) != 1:
            raise RuntimeError("The original FFN did not reach its single TP boundary")
        captured["pre_ar2"] = reductions[0]
    finally:
        handle.remove()
        runner._forward_impl = original_body
        group.all_reduce = original_reduce


EXPECTED_DOWN_TILING = {'layout': 1, 'raw_hex': '0100000040000000001800008000000080000000400000000018000080000000200000000004000080000000020000000600000002000000060000000000000000000000000000000000000000100600000002000000000001000000010000000100000001000000010000000100000000000000000000000200000001000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'raw_sha256': 'c2b73af9d605ad8e64ea551f288d188a0cf636828a5a0471ea7c63d616f50999'}


def check_down_tiling(torch, value, expected):
    if value.dtype != torch.uint8 or value.ndim != 1 or value.numel() < 200:
        raise ValueError("Down tiling requires at least 200 explicit bytes")
    payload = bytes(value[:200].detach().cpu().tolist())
    if payload.hex() != expected["raw_hex"]:
        raise RuntimeError("Prepared Down tiling differs from the K128 host tiler")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected["raw_sha256"]:
        raise RuntimeError("The Down tiling byte oracle is inconsistent")
    return {"address": value.data_ptr(), "bytes": 200,
            "sha256": actual, "pass": True}


def run_decoder_forward(torch, layers, config, state, args):
    import torch_npu
    from attention_check import metric
    from vendor_hook_forward import snapshot, source_identity
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    library = Path(__file__).with_name("ffn_down_k128.so")
    if hashlib.sha256(library.read_bytes()).hexdigest() != LIBRARY_SHA:
        raise RuntimeError("The accepted complete-FFN library changed")
    torch.ops.load_library(str(library))
    native = torch.ops.tilert_ffn_down_k128
    report = {
        "scope": "Actual post_attention_layernorm plus original MLP versus complete native FFN; explicit post-AR1 and retained-residual inputs; local pre-AR2 only",
        "library_sha256": LIBRARY_SHA,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "widths": args.forward_widths, "iterations": args.forward_replays,
        "blocks": 24, "tp_geometry": 16, "tp_collectives_executed": 0,
        "attention_executed": False, "synthetic_parameters": True,
        "native_graph_replay": True, "reference_graph_replay": False,
        "immediate_graph_consumers": ["combined", "routed", "shared", "retained"],
        "consumer_storage": "Preallocated outside capture and retained by owner",
        "consumer_poison": "All four outputs poisoned before every replay",
        "down_tiling_checks": [],
        "layers": [], "cases": [], "pass": False,
    }
    output = args.output / "decoder-ffn.json"

    def persist():
        output.write_text(json.dumps(report, indent=2) + "\n")

    def nd(value):
        return torch_npu.npu_format_cast(value.to("npu").contiguous(), 2)

    owners = []
    persist()
    try:
        with torch.inference_mode():
            for index, layer in zip((6, 7), layers):
                routed = layer.mlp.experts.routed_experts
                scheme = routed.quant_method.quant_method
                scale_path = Path(inspect.getfile(type(scheme)))
                if hashlib.sha256(scale_path.read_bytes()).hexdigest() != SCALE_SOURCE_SHA:
                    raise RuntimeError("The actual scale encoding implementation changed")
                cfg = config.model_config.hf_config
                expected = {"hidden_size": 6144, "n_routed_experts": 256,
                            "num_experts_per_tok": 8, "n_group": 1,
                            "topk_group": 1, "routed_scaling_factor": 2.5,
                            "scoring_func": "sigmoid", "norm_topk_prob": True}
                if any(getattr(cfg, key) != value for key, value in expected.items()):
                    raise RuntimeError("The actual router does not match the native FFN contract")
                scales = [unpack_scale(torch, getattr(routed, name), columns)
                          for name, columns in (("w13_weight_scale", 256),
                                                ("w2_weight_scale", 6144))]
                weights = [routed.w13_weight, routed.w13_scale_bias, nd(scales[0]),
                           routed.w2_weight, routed.w2_scale_bias, nd(scales[1])]
                shared = shared_weights(torch, torch_npu, layer.mlp.experts.ascend_shared_experts)
                gamma = layer.post_attention_layernorm.weight.detach()
                gate = nd(torch_npu.npu_format_cast(layer.mlp.gate.weight.detach(), 2).t())
                bias = layer.mlp.gate.e_score_correction_bias.detach()
                epsilon = layer.post_attention_layernorm.variance_epsilon
                report["layers"].append({
                    "layer": index, "norm": source_identity(type(layer.post_attention_layernorm).forward),
                    "mlp": source_identity(type(layer.mlp).forward),
                    "gmm1": source_identity(type(scheme).apply_gmm1_act_quant),
                    "gmm2": source_identity(type(scheme).apply_gmm2),
                    "scale_source_sha256": SCALE_SOURCE_SHA,
                    "scale_fp32_sha256": [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in scales],
                    "routed_weight_data_ptrs": [weights[i].data_ptr() for i in (0, 3)],
                    "epsilon": epsilon, "router": expected,
                })
                persist()
                for rows in args.forward_widths:
                    hidden = nd(torch.zeros(rows, 6144, dtype=torch.bfloat16))
                    residual = nd(torch.zeros_like(hidden, device="cpu"))
                    bridge, body, sync = native.prepare(hidden, 24, 1)
                    report["down_tiling_checks"].append({
                        "layer": index, "rows": rows, "stage": "prepared",
                        **check_down_tiling(torch, body[7], EXPECTED_DOWN_TILING)})
                    consumer_sources = {
                        "combined": bridge[10], "routed": body[9],
                        "shared": body[19], "retained": bridge[1],
                    }
                    consumers = {name: torch.empty_like(value)
                                 for name, value in consumer_sources.items()}
                    owners.append((bridge, body, consumers))

                    def call_native():
                        native.forward_out(hidden.clone(), residual.clone(), gamma, gate, bias,
                                           weights, shared, bridge, body, epsilon, 24, 1, sync)
                        for name, value in consumer_sources.items():
                            consumers[name].copy_(value)

                    for unused in range(2):
                        call_native()
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        call_native()
                    try:
                        for iteration in range(args.forward_replays):
                            generator = torch.Generator().manual_seed(73023 + index * 10000 + rows * 100 + iteration)
                            x = torch.randn(rows, 6144, generator=generator).bfloat16()
                            r = torch.randn(rows, 6144, generator=generator).bfloat16()
                            hidden.copy_(x)
                            residual.copy_(r)
                            with set_ascend_forward_context(None, config, num_tokens=rows):
                                with observe_original(torch, layer, state.get_tp_group()) as values:
                                    normalized, retained = layer.post_attention_layernorm(hidden.clone(), residual.clone())
                                    values["normalized"] = normalized.clone()
                                    values["retained"] = retained.clone()
                                    values["combined"] = layer.mlp(normalized).clone()
                            torch.npu.synchronize()
                            reference = {name: snapshot(torch, value) for name, value in values.items()}
                            for name in consumers:
                                reference["consumer." + name] = reference[name]
                            for value in (bridge[0], bridge[1], bridge[3], bridge[10],
                                          body[9], body[19], *consumers.values()):
                                value.fill_(float("nan"))
                            graph.replay()
                            torch.npu.synchronize()
                            actual = {"normalized": bridge[0], "retained": bridge[1],
                                      "logits": bridge[3], "combined": bridge[10],
                                      "pre_ar2": bridge[10], "routed": body[9], "shared": body[19]}
                            actual.update({"consumer." + name: value
                                           for name, value in consumers.items()})
                            actual = {name: snapshot(torch, value) for name, value in actual.items()}
                            report["down_tiling_checks"].append({
                                "layer": index, "rows": rows,
                                "stage": "replay", "iteration": iteration,
                                **check_down_tiling(torch, body[7], EXPECTED_DOWN_TILING)})
                            checks = {name: metric(torch, actual[name], value)
                                      for name, value in reference.items()}
                            passed = len(checks) == 11 and all(v["pass"] and v["bitwise_equal"] for v in checks.values())
                            case = {"layer": index, "rows": rows, "iteration": iteration,
                                    "pass": passed, "checks": checks}
                            report["cases"].append(case)
                            persist()
                            print(json.dumps({"decoder_ffn_case": case}), flush=True)
                            if not passed:
                                dump = args.output / f"decoder-ffn-layer{index}-m{rows}-r{iteration}.pt"
                                torch.save({"hidden": x, "residual": r,
                                            "reference": reference, "native": actual}, dump)
                                case["failure_dump"] = str(dump)
                                persist()
                                raise RuntimeError("The complete native FFN differs from the original FFN")
                    finally:
                        graph.reset()
            report["pass"] = True
            persist()
            return report
    except Exception:
        report["error"] = traceback.format_exc()
        persist()
        raise
