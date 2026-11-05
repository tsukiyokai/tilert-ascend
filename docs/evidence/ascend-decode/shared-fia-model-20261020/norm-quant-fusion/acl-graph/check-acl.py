"""Check one norm/quant graph with the installed ACLGraphWrapper.

Each arm runs in a fresh process and directory. CPU mode uses FakeTensor
NPU inputs and blocks lazy device initialization. Device mode compiles
the same M8 subgraph using the production ACL capture scope and optimizer.
The single-die norm subgraph does not exercise attention or TP reductions.
"""

import argparse
import ast
from collections import Counter
from contextlib import ExitStack
import copy
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

import torch
import torch_npu
import vllm_ascend.vllm_ascend_C
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx


ROOT = Path(__file__).resolve().parent
INPUT_SHA = "5e59d7dc1f27f545c21ea4f9fb58efb9015ea5fc8b8ca73167c1437d384c6673"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def norm_quant(x, residual, gamma, beta):
    normalized, _, retained = torch.ops._C_ascend.npu_add_rms_norm_bias(
        x, residual, gamma, beta, 1e-5,
    )
    codes, scale = torch_npu.npu_dynamic_quant(normalized,
                                              dst_type=torch.int8)
    return codes, scale, retained


def source_identity(*objects):
    paths = {Path(inspect.getfile(inspect.unwrap(value))).resolve()
             for value in objects}
    return {str(path): digest(path) for path in sorted(paths)}


def save_graph(directory, name, graph):
    graph.recompile()
    path = directory / (name + ".py")
    path.write_text(graph.code)
    return {
        "path": str(path), "sha256": digest(path),
        "operators": dict(Counter(str(node.target)
                                   for node in graph.graph.nodes
                                   if node.op == "call_function")),
    }


def actual_config(enabled):
    from tilert.models.glm_5_2_ascend.cann_model import create_config
    from vllm_ascend.ascend_config import (
        clear_ascend_config, init_ascend_config,
    )

    config = create_config(
        "/workspace/models/GLM-5.2-w4a8c8", 9344, 0.02,
        compile_target=True, super_kernel=True, graph_width=8,
    )
    options = config.additional_config["ascend_compilation_config"]
    options["fuse_norm_quant"] = enabled
    clear_ascend_config()
    ascend = init_ascend_config(config).ascend_compilation_config
    if ascend.fuse_norm_quant != enabled:
        raise ValueError("Validated fusion setting differs from requested arm")
    return config, ascend


def inspect_graphs(args, directory, frozen):
    import npugraph_ex as nge
    import npugraph_ex.npu_fx_compiler as nfx
    try:
        from npugraph_ex.configs.compiler_config import _process_kwargs_options
    except ImportError:
        from npugraph_ex.configs import npugraphex_config

        _process_kwargs_options = npugraphex_config._process_kwargs_options
    from vllm.compilation.passes.inductor_pass import pass_context
    from vllm.config import set_current_vllm_config
    from vllm.config.compilation import Range
    from vllm.forward_context import set_forward_context
    from tilert.models.glm_5_2_ascend.cann_model import create_config
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.compilation.compiler_interface import _configure_backend
    from vllm_ascend.compilation.graph_fusion_pass_manager import (
        GraphFusionPassManager,
    )
    from vllm_ascend.device.hardware_profile import (
        HardwareCapability, get_current_hardware_profile,
    )
    from vllm_ascend.compilation.updatable_graph import UpdatableGraph
    from vllm_ascend.utils import super_kernel_scope
    from vllm_ascend.worker.model_runner_v1 import graph_capture

    config, ascend = actual_config(args.flag == "on")
    backend_config = nge.CompilerConfig()
    _configure_backend(backend_config, ascend, config,
                       process_kwargs_options=_process_kwargs_options)
    manager = GraphFusionPassManager()
    device = torch.device(args.device if args.mode == "device" else "npu:0")
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    with (fake_mode, set_current_vllm_config(config),
          pass_context(Range(1, 16))):
        manager.configure(config)
        inputs = [torch.empty((8, 6144), device=device, dtype=torch.bfloat16)
                  for _ in range(2)]
        inputs += [torch.empty((6144,), device=device, dtype=torch.bfloat16)
                   for _ in range(2)]
        graph = make_fx(norm_quant)(*inputs)
        graph.graph.eliminate_dead_code()
        stages = {"before": save_graph(directory, "before", graph)}
        manager(graph)
        stages["vllm"] = save_graph(directory, "after-vllm", graph)
        before_backend = copy.deepcopy(graph)
        observer = nfx.GraphTransformObserver(graph, inputs, backend_config)
        optimized = nfx._optimize_fx(graph, backend_config, observer)
        stages["backend"] = save_graph(directory, "after-backend", optimized)
    profile = get_current_hardware_profile()
    result = {
        "stages": stages,
        "passes": [type(value).__name__ for value in manager.passes],
        "validated_ascend_config": asdict(ascend),
        "profile": str(profile),
        "supports_norm_quant_fusion": profile.supports(
            HardwareCapability.GRAPH_NORM_QUANT_FUSION,
        ),
        "backend_options": {
            name: getattr(getattr(backend_config, name), "value",
                          getattr(backend_config, name))
            for name in ("force_eager", "static_kernel_compile",
                         "super_kernel_optimize", "pattern_fusion_pass")
        },
        "config_factory_source": str(args.source.resolve()),
        "vllm_config_hash": config.compute_hash(),
        "use_v2_model_runner": config.use_v2_model_runner,
        "source_sha256": source_identity(
            GraphFusionPassManager, _configure_backend, type(backend_config),
            nfx._NpuFxCompiler, create_config, ACLGraphWrapper,
            UpdatableGraph, super_kernel_scope, graph_capture,
            set_forward_context,
        ),
    }
    enabled = args.flag == "on"
    result["operator_checks"] = {
        name: operator_contract(stage["operators"],
                                enabled and name != "before")
        for name, stage in stages.items()
    }
    return before_backend, config, ascend, result


def operator_contract(operators, fused):
    expected = {
        "npu.npu_add_rms_norm_dynamic_quant.default": int(fused),
        "_C_ascend.npu_add_rms_norm_bias.default": int(not fused),
        "npu.npu_dynamic_quant.default": int(not fused),
    }
    return all(operators.get(name, 0) == count
               for name, count in expected.items())


def compare(left, right):
    a, b = left.detach().cpu(), right.detach().cpu()
    return {
        "bitexact": torch.equal(a.contiguous().view(torch.uint8),
                                b.contiguous().view(torch.uint8)),
        "different": int((a != b).sum()),
        "maxabs": float((a.double() - b.double()).abs().max()),
    }


def device_run(args, directory, frozen, graph, config, ascend):
    from vllm.compilation import monitor
    from vllm.config import CUDAGraphMode, set_current_vllm_config
    from vllm.config.compilation import Range
    from vllm.forward_context import (
        BatchDescriptor, get_forward_context, set_forward_context,
    )
    from vllm_ascend.attention.sfa_v1 import AscendSFABackend
    from vllm_ascend.compilation.acl_graph import (
        ACLGraphWrapper, get_graph_params, set_graph_params,
    )
    from vllm_ascend.compilation.compiler_interface import AscendCompiler
    from vllm_ascend.utils import use_updatable_graph
    from vllm_ascend.worker.model_runner_v1 import graph_capture

    device = torch.device(args.device)
    inputs = [frozen[name].repeat(8, 1).to(device)
              for name in ("x", "residual")]
    inputs += [frozen[name].to(device) for name in ("gamma", "beta")]
    compiler = AscendCompiler()
    compiler.compute_hash(config)
    compiler.initialize_cache(str(directory / "cache"))
    compiled, handle = compiler.compile(
        graph, inputs, {}, Range(1, 16), key="norm-m8",
    )
    if config.use_v2_model_runner:
        raise ValueError("Expected the production V1 forward context")
    if use_updatable_graph(AscendSFABackend):
        raise ValueError("Expected SFA to use direct graph replay")
    params = get_graph_params()
    if params is None:
        set_graph_params([8])
    elif 8 not in params.workspaces:
        raise ValueError("Graph parameters omit M8")
    wrapper = ACLGraphWrapper(compiled, config, CUDAGraphMode.FULL)
    wrapper.set_attn_backend(AscendSFABackend)
    if not wrapper.enable_super_kernel:
        raise ValueError("Production ACL SuperKernel is disabled")
    descriptor = BatchDescriptor(8, 1, uniform=True)
    addresses = [value.data_ptr() for value in inputs]
    with (torch.inference_mode(), set_current_vllm_config(config),
          set_forward_context({}, config, num_tokens=8,
                              cudagraph_runtime_mode=CUDAGraphMode.FULL,
                              batch_descriptor=descriptor)):
        warm = tuple(value.cpu().clone() for value in compiled(*inputs))
        caller = torch.npu.current_stream(device)
        enabled = monitor.cudagraph_capturing_enabled
        monitor.set_cudagraph_capturing_enabled(True)
        try:
            with graph_capture(device) as capture:
                wrapper(*inputs)
            caller.wait_stream(capture.stream)
        finally:
            monitor.set_cudagraph_capturing_enabled(enabled)
        entry = wrapper.concrete_aclgraph_entries[descriptor]
        if entry.aclgraph is None:
            raise RuntimeError("ACLGraphWrapper did not capture")
        get_forward_context().capturing = False
        variants = [
            ("original", frozen["x"], frozen["residual"]),
            ("changed", -frozen["x"], frozen["residual"].roll(1, -1)),
            ("restored", frozen["x"], frozen["residual"]),
        ]
        snapshots, comparisons = [], []
        names = ("q", "scale", "residual")
        for name, x, residual in variants:
            inputs[0].copy_(x.repeat(8, 1))
            inputs[1].copy_(residual.repeat(8, 1))
            output = wrapper(*inputs)
            actual = tuple(value.cpu().clone() for value in output)
            eager = tuple(value.cpu().clone()
                          for value in norm_quant(*inputs))
            if args.flag == "on":
                fused = torch.ops.npu.npu_add_rms_norm_dynamic_quant.default(
                    *inputs[:3], beta=inputs[3], epsilon=1e-5,
                    output_mask=[True, False],
                )
                expected = tuple(fused[index].cpu().clone()
                                 for index in (0, 3, 2))
            else:
                expected = eager
            comparisons.append({
                "input": name,
                "expected_vs_graph": {
                    field: compare(a, b)
                    for field, a, b in zip(names, expected, actual)
                },
                "eager_vs_graph": {
                    field: compare(a, b)
                    for field, a, b in zip(names, eager, actual)
                },
            })
            snapshots.append({"input": name, "x": x, "residual": residual,
                              "eager": eager, "expected": expected,
                              "graph": actual})
        torch.npu.synchronize(device)
    artifact = Path(handle[1])
    code = artifact.read_text()
    (directory / "compiled-ir.py").write_text(code)
    torch.save({"inputs": frozen, "compiled_warm": warm,
                "graph_replays": snapshots}, directory / "outputs.pt")
    operators = Counter(
        ast.unparse(node.func).removeprefix("torch.ops.")
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
    )
    exact = all(
        value["bitexact"] for entry in comparisons
        for value in entry["expected_vs_graph"].values()
    )
    storage = addresses == [value.data_ptr() for value in inputs]
    operator_check = operator_contract(operators, args.flag == "on")
    result = {
        "passed": exact and storage and operator_check,
        "compiled_ir_sha256": digest(directory / "compiled-ir.py"),
        "compiled_ir_operators": dict(operators),
        "compiled_ir_operator_check": operator_check,
        "compiled_handle": handle,
        "outer_graph": {
            "wrapper": type(wrapper).__name__,
            "graph": type(entry.aclgraph).__name__,
            "captures": 1, "replays": 3,
            "scope": "full_model",
            "super_kernel_enabled": wrapper.enable_super_kernel,
            "optimizer_options": {"dcci_after_kernel_end": [".*"]},
            "forward_context": "V1 inner vLLM context; no TP or attention",
        },
        "persistent_input_addresses": addresses,
        "persistent_addresses_preserved": storage,
        "comparisons": comparisons,
        "outputs_sha256": digest(directory / "outputs.pt"),
        "peak_allocated": torch.npu.max_memory_allocated(device),
    }
    entry.aclgraph.reset()
    wrapper.concrete_aclgraph_entries.clear()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "device"), required=True)
    parser.add_argument("--flag", choices=("on", "off"), required=True)
    parser.add_argument("--device", default="npu:9")
    parser.add_argument("--source", type=Path,
                        default=Path("/workspace/src/TileRT-Ascend"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.source.resolve())]
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output)
    inputs_path = ROOT.parent / "inputs.pt"
    if digest(inputs_path) != INPUT_SHA:
        raise ValueError("Frozen operand bundle differs")
    frozen = torch.load(inputs_path, map_location="cpu", weights_only=True)
    report = {
        "status": "running", "mode": args.mode, "flag": args.flag,
        "input_kind": "eight repeats of actual frozen M1 post-attention row",
        "inputs_sha256": INPUT_SHA, "script_sha256": digest(__file__),
        "timing_measured": False,
        "npu_initialized_before": torch.npu.is_initialized(),
    }
    try:
        with ExitStack() as stack:
            if args.mode == "cpu":
                stack.enter_context(patch.object(torch.npu, "current_device",
                                                 return_value=0))
                stack.enter_context(patch.object(
                    torch.npu, "_lazy_init",
                    side_effect=RuntimeError("CPU check blocks NPU init"),
                ))
            else:
                torch.npu.set_device(torch.device(args.device))
                properties = torch.npu.get_device_properties(args.device)
                fraction = 512 * 2 ** 20 / properties.total_memory
                torch.npu.set_per_process_memory_fraction(
                    fraction, args.device,
                )
            graph, config, ascend, observed = inspect_graphs(
                args, args.output, frozen,
            )
            report.update(observed)
            if not all(report["operator_checks"].values()):
                raise RuntimeError("Observed FX operator counts differ")
            if args.mode == "device":
                report["device_result"] = device_run(
                    args, args.output, frozen, graph, config, ascend,
                )
                if not report["device_result"]["passed"]:
                    raise RuntimeError("Compiled IR or replay values differ")
            elif (report["npu_initialized_before"]
                  or torch.npu.is_initialized()):
                raise RuntimeError("CPU check initialized NPU")
            report["status"] = "pass"
    except Exception as error:
        report.update(status="failed", error=repr(error),
                      traceback=traceback.format_exc())
    report["npu_initialized_after"] = torch.npu.is_initialized()
    (args.output / "result.json").write_text(
        json.dumps(report, indent=2) + "\n",
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
