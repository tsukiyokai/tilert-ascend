"""Observe the original shared FFN and native payload on a retained failure."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import MethodType


PARENT_SHA = "806866e0b5ba358519202760ae6651aaaebc598ec311816237515326717d39a7"
FAILURE_SHA = "4a2ba25892268b56e645bcaf6af95790d72eef18f008fefda6052a8f25b48e1e"


@contextmanager
def replace_attribute(owner, name, replacement):
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield original
    finally:
        setattr(owner, name, original)


class SharedTaps:
    """Clone values on their producer stream without replacing arithmetic."""

    def __init__(self, torch, torch_npu):
        self.torch = torch
        self.torch_npu = torch_npu
        self.values = {}
        self.calls = {"prepare": 0, "activate": 0, "down": 0}

    @contextmanager
    def observe(self, shared):
        prepare = shared._prepare_gate_up_input
        finish = shared._run_a8_int_mlp

        def prepared(unused_self, hidden):
            result = prepare(hidden)
            self.calls["prepare"] += 1
            self.values["normalized"] = hidden.clone()
            self.values["codes"], self.values["input_scale"] = (
                value.clone() for value in result)
            return result

        def finished(unused_self, submitted, milestones):
            self.values["gate_int32"] = submitted.gate_up.clone()
            gate = shared.layer.gate_up_proj
            down = shared.layer.down_proj
            self.values["gate_weight"] = gate.weight.detach().clone()
            self.values["gate_scale"] = gate.weight_scale_fp32.detach().clone()
            self.values["down_weight"] = down.weight.detach().clone()
            self.values["down_scale"] = down.weight_scale.detach().clone()
            activation = self.torch.ops._C_ascend.npu_dequant_swiglu_quant
            matmul = self.torch_npu.npu_quant_matmul

            def activated(*args, **kwargs):
                result = activation(*args, **kwargs)
                self.calls["activate"] += 1
                self.values["middle"], self.values["middle_scale"] = (
                    value.clone() for value in result)
                return result

            def projected(*args, **kwargs):
                if args[1].data_ptr() != down.weight.data_ptr():
                    raise RuntimeError("The observed matmul is not shared Down")
                result = matmul(*args, **kwargs)
                self.calls["down"] += 1
                self.values["output"] = result.clone()
                return result

            with replace_attribute(
                    self.torch.ops._C_ascend, "npu_dequant_swiglu_quant",
                    activated), replace_attribute(
                        self.torch_npu, "npu_quant_matmul", projected):
                return finish(submitted, milestones)

        with replace_attribute(shared, "_prepare_gate_up_input",
                               MethodType(prepared, shared)), replace_attribute(
                shared, "_run_a8_int_mlp", MethodType(finished, shared)):
            yield


def shared_payload(torch, bridge, body):
    """Read the initialized M32 shared regions of the frozen 8430 ABI."""
    rows, capacity = 32, 32
    payload = body[17]
    partial = capacity * 6144
    middle = partial + 6 * 256 * capacity * 4
    scale = middle + capacity * 128
    down = scale + capacity * 32
    parts = payload[partial:middle].view(torch.int32)
    gate = parts.reshape(6, 4, capacity, 64).sum(0, dtype=torch.int32)
    return {
        "normalized": bridge[0], "codes": bridge[6],
        "input_scale": bridge[7],
        "gate_int32": gate.permute(1, 0, 2).reshape(rows, 256),
        "middle": payload[middle:scale].view(torch.int8).reshape(rows, 128),
        "middle_scale": payload[scale:down].view(torch.float32).reshape(
            rows, 8)[:, 0],
        "down_int32": payload[down:down + capacity * 6144 * 4].view(
            torch.int32).reshape(24, capacity, 256).permute(1, 0, 2).reshape(
                rows, 6144),
        "output": body[19],
    }


def save_diagnostic(torch, torch_npu, args, taps, native_arguments):
    def snapshot(value):
        return torch_npu.npu_format_cast(value.detach(), 2).cpu().clone()

    parent = torch.load(args.ffn_shared_parent, weights_only=True,
                        map_location="cpu")
    failed = args.output / "decoder-ffn-layer6-m32-r2.pt"
    current = torch.load(failed, weights_only=True, map_location="cpu")
    reproduce = {}
    for name in ("hidden", "residual"):
        reproduce[name] = torch.equal(parent[name].view(torch.uint8),
                                      current[name].view(torch.uint8))
    for arm in ("reference", "native"):
        for name, value in parent[arm].items():
            reproduce[f"{arm}.{name}"] = torch.equal(
                value.view(torch.uint8), current[arm][name].view(torch.uint8))
    if not all(reproduce.values()):
        raise RuntimeError("The shared diagnostic changed the retained failure")
    if taps.calls != {"prepare": 1, "activate": 1, "down": 1}:
        raise RuntimeError(f"Unexpected original shared call counts: {taps.calls}")
    bridge, body = native_arguments[7:9]
    native = shared_payload(torch, bridge, body)
    reference = {name: snapshot(value) for name, value in taps.values.items()}
    native = {name: snapshot(value) for name, value in native.items()}
    artifact = args.output / "decoder-shared-diagnostic.pt"
    torch.save({"reference": reference, "native": native,
                "parent_reproduced": reproduce}, artifact)
    checks = {}
    for name, actual in native.items():
        if name not in reference:
            continue
        expected = reference[name].reshape(actual.shape)
        different = actual != expected
        checks[name] = {
            "bitwise_equal": torch.equal(actual.contiguous().view(torch.uint8),
                                         expected.contiguous().view(torch.uint8)),
            "different_elements": int(different.sum()),
            "coordinates": different.nonzero().tolist()[:24],
        }
    report = {
        "scope": "Original shared FFN stages versus frozen native payload",
        "parent_failure_sha256": FAILURE_SHA,
        "parent_reproduced": reproduce, "calls": taps.calls,
        "checks": checks,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "native_graph_replay": True, "original_arithmetic_replaced": False,
        "internal_device_synchronizations_added": 0,
        "timing": False, "diagnostic_complete": True,
    }
    (args.output / "decoder-shared-diagnostic.json").write_text(
        json.dumps(report, indent=2) + "\n")


def run_decoder_forward(torch, layers, config, state, args):
    import decoder_ffn
    import torch_npu

    if hashlib.sha256(Path(decoder_ffn.__file__).read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError("The original complete-FFN fixture changed")
    if hashlib.sha256(args.ffn_shared_parent.read_bytes()).hexdigest() != FAILURE_SHA:
        raise RuntimeError("The retained shared failure changed")
    if args.forward_widths != [32] or args.forward_replays != 3:
        raise ValueError("Replay M32 iterations 0, 1 and the retained failure 2")
    torch.ops.load_library(str(Path(__file__).with_name("ffn_active_token_join.so")))
    native = torch.ops.tilert_ffn_active_token_join
    original_call = native.forward_out
    original_observer = decoder_ffn.observe_original
    taps = SharedTaps(torch, torch_npu)
    native_arguments = []
    reference_calls = 0

    def record_native(*arguments):
        native_arguments[:] = arguments
        return original_call(*arguments)

    @contextmanager
    def observe(torch_module, layer, group):
        nonlocal reference_calls
        reference_calls += 1
        with original_observer(torch_module, layer, group) as values:
            if reference_calls == 3:
                with taps.observe(layer.mlp.experts.ascend_shared_experts):
                    yield values
            else:
                yield values

    with replace_attribute(native, "forward_out", record_native), \
            replace_attribute(decoder_ffn, "observe_original", observe):
        try:
            result = decoder_ffn.run_decoder_forward(
                torch, layers, config, state, args)
        except RuntimeError as error:
            if str(error) == "The complete native FFN differs from the original FFN":
                save_diagnostic(torch, torch_npu, args, taps, native_arguments)
            raise
    raise RuntimeError(f"The retained numerical failure did not recur: {result}")
