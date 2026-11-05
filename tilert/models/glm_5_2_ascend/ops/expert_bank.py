"""Optional TP16 W4A8 expert bank using two native CANN grouped matmuls."""

from dataclasses import dataclass
import json
from pathlib import Path
import struct


EXPERTS = 256
HIDDEN = 6144
LOCAL_INTERMEDIATE = 128
TOP_K = 8


def checkpoint_contract(directory, layer, rank):
    """Check tensor headers only; importing this module does not load Torch."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    expected = {
        "hidden_size": HIDDEN, "moe_intermediate_size": 2048,
        "n_routed_experts": EXPERTS, "num_experts_per_tok": TOP_K,
        "num_hidden_layers": 78, "num_nextn_predict_layers": 1,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"Expert bank requires {name}={value}")
    if type(rank) is not int or not 0 <= rank < 16:
        raise ValueError("Expert bank requires a TP16 rank")
    if type(layer) is not int or not 3 <= layer <= 78:
        raise ValueError("Expected a routed layer from 3 through MTP 78")
    description = directory / "quant_model_description.json"
    quant = json.loads(description.read_text())
    index = directory / "quant_model_weights.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    headers = {}
    checked = 0
    for expert in range(EXPERTS):
        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
        for projection, rows, columns, partitions in (
            ("gate_proj", 2048, HIDDEN, 1),
            ("up_proj", 2048, HIDDEN, 1),
            ("down_proj", HIDDEN, 2048, 16),
        ):
            name = prefix + "." + projection
            if quant.get(name + ".weight") != "W4A8_DYNAMIC":
                raise ValueError(f"Expected W4A8_DYNAMIC: {name}")
            for suffix, dtype, shape in (
                ("weight", "I8", [rows // 2, columns]),
                ("weight_scale", "F32", [rows, 1]),
                ("weight_offset", "F32", [rows, 1]),
                ("scale_bias", "F32", [rows, partitions]),
            ):
                key = name + "." + suffix
                filename = weight_map[key]
                if filename not in headers:
                    with (directory / filename).open("rb") as source:
                        size = struct.unpack("<Q", source.read(8))[0]
                        headers[filename] = json.loads(source.read(size))
                entry = headers[filename][key]
                if entry["dtype"] != dtype or entry["shape"] != shape:
                    raise ValueError(f"Unexpected tensor metadata: {key}")
                checked += 1
    return {
        "layer": layer, "rank": rank, "tp_size": 16,
        "tensor_headers_checked": checked, "files": sorted(headers),
        "gate_up": {"weight_int32": [256, 6144, 32],
                    "scale_int64": [256, 1, 256], "bias_fp32": [256, 256]},
        "down": {"weight_int32": [256, 128, 768],
                 "scale_int64": [256, 1, 6144], "bias_fp32": [256, 6144]},
        "resident_bytes": 321650688, "group_list_type": 1,
        "group_list": "256 nonnegative INT64 counts, including empty groups",
        "activation": "A8 per original input row; down A8 per local K128 row",
        "output": "routed contribution before shared expert and TP reduction",
        "offset_values_checked": False,
    }


@dataclass
class DeviceRoutes:
    order: object
    token_ids: object
    inverse: object
    counts: object


def device_routes(choices):
    """Pack valid device top-8 IDs; data validity is the router's contract."""
    import torch

    if (choices.ndim != 2 or choices.shape[0] not in (1, 2, 4)
            or choices.shape[1] != TOP_K
            or choices.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Routes must be integer [M,8], M in 1/2/4")
    flat = choices.reshape(-1).long()
    order = flat.argsort(stable=True)
    counts = torch.zeros(EXPERTS, dtype=torch.int64, device=choices.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    inverse = torch.empty_like(order, dtype=torch.int32)
    inverse.scatter_(0, order, torch.arange(
        flat.numel(), device=choices.device, dtype=torch.int32
    ))
    return DeviceRoutes(order, order // TOP_K, inverse, counts)


@dataclass
class ProjectionBank:
    weight: object
    scale: object
    scale_bias: object

    @classmethod
    def allocate(cls, inputs, outputs, device):
        import torch

        return cls(
            torch.empty(EXPERTS, inputs, outputs // 8,
                        dtype=torch.int32, device=device),
            torch.empty(EXPERTS, 1, outputs, dtype=torch.int64, device=device),
            torch.empty(EXPERTS, outputs, dtype=torch.float32, device=device),
        )

    def copy_expert(self, expert, projection):
        self.weight[expert].copy_(projection.weight[0])
        self.scale[expert].copy_(projection.scale[0])
        self.scale_bias[expert].copy_(projection.scale_bias[0])

    def apply_quantized(self, activation, scale, counts):
        import torch
        import torch_npu

        return torch_npu.npu_grouped_matmul(
            x=[activation.clone()], weight=[self.weight],
            scale=[self.scale], bias=[self.scale_bias],
            per_token_scale=[scale], group_list=counts,
            split_item=2, group_type=0, group_list_type=1,
            output_dtype=torch.bfloat16,
        )[0]


@dataclass
class ExpertBankTrace:
    routes: DeviceRoutes
    input_quantized: object
    input_scale: object
    packed_quantized: object
    packed_scale: object
    gate_up: object
    intermediate: object
    down_quantized: object
    down_scale: object
    expert_output: object
    output: object


@dataclass
class NpuExpertBank:
    """One routed layer/rank; the caller owns shared FFN and TP reduction.

    Construction copies each temporary expert into preallocated storage,
    then releases it. Runtime routes stay on the device. Each row's eight
    IDs must be distinct and in [0,256), as provided by the native router.
    """

    gate_up: ProjectionBank
    down: ProjectionBank
    layer: int
    rank: int

    @classmethod
    def from_checkpoint(cls, checkpoint, layer, rank, device="npu:0"):
        import torch

        from .expert import NpuExpert

        checkpoint_contract(checkpoint.directory, layer, rank)
        gate_up = ProjectionBank.allocate(HIDDEN, 2 * LOCAL_INTERMEDIATE,
                                          device)
        down = ProjectionBank.allocate(LOCAL_INTERMEDIATE, HIDDEN, device)
        with torch.inference_mode():
            for expert in range(EXPERTS):
                source = checkpoint.expert(layer, expert, rank)
                temporary = NpuExpert.from_checkpoint(source, device)
                gate_up.copy_expert(expert, temporary.gate_up)
                down.copy_expert(expert, temporary.down)
                del temporary, source
        return cls(gate_up, down, layer, rank)

    def trace(self, x, choices, probabilities):
        import torch
        import torch_npu

        if (x.dtype != torch.bfloat16 or x.ndim != 2
                or x.shape[1] != HIDDEN or x.shape[0] not in (1, 2, 4)):
            raise ValueError("Expert bank input must be BF16 [M,6144]")
        if (choices.shape != (x.shape[0], TOP_K)
                or probabilities.shape != choices.shape):
            raise ValueError("Expert routes and probabilities must be [M,8]")
        device = self.gate_up.weight.device
        inputs = (x, choices, probabilities)
        if any(value.device != device for value in inputs):
            raise ValueError("Input, routes and bank must share one device")
        with torch.inference_mode():
            routes = device_routes(choices)
            activation, scale = torch_npu.npu_dynamic_quant(
                x, dst_type=torch.int8
            )
            packed = activation.index_select(0, routes.token_ids)
            packed_scale = scale.index_select(0, routes.token_ids)
            gate_up = self.gate_up.apply_quantized(
                packed, packed_scale, routes.counts
            )
            intermediate = torch_npu.npu_swiglu(gate_up, dim=-1)
            down_input, down_scale = torch_npu.npu_dynamic_quant(
                intermediate, dst_type=torch.int8
            )
            expert_output = self.down.apply_quantized(
                down_input, down_scale, routes.counts
            )
            output = torch_npu.npu_moe_token_unpermute(
                expert_output, routes.inverse, probs=probabilities.to(x.dtype)
            )
        return ExpertBankTrace(
            routes, activation, scale, packed, packed_scale, gate_up,
            intermediate, down_input, down_scale, expert_output, output,
        )

    def __call__(self, x, choices, probabilities):
        return self.trace(x, choices, probabilities).output
