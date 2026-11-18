from contextlib import ExitStack
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class QuantizedWeight:

    values: torch.Tensor
    scale: torch.Tensor
    bits: int
    scale_bias: torch.Tensor | None = None

    @property
    def out_features(self) -> int:
        return self.values.shape[0] * (8 // self.bits)

    @property
    def in_features(self) -> int:
        return self.values.shape[1]

    def unpack(self) -> torch.Tensor:
        if self.bits == 8:
            return self.values
        low = ((self.values & 15) ^ 8) - 8
        high = self.values >> 4
        return torch.stack((low, high), dim=1).reshape(
            self.out_features, self.in_features
        )

    def dequantize(self) -> torch.Tensor:
        return self.unpack().float() * self.scale[:, None]

    def reference(
        self, activation: torch.Tensor, activation_scale: torch.Tensor
    ) -> torch.Tensor:
        product = activation.to(torch.int32) @ self.unpack().to(torch.int32).T
        return product.float() * self.scale * activation_scale.reshape(-1, 1)


@dataclass(frozen=True)
class ExpertWeights:
    gate: QuantizedWeight
    up: QuantizedWeight
    down: QuantizedWeight


class ModelSlimCheckpoint:

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        index = self.directory / "quant_model_weights.safetensors.index.json"
        self.weight_map = json.loads(index.read_text())["weight_map"]
        description = self.directory / "quant_model_description.json"
        self.quantization = json.loads(description.read_text())

    def tensor(self, name: str, selection=None) -> torch.Tensor:
        path = self.directory / self.weight_map[name]
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            if selection is None:
                value = checkpoint.get_tensor(name)
            else:
                value = checkpoint.get_slice(name)[selection]
            return value.clone().contiguous()

    def tensors(self, names):
        with ExitStack() as stack:
            shards = {}
            for name in names:
                filename = self.weight_map[name]
                if filename not in shards:
                    shards[filename] = stack.enter_context(safe_open(
                        self.directory / filename, framework="pt",
                        device="cpu",
                    ))
                value = shards[filename].get_tensor(name)
                yield name, value.clone().contiguous()

    def shape(self, name: str) -> tuple[int, ...]:
        path = self.directory / self.weight_map[name]
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            return tuple(checkpoint.get_slice(name).get_shape())

    def linear(
        self,
        name: str,
        output_range: tuple[int, int] | None = None,
        input_range: tuple[int, int] | None = None,
    ) -> QuantizedWeight:
        kind = self.quantization[name + ".weight"]
        if kind not in ("W4A8_DYNAMIC", "W8A8_DYNAMIC"):
            raise ValueError(f"{name}: unsupported quantization {kind}")
        bits = 4 if kind == "W4A8_DYNAMIC" else 8
        packed_rows, columns = self.shape(name + ".weight")
        packing = 8 // bits
        rows = packed_rows * packing
        out_start, out_end = output_range or (0, rows)
        in_start, in_end = input_range or (0, columns)
        if not (0 <= out_start < out_end <= rows):
            raise ValueError(f"{name}: output range exceeds {rows} channels")
        if not (0 <= in_start < in_end <= columns):
            raise ValueError(f"{name}: input range exceeds {columns} channels")
        if out_start % packing or out_end % packing:
            raise ValueError(f"{name}: W4 output boundaries must be even")

        output = slice(out_start, out_end)
        packed_output = slice(out_start // packing, out_end // packing)
        values = self.tensor(
            name + ".weight", (packed_output, slice(in_start, in_end))
        )
        scale = self.tensor(name + ".weight_scale", output).flatten().float()
        offset = self.tensor(name + ".weight_offset", output)
        if torch.count_nonzero(offset).item():
            raise ValueError(f"{name}: nonzero weight offsets are unsupported")

        scale_bias = None
        if bits == 4:
            partitions = self.shape(name + ".scale_bias")[1]
            partition_size = columns // partitions
            if in_start % partition_size or in_end % partition_size:
                raise ValueError(
                    f"{name}: input boundaries must align to "
                    f"{partition_size}-channel scale_bias partitions"
                )
            selection = (
                output,
                slice(in_start // partition_size, in_end // partition_size),
            )
            scale_bias = self.tensor(name + ".scale_bias", selection).sum(dim=1)
        return QuantizedWeight(values, scale, bits, scale_bias)

    def expert(self, layer: int, expert: int, rank: int) -> ExpertWeights:
        if not 0 <= rank < 16:
            raise ValueError("TP16 rank must be between 0 and 15")
        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
        channels = (rank * 128, (rank + 1) * 128)
        return ExpertWeights(
            gate=self.linear(prefix + ".gate_proj", output_range=channels),
            up=self.linear(prefix + ".up_proj", output_range=channels),
            down=self.linear(prefix + ".down_proj", input_range=channels),
        )
