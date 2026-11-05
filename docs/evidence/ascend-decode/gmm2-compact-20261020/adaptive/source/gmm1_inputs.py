#!/usr/bin/env python3
"""Synthetic argument set for the fused W4A8 expert GMM operator.

The tensor contract -- shapes, dtypes, NPU formats and scalar arguments --
is the one replayed by
``/workspace/src/TileRT-Ascend/tests/glm52/replay_cann_gmm1.py``. That
harness needs a GLM-5.2-W4A8 checkpoint only to obtain a ``VllmConfig``;
nothing downstream of ``process_weights_after_loading`` reads the model
files. Here the raw expert bank is random and the config is assembled from
library defaults, so the production postprocessing in
``vllm_ascend.quantization.methods.w4a8.w4a8`` still performs the
FRACTAL_NZ cast, the INT32 repack and the INT64 scale-carrier packing.

One TP16 rank of GLM-5.2-W4A8 is modelled: 256 experts, 6144 hidden
channels, 128 gate plus 128 up channels per expert.
"""

import argparse
from dataclasses import dataclass
import functools
import os
import time
from unittest.mock import patch

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch  # noqa: E402
import torch_npu  # noqa: E402

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm_ascend.ascend_config import init_ascend_config  # noqa: E402
from vllm_ascend.quantization.configs.modelslim_config import (  # noqa: E402
    AscendModelSlimConfig,
)
from vllm_ascend.quantization.methods.w4a8 import w4a8  # noqa: E402
from vllm_ascend.utils import enable_custom_op  # noqa: E402


TP_WORLD_SIZE = 16
EXPERTS = 256
HIDDEN = 6144
GATE_CHANNELS = 128
CHANNELS = 2 * GATE_CHANNELS
PACKED_BYTES = CHANNELS // 2
W2_ROWS = 3072
CODE_LIMIT = 127
FRACTAL_NZ = 29
INT4_LIMITS = (-8, 8)
ACTIVATION_LIMITS = (-CODE_LIMIT, CODE_LIMIT + 1)
WEIGHT_SCALE_LIMITS = (0.005, 0.02)
ACTIVATION_SCALE_LIMITS = (0.004, 0.02)
WEIGHT_BUDGET_BYTES = 2048 * 2 ** 20
ALLOCATOR_BYTES = 3584 * 2 ** 20

# The kernel splits the INT8 activation into two signed INT4 digits whose
# recomposition is x - 8; weight_assist_matrix carries back the 8.
MSD_BIAS = 8

DEQUANT_MODE = 0
GROUP_LIST_TYPE = 1
SWIGLU_LIMIT = 0.0


def bank_bytes(experts=EXPERTS):
    """Resident bytes of one processed bank: weight plus both scale tensors."""
    return experts * (HIDDEN * PACKED_BYTES + CHANNELS * 8 + CHANNELS * 4)


@dataclass(frozen=True)
class Bank:
    """One independent copy of the expert weights, sized to defeat L2."""

    weight: list
    weight_scale: list
    weight_assist_matrix: list

    @property
    def nbytes(self):
        groups = (self.weight, self.weight_scale, self.weight_assist_matrix)
        return sum(tensor.untyped_storage().nbytes()
                   for group in groups for tensor in group)


@dataclass(frozen=True)
class GmmArguments:
    """Everything one operator call needs, with the banks kept separate."""

    x: torch.Tensor
    x_scale: torch.Tensor
    group_list: torch.Tensor
    banks: list
    active_experts: list
    rows_per_expert: int
    reference: dict | None

    def kwargs(self, bank=0):
        chosen = self.banks[bank % len(self.banks)]
        return {"x": self.x,
                "weight": chosen.weight,
                "weight_scale": chosen.weight_scale,
                "x_scale": self.x_scale,
                "group_list": self.group_list,
                "weight_assist_matrix": chosen.weight_assist_matrix,
                "dequant_mode": DEQUANT_MODE,
                "group_list_type": GROUP_LIST_TYPE,
                "swiglu_limit": SWIGLU_LIMIT}


@functools.cache
def quant_method():
    """The production W4A8 MoE method, configured without a checkpoint.

    ``AscendW4A8DynamicFusedMoEMethod.__init__`` reads only the quantization
    name, the per-group flag, the expert-parallel flag and the v2-runner
    flag, so a default ``VllmConfig`` plus an empty ModelSlim description is
    enough. ``enable_fused_mc2=0`` makes ``use_cann_megamoe`` return False
    before it asks for an expert-parallel process group.
    """
    config = VllmConfig()
    config.quant_config = AscendModelSlimConfig({})
    config.additional_config = {"enable_fused_mc2": 0}
    init_ascend_config(config)
    with set_current_vllm_config(config):
        with patch.object(w4a8, "get_tensor_model_parallel_world_size",
                          return_value=TP_WORLD_SIZE), \
             patch.object(w4a8, "get_mc2_group", return_value=None):
            method = w4a8.AscendW4A8DynamicFusedMoEMethod()
    if method.use_expert_weight_list or method.tp_size != TP_WORLD_SIZE:
        raise RuntimeError("The W4A8 method did not take the single-tensor "
                           "per-expert layout this builder depends on")
    return config, method


def _packed_weight(device, experts):
    """Random packed INT4 weights plus the per-channel sums they imply.

    One INT8 byte holds channel ``2j`` in its low nibble and channel
    ``2j + 1`` in its high nibble, the layout ``_pack_int4_to_int8`` writes
    (w4a8.py:362-371). The channel sums feed the MSD correction, so the
    unpacked channels are generated first, one expert at a time, and never
    held for the whole bank.
    """
    weight = torch.empty((experts, PACKED_BYTES, HIDDEN), dtype=torch.int8,
                         device=device)
    channel_sum = torch.empty((experts, CHANNELS), dtype=torch.float32,
                              device=device)
    for expert in range(experts):
        channels = torch.randint(*INT4_LIMITS, (CHANNELS, HIDDEN),
                                 dtype=torch.int8, device=device)
        weight[expert] = (channels[1::2] * 16) | (channels[0::2] & 15)
        channel_sum[expert] = channels.float().sum(-1)
    return weight, channel_sum


def _raw_bank(device, experts):
    """The six raw parameters ``process_weights_after_loading`` consumes.

    GMM1 reads only the w13 group. The w2 group stays zero, as in the
    replay harness, because postprocessing still rewrites it in place.
    """
    weight, channel_sum = _packed_weight(device, experts)
    scale = torch.empty((experts, CHANNELS, 1), dtype=torch.float32,
                        device=device).uniform_(*WEIGHT_SCALE_LIMITS)
    # Cancels the -8 the MSD activation split leaves behind, the same
    # correction update_bias_compressed_tensors derives (w4a8.py:421-427).
    assist = MSD_BIAS * scale.squeeze(-1) * channel_sum
    raw = {"w13_weight": weight,
           "w2_weight": torch.zeros((experts, W2_ROWS, GATE_CHANNELS),
                                    dtype=torch.int8, device=device),
           "w13_weight_scale": scale,
           "w2_weight_scale": torch.zeros((experts, HIDDEN, 1),
                                          dtype=torch.float32, device=device),
           "w13_scale_bias": assist.unsqueeze(-1),
           "w2_scale_bias": torch.zeros((experts, HIDDEN, 1),
                                        dtype=torch.float32, device=device)}
    bank = torch.nn.Module()
    for name, tensor in raw.items():
        bank.register_parameter(
            name, torch.nn.Parameter(tensor, requires_grad=False))
    return bank


def _capture_reference(bank, active_experts):
    """Raw w13 rows of the active experts, kept on the host for replay."""
    index = torch.tensor(active_experts, dtype=torch.int64,
                         device=bank.w13_weight.device)
    return {"weight": bank.w13_weight.data[index].cpu(),
            "scale": bank.w13_weight_scale.data[index].cpu(),
            "assist": bank.w13_scale_bias.data[index].cpu()}


def _processed_bank(device, experts, active_experts):
    """Run the production postprocessing and keep only what GMM1 reads."""
    config, method = quant_method()
    bank = _raw_bank(device, experts)
    reference = None
    if active_experts is not None:
        reference = _capture_reference(bank, active_experts)
    with set_current_vllm_config(config):
        method.process_weights_after_loading(bank)
        weight, scale, _, _ = method._get_mlp_weights(bank)
        assist, _ = method._get_mlp_scale_bias(bank)
    built = Bank(weight=weight, weight_scale=scale,
                 weight_assist_matrix=assist)
    for name in ("w2_weight", "w2_weight_scale", "w2_scale_bias"):
        delattr(bank, name)
    _check_bank(built, experts)
    return built, reference


def _check_bank(bank, experts):
    """Reject a bank that the operator would read as a different tensor."""
    weight, scale = bank.weight[0], bank.weight_scale[0]
    assist = bank.weight_assist_matrix[0]
    if weight.shape != (experts, HIDDEN, PACKED_BYTES // 4):
        raise RuntimeError(f"weight shape {tuple(weight.shape)}")
    if weight.dtype != torch.int32 or not weight.is_contiguous():
        raise RuntimeError(f"weight dtype {weight.dtype}")
    if torch_npu.get_npu_format(weight) != FRACTAL_NZ:
        raise RuntimeError(f"weight format {torch_npu.get_npu_format(weight)}")
    if scale.shape != (experts, CHANNELS) or scale.dtype != torch.int64:
        raise RuntimeError(f"weight_scale {tuple(scale.shape)} {scale.dtype}")
    if assist.shape != (experts, CHANNELS) or assist.dtype != torch.float32:
        raise RuntimeError(f"assist {tuple(assist.shape)} {assist.dtype}")


def _group_list(device, experts, active, rows_per_expert):
    """Per-expert row counts for ``group_list_type=1``, spread evenly."""
    chosen = [index * experts // active for index in range(active)]
    counts = torch.zeros(experts, dtype=torch.int64)
    counts[chosen] = rows_per_expert
    return counts.to(device), chosen


def build(device, experts=EXPERTS, active=8, banks=1, *, rows_per_expert=1,
          seed=0, capture_reference=False):
    """Argument set for ``_C_ascend::grouped_matmul_swiglu_quant_v2``.

    ``active`` experts each receive ``rows_per_expert`` rows, so the token
    count is their product: ``active=8, rows_per_expert=1`` is batch-1
    decode with top-8 routing, ``active=32`` is batch-4 decode whose tokens
    picked disjoint experts, and ``active=8, rows_per_expert=4`` is the
    degenerate batch-4 case where all four tokens picked the same experts.
    """
    if not 1 <= active <= experts:
        raise ValueError(f"active must be within 1..{experts}, got {active}")
    if rows_per_expert < 1:
        raise ValueError(f"rows_per_expert must be positive, got "
                         f"{rows_per_expert}")
    requested = banks * bank_bytes(experts)
    if banks < 1 or requested > WEIGHT_BUDGET_BYTES:
        raise ValueError(f"{banks} banks need {requested} bytes, over the "
                         f"{WEIGHT_BUDGET_BYTES} byte weight budget")
    # _process_scale sends the INT64 carrier through .npu() with no index,
    # so the current device has to be the one the caller asked for.
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(seed)
    group_list, chosen = _group_list(device, experts, active, rows_per_expert)
    rows = active * rows_per_expert
    x = torch.randint(*ACTIVATION_LIMITS, (rows, HIDDEN), dtype=torch.int8,
                      device=device)
    x_scale = torch.empty(rows, dtype=torch.float32, device=device)
    x_scale.uniform_(*ACTIVATION_SCALE_LIMITS)
    built, reference = [], None
    for index in range(banks):
        capture = chosen if capture_reference and index == 0 else None
        bank, captured = _processed_bank(device, experts, capture)
        built.append(bank)
        reference = reference or captured
    return GmmArguments(x=x, x_scale=x_scale, group_list=group_list,
                        banks=built, active_experts=chosen,
                        rows_per_expert=rows_per_expert, reference=reference)


def _unpack_int4(packed):
    """Signed INT4 channels from bytes holding 2j low and 2j+1 high."""
    widened = packed.to(torch.int32)
    low = widened & 15
    channels = torch.empty((CHANNELS, HIDDEN), dtype=torch.int32)
    channels[0::2] = torch.where(low >= MSD_BIAS, low - 16, low)
    channels[1::2] = torch.div(widened - low, 16, rounding_mode="floor")
    return channels


def reference_codes(arguments):
    """Host replay of the kernel: MSD dequant, SwiGLU, per-row INT8 quant.

    Mirrors ``w4_msd_product`` in
    ``tilert/models/glm_5_2_ascend/reference_weights.py``, which reproduces
    the CANN 9.2.0 API-dequant order: FP16 rounding of each scaled partial
    product, the high digit weighted by 16, then the correction.
    """
    reference = arguments.reference
    x = arguments.x.cpu()
    x_scale = arguments.x_scale.cpu()
    codes, scales = [], []
    for index in range(len(arguments.active_experts)):
        rows = slice(index * arguments.rows_per_expert,
                     (index + 1) * arguments.rows_per_expert)
        weight = _unpack_int4(reference["weight"][index]).float()
        scale = reference["scale"][index].squeeze(-1)
        activation = x[rows].to(torch.int32)
        high = torch.div(activation, 16, rounding_mode="floor").float()
        low = (activation & 15).float() - MSD_BIAS
        product = ((high @ weight.T) * scale).half().float() * 16
        product = product + ((low @ weight.T) * scale).half().float()
        product = product + reference["assist"][index].squeeze(-1)
        product = product * x_scale[rows, None]
        gate = product[:, :GATE_CHANNELS]
        up = product[:, GATE_CHANNELS:]
        activated = torch.nn.functional.silu(gate) * up
        row_scale = torch.div(activated.abs().amax(-1), CODE_LIMIT)
        factor = torch.div(torch.ones_like(row_scale), row_scale)
        codes.append((activated * factor[:, None]).half().round()
                     .to(torch.int8))
        scales.append(row_scale)
    return torch.cat(codes), torch.cat(scales)


def _smoke_test(args):
    torch.npu.set_device(args.device)
    capacity = torch.npu.get_device_properties(args.device).total_memory
    torch.npu.set_per_process_memory_fraction(ALLOCATOR_BYTES / capacity)
    torch.npu.reset_peak_memory_stats()
    if not enable_custom_op():
        raise RuntimeError("The fused W4A8 GMM operator is not registered")
    operator = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2

    start = time.perf_counter()
    arguments = build(args.device, active=args.active, banks=args.banks,
                      rows_per_expert=args.rows_per_expert,
                      capture_reference=args.reference)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    print(f"schema {operator.default._schema}")
    print(f"device {args.device} experts {EXPERTS} active {args.active} "
          f"rows_per_expert {args.rows_per_expert} tokens "
          f"{arguments.x.shape[0]} banks {args.banks}")
    print(f"active_experts {arguments.active_experts}")
    print(f"build {elapsed:.3f}s total, {elapsed / args.banks:.3f}s per bank")
    print(f"bank_bytes {arguments.banks[0].nbytes} "
          f"({arguments.banks[0].nbytes / 2 ** 20:.2f} MiB) predicted "
          f"{bank_bytes()}")
    for name, tensor in (("x", arguments.x), ("x_scale", arguments.x_scale),
                         ("group_list", arguments.group_list),
                         ("weight", arguments.banks[0].weight[0]),
                         ("weight_scale", arguments.banks[0].weight_scale[0]),
                         ("assist",
                          arguments.banks[0].weight_assist_matrix[0])):
        print(f"input {name} {tuple(tensor.shape)} {tensor.dtype} "
              f"format {int(torch_npu.get_npu_format(tensor))}")
    print(f"group_list sum {int(arguments.group_list.sum())} nonzero "
          f"{int((arguments.group_list != 0).sum())}")

    for bank in range(args.banks):
        output, output_scale = operator(**arguments.kwargs(bank))
        torch.npu.synchronize()
        amax = output.to(torch.int16).abs().amax(-1)
        dequantized = output.float() * output_scale[:, None]
        finite = (bool(torch.isfinite(output_scale).all())
                  and bool(torch.isfinite(dequantized).all()))
        print(f"bank {bank} output {tuple(output.shape)} {output.dtype} "
              f"format {int(torch_npu.get_npu_format(output))} "
              f"output_scale {tuple(output_scale.shape)} {output_scale.dtype}")
        print(f"bank {bank} finite {finite} "
              f"positive_scale {bool((output_scale > 0).all())} "
              f"amax_invariant {bool((amax == CODE_LIMIT).all())} "
              f"amax_min {int(amax.min())} "
              f"scale_range [{float(output_scale.min()):.6g}, "
              f"{float(output_scale.max()):.6g}] "
              f"abs_mean {float(dequantized.abs().mean()):.6g}")
        if args.reference:
            codes, row_scale = reference_codes(arguments)
            difference = (output.cpu().to(torch.int16)
                          - codes.to(torch.int16)).abs()
            relative = ((output_scale.cpu() - row_scale).abs()
                        / row_scale).max()
            print(f"bank {bank} reference exact "
                  f"{int((difference == 0).sum())}/{difference.numel()} "
                  f"max_code_diff {int(difference.max())} "
                  f"max_scale_rel_error {float(relative):.3g}")
    print(f"peak_reserved_bytes {torch.npu.max_memory_reserved()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:7")
    parser.add_argument("--active", type=int, default=8)
    parser.add_argument("--rows-per-expert", type=int, default=1)
    parser.add_argument("--banks", type=int, default=1)
    parser.add_argument("--reference", action="store_true",
                        help="replay the kernel arithmetic on the host")
    _smoke_test(parser.parse_args())


if __name__ == "__main__":
    main()
