"""Same-owner native stages and the two explicit FP32 HCCL reduction cuts."""

import hashlib
from pathlib import Path


NORM_SHA = "40a9e3c04d22355cd51228c1dff7d3dc516d3442253ee3310a6cf95c01182027"


def prepare(torch, cases):
    from local_chain import BLOCKS

    library = Path(__file__).with_name("native_norm.so")
    if hashlib.sha256(library.read_bytes()).hexdigest() != NORM_SHA:
        raise RuntimeError("The frozen norm library changed")
    torch.ops.load_library(str(library))
    views = []
    for case in cases:
        local = case.local
        blocks, sync = torch.ops.tilert.norm_prepare(local.norm[0], BLOCKS)
        if blocks != BLOCKS or sync != local.sync:
            raise RuntimeError("The native norm and layer launch geometry differ")
        rstd = local.norm[5].view(local.rows, 1)
        if (rstd.data_ptr() != local.norm[5].data_ptr()
                or rstd.untyped_storage().data_ptr()
                != local.norm[5].untyped_storage().data_ptr()):
            raise RuntimeError("The norm view replaced its original rstd storage")
        views.append(rstd)
    return views


class Reductions:
    """Retain four FP32 collective owners and their original BF16 destinations."""

    def __init__(self, torch, cases, group, cpu_group):
        self.torch, self.cases, self.group = torch, cases, group
        self.cpu_group = cpu_group
        self.buffers = {
            (number, phase): torch.empty_like(case.output, dtype=torch.float32)
            for number, case in zip((6, 7), cases) for phase in ("ar1", "ar2")}
        self.pointers = {key: value.data_ptr() for key, value in self.buffers.items()}
        if (len(set(self.pointers.values())) != 4
                or torch.distributed.get_world_size(group) != 2):
            raise RuntimeError("Use four distinct FP32 buffers and real HCCL2")

    def check_owners(self):
        if self.pointers != {key: value.data_ptr() for key, value in self.buffers.items()}:
            raise RuntimeError("A collective temporary changed storage")

    def sum(self, number, phase, partial, result):
        self.check_owners()
        temporary = self.buffers[number, phase]
        temporary.copy_(partial)
        self.torch.distributed.all_reduce(
            temporary, op=self.torch.distributed.ReduceOp.SUM, group=self.group)
        result.copy_(temporary)

    def literal(self, number, phase, partial, result, record):
        from local_chain import snapshot, tensor_witness
        from torch.utils._python_dispatch import TorchDispatchMode

        torch = self.torch

        class Observe(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.operators = []

            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                self.operators.append(str(function))
                return function(*args, **(kwargs or {}))

        # Only dtype operations are recorded; CPU observations remain outside
        # capture and events. The underlying CANN kernels are not inferred.
        with Observe() as literal_up:
            temporary = partial.float()
        converted = snapshot(torch, temporary)
        torch.distributed.all_reduce(
            temporary, op=torch.distributed.ReduceOp.SUM, group=self.group)
        reduced = snapshot(torch, temporary)
        with Observe() as literal_down:
            narrowed = temporary.bfloat16()
        result.copy_(narrowed)
        actual = snapshot(torch, result)
        partials = [None, None]
        torch.distributed.all_gather_object(
            partials, snapshot(torch, partial), group=self.cpu_group)
        ordered = partials[0].float() + partials[1].float()

        preallocated = self.buffers[number, phase]
        with Observe() as prepared_up:
            preallocated.copy_(partial)
        prepared_converted = snapshot(torch, preallocated)
        torch.distributed.all_reduce(
            preallocated, op=torch.distributed.ReduceOp.SUM, group=self.group)
        prepared_reduced = snapshot(torch, preallocated)
        with Observe() as prepared_down:
            result.copy_(preallocated)
        prepared_result = snapshot(torch, result)
        from attention_check import metric

        checks = {name: metric(torch, left, right) for name, left, right in (
            ("ordered_fp32_sum", reduced, ordered),
            ("converted", prepared_converted, converted),
            ("reduced", prepared_reduced, reduced),
            ("narrowed", prepared_result, actual))}
        witnesses = [tensor_witness(torch, value) for value in partials]
        passed = (all(value["pass"] and value["bitwise_equal"] for value in checks.values())
                  and witnesses[0]["sha256"] != witnesses[1]["sha256"])
        statuses = [None, None]
        torch.distributed.all_gather_object(statuses, passed, group=self.cpu_group)
        if not all(statuses):
            raise RuntimeError("Preallocated conversion differs from the literal FP32 path")
        record.append({
            "layer": number, "phase": phase, "checks": checks,
            "operators": {"literal_up": literal_up.operators,
                          "literal_down": literal_down.operators,
                          "prepared_up": prepared_up.operators,
                          "prepared_down": prepared_down.operators},
            "operator_scope": "Actual Torch dispatcher entries; no CANN kernel identity claim",
            "partials": witnesses,
            "reduced_fp32": tensor_witness(torch, reduced),
            "output": tensor_witness(torch, actual)})


def separate(case, rstd, reduce):
    from local_chain import BLOCKS

    local = case.local
    local.torch.ops.tilert.norm_quant_out(
        *local.norm[:3], None, local.norm[3], local.norm[4], rstd,
        *local.norm[6:8], local.input_norm.variance_epsilon, BLOCKS, local.sync)
    if local.mode == "shared":
        local.torch.ops.tilert.shared_full_attention_out(
            *local.norm[6:8], *local.main, *local.main_metadata[:3],
            *local.caches, local.selected, *local.main_metadata[3:],
            *local.tail, None, local.scratch, local.tiling, local.partial,
            *local.scalars, BLOCKS, local.sync)
    else:
        local.torch.ops.tilert.indexed_attention_out(
            local.norm[3], *local.norm[6:8], local.main, local.front, local.tail,
            None, local.main_metadata, local.index_metadata, local.caches,
            local.scratch, local.tiling, local.partial, *local.scalars,
            BLOCKS, local.sync)
    reduce(local.partial, case.attention_sum, "ar1")
    local.ffn.native.forward_out(
        case.attention_sum, local.norm[4], local.ffn.gamma, local.ffn.gate,
        local.ffn.bias, local.ffn.weights, local.ffn.shared,
        local.bridge, local.body, local.ffn.epsilon, BLOCKS, 1, local.sync)
    reduce(local.bridge[10], case.output, "ar2")
