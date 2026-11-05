"""Fuse TP16 MLA value-up, dynamic quantization and local output projection.

The vendor attention implementation needs the explicit tail hook in
patches/vllm-ascend/mla-tail-hook.patch. SFA, KV writes, index selection and
TP reduction keep their original call sites. Native workspaces are shared
across layers on one stream and reserved before graph capture.
"""

import os
from pathlib import Path
from types import MethodType


HEADS = 4
LATENT = 512
VALUE = 256
HIDDEN = 6144
MAX_TOKENS = 32
LIBRARY_DIR = Path(os.environ.get(
    "TILERT_MLA_TAIL_LIB_DIR",
    Path(__file__).resolve().parents[3] / "tests/glm52/mla_tail/build",
))


def pack_value_weights(weight):
    """ND BF16 [4,512,256] to 16 independent [512,64] CUBE tiles."""
    import torch

    if (weight.shape != (HEADS, LATENT, VALUE)
            or weight.dtype != torch.bfloat16):
        raise ValueError("MLA value weights require BF16 [4,512,256]")
    return weight.reshape(HEADS, LATENT, 4, 64).permute(
        0, 2, 1, 3,
    ).contiguous().reshape(16, LATENT, 64)


def pack_output_weights(weight):
    """ND INT8 [1024,6144] to 24 independent [1024,256] CUBE tiles."""
    import torch

    if (weight.shape != (HEADS * VALUE, HIDDEN)
            or weight.dtype != torch.int8):
        raise ValueError("MLA output weights require INT8 [1024,6144]")
    return weight.reshape(HEADS * VALUE, 24, 256).permute(
        1, 0, 2,
    ).contiguous()


class WorkspacePool:
    """Keep each live stream's native scratch and tiling tensors owned."""

    def __init__(self, blocks=0):
        self.blocks = blocks
        self.entries = {}
        self.capture_reserves = {}

    @staticmethod
    def _capacity(reference):
        rows = reference.shape[0]
        if not 1 <= rows <= MAX_TOKENS:
            raise ValueError("MLA tail requires 1..32 token rows")
        return 16 if rows <= 16 else 32

    def reserve_capture(self, reference):
        import torch

        capacity = self._capacity(reference)
        self.capture_reserves[(reference.device, capacity)] = (
            torch.ops.tilert.mla_tail_prepare(reference, capacity, self.blocks)
        )

    def get(self, reference):
        import torch

        capacity = self._capacity(reference)
        stream = torch.npu.current_stream(reference.device)
        key = (reference.device, stream.npu_stream, capacity)
        if key not in self.entries:
            if torch.npu.is_current_stream_capturing():
                reserve_key = (reference.device, capacity)
                if reserve_key not in self.capture_reserves:
                    raise RuntimeError(
                        "Reserve MLA tail workspace before NPU graph capture"
                    )
                self.entries[key] = self.capture_reserves.pop(reserve_key)
            else:
                self.entries[key] = torch.ops.tilert.mla_tail_prepare(
                    reference, capacity, self.blocks,
                )
        return capacity, self.entries[key]


def load():
    import torch

    torch.ops.load_library(str(LIBRARY_DIR / "mla_tail_torch.so"))


def install(impl, pool):
    """Bind an ungated TP16 W8A8_DYNAMIC output projection to the tail."""
    import torch
    import torch_npu
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
        AscendW8A8DynamicLinearMethod,
    )

    if not hasattr(impl, "_project_attention_output"):
        raise RuntimeError(
            "Apply patches/vllm-ascend/mla-tail-hook.patch to vLLM-Ascend "
            "before enabling the MLA tail"
        )
    projection = impl.o_proj
    scheme = projection.quant_method.quant_method
    if (not isinstance(scheme, AscendW8A8DynamicLinearMethod)
            or scheme.act_quant_type != torch.int8
            or projection.tp_size != 16
            or not projection.input_is_parallel
            or getattr(projection, "bias", None) is not None
            or getattr(impl, "g_proj", None) is not None):
        raise ValueError("MLA tail requires ungated TP16 W8A8_DYNAMIC")
    value_weight = pack_value_weights(
        torch_npu.npu_format_cast(impl.W_UV.detach(), 2),
    )
    output_weight = pack_output_weights(
        torch_npu.npu_format_cast(projection.weight.detach(), 2),
    )
    # A contiguous four-dimensional view can retain NCHW format metadata
    # after reshaping back to three dimensions. Normalize the packed tiles.
    value_weight = torch_npu.npu_format_cast(value_weight, 2)
    output_weight = torch_npu.npu_format_cast(output_weight, 2)
    output_scale = projection.weight_scale.detach().float().contiguous()
    if output_scale.shape != (HIDDEN,):
        raise ValueError("MLA output scales require 6144 channels")
    original = impl._project_attention_output

    def project(self, latent, gate_hidden_states, output,
                gather_full_o_proj, input_is_latent):
        if (not input_is_latent or gather_full_o_proj
                or gate_hidden_states is not None
                or not 1 <= latent.shape[0] <= MAX_TOKENS):
            return original(latent, gate_hidden_states, output,
                            gather_full_o_proj, input_is_latent)
        latent = latent.reshape(-1, HEADS, LATENT).contiguous()
        capacity, prepared = pool.get(latent)
        scratch, tiling, blocks, sync_base = prepared
        # The collective must consume a separate rank-local output.
        partial = torch.empty_like(output)
        torch.ops.tilert.mla_tail_out(
            latent, value_weight, output_weight, output_scale, None,
            scratch, tiling, partial, capacity, blocks, sync_base,
        )
        if projection.reduce_results:
            partial = tensor_model_parallel_all_reduce(partial)
        output.copy_(partial)
        return output

    impl._project_attention_output = MethodType(project, impl)

    def restore():
        impl._project_attention_output = original

    return restore
