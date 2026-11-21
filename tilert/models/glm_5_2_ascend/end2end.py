import ctypes
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
import time
from typing import Any

from .config import (
    HIDDEN_SIZE,
    MAIN_LAYERS,
    PAGE_SIZE,
    RMS_NORM_EPS,
    RuntimeConfig,
    VOCAB_SIZE,
    WIDTHS,
)


def _round_state_tensor(
    *,
    width: int,
    prefix_len: int,
    draft_prefix_len: int,
    max_pos: int,
    first_token: int,
    init_ema: float,
    eos_token_id: int,
    device,
):
    import torch

    cpu_u8 = torch.zeros((256,), dtype=torch.uint8)
    i32 = cpu_u8[:64].view(torch.int32)
    i64 = cpu_u8[:128].view(torch.int64)
    i32[0] = int(width)
    i32[1] = 0
    i32[2] = 0
    i32[3] = 0
    i32[4] = int(prefix_len)
    i32[5] = int(draft_prefix_len)
    i32[6] = 1
    i32[7] = int(max_pos)
    i32[8] = int(width)
    i32[9] = int(width)
    i32[10] = 0
    i32[11] = 0
    i32[12] = int(eos_token_id)
    i32[13] = 0
    i64[7] = 0
    i64[8] = struct.unpack("<q", struct.pack("<d", float(init_ema)))[0]
    i64[9] = int(first_token)
    return cpu_u8.to(device)


def _serial_target_hidden(
    *,
    device,
    cap_stream,
    rank: int,
    world_size: int,
    width: int,
    arena,
    prefix,
    chain,
    t_input_ids,
    dyn_w,
    full_q_len,
    full_seq_len,
    mc2_mask,
    t_slots_i64,
    target_cos,
    target_sin,
    t_emb_partial,
    t_emb_out,
    l0_normed,
    l77_output,
    l77_bridge1,
    ops_cdll,
    tokens: list[int],
    pos_limit: int,
):
    import torch
    import torch_npu

    local_vocab = VOCAB_SIZE // world_size
    vocab_start = rank * local_vocab
    vocab_end = (rank + 1) * local_vocab
    lm_head_w_t = arena.lm_head_w.t()
    l0_gamma = arena.layers[0].input_gamma
    total_need = pos_limit - 8191
    h_chunks = []

    with torch.npu.stream(cap_stream):
        stream_ptr = ctypes.c_void_p(cap_stream.npu_stream)
        full_q_len.fill_(width)
        torch.ops.tilert.set_active_tokens_ptr(dyn_w, True, True)
        causal_masks = [(1 << (j + 1)) - 1 for j in range(width)]
        for start in range(0, total_need - 1, width):
            chunk_len = min(width, total_need - 1 - start)
            toks = tokens[start : start + chunk_len]
            if len(toks) < width:
                toks = toks + [toks[-1]] * (width - len(toks))
            pos_base = 8192 + start
            t_input_ids.copy_(torch.tensor(toks, dtype=torch.int32, device=device))
            slots = torch.arange(pos_base, pos_base + width, dtype=torch.int64, device=device)
            t_slots_i64.copy_(slots)
            target_cos.copy_(arena.cos_cache[pos_base : pos_base + width])
            target_sin.copy_(arena.sin_cache[pos_base : pos_base + width])
            masks_step = [
                causal_masks[j] if j < chunk_len else (1 << j) for j in range(width)
            ] + [0] * (32 - width)
            arena.tree_mask_i32.copy_(torch.tensor(masks_step, dtype=torch.int32, device=device))
            mc2_mask[:chunk_len].fill_(1)
            if chunk_len < width:
                mc2_mask[chunk_len:].zero_()
            dyn_w[0, 0] = chunk_len
            dyn_w[1, 0] = pos_base + chunk_len
            dyn_w[0, 1] = width
            dyn_w[1, 1] = pos_base + width
            full_seq_len.fill_(pos_base + width)

            ops_cdll.tilert_dspine_embed_lookup_launch(
                stream_ptr,
                t_input_ids.data_ptr(),
                dyn_w.data_ptr(),
                arena.embed_w.data_ptr(),
                t_emb_partial.data_ptr(),
                width,
                HIDDEN_SIZE,
                vocab_start,
                vocab_end,
            )
            torch.ops.tilert.onesided_all_reduce_out(t_emb_partial, t_emb_out)
            l0_normed.copy_(torch_npu.npu_rms_norm(t_emb_out, l0_gamma, RMS_NORM_EPS)[0])
            chain.launch()
            norm_out = torch_npu.npu_add_rms_norm(
                l77_output, l77_bridge1, arena.norm_w, RMS_NORM_EPS
            )[0]
            _ = torch.ops.aten.mm.dtype(norm_out, lm_head_w_t, torch.float32)
            h_chunks.append(norm_out[:chunk_len].clone())
        torch.npu.synchronize(device)
    return torch.cat(h_chunks, dim=0)


class Glm52AscendShowHands:

    def __init__(
        self,
        cfg: RuntimeConfig,
        case_dir: Path,
        mtp_prefix_path: Path,
        build_dir: Path,
        k_sfa: int = 1024,
        width: int = 16,
        main_prefix_path: Path | None = None,
        rank: int | None = None,
        tp_group: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.case_dir = Path(case_dir)
        self.mtp_prefix_path = Path(mtp_prefix_path)
        self.build_dir = Path(build_dir)
        self.k_sfa = int(k_sfa)
        self.width = int(width)
        if self.width not in WIDTHS:
            raise ValueError(f"Unsupported width: {self.width}")
        self.main_prefix_path = Path(main_prefix_path) if main_prefix_path is not None else None
        self.rank = int(cfg.rank if rank is None else rank)
        self.world_size = int(cfg.world_size)
        self.tp_group = tp_group
        self.device = None
        self.libraries = None
        self.metadata_lib = None
        self.tree_lib = None
        self.ffn_ops = None
        self.onesided = None
        self.prefix = None
        self.arena = None
        self.cap_stream = None
        self.target_win = None
        self.l78_win = None
        self.prepared_chain = None
        self.drafter = None
        self.g_round = None
        self._owners: list = []
        self._arena_pool: dict = {}
        self.compiled_so_dir = str(self.build_dir)
        self.head_warmed = False
        self.head_execution_before_timing: dict[str, Any] = {}

    @property
    def kernel_prof_enabled(self) -> bool:
        from .abi import kernel_prof_enabled

        if self.libraries is None:
            return False
        return kernel_prof_enabled(self.libraries.tree_manifest())

    def capture_prof_acc(self, *, sync: bool = True, cpu: bool = True):
        if self.prepared_chain is None:
            raise RuntimeError("Call load() before capture_prof_acc()")
        return self.prepared_chain.capture_prof_acc(sync=sync, cpu=cpu)

    def _resolve_libraries(self):
        from .abi import Libraries

        wl_path = Path(self.cfg.whole_layer_json)
        if not wl_path.is_file():
            wl_path = self.build_dir / "whole_layer.json"
        if not wl_path.is_file():
            raise FileNotFoundError(f"Missing whole_layer.json: {wl_path}")
        self.compiled_so_dir = str(wl_path.parent)
        libs = Libraries.from_json(wl_path)
        libs.verify()
        return libs

    def _build_chain_steps(self, l0_normed, t_emb_out):
        import torch

        dev = self.device
        pages = self.prefix.pages
        table_pages = int(self.prefix.block_table.shape[1])
        w = self.width

        self._arena_pool.clear()
        shared_scratch = {}
        prepared_layers = []
        prev_out = None
        prev_res = None

        for li in range(MAIN_LAYERS):
            lw = self.arena.layers[li]
            caches = [
                self.prefix.views[f"mla_latent.{li}"],
                self.prefix.views[f"mla_rope.{li}"],
            ]
            if lw.mode != "shared":
                caches.append(self.prefix.views[f"index.{li}"])
            if lw.mode == "int8":
                caches.append(self.prefix.views[f"index_scale.{li}"].view(pages, PAGE_SIZE))

            scratch_key = (w, pages, table_pages)
            pool_key = (w, lw.mode, pages, table_pages, li % 2)
            if pool_key not in self._arena_pool:
                norm_outputs = [
                    torch.zeros((w, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev),
                    torch.zeros((w, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev),
                    torch.zeros((w,), dtype=torch.float32, device=dev),
                    torch.zeros((w, HIDDEN_SIZE), dtype=torch.int8, device=dev),
                    torch.zeros((w, 1), dtype=torch.float32, device=dev),
                ]
                if lw.mode == "shared":
                    layer_scratch, tiling, blocks, sync_a = (
                        torch.ops.tilert.shared_full_attention_prepare(
                            norm_outputs[3], pages, table_pages, 24
                        )
                    )
                    offsets = None
                else:
                    layer_scratch, tiling, offsets, blocks, sync_a = (
                        torch.ops.tilert.indexed_attention_prepare(
                            norm_outputs[3],
                            pages,
                            table_pages,
                            pages,
                            table_pages,
                            lw.mode == "int8",
                            24,
                        )
                    )
                if scratch_key not in shared_scratch:
                    if lw.mode == "bf16":
                        shared_scratch[scratch_key] = layer_scratch
                    else:
                        max_scratch, _, _, _, _ = torch.ops.tilert.indexed_attention_prepare(
                            norm_outputs[3],
                            pages,
                            table_pages,
                            pages,
                            table_pages,
                            False,
                            24,
                        )
                        shared_scratch[scratch_key] = max_scratch
                scratch = shared_scratch[scratch_key]
                if (
                    scratch.numel() < layer_scratch.numel()
                    or scratch.dtype != layer_scratch.dtype
                ):
                    raise RuntimeError(
                        f"Shared scratch mismatch: {scratch.numel()}/{scratch.dtype}"
                    )
                del layer_scratch
                partial = torch.zeros((w, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)
                bridge, body, sync = self.ffn_ops.prepare(partial, 24, 1)
                if (
                    blocks != 24
                    or not sync
                    or sync_a != sync
                    or len(bridge) != 12
                    or len(body) != 21
                ):
                    raise RuntimeError("Chain workspace geometry or sync mismatch")
                attention_sum = torch.zeros((w, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)
                output = torch.zeros((w, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)
                self._arena_pool[pool_key] = (
                    norm_outputs,
                    scratch,
                    tiling,
                    offsets,
                    sync_a,
                    partial,
                    bridge,
                    body,
                    attention_sum,
                    output,
                )
            (
                norm_outputs,
                scratch,
                tiling,
                offsets,
                sync,
                partial,
                bridge,
                body,
                attention_sum,
                output,
            ) = self._arena_pool[pool_key]
            selected = lw.topk[:w].view(w, 1, 2048)

            hs_in = l0_normed if li == 0 else prev_out
            res_in = t_emb_out if li == 0 else prev_res
            norm = [hs_in, res_in, lw.input_gamma, *norm_outputs]
            main_meta = [
                self.target_sin,
                self.target_cos,
                self.t_slots_i64,
                self.dyn_w[0, :1],
                self.dyn_w[1, :1],
                self.prefix.block_table,
            ]
            idx_meta = (
                [
                    self.target_sin,
                    self.target_cos,
                    self.t_slots_i64,
                    self.full_q_len,
                    self.full_seq_len,
                    self.prefix.block_table,
                ]
                if lw.mode != "shared"
                else None
            )
            caches_pad = [*caches, *([None] * (4 - len(caches)))]
            addrs = [
                *norm[3:4],
                *norm[6:8],
                *lw.main,
                *(lw.front or [None] * 6),
                *lw.tail,
                None,
                *main_meta,
                *(idx_meta or [None] * 6),
                *caches_pad,
                scratch,
                tiling,
                partial,
            ]
            dims = (
                w,
                pages,
                table_pages,
                pages,
                table_pages,
                24,
                int(lw.mode == "int8"),
                int(bool(lw.front) and lw.front[1].dtype == torch.bfloat16),
                0,
            )
            groups = (norm, addrs, lw.post, lw.routed, bridge, body, lw.shared)
            from .abi import LocalRecords, PreparedLayer

            local = LocalRecords(
                self.metadata_lib,
                *groups,
                selected,
                dims,
                lw.scalars,
                sync,
                lw.mode != "shared",
                prepare_fn=getattr(self.tree_lib, "prepare_records", None),
            )
            if li == 0:
                local.indexed = 1 | 2 | 4
            elif li < 3:
                local.indexed = 1 | 2
            else:
                local.indexed = 1 if lw.mode != "shared" else 0
            prepared_layers.append(PreparedLayer(local, self.target_win, attention_sum, output))
            prev_out = output
            prev_res = bridge[1]

        self._l77_output = prev_out
        self._l77_bridge1 = prev_res
        return prepared_layers

    def load(self) -> None:
        import torch
        import torch_npu

        from .abi import (
            FFN_NAMESPACE,
            FrozenPrefixArena,
            MetadataLibrary,
            OneSidedReduce,
            PreparedChain,
            SHMEM_HEAP_BYTES,
            TreeLibrary,
            Window,
            _alloc_aligned,
        )
        from .dspine import CalibratedL78V3Runner, ZeroLagHorizonController, load_dspine_ops
        from .weights import build_model_arena

        if (
            os.environ.get("HCCL_OP_EXPANSION_MODE") != "AIV"
            or os.environ.get("HCCL_DETERMINISTIC", "").lower() != "true"
        ):
            raise RuntimeError("Require HCCL_OP_EXPANSION_MODE=AIV and HCCL_DETERMINISTIC=true")

        self.device = torch.device(self.cfg.device)
        torch.npu.set_device(self.device)
        torch.npu.config.allow_internal_format = True

        if self.tp_group is None and torch.distributed.is_initialized():
            self.tp_group = torch.distributed.new_group(
                backend="gloo", timeout=timedelta(hours=24)
            )

        self.libraries = self._resolve_libraries()
        for path in (self.libraries.indexed, self.libraries.shared, self.libraries.ffn):
            torch.ops.load_library(str(path))
        self.ffn_ops = getattr(torch.ops, FFN_NAMESPACE)
        self.metadata_lib = MetadataLibrary(self.libraries.metadata, None)
        self.tree_lib = TreeLibrary(self.libraries.tree, None)

        if self.tp_group is not None:
            torch.distributed.barrier(group=self.tp_group)
        onesided_ep = os.environ.get("TILERT_ONESIDED_ENDPOINT", "tcp://127.0.0.1:29797")
        whole_ep = os.environ.get("TILERT_WHOLE_LAYER_ENDPOINT", "tcp://127.0.0.1:29590")
        self.onesided = OneSidedReduce(self.rank, self.device, self.world_size, onesided_ep)
        self.tree_lib.initialize(
            self.device.index, self.rank, self.world_size, whole_ep, SHMEM_HEAP_BYTES
        )

        ops_dir = self.build_dir / "dspine_ops"
        _, self.ops_cdll = load_dspine_ops(self.rank, out_dir=ops_dir, world_size=self.world_size)

        self.prefix = FrozenPrefixArena.from_case(
            self.case_dir,
            self.mtp_prefix_path,
            self.device,
            capacity_tokens=self.cfg.capacity_tokens,
            main_prefix_path=self.main_prefix_path,
        )
        self.arena = build_model_arena(self.cfg, self.libraries, load_draft=True)
        if self.arena.draft is not None and self.arena.draft.cal_wt.ops is None:
            self.arena.draft.cal_wt.ops = self.ops_cdll

        dev = self.device
        w = self.width
        self.cap_stream = torch.npu.Stream(device=dev)
        with torch.npu.stream(self.cap_stream):
            self.target_win = Window(self.tree_lib, max(WIDTHS))
            self.l78_win = Window(self.tree_lib, 32)

        def alloc_buf(shape, dtype, align=64):
            nbytes = 1
            for d in shape:
                nbytes *= int(d)
            nbytes *= torch.empty((), dtype=dtype).element_size()
            raw, view = _alloc_aligned(max(64, nbytes), align, dev, zero=True)
            self._owners.append(raw)
            return view[:nbytes].view(dtype).view(shape)

        self.t_input_ids = alloc_buf((w,), torch.int32)
        self.t_positions = alloc_buf((w,), torch.int64)
        self.t_slots = alloc_buf((w,), torch.int32)
        self.t_slots_i64 = alloc_buf((w,), torch.int64)
        self.dyn_w = alloc_buf((2, 8), torch.int32)
        self.dyn_w[0, 0] = w
        self.dyn_w[0, 1] = w
        self.full_q_len = alloc_buf((1,), torch.int32)
        self.full_q_len.fill_(w)
        self.full_seq_len = alloc_buf((1,), torch.int32)
        self.full_seq_len.fill_(self.prefix.prefix_len + w)
        self.mc2_mask = alloc_buf((w,), torch.uint8)
        self.mc2_mask.fill_(1)
        self.target_cos = alloc_buf((w, 64), torch.bfloat16)
        self.target_sin = alloc_buf((w, 64), torch.bfloat16)
        self.t_emb_partial = torch.zeros((w, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)
        self.t_emb_out = torch.zeros((w, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)
        self.l0_normed = torch.zeros((w, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)
        self.target_hidden = torch.zeros((w, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)
        self.target_tokens = alloc_buf((w,), torch.int64)
        self.target_argmax_work = alloc_buf((48, 16), torch.float32)

        full_seq_ptrs_cpu = torch.zeros((16,), dtype=torch.int64)
        full_seq_ptrs_cpu[0] = int(self.full_seq_len.data_ptr())
        self.full_seq_ptrs_dev = alloc_buf((16,), torch.int64)
        self.full_seq_ptrs_dev.copy_(full_seq_ptrs_cpu.to(dev))

        rope_dsts_cpu = torch.zeros((1024,), dtype=torch.int64)
        rope_dsts_cpu[0] = int(self.target_cos.data_ptr())
        rope_dsts_cpu[1] = int(self.target_sin.data_ptr())
        self.rope_dsts_dev = alloc_buf((1024,), torch.int64)
        self.rope_dsts_dev.copy_(rope_dsts_cpu.to(dev))

        steps_layers = self._build_chain_steps(self.l0_normed, self.t_emb_out)
        chain_steps_buf = self.arena.bind_chain_step_buffers(MAIN_LAYERS)
        self.prepared_chain = PreparedChain(
            steps_layers,
            chain_steps_buf,
            copy_stream=None,
            mask_ptr=int(self.arena.tree_mask_i32.data_ptr()),
            peer_weights_ptr=int(self.arena.peer_weights_ptr),
        )

        self.state_dev = alloc_buf((256,), torch.uint8)
        self.receipt_ring_dev = alloc_buf((4, 16), torch.int32)
        self.receipt_toks_dev = alloc_buf((4, w), torch.int64)
        self.receipt_ring_cpu = torch.zeros((4, 16), dtype=torch.int32, pin_memory=True)
        self.receipt_toks_cpu = torch.zeros((4, w), dtype=torch.int64, pin_memory=True)

        self.target_ctx = {
            "width": w,
            "t_input_ids": self.t_input_ids,
            "t_positions": self.t_positions,
            "t_slots": self.t_slots,
            "t_slots_i64": self.t_slots_i64,
            "dyn_w": self.dyn_w,
            "full_seq_ptrs_dev": self.full_seq_ptrs_dev,
            "num_full_seq_ptrs": 1,
            "mc2_mask_ptr": int(self.mc2_mask.data_ptr()),
            "has_mc2_mask": 1,
            "rope_dsts_dev": self.rope_dsts_dev,
            "num_rope_dsts": 1,
            "tree_mask_i32": self.arena.tree_mask_i32,
            "target_hidden": self.target_hidden,
            "target_tokens": self.target_tokens,
        }
        self.hctrl = ZeroLagHorizonController()
        self.drafter = CalibratedL78V3Runner(
            self.arena,
            self.prefix,
            self.metadata_lib,
            self.ffn_ops,
            self.l78_win,
            self.target_ctx,
            self.tp_group,
            self.prefix.first_token_id,
            b_scheds=[
                [4, 4, 4, 4, 4, 2, 2],
                [4, 4, 4, 4, 2, 2, 2, 2, 1, 1],
                [4, 4, 4, 4, 2, 2, 2, 2, 1, 1],
            ],
            horizon_ctrl=self.hctrl,
            inv_temp=self.cfg.inv_temp,
            beta_margin=0.0,
            beta_depth=0.0,
            k_sfa=self.k_sfa,
            min_w_act=10,
            t_base_ms=41.5,
            c_w_ms=0.68,
            narrow_topk=True,
            l78_build_dir=self.build_dir / "dspine_l78",
        )
        self._capture_g_round()

    def _emit_target_stage(self):
        import torch
        import torch_npu

        w = self.width
        local_vocab = VOCAB_SIZE // self.world_size
        vocab_start = self.rank * local_vocab
        vocab_end = (self.rank + 1) * local_vocab
        stream_ptr = ctypes.c_void_p(torch.npu.current_stream(self.device).npu_stream)
        if self.world_size == 16:
            st = self.ops_cdll.tilert_dspine_target_prolog_launch(
                stream_ptr,
                self.drafter.sync_addr,
                self.dyn_w.data_ptr(),
                self.t_input_ids.data_ptr(),
                self.arena.embed_w.data_ptr(),
                self.arena.layers[0].input_gamma.data_ptr(),
                self.t_emb_out.data_ptr(),
                self.l0_normed.data_ptr(),
                self.drafter.comm_win,
                w,
                VOCAB_SIZE,
                vocab_start,
                vocab_end,
                self.rank,
                self.world_size,
                RMS_NORM_EPS,
            )
            if st != 0:
                raise RuntimeError(f"target_prolog failed: {st}")
            self.prepared_chain.launch()
            st = self.ops_cdll.tilert_dspine_target_add_rmsnorm_launch(
                stream_ptr,
                self.dyn_w.data_ptr(),
                self._l77_output.data_ptr(),
                self._l77_bridge1.data_ptr(),
                self.arena.norm_w.data_ptr(),
                self.target_hidden.data_ptr(),
                w,
                RMS_NORM_EPS,
            )
            if st != 0:
                raise RuntimeError(f"target_add_rmsnorm failed: {st}")
            logits = torch.ops.aten.mm.dtype(
                self.target_hidden, self.arena.lm_head_w.t(), torch.float32
            )
            st = self.ops_cdll.tilert_dspine_target_argmax_launch(
                stream_ptr,
                self.drafter.sync_addr,
                self.dyn_w.data_ptr(),
                logits.data_ptr(),
                self.target_argmax_work.data_ptr(),
                self.target_tokens.data_ptr(),
                self.drafter.comm_win,
                w,
                local_vocab,
                vocab_start,
                self.rank,
                self.world_size,
            )
            if st != 0:
                raise RuntimeError(f"target_argmax failed: {st}")
            return

        st = self.ops_cdll.tilert_dspine_embed_lookup_launch(
            stream_ptr,
            self.t_input_ids.data_ptr(),
            self.dyn_w.data_ptr(),
            self.arena.embed_w.data_ptr(),
            self.t_emb_partial.data_ptr(),
            w,
            HIDDEN_SIZE,
            vocab_start,
            vocab_end,
        )
        if st != 0:
            raise RuntimeError(f"embed_lookup failed: {st}")
        torch.ops.tilert.onesided_all_reduce_out(self.t_emb_partial, self.t_emb_out)
        l0_norm = torch_npu.npu_rms_norm(
            self.t_emb_out, self.arena.layers[0].input_gamma, RMS_NORM_EPS
        )[0]
        self.l0_normed.copy_(l0_norm)
        self.prepared_chain.launch()
        self.target_hidden.copy_(
            torch_npu.npu_add_rms_norm(
                self._l77_output, self._l77_bridge1, self.arena.norm_w, RMS_NORM_EPS
            )[0]
        )
        logits = torch.ops.aten.mm.dtype(
            self.target_hidden, self.arena.lm_head_w.t(), torch.float32
        )
        local_max, local_idx = logits.max(dim=-1)
        top_toks = torch.ops.tilert.onesided_top_tokens(local_max, local_idx, vocab_start)
        self.target_tokens.copy_(top_toks)

    def _reset_round_state(self, max_steps: int, eos_token_id: int = -1) -> int:
        import torch

        max_pos = self.prefix.prefix_len + int(max_steps)
        init_state = _round_state_tensor(
            width=self.width,
            prefix_len=self.prefix.prefix_len,
            draft_prefix_len=self.prefix.mtp_prefix_len,
            max_pos=max_pos,
            first_token=self.prefix.first_token_id,
            init_ema=self.hctrl.init_ema,
            eos_token_id=eos_token_id,
            device=self.device,
        )
        self.state_dev.copy_(init_state)
        self.receipt_ring_dev.zero_()
        self.receipt_toks_dev.zero_()
        self.receipt_ring_cpu.zero_()
        self.receipt_toks_cpu.zero_()
        self.arena.tree_mask_i32.zero_()
        self.dyn_w.zero_()
        self.dyn_w[0, 0] = self.width
        self.dyn_w[0, 1] = self.width
        self.dyn_w[1, 0] = self.prefix.prefix_len + self.width
        self.dyn_w[1, 1] = self.prefix.prefix_len + self.width
        self.full_seq_len.fill_(self.prefix.prefix_len + self.width)
        self.mc2_mask.fill_(1)
        torch.ops.tilert.set_active_tokens_ptr(self.dyn_w, True, True)
        torch.npu.synchronize(self.device)
        return max_pos

    def _emit_round_body(self, max_pos: int, eos_token_id: int = -1):
        self.drafter.emit_prolog_prepare_meta(self.state_dev, self.receipt_ring_dev)
        self.drafter.emit_draft_stage()
        self._emit_target_stage()
        self.drafter.emit_resolve_and_advance(
            self.state_dev,
            self.receipt_ring_dev,
            self.receipt_toks_dev,
            max_pos,
            eos_token_id,
        )
        self.drafter.emit_compact_and_commit_seed(include_compact=True)
        self.receipt_ring_cpu.copy_(self.receipt_ring_dev, non_blocking=True)
        self.receipt_toks_cpu.copy_(self.receipt_toks_dev, non_blocking=True)

    def _capture_g_round(self) -> None:
        import torch

        dev = self.device
        max_pos = self.prefix.prefix_len + 1023
        with torch.npu.device(dev), torch.inference_mode():
            self.prefix.restore()
            self._reset_round_state(1023, -1)
            self.drafter.seed_round0_eager()

            caller = torch.npu.current_stream(dev)
            with torch.npu.stream(self.cap_stream):
                torch.ops.tilert.set_active_tokens_ptr(self.dyn_w, True, True)
                self._emit_round_body(max_pos, -1)
            self.cap_stream.synchronize()

            self.prefix.restore()
            self._reset_round_state(1023, -1)
            self.drafter.seed_round0_eager()

            pool = torch.npu.graph_pool_handle()
            self.g_round = torch.npu.NPUGraph()
            torch.ops.tilert.set_active_tokens_ptr(self.dyn_w, True, True)
            with torch.npu.graph(self.g_round, pool=pool, stream=self.cap_stream):
                self._emit_round_body(max_pos, -1)
            caller.wait_stream(self.cap_stream)
            torch.npu.synchronize(dev)

            self.prefix.restore()
            self._reset_round_state(1023, -1)
            self.head_warmed = True
            self.head_execution_before_timing = {
                "wrapped": False,
                "active_graph_width": self.width,
                "graph_widths": [self.width],
            }

    def _run_show_hands_trial(
        self, max_steps: int, check_digests: bool = False
    ) -> dict[str, Any]:
        import torch

        dev = self.device
        w = self.width
        max_pos = self.prefix.prefix_len + int(max_steps)
        with torch.npu.device(dev), torch.inference_mode():
            self.prefix.restore()
            self._reset_round_state(max_steps, -1)
            self.drafter.seed_round0_eager()
            ev = [
                torch.npu.Event(enable_timing=False),
                torch.npu.Event(enable_timing=False),
            ]
            torch.npu.synchronize(dev)
            if self.tp_group is not None:
                torch.distributed.barrier(group=self.tp_group)
            t0 = time.perf_counter()

            self.g_round.replay()
            ev[0].record()
            self.g_round.replay()
            ev[1].record()
            in_flight = 2
            r_idx = 0
            out_toks = [int(self.prefix.first_token_id)]
            accepted_per_round = []
            round_traces = []
            seed_positions = [self.prefix.mtp_prefix_len]
            main_start = self.prefix.prefix_len

            while r_idx < max_steps + 2:
                ev[r_idx & 1].synchronize()
                in_flight -= 1
                rs = r_idx & 3
                rec = self.receipt_ring_cpu[rs].tolist()
                acc = int(rec[1])
                final_pos = int(rec[2])
                w_act = int(rec[3])
                h_lvl = int(rec[4])
                kv_qlen = int(rec[5])
                done = bool(int(rec[6]))
                toks = self.receipt_toks_cpu[rs, :acc].tolist()
                out_toks.extend(int(t) for t in toks)
                accepted_per_round.append(acc)
                seed_positions.append(main_start + acc - 1)
                main_start = final_pos
                round_traces.append(
                    {
                        "round_idx": r_idx,
                        "width": w,
                        "min_w_act": self.drafter.min_w_act,
                        "w_act": w_act,
                        "mc2_mask": [i < w_act for i in range(w)],
                        "dcci_issued": True,
                        "collective_rows": self.drafter.COMMIT_M,
                        "l78_gemm_chunk_rows": 8,
                        "kv_commit_q_len": kv_qlen,
                        "kv_commit_wrote": kv_qlen > 0,
                        "accepted": acc,
                        "final_pos": final_pos,
                        "horizon_level": h_lvl,
                        "done": done,
                    }
                )
                r_idx += 1
                if done:
                    break
                self.g_round.replay()
                ev[(r_idx + 1) & 1].record()
                in_flight += 1

            if in_flight > 0:
                ev[r_idx & 1].synchronize()
                drain_q = int(self.receipt_ring_cpu[r_idx & 3, 7].item())
                round_traces[-1]["drain_q_len"] = drain_q

            if self.tp_group is not None:
                torch.distributed.barrier(group=self.tp_group)
            dt_ms = (time.perf_counter() - t0) * 1000.0

        if self.tp_group is not None:
            dt_list = [None] * self.world_size
            torch.distributed.all_gather_object(dt_list, dt_ms, group=self.tp_group)
        else:
            dt_list = [dt_ms] * self.world_size
        out_sha = hashlib.sha256(json.dumps(out_toks).encode()).hexdigest()
        sum_acc = sum(accepted_per_round)
        res: dict[str, Any] = {
            "rank_dt_ms": [float(x) for x in dt_list],
            "local_dt_ms": float(dt_ms),
            "max_tpot_ms": max(float(x) for x in dt_list) / float(max_steps),
            "rounds": len(accepted_per_round),
            "sum_accepted": sum_acc,
            "mean_accepted": sum_acc / max(1, len(accepted_per_round)),
            "accepted_per_round": accepted_per_round,
            "output_token_ids": out_toks,
            "output_sha256": out_sha,
            "target_pos": int(round_traces[-1]["final_pos"]),
            "draft_committed_pos": int(round_traces[-1]["final_pos"]),
            "seed_positions": seed_positions,
            "round_traces": round_traces,
            "eos_drained_cleanly": round_traces[-1].get("drain_q_len") == 0,
        }
        if check_digests:
            final_p = int(round_traces[-1]["final_pos"])
            target_dig = self.prefix.main_state(final_p)
            draft_dig = self.prefix.draft_state(final_p)
            if self.tp_group is not None:
                all_checks = [None] * self.world_size
                torch.distributed.all_gather_object(
                    all_checks,
                    {
                        "out_sha": out_sha,
                        "target_digest": target_dig,
                        "draft_digest": draft_dig,
                    },
                    group=self.tp_group,
                )
            else:
                all_checks = [
                    {
                        "out_sha": out_sha,
                        "target_digest": target_dig,
                        "draft_digest": draft_dig,
                    }
                ]
            same_tok = all(c["out_sha"] == out_sha for c in all_checks)
            same_tgt = all(c["target_digest"] == all_checks[0]["target_digest"] for c in all_checks)
            same_dft = all(c["draft_digest"] == all_checks[0]["draft_digest"] for c in all_checks)
            res["main_state"] = target_dig
            res["draft_state"] = draft_dig
            res["all_ranks_target_exact"] = same_tok and same_tgt
            res["all_ranks_cache_parity"] = same_tgt and same_dft
        return res

    def warmup_and_run(
        self,
        warmups: int = 2,
        formals: int = 5,
        max_steps: int = 1023,
        check_serial: bool = False,
    ) -> dict[str, Any]:
        import torch

        if self.g_round is None:
            raise RuntimeError("Call load() before warmup_and_run()")
        warmups = int(warmups)
        formals = int(formals)
        if warmups < 0 or formals < 1:
            raise ValueError("Require warmups >= 0 and formals >= 1")

        total_runs = warmups + formals
        samples = []
        trials = []
        for idx in range(total_runs):
            is_warm = idx < warmups
            res = self._run_show_hands_trial(max_steps, check_digests=(idx == total_runs - 1))
            samples.append(res)
            trials.append(
                {
                    "trial_idx": idx,
                    "is_warmup": is_warm,
                    "arm": "candidate",
                    "rank_dt_ms": res["rank_dt_ms"],
                    "max_tpot_ms": res["max_tpot_ms"],
                    "rounds": res["rounds"],
                    "accepted_per_round": res["accepted_per_round"],
                    "mean_accepted": res["mean_accepted"],
                    "first_token_id": int(self.prefix.first_token_id),
                    "output_token_ids": res["output_token_ids"],
                    "output_sha256": res["output_sha256"],
                    "eos_drained_cleanly": bool(res["eos_drained_cleanly"]),
                }
            )

        last = samples[-1]
        kernel_prof_acc = (
            self.capture_prof_acc().tolist() if self.kernel_prof_enabled else None
        )
        warm_ms = [t["max_tpot_ms"] for t in trials[:warmups]]
        form_raw = [t["max_tpot_ms"] for t in trials[warmups:]]
        form_sorted = sorted(form_raw)
        pos = 0.9 * (len(form_sorted) - 1)
        lo_i, hi_i = math.floor(pos), math.ceil(pos)
        p90 = (
            form_sorted[lo_i]
            if lo_i == hi_i
            else form_sorted[lo_i] * (1.0 - (pos - lo_i)) + form_sorted[hi_i] * (pos - lo_i)
        )

        serial_draft_match = None
        serial_draft_diffs = None
        ekv_serial_diffs = None
        if check_serial:
            pos_limit = int(last["draft_committed_pos"])
            prev_tail = {
                k: v[8191:pos_limit].cpu().clone() for k, v in self.prefix.draft_cache.items()
            }
            self.prefix.restore()
            self.arena.tree_mask_i32.zero_()
            hidden_seq = _serial_target_hidden(
                device=self.device,
                cap_stream=self.cap_stream,
                rank=self.rank,
                world_size=self.world_size,
                width=self.width,
                arena=self.arena,
                prefix=self.prefix,
                chain=self.prepared_chain,
                t_input_ids=self.t_input_ids,
                dyn_w=self.dyn_w,
                full_q_len=self.full_q_len,
                full_seq_len=self.full_seq_len,
                mc2_mask=self.mc2_mask,
                t_slots_i64=self.t_slots_i64,
                target_cos=self.target_cos,
                target_sin=self.target_sin,
                t_emb_partial=self.t_emb_partial,
                t_emb_out=self.t_emb_out,
                l0_normed=self.l0_normed,
                l77_output=self._l77_output,
                l77_bridge1=self._l77_bridge1,
                ops_cdll=self.ops_cdll,
                tokens=last["output_token_ids"],
                pos_limit=pos_limit,
            )
            with torch.npu.stream(self.cap_stream):
                serial_dig, serial_draft_diffs, ekv_serial_diffs = (
                    self.drafter.run_serial_commit_pass(
                        last["output_token_ids"],
                        hidden_seq,
                        pos_limit,
                        last["seed_positions"],
                        prev_tail,
                    )
                )
            serial_draft_match = serial_dig == last["draft_state"]

        prompt_json = self.case_dir / "prompt.json"
        if prompt_json.is_file():
            pdata = json.loads(prompt_json.read_text(encoding="utf-8"))
            case_id = str(pdata.get("case_id", self.case_dir.name))
            prompt_sha256 = str(pdata["prompt_sha256"])
        else:
            case_id = self.case_dir.name
            packed_p = json.dumps(list(self.prefix.prompt_token_ids), separators=(",", ":"))
            prompt_sha256 = hashlib.sha256(packed_p.encode("utf-8")).hexdigest()

        traces = last["round_traces"]
        w_hist = sorted({int(t["w_act"]) for t in traces})
        rt_inv = {
            "w_act_min": int(self.drafter.min_w_act),
            "w_act_max": max(w_hist),
            "w_act_history": w_hist,
            "collective_split_max_rows": int(self.drafter.COMMIT_M),
            "l78_gemm_chunk_max_rows": 8,
            "in_flight_depth": 2,
            "draft_sfa_topk": int(self.k_sfa),
            "target_sfa_topk": 2048,
        }
        rank_dt = float(last["rank_dt_ms"][self.rank])
        loaded_has_vllm = any(
            m == "vllm"
            or m.startswith("vllm.")
            or m == "vllm_ascend"
            or m.startswith("vllm_ascend.")
            or m == "tilert.pd_vllm"
            or m.startswith("tilert.pd_vllm.")
            for m in sys.modules
        )
        draft_head_distinct = bool(
            self.arena is not None
            and self.arena.draft is not None
            and self.arena.draft.head_w_shard.data_ptr() != self.arena.lm_head_w.data_ptr()
            and not torch.equal(self.arena.draft.head_w_shard, self.arena.lm_head_w)
        )
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "case_id": case_id,
            "prompt_sha256": prompt_sha256,
            "first_token_id": int(self.prefix.first_token_id),
            "output_token_ids": last["output_token_ids"],
            "output_sha256": last["output_sha256"],
            "rounds": last["rounds"],
            "accepted_per_round": last["accepted_per_round"],
            "horizon_levels": [int(t["horizon_level"]) for t in traces],
            "dt_ms": rank_dt,
            "tpot_ms": rank_dt / float(max_steps),
            "main_state": last["main_state"],
            "draft_state": last["draft_state"],
            "serial_draft_match": serial_draft_match,
            "serial_draft_diffs": serial_draft_diffs,
            "ekv_serial_diffs": ekv_serial_diffs,
            "runtime_invariants": rt_inv,
            "round_traces": traces,
            "trials": trials,
            "warmups_ms": warm_ms,
            "formal_ms": form_raw,
            "median_tpot_ms": float(statistics.median(form_raw)),
            "min_tpot_ms": float(min(form_raw)),
            "p90_tpot_ms": float(p90),
            "sum_accepted": last["sum_accepted"],
            "mean_accepted": last["mean_accepted"],
            "target_pos": last["target_pos"],
            "draft_committed_pos": last["draft_committed_pos"],
            "all_ranks_target_exact": bool(last["all_ranks_target_exact"]),
            "all_ranks_cache_parity": bool(last["all_ranks_cache_parity"]),
            "loaded_modules_has_vllm": bool(loaded_has_vllm),
            "compiled_so_dir": self.compiled_so_dir,
            "head_warmed": bool(self.head_warmed),
            "head_execution_before_timing": dict(self.head_execution_before_timing),
            "eos_drained_cleanly": bool(last["eos_drained_cleanly"]),
            "draft_head_distinct_from_lm_head": draft_head_distinct,
            "kernel_prof_acc": kernel_prof_acc,
        }

    def close(self) -> None:
        import torch

        if self.device is not None:
            torch.npu.synchronize(self.device)
        if hasattr(getattr(torch.ops, "tilert", None), "set_active_tokens_ptr") and hasattr(
            self, "dyn_w"
        ):
            torch.ops.tilert.set_active_tokens_ptr(self.dyn_w, False, False)
        if self.g_round is not None:
            self.g_round.reset()
            self.g_round = None
        if self.drafter is not None:
            self.drafter.close()
            self.drafter = None
        if self.target_win is not None and getattr(self.target_win, "handle", 0) != 0:
            self.target_win.close()
            self.target_win = None
        if self.l78_win is not None and getattr(self.l78_win, "handle", 0) != 0:
            self.l78_win.close()
            self.l78_win = None
        if self.tree_lib is not None:
            try:
                self.tree_lib.finalize()
            except Exception:
                pass
            self.tree_lib = None
        if self.onesided is not None:
            self.onesided.close()
            self.onesided = None
        if self.arena is not None:
            self.arena.close()
            self.arena = None
        self.prepared_chain = None
        self._arena_pool.clear()
        self._owners.clear()
        gc.collect()
        if self.device is not None:
            torch.npu.empty_cache()


DecodeEngine = Glm52AscendShowHands
