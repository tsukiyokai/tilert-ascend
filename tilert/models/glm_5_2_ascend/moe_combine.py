"""Combine the shared and routed MoE outputs with a plain add.

vLLM-Ascend overrides upstream's inline combine for one reason: to route it
through `torch._foreach_add`, whose launch count does not grow with the tensor
list. It passes a list of one. At matched shapes in a single capture the
multi-tensor kernel costs 21.282 us against the plain add's 1.820, and its
cost is flat in row count -- 21.282 at one row, 21.442 at eight -- so all of
it is structure rather than work. Over 75 MoE layers that is 1.4 ms a round.

Both compute an elementwise add of two BF16 tensors, so the substitution is
bit-exact. The vendor's own comment retires the override once a superkernel
add exists; `enable_super_kernel` is on and the capture's plain `Add` runs on
AI_VECTOR_CORE.

Evidence: docs/evidence/ascend-decode/critical-path-20261020.
"""


def _install_moe_superkernel_fusion(restorers, model=None):
    import os
    import torch
    import torch.distributed as dist

    try:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.ops.fused_moe.dataclass.shared_experts import (
            PreparedSharedExpertInput,
            RoutedMoEMilestones,
        )
        from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
        from vllm_ascend.ops.fused_moe.moe_comm_method import (
            get_moe_comm_method,
        )
        from vllm_ascend.ops.fused_moe.prepare_finalize import (
            PrepareAndFinalizeWithAll2All,
            PrepareAndFinalizeWithMC2,
            _pad_tokens_with_cat,
        )
        from vllm_ascend.ops.fused_moe.shared_experts import (
            PreparedSharedExpertExecution,
        )
    except ImportError:
        return

    orig_prepare = PrepareAndFinalizeWithMC2.prepare
    orig_finalize = PrepareAndFinalizeWithAll2All.finalize
    orig_forward_impl = AscendMoERunner._forward_impl
    orig_scale = AscendMoERunner._maybe_apply_routed_scale_to_output
    orig_reduce_shared = AscendMoERunner._maybe_reduce_shared_expert_output
    orig_reduce_final = AscendMoERunner._maybe_reduce_final_output
    try:
        import vllm_ascend.ascend_forward_context as afc
        from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommType

        orig_select_moe = afc.select_moe_comm_method

        def _forced_select_moe(num_tokens, vllm_config, *args, **kwargs):
            force_ag = os.environ.get("TRT_MOE_ALLGATHER", "0") == "1"
            if force_ag and getattr(vllm_config.parallel_config, "enable_expert_parallel", False):
                return MoECommType.ALLGATHER
            return orig_select_moe(num_tokens, vllm_config, *args, **kwargs)

        afc.select_moe_comm_method = _forced_select_moe
        restorers.append(lambda: setattr(afc, "select_moe_comm_method", orig_select_moe))
    except Exception:
        pass
    from vllm.compilation import monitor
    from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
    pre_gather_scale = os.environ.get("TRT_PRE_GATHER_SCALE", "0") == "1"
    slice_router = os.environ.get("TRT_SLICE_ROUTER", "1") == "1"
    shared_after_routed = os.environ.get("TRT_SHARED_AFTER_ROUTED", "0") == "1"
    zero_comm_moe = os.environ.get("TRT_ZERO_COMM_MOE", "1") == "1"
    fuse_ar_norm = (
        os.environ.get("TRT_FUSE_AR_NORM", "1") == "1"
        and hasattr(
            getattr(torch.ops, "tilert", None),
            "onesided_all_reduce_rmsnorm_quant",
        )
    )
    pending_ar = {}
    pending_quant = {}
    final_norm_w_ptr = None
    if model is not None:
        fnorm = getattr(getattr(model, "model", None), "norm", None)
        if fnorm is not None and hasattr(fnorm, "weight"):
            final_norm_w_ptr = fnorm.weight.data_ptr()

    def _defer_ar(partial, out=None, routed=None, scale=1.0):
        if out is None:
            out = torch.empty_like(partial)
        pending_ar[out.data_ptr()] = (partial, out, routed, float(scale))
        return out

    def prepare(
        self,
        hidden_states,
        router_logits,
        replace_allreduce=False,
        quant_type=None,
    ):
        self.replace_allreduce = replace_allreduce
        if (
            zero_comm_moe
            and not self.replace_allreduce
            and self.tp_size > 1
            and hidden_states.shape[0] <= 32
        ):
            self.num_tokens, _ = hidden_states.shape
            from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import (
                MoEPrepareOutput,
            )

            return MoEPrepareOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                mc2_mask=None,
                padded_hidden_states_shape=hidden_states.shape,
                pertoken_scale=None,
            )
        if (
            not self.replace_allreduce
            and self.tp_size > 1
            and router_logits is not None
            and router_logits.shape[0] != hidden_states.shape[0]
        ):
            self.num_tokens, _ = hidden_states.shape
            target_pad_length = _EXTRA_CTX.padded_num_tokens
            pad_size = target_pad_length - self.num_tokens
            if pad_size > 0:
                hidden_states = _pad_tokens_with_cat(
                    hidden_states, target_pad_length
                )
            padded_hidden_states_shape = hidden_states.shape
            hidden_states = torch.tensor_split(
                hidden_states, self.tp_size, dim=0
            )[self.tp_rank]
            mc2_mask = _EXTRA_CTX.mc2_mask
            if self.tp_size > 1 and mc2_mask is not None:
                mc2_mask = torch.tensor_split(mc2_mask, self.tp_size, dim=0)[
                    self.tp_rank
                ]
            from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import (
                MoEPrepareOutput,
            )

            return MoEPrepareOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                mc2_mask=mc2_mask,
                padded_hidden_states_shape=padded_hidden_states_shape,
                pertoken_scale=None,
            )
        return orig_prepare(
            self,
            hidden_states,
            router_logits,
            replace_allreduce=replace_allreduce,
            quant_type=quant_type,
        )

    def finalize(
        self,
        hidden_states,
        reduce_results,
        padded_hidden_states_shape=None,
    ):
        if getattr(self, "_defer_finalize", False):
            self._pending_finalize = (
                hidden_states,
                reduce_results,
                padded_hidden_states_shape,
            )
            return hidden_states
        if (
            zero_comm_moe
            and not self.replace_allreduce
            and self.tp_size > 1
            and self.num_tokens <= 32
            and hidden_states.shape == padded_hidden_states_shape
        ):
            return hidden_states
        if not self.replace_allreduce:
            if self.tp_size > 1:
                assert padded_hidden_states_shape is not None
                gathered = torch.empty(
                    padded_hidden_states_shape,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                dist.all_gather_into_tensor(
                    gathered,
                    hidden_states,
                    group=self.moe_config.tp_group.device_group,
                )
                hidden_states = gathered
            if self.num_tokens < hidden_states.shape[0]:
                hidden_states = hidden_states[: self.num_tokens]
        return hidden_states

    def _forward_impl(
        self,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids=None,
    ):
        shared_experts = self.ascend_shared_experts
        if (
            shared_experts is None
            or self.moe_config.is_sequence_parallel
            or getattr(shared_experts, "multistream_overlap", False)
            or self._can_overlap_sp_shared_with(self.routed_input_transform)
            or self._can_overlap_sp_shared_with(self.routed_output_transform)
        ):
            return orig_forward_impl(
                self,
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
            )
        comm = get_moe_comm_method(_EXTRA_CTX.moe_comm_type, self.moe_config)
        pf = getattr(comm, "prepare_finalize", None)
        if not isinstance(pf, PrepareAndFinalizeWithAll2All):
            return orig_forward_impl(
                self,
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
            )
        with self._sequence_parallel_context():
            shared_hidden = (
                shared_experts_input
                if shared_experts_input is not None
                else hidden_states
            )
            prepared_shared = shared_experts.prepare_input_before_routed(
                shared_hidden
            )
            router_hidden = shared_hidden
            if (
                slice_router
                and not (zero_comm_moe and hidden_states.shape[0] <= 32)
                and isinstance(pf, PrepareAndFinalizeWithMC2)
                and pf.tp_size > 1
                and not getattr(pf, "replace_allreduce", False)
            ):
                num_tok, _ = hidden_states.shape
                target_pad = _EXTRA_CTX.padded_num_tokens
                padded_hs = (
                    _pad_tokens_with_cat(hidden_states, target_pad)
                    if target_pad > num_tok
                    else hidden_states
                )
                router_hidden = torch.tensor_split(
                    padded_hs, pf.tp_size, dim=0
                )[pf.tp_rank]
            (
                router_logits,
                shared_input_ready,
                router_output_ready,
                submitted_gate_up,
            ) = self._prepare_router_and_milestones(
                prepared_shared,
                router_hidden,
                router_logits,
            )
            init_milestones = RoutedMoEMilestones(
                shared_input_ready=shared_input_ready,
                router_output_ready=router_output_ready,
            )
            if submitted_gate_up is None and not shared_after_routed:
                mode = shared_experts.parallel_mode()
                exec_hidden, local_dp_meta = (
                    shared_experts._prepare_execution_input(
                        prepared_shared,
                        init_milestones,
                        mode,
                    )
                )
                gate_up_input = shared_experts._prepare_gate_up_input(
                    exec_hidden
                )
                prepared_execution = PreparedSharedExpertExecution(
                    hidden_states=exec_hidden,
                    gate_up_input=gate_up_input,
                    local_dp_metadata=local_dp_meta,
                )
                submitted_gate_up = shared_experts.enqueue_gate_up_after_router(
                    prepared_execution,
                    router_output_ready,
                )
            dispatcher = getattr(comm, "token_dispatcher", None)
            if (
                not shared_after_routed
                and dispatcher is not None
                and not getattr(dispatcher, "_tilert_wrapped_combine", False)
            ):
                orig_token_combine = dispatcher.token_combine

                def _wrapped_token_combine(*args, **kwargs):
                    hook = getattr(dispatcher, "_pre_combine_hook", None)
                    if hook is not None:
                        dispatcher._pre_combine_hook = None
                        hook()
                    return orig_token_combine(*args, **kwargs)

                dispatcher.token_combine = _wrapped_token_combine
                dispatcher._tilert_wrapped_combine = True
                restorers.append(
                    lambda d=dispatcher, o=orig_token_combine: (
                        setattr(d, "token_combine", o),
                        setattr(d, "_tilert_wrapped_combine", False),
                    )
                )
            shared_out_holder = [None]
            if not shared_after_routed and dispatcher is not None:
                dispatcher._pre_combine_hook = lambda: (
                    shared_out_holder.__setitem__(
                        0,
                        shared_experts.forward(
                            prepared_shared,
                            init_milestones,
                            defer_output_wait=False,
                            submitted_gate_up=submitted_gate_up,
                        ),
                    )
                )
            orig_overlap_gmm1 = getattr(
                comm, "shared_activation_overlaps_routed_gmm1", False
            )
            comm.shared_activation_overlaps_routed_gmm1 = False
            pf._defer_finalize = True
            pf._pending_finalize = None
            try:
                routed_out, milestones = self.routed_experts.forward_impl(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    input_ids=input_ids,
                )
            finally:
                comm.shared_activation_overlaps_routed_gmm1 = orig_overlap_gmm1
                pf._defer_finalize = False
                if dispatcher is not None:
                    dispatcher._pre_combine_hook = None
            shared_out = shared_out_holder[0]
            if shared_out is None:
                assert isinstance(milestones, RoutedMoEMilestones)
                milestones.shared_input_ready = shared_input_ready
                milestones.router_output_ready = router_output_ready
                if prepared_shared.is_gathered:
                    milestones.routed_finalize_done = (
                        torch.npu.current_stream().record_event()
                    )
                shared_out = shared_experts.forward(
                    prepared_shared,
                    milestones,
                    defer_output_wait=False,
                    submitted_gate_up=submitted_gate_up,
                )
            pending = pf._pending_finalize
            pf._pending_finalize = None
            if pending is not None:
                routed_slice, reduce_results, padded_shape = pending
                split_factor = max(
                    1,
                    getattr(pf, "tp_size", 1),
                    getattr(self.moe_config, "ep_size", 1),
                )
                if (
                    _can_fuse_onesided_scale_add(self, shared_out, routed_slice)
                    and not pf.replace_allreduce
                    and (
                        shared_out.numel() == routed_slice.numel()
                        or shared_out.numel() == routed_slice.numel() * split_factor
                    )
                ):
                    return shared_out, routed_slice
                if (
                    pre_gather_scale
                    and self.routed_scaling_factor != 1.0
                    and routed_slice.dtype != torch.float16
                ):
                    routed_slice = routed_slice * self.routed_scaling_factor
                routed_out = pf.finalize(
                    routed_slice, reduce_results, padded_shape
                )
            return shared_out, routed_out

    def _can_fuse_onesided_scale_add(self, shared_output, fused_output=None):
        return (
            shared_output is not None
            and isinstance(shared_output, torch.Tensor)
            and shared_output.dtype == torch.bfloat16
            and (
                fused_output is None
                or fused_output.dtype in (torch.bfloat16, torch.float32)
            )
            and self.moe_config.ep_size > 1
            and not self.moe_config.is_sequence_parallel
            and getattr(get_tp_group(), "_tilert_onesided", False)
            and hasattr(
                getattr(torch.ops, "tilert", None),
                "onesided_all_reduce_scale_add",
            )
        )

    def _maybe_apply_routed_scale_to_output(self, shared_output, fused_output):
        if _can_fuse_onesided_scale_add(self, shared_output, fused_output):
            combined = torch.ops.tilert.onesided_all_reduce_scale_add(
                shared_output,
                fused_output,
                float(self.routed_scaling_factor),
            )
            return None, combined
        if (
            pre_gather_scale
            and shared_output is not None
            and fused_output.dtype != torch.float16
            and not self.moe_config.is_sequence_parallel
        ):
            return shared_output, fused_output
        return orig_scale(self, shared_output, fused_output)

    def _maybe_reduce_shared_expert_output(
        self, shared_output, fused_output_is_reduced=None
    ):
        if _can_fuse_onesided_scale_add(self, shared_output):
            return shared_output
        if (
            shared_output is not None
            and self.moe_config.ep_size > 1
            and not self.moe_config.is_sequence_parallel
        ):
            return tensor_model_parallel_all_reduce(shared_output)
        return orig_reduce_shared(self, shared_output, fused_output_is_reduced)

    def _maybe_reduce_final_output(
        self, states, trunc_size, output_is_reduced=None
    ):
        if (
            self.moe_config.ep_size > 1
            and not self.moe_config.is_sequence_parallel
        ):
            if trunc_size is not None and trunc_size > 0:
                return states[..., :trunc_size]
            return states
        return orig_reduce_final(self, states, trunc_size, output_is_reduced)

    PrepareAndFinalizeWithMC2.prepare = prepare
    PrepareAndFinalizeWithAll2All.finalize = finalize
    AscendMoERunner._forward_impl = _forward_impl
    AscendMoERunner._maybe_apply_routed_scale_to_output = (
        _maybe_apply_routed_scale_to_output
    )
    AscendMoERunner._maybe_reduce_shared_expert_output = (
        _maybe_reduce_shared_expert_output
    )
    AscendMoERunner._maybe_reduce_final_output = _maybe_reduce_final_output
    restorers.append(
        lambda: setattr(PrepareAndFinalizeWithMC2, "prepare", orig_prepare)
    )
    restorers.append(
        lambda: setattr(PrepareAndFinalizeWithAll2All, "finalize", orig_finalize)
    )
    restorers.append(
        lambda: setattr(AscendMoERunner, "_forward_impl", orig_forward_impl)
    )
    restorers.append(
        lambda: setattr(
            AscendMoERunner,
            "_maybe_apply_routed_scale_to_output",
            orig_scale,
        )
    )
    restorers.append(
        lambda: setattr(
            AscendMoERunner,
            "_maybe_reduce_shared_expert_output",
            orig_reduce_shared,
        )
    )
    restorers.append(
        lambda: setattr(
            AscendMoERunner,
            "_maybe_reduce_final_output",
            orig_reduce_final,
        )
    )
    if zero_comm_moe:
        import torch._library.simple_registry as sreg
        import torch_npu
        from vllm.distributed.parallel_state import get_ep_group
        from vllm_ascend.ops.fused_moe.token_dispatcher import (
            TokenDispatcherWithMC2,
        )
        from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import (
            MoETokenDispatchOutput,
            MoEAllGatherCombineMetadata,
        )

        entry_shared = sreg.singleton.find("vllm::moe_forward_shared")
        if entry_shared is not None and len(entry_shared.fake_impl.kernels) > 0:
            orig_fake_shared = entry_shared.fake_impl.kernels[-1].func

            def _zero_comm_moe_forward_shared_fake(
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                shared_experts_input: torch.Tensor | None,
                input_ids: torch.Tensor | None,
                layer_name,
                hidden_dim_unpadded: int,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                shared_out, _ = orig_fake_shared(
                    hidden_states,
                    router_logits,
                    shared_experts_input,
                    input_ids,
                    layer_name,
                    hidden_dim_unpadded,
                )
                if hidden_dim_unpadded > 0:
                    fused_out = hidden_states.new_empty(
                        (*hidden_states.shape[:-1], hidden_dim_unpadded),
                        dtype=torch.float32,
                    )
                else:
                    fused_out = hidden_states.new_empty(
                        hidden_states.shape,
                        dtype=torch.float32,
                    )
                return shared_out, fused_out

            entry_shared.fake_impl.kernels[-1].func = _zero_comm_moe_forward_shared_fake
            restorers.append(
                lambda k=entry_shared.fake_impl.kernels[-1], o=orig_fake_shared: setattr(
                    k, "func", o
                )
            )

        entry_fwd = sreg.singleton.find("vllm::moe_forward")
        if entry_fwd is not None and len(entry_fwd.fake_impl.kernels) > 0:
            orig_fake_fwd = entry_fwd.fake_impl.kernels[-1].func

            def _zero_comm_moe_forward_fake(
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                shared_experts_input: torch.Tensor | None,
                input_ids: torch.Tensor | None,
                layer_name,
                hidden_dim_unpadded: int,
            ) -> torch.Tensor:
                if hidden_dim_unpadded > 0:
                    return hidden_states.new_empty(
                        (*hidden_states.shape[:-1], hidden_dim_unpadded),
                        dtype=torch.float32,
                    )
                return hidden_states.new_empty(
                    hidden_states.shape,
                    dtype=torch.float32,
                )

            entry_fwd.fake_impl.kernels[-1].func = _zero_comm_moe_forward_fake
            restorers.append(
                lambda k=entry_fwd.fake_impl.kernels[-1], o=orig_fake_fwd: setattr(
                    k, "func", o
                )
            )

        orig_td_dispatch = TokenDispatcherWithMC2.token_dispatch
        orig_td_combine = TokenDispatcherWithMC2.token_combine

        def _zero_comm_dispatch(self, token_dispatch_input):
            hs = token_dispatch_input.hidden_states
            if hs.shape[0] > 32:
                return orig_td_dispatch(self, token_dispatch_input)

            topk_ids = token_dispatch_input.topk_ids
            topk_weights = token_dispatch_input.topk_weights
            num_tokens = hs.shape[0]
            top_k = topk_ids.shape[1]
            num_local_experts = 32
            ep_rank = getattr(self, "ep_rank_id", None)
            if ep_rank is None or ep_rank < 0:
                ep_rank = (
                    dist.get_rank()
                    if dist.is_initialized()
                    else int(os.environ.get("RANK", 0))
                )
            first_expert_idx = ep_rank * num_local_experts
            last_expert_idx = first_expert_idx + num_local_experts
            global_num_experts = 256

            mask = (topk_ids >= first_expert_idx) & (topk_ids < last_expert_idx)
            local_topk_weights = topk_weights * mask

            sorted_hs, expanded_row_idx, expert_tokens, dynamic_scale = (
                torch_npu.npu_moe_init_routing_v2(
                    hs,
                    topk_ids,
                    scale=None,
                    active_num=num_tokens * top_k,
                    expert_num=global_num_experts,
                    expert_tokens_num_type=1,
                    expert_tokens_num_flag=True,
                    active_expert_range=[first_expert_idx, last_expert_idx],
                    quant_mode=1,
                )
            )
            return MoETokenDispatchOutput(
                hidden_states=sorted_hs,
                dynamic_scale=dynamic_scale,
                group_list=expert_tokens.to(torch.int64),
                group_list_type=1,
                combine_metadata=MoEAllGatherCombineMetadata(
                    topk_weights=local_topk_weights,
                    expanded_row_idx=expanded_row_idx,
                    restore_shape=hs.shape,
                ),
            )

        def _zero_comm_combine(self, hidden_states, combine_metadata, bias=None):
            hook = getattr(self, "_pre_combine_hook", None)
            if hook is not None:
                self._pre_combine_hook = None
                hook()
            if isinstance(combine_metadata, MoEAllGatherCombineMetadata):
                return torch_npu.npu_moe_token_unpermute(
                    permuted_tokens=hidden_states.to(torch.float32),
                    sorted_indices=combine_metadata.expanded_row_idx,
                    probs=combine_metadata.topk_weights.to(torch.float32),
                )
            return orig_td_combine(self, hidden_states, combine_metadata, bias=bias)

        TokenDispatcherWithMC2.token_dispatch = _zero_comm_dispatch
        TokenDispatcherWithMC2.token_combine = _zero_comm_combine
        restorers.append(
            lambda: setattr(
                TokenDispatcherWithMC2, "token_dispatch", orig_td_dispatch
            )
        )
        restorers.append(
            lambda: setattr(
                TokenDispatcherWithMC2, "token_combine", orig_td_combine
            )
        )
    if fuse_ar_norm:
        import torch_npu
        import vllm_ascend.vllm_ascend_C  # noqa: F401
        from .cann_graph import CannGraph

        reduce_obj = getattr(get_tp_group(), "_tilert_onesided_reduce", None)
        orig_reduce_sum = reduce_obj.sum if reduce_obj is not None else None
        orig_scale_add_pkt = torch.ops.tilert.onesided_all_reduce_scale_add
        orig_norm_bias_pkt = torch.ops._C_ascend.npu_add_rms_norm_bias
        orig_dq_pkt = torch.ops.npu.npu_dynamic_quant
        orig_dq_fn = torch_npu.npu_dynamic_quant
        orig_cann_capture = CannGraph._capture

        fuse_attn = os.environ.get("TRT_FUSE_ATTN_NORM", "1") == "1"
        fuse_moe = os.environ.get("TRT_FUSE_MOE_NORM", "1") == "1"

        def _wrapped_reduce_sum(value):
            if (
                fuse_attn
                and isinstance(value, torch.Tensor)
                and value.dtype == torch.bfloat16
                and value.numel() == 196608
                and value.is_contiguous()
            ):
                out = (
                    getattr(reduce_obj, "target_out", None)
                    if reduce_obj is not None and not isinstance(reduce_obj, bool)
                    else None
                )
                if (
                    reduce_obj is not None
                    and not isinstance(reduce_obj, bool)
                    and hasattr(reduce_obj, "target_out")
                ):
                    reduce_obj.target_out = None
                if (
                    out is not None
                    and out.dtype == value.dtype
                    and out.numel() == value.numel()
                    and out.is_contiguous()
                ):
                    return _defer_ar(value, out=out, routed=None, scale=1.0)
                return _defer_ar(value, out=None, routed=None, scale=1.0)
            return orig_reduce_sum(value)

        def _defer_scale_add(shared_output, fused_output, scale):
            if (
                fuse_moe
                and isinstance(shared_output, torch.Tensor)
                and shared_output.dtype == torch.bfloat16
                and shared_output.numel() == 196608
                and shared_output.is_contiguous()
            ):

                return _defer_ar(
                    shared_output,
                    out=None,
                    routed=(
                        fused_output.contiguous()
                        if fused_output is not None
                        else None
                    ),
                    scale=float(scale),
                )
            return orig_scale_add_pkt(shared_output, fused_output, scale)

        _defer_scale_add.default = _defer_scale_add

        fused_norm_produced = 0
        fused_dq_consumed = 0

        def _fused_norm_bias(x1, x2, weight, bias=None, eps=1e-5):
            nonlocal fused_norm_produced
            if pending_ar:
                entry = pending_ar.pop(x1.data_ptr(), None)
                if entry is not None:
                    partial, out_buf, routed, scale = entry
                    need_quant = (
                        final_norm_w_ptr is None
                        or weight.data_ptr() != final_norm_w_ptr
                    )
                    norm_out, res_out, q_out, s_out = (
                        torch.ops.tilert.onesided_all_reduce_rmsnorm_quant(
                            partial,
                            out_buf,
                            routed,
                            scale,
                            x2,
                            weight,
                            bias,
                            float(eps),
                            True,
                            need_quant,
                        )
                    )
                    if need_quant:
                        fused_norm_produced += 1
                        pending_quant[norm_out.data_ptr()] = (
                            norm_out,
                            q_out,
                            s_out,
                        )
                    dummy_rstd = torch.empty(
                        x1.shape[0], dtype=torch.float32, device=x1.device
                    )
                    return norm_out, dummy_rstd, res_out
            return orig_norm_bias_pkt(x1, x2, weight, bias, eps)

        _fused_norm_bias.default = _fused_norm_bias

        def _fused_dq_op(x, *args, **kwargs):
            nonlocal fused_dq_consumed
            if pending_quant:
                q_entry = pending_quant.pop(x.data_ptr(), None)
                if q_entry is not None:
                    fused_dq_consumed += 1
                    return q_entry[1], q_entry[2]
            return orig_dq_pkt(x, *args, **kwargs)

        _fused_dq_op.default = _fused_dq_op

        def _fused_dq_fn(x, *args, **kwargs):
            nonlocal fused_dq_consumed
            if pending_quant:
                q_entry = pending_quant.pop(x.data_ptr(), None)
                if q_entry is not None:
                    fused_dq_consumed += 1
                    return q_entry[1], q_entry[2]
            return orig_dq_fn(x, *args, **kwargs)

        def _wrapped_cann_capture(self_graph):
            nonlocal fused_norm_produced, fused_dq_consumed
            fused_norm_produced = 0
            fused_dq_consumed = 0
            pending_ar.clear()
            pending_quant.clear()
            if reduce_obj is not None:
                reduce_obj.sum = _wrapped_reduce_sum
            setattr(
                torch.ops.tilert,
                "onesided_all_reduce_scale_add",
                _defer_scale_add,
            )
            setattr(
                torch.ops._C_ascend,
                "npu_add_rms_norm_bias",
                _fused_norm_bias,
            )
            setattr(torch.ops.npu, "npu_dynamic_quant", _fused_dq_op)
            torch_npu.npu_dynamic_quant = _fused_dq_fn
            try:
                ret = orig_cann_capture(self_graph)
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                if rank == 0:
                    print(
                        f"[FUSED_PROBE] Capture done! produced_norms={fused_norm_produced}, "
                        f"consumed_dq={fused_dq_consumed}, remaining_unconsumed={len(pending_quant)}",
                        flush=True,
                    )
                return ret
            finally:
                pending_ar.clear()
                pending_quant.clear()
                if reduce_obj is not None:
                    reduce_obj.sum = orig_reduce_sum
                setattr(
                    torch.ops.tilert,
                    "onesided_all_reduce_scale_add",
                    orig_scale_add_pkt,
                )
                setattr(
                    torch.ops._C_ascend,
                    "npu_add_rms_norm_bias",
                    orig_norm_bias_pkt,
                )
                setattr(torch.ops.npu, "npu_dynamic_quant", orig_dq_pkt)
                torch_npu.npu_dynamic_quant = orig_dq_fn

        CannGraph._capture = _wrapped_cann_capture
        restorers.append(
            lambda: setattr(CannGraph, "_capture", orig_cann_capture)
        )

    _install_sfa_superkernel_fusion(restorers, model=model)


def _install_sfa_superkernel_fusion(restorers, model=None):
    import os
    import torch

    try:
        import torch_npu
        from vllm.distributed import get_tp_group
        from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
        from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
    except ImportError:
        return

    skip_topk_copy = os.environ.get("TRT_SKIP_TOPK_COPY", "1") == "1"
    inplace_o_proj = os.environ.get("TRT_INPLACE_O_PROJ", "1") == "1"
    wk_mm_v3 = os.environ.get("TRT_WK_MM_V3", "0") == "1"
    local_embed = os.environ.get("TRT_LOCAL_EMBED", "1") == "1"

    if local_embed and model is not None:
        import torch.distributed as dist
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        embed = getattr(getattr(model, "model", None), "embed_tokens", None)
        if (
            isinstance(embed, VocabParallelEmbedding)
            and not isinstance(embed, ParallelLMHead)
            and embed.tp_size > 1
            and embed.weight.dtype == torch.bfloat16
            and embed.shard_indices.num_added_elements == 0
        ):
            n_org = embed.shard_indices.num_org_elements
            shard = embed.weight.data[:n_org].contiguous()
            full_w = torch.empty(
                (n_org * embed.tp_size, embed.weight.shape[1]),
                dtype=embed.weight.dtype,
                device=embed.weight.device,
            )
            dist.all_gather_into_tensor(
                full_w, shard, group=get_tp_group().device_group
            )
            full_w.add_(0.0)
            embed._tilert_full_weight = full_w
            orig_embed_fwd = embed.forward
            embed.forward = lambda input_, _w=full_w: (
                torch.nn.functional.embedding(input_, _w)
            )
            restorers.append(
                lambda e=embed, o=orig_embed_fwd: (
                    setattr(e, "forward", o),
                    delattr(e, "_tilert_full_weight"),
                )
            )

    orig_get_topk = AscendSFAImpl._get_indexcache_topk_indices
    orig_update_topk = AscendSFAImpl._update_indexcache_topk_indices
    orig_finalize_o_proj = AscendSFAImpl._finalize_o_proj
    orig_indexer_forward = AscendSFAIndexerBackend.forward
    last_topk_by_buf = {}

    def _update_indexcache_topk_indices(self, topk_indices):
        if skip_topk_copy and self.topk_indices_buffer is not None:
            last_topk_by_buf[self.topk_indices_buffer.data_ptr()] = topk_indices
            self.topk_indices_buffer._tilert_last_topk = topk_indices
            return
        return orig_update_topk(self, topk_indices)

    def _get_indexcache_topk_indices(self, num_tokens):
        if skip_topk_copy and self.topk_indices_buffer is not None:
            cached = last_topk_by_buf.get(self.topk_indices_buffer.data_ptr())
            if cached is not None:
                topk_indices = cached[:num_tokens]
                if topk_indices.dim() == 2:
                    topk_indices = topk_indices.unsqueeze(1)
                return topk_indices
        return orig_get_topk(self, num_tokens)

    def _finalize_o_proj(self, attn_output, output, gather_full_o_proj):
        if inplace_o_proj:
            reduce = getattr(get_tp_group(), "_tilert_onesided_reduce", None)
            if (
                reduce is not None
                and not isinstance(reduce, bool)
                and hasattr(reduce, "target_out")
            ):
                reduce.target_out = output
                try:
                    res = self.o_proj(attn_output)[0]
                finally:
                    reduce.target_out = None
                if res.data_ptr() != output.data_ptr():
                    output[...] = res
                return output
        return orig_finalize_o_proj(
            self, attn_output, output, gather_full_o_proj
        )

    def _install_wk_mm(holder):
        wk = getattr(holder, "wk_weights_proj", None)
        if wk is None or not hasattr(wk, "weight") or getattr(wk, "_tilert_wk_mm_installed", False):
            return
        w_t = wk.weight.t()
        orig_wk_fwd = wk.forward
        wk.forward = lambda x, _wt=w_t: (torch.mm(x, _wt), None)
        wk._tilert_wk_mm_installed = True
        restorers.append(
            lambda w=wk, o=orig_wk_fwd: (
                setattr(w, "forward", o),
                setattr(w, "_tilert_wk_mm_installed", False),
            )
        )

    if wk_mm_v3 and model is not None:
        for mod in model.modules():
            if hasattr(mod, "wk_weights_proj"):
                _install_wk_mm(mod)

    def _indexer_forward(self, *args, **kwargs):
        if wk_mm_v3:
            _install_wk_mm(self)
        return orig_indexer_forward(self, *args, **kwargs)

    AscendSFAImpl._update_indexcache_topk_indices = (
        _update_indexcache_topk_indices
    )
    AscendSFAImpl._get_indexcache_topk_indices = _get_indexcache_topk_indices
    AscendSFAImpl._finalize_o_proj = _finalize_o_proj
    AscendSFAIndexerBackend.forward = _indexer_forward
    restorers.append(
        lambda: setattr(
            AscendSFAImpl,
            "_update_indexcache_topk_indices",
            orig_update_topk,
        )
    )
    restorers.append(
        lambda: setattr(
            AscendSFAImpl,
            "_get_indexcache_topk_indices",
            orig_get_topk,
        )
    )
    restorers.append(
        lambda: setattr(
            AscendSFAImpl,
            "_finalize_o_proj",
            orig_finalize_o_proj,
        )
    )
    restorers.append(
        lambda: setattr(
            AscendSFAIndexerBackend,
            "forward",
            orig_indexer_forward,
        )
    )


def install(model=None):
    """Send one-element combines to a plain add; return how to undo it."""
    import torch

    original = torch._foreach_add

    def combine(tensors, others, **rest):
        if len(tensors) == 1 and len(others) == 1 and not rest:
            return [tensors[0] + others[0]]
        return original(tensors, others, **rest)

    torch._foreach_add = combine
    restorers = [lambda: setattr(torch, "_foreach_add", original)]
    _install_moe_superkernel_fusion(restorers, model=model)

    def restore():
        for undo in reversed(restorers):
            undo()

    return restore

