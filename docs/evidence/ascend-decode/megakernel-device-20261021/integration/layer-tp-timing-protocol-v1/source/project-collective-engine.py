def _choose_collective_engine(aiv):
    """Expand collectives on the vector cores rather than the device AI CPU.

    A3 defaults to AI_CPU. At decode sizes the vector path is 1.6 to 1.9 times
    faster -- sixteen ranks over twelve rows of 6144 bf16: all_reduce 300.56 us
    to 183.40, reduce_scatter 318.09 to 177.41, all_gather 326.74 to 175.40 --
    and the gain is the per-call floor rather than transport, since the only
    case that barely moves is the one actually carrying a large payload.

    The variable is read once when HCCL loads, so this has to run before any
    communicator exists and cannot be changed afterwards; setting it from
    inside the process works, measured at 181.50 us against 290.44 unset.

    Determinism comes with it. The vector path reduces in its own order and is
    not reproducible on its own -- two runs of the same sixteen-rank all_reduce
    gave different digests -- while the AI CPU path is. `HCCL_DETERMINISTIC`
    restores reproducibility for 6 to 8 percent: 189 us against 175
    non-deterministic and 284 to 350 on the AI CPU. Bit-exactness is what every
    comparison in this project is checked against, so it is not optional.
    """
    if not aiv:
        return
    os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
    os.environ.setdefault("HCCL_DETERMINISTIC", "true")


class CannModel:
    """One request, TP16, full-P injection and fixed-width target forwards.

    ``rank`` is the logical TP rank; ``device`` selects the process's NPU.
    The caller supplies all 16 visible devices without a visibility rotation.
    Worker construction and loading execute device code. No scheduler,
    speculative model, profiling pass, or graph capture is created here.
    """

    def __init__(
        self, checkpoint, device, rank, init_method="env://", capacity=9344,
        gpu_memory_utilization=0.85, *, compile_target=False,
        super_kernel=False, attention_kernel="sfa", fp32_head=False,
        expert_parallel=False, onesided_reduce=None, graph_width=4,
        index_topk=None, plain_moe_combine=False, routed_kernel=False,
        mla_tail=False, routed_shared_kernel=False, native_attention=False,
        narrow_bank=False, aiv_collectives=True, rope_graph=False,
        gather_fia_max_width=None,
        mtp_tokens=0, layers=None,
        hoist_metadata=False, hoist_forward_context=False,
    ):
        if type(rank) is not int or not 0 <= rank < 16:
            raise ValueError("GLM-5.2 requires a logical TP rank in [0, 16)")
        routed_kernel = routed_kernel or routed_shared_kernel
        if routed_kernel and expert_parallel:
            raise ValueError("The routed kernel requires TP16 with EP off")
        if routed_kernel and narrow_bank:
            raise ValueError("The routed kernel owns its expert grouping")
        if native_attention and (mla_tail or attention_kernel != "sfa"):
            raise ValueError("Native attention includes the MLA tail and requires SFA")
        self._mla_tail_enabled = mla_tail
        self._native_attention_enabled = native_attention
        self._routed_workspace_pool = None
        self._routed_shared_workspace_pool = None
        self._mla_tail_workspace_pool = None
        self._native_attention_workspace_pool = None
        self._restore_routed = None
        self._restore_routed_shared = []
        self._restore_mla_tail = []
        self._restore_native_attention = []
        _choose_collective_engine(aiv_collectives)
        self.rope_graph = rope_graph
