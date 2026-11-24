"""Separate constructor TP16, singleton MC2 and two-rank oracle communicators."""

from contextlib import contextmanager
from datetime import timedelta
import hashlib


def initialize_constructor_groups(torch, rank, args):
    from vendor_hook_runtime212_device import GROUP_FIELDS, identity
    from vllm.distributed import parallel_state as state

    local_rank = rank if rank < 2 else 0
    state.init_distributed_environment(
        world_size=16, rank=rank, local_rank=local_rank, backend="gloo",
        distributed_init_method=f"file://{args.output / 'rendezvous'}",
        timeout=timedelta(seconds=args.timeout))
    if any(getattr(state, field) is not None for field in GROUP_FIELDS):
        raise RuntimeError("The isolated process already owns constructor groups")
    tp = state.init_model_parallel_group(
        [list(range(16))], local_rank, "gloo", use_device_communicator=False,
        group_name="tp_shape_16")
    singleton = state.init_model_parallel_group(
        [[index] for index in range(16)], local_rank, "gloo",
        use_device_communicator=False, group_name="tp_shape_singleton")
    for field in GROUP_FIELDS:
        setattr(state, field, tp if field in ("_TP", "_ETP") else singleton)
    witness = torch.tensor(rank, dtype=torch.int64)
    torch.distributed.all_reduce(witness, group=tp.cpu_group)
    if witness.item() != 120 or tp.rank_in_group != rank:
        raise RuntimeError("The real TP16 Gloo constructor group differs")
    return state, (tp, singleton), {
        "tp_world_size": 16, "tp_rank": rank, "cpu_rank_sum": witness.item(),
        "coordinator": identity(type(tp)), "device_communicator": None,
        "scope": "Constructor geometry only; no TP16 device collective"}


def create_communication_groups(torch, state, rank):
    from vllm_ascend.utils import create_hccl_pg_options

    torch.distributed.barrier(group=state.get_world_group().cpu_group)
    singleton = None
    for target in (0, 1):
        if rank == target:
            singleton = state.init_model_parallel_group(
                [[target]], rank, "hccl", group_name="mc2",
                use_device_communicator=False)
        else:
            device = torch.distributed.new_group(
                [target], backend="hccl", pg_options=create_hccl_pg_options("mc2"))
            cpu = torch.distributed.new_group([target], backend="gloo")
            if any(group != torch.distributed.GroupMember.NON_GROUP_MEMBER
                   for group in (device, cpu)):
                raise RuntimeError("An unrelated rank joined a singleton MC2 group")
    device = torch.distributed.new_group(
        [0, 1], backend="hccl", pg_options=create_hccl_pg_options("tp"))
    cpu = torch.distributed.new_group([0, 1], backend="gloo")
    if rank >= 2:
        if (singleton is not None or torch.npu.is_initialized()
                or device != torch.distributed.GroupMember.NON_GROUP_MEMBER
                or cpu != torch.distributed.GroupMember.NON_GROUP_MEMBER):
            raise RuntimeError("A CPU participant joined a device group")
    return singleton, device, cpu


@contextmanager
def device_groups_context(torch, state, stage, rank, args):
    from vllm_ascend.distributed import parallel_state as ascend_state
    from vllm_ascend.patch.worker.patch_distributed import GroupCoordinatorPatch
    from vendor_hook_runtime212_device import identity

    if ascend_state._MC2 is not None or state.GroupCoordinator is not GroupCoordinatorPatch:
        raise RuntimeError("Use the original worker coordinator and a fresh MC2 context")
    mc2, device, cpu = create_communication_groups(torch, state, rank)
    if (mc2.ranks != [rank] or mc2.rank_in_group != 0 or mc2.world_size != 1
            or torch.distributed.get_world_size(device) != 2
            or torch.distributed.get_rank(device) != rank):
        raise RuntimeError("The MC2 and two-rank oracle groups differ")
    name = mc2.device_group._get_backend(torch.device("npu")).get_hccl_comm_name(0)
    if not name:
        raise RuntimeError("The actual singleton HCCL communicator is empty")
    args.oracle_hccl, args.oracle_cpu = device, cpu
    ascend_state._MC2 = mc2
    stage("actual_mc2_and_hccl2_groups", lambda: (None, {
        "coordinator": identity(type(mc2)), "mc2_ranks": [rank],
        "tp_constructor_size": state.get_tp_group().world_size,
        "oracle_hccl_ranks": [0, 1], "oracle_rank": rank,
        "comm_name_sha256": hashlib.sha256(str(name).encode()).hexdigest(),
        "tp16_device_collective": False}))
    try:
        yield mc2
    finally:
        ascend_state._MC2 = None
        torch.npu.synchronize()
        torch.distributed.barrier(group=cpu)
        mc2.destroy()
        torch.distributed.destroy_process_group(device)
        torch.distributed.destroy_process_group(cpu)
