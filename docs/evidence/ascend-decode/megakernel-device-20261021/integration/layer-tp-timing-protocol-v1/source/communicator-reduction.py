    It can use the `cpu_group` to initialize the communicator.
    If the device has PyTorch integration (PyTorch can recognize its
    communication backend), the `device_group` will also be given.
    """

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        use_all2all: bool = False,
    ):
        self.device = device or torch.device("cpu")
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.unique_name = unique_name

        # Check if this is a stateless process group
        from torch.distributed.distributed_c10d import _world

        is_stateless = _world.pg_map.get(cpu_group, None) is None

        if is_stateless:
            # For stateless groups, we can't use torch.distributed methods
            self.rank = cpu_group.rank()
            self.world_size = cpu_group.size()
            assert global_ranks is not None
            assert global_world_size is not None
            self.ranks = global_ranks
            self.global_rank = self.ranks[self.rank]
            self.global_world_size = global_world_size
            self.rank_in_group = self.rank
        else:
            self.rank = dist.get_rank(cpu_group)
            self.world_size = dist.get_world_size(cpu_group)
            self.ranks = dist.get_process_group_ranks(cpu_group)
            self.global_rank = dist.get_rank()
            self.global_world_size = dist.get_world_size()
            self.rank_in_group = dist.get_group_rank(self.cpu_group, self.global_rank)

        all2all_backend = None
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if config is not None:
            all2all_backend = config.parallel_config.all2all_backend

        self.is_ep_communicator = unique_name.split(":")[0] == "ep"
        self.use_all2all = self.is_ep_communicator and use_all2all
        self.all2all_backend = all2all_backend
        self.all2all_manager: All2AllManagerBase | None = None

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(input_, group=self.device_group)
        return input_
