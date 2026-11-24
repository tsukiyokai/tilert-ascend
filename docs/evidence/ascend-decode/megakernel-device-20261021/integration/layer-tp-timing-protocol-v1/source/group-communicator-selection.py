    def _init_device_communicator(self) -> None:
        self.device = torch.npu.current_device()
        if self.use_device_communicator and self.world_size > 1:
            self.device_communicator = NPUCommunicator(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
            )
