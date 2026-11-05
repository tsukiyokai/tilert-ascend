    def _install_tp_sum(self):
        from vllm.distributed import get_tp_group

        self._tp_sum_group = get_tp_group()
        self._tp_sum_original = self._tp_sum_group._all_reduce_out_place
        self._tp_sum_had_override = (
            "_all_reduce_out_place" in vars(self._tp_sum_group)
        )
        self._tp_sum_group._all_reduce_out_place = self._tp_reduce
        self.tp_reduction = "float32"
        if self.onesided_endpoint is not None:
            from .onesided_reduce import install

            self._onesided = install(self, self.onesided_endpoint)

    def _tp_reduce(self, value):
        import torch

        if value.dtype == torch.bfloat16:
            return self._tp_sum_original(value.float()).bfloat16()
        return self._tp_sum_original(value)
