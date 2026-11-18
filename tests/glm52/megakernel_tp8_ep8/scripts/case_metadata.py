from contextlib import contextmanager
from types import MethodType


@contextmanager
def exact_rows(target):
    original = target._metadata

    def exact_meta(self, width_arg, *, graphed=False, depths=None, host_values=None):
        ea = getattr(self, "_exact_active", None)
        res = original(width_arg, graphed=graphed, depths=depths, host_values=host_values)
        if ea is not None and graphed and self._hoisted.get(width_arg) is not None:
            self._active_rows = ea
            if self._dynamic_mc2_mask and self.graph is not None and self.graph.capture_count:
                if self._last_mc2_active.get(width_arg, width_arg) != ea:
                    from vllm_ascend.ascend_forward_context import (
                        _hoisted_mc2_masks,
                        get_mc2_mask,
                    )

                    mask = (
                        _hoisted_mc2_masks.get((width_arg, width_arg))
                        if _hoisted_mc2_masks is not None
                        else None
                    )
                    if mask is None:
                        reserved = get_mc2_mask()
                        if reserved is not None:
                            mask = reserved[:width_arg]
                    if mask is not None:
                        mask.copy_(self._mc2_host_masks[width_arg][ea], non_blocking=True)
                    self._last_mc2_active[width_arg] = ea
            if self._dynamic_sfa_len:
                s_idx = self._sfa_len_idx % len(self._sfa_len_slots)
                self._sfa_len_idx += 1
                s_len = self._sfa_len_slots[s_idx]
                s_len[0, 0] = ea
                s_len[1, 0] = self.position + ea
                s_len[0, 1] = width_arg
                s_len[1, 1] = self.position + width_arg
                self._dyn_lens_dev[width_arg].copy_(s_len, non_blocking=True)
        return res

    target._metadata = MethodType(exact_meta, target)
    try:
        yield
    finally:
        target._exact_active = None
        target._metadata = original
