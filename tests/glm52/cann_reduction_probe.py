"""An eager diagnostic copy/dtype intervention on one existing TP group."""


_MISSING = object()
_METHOD = "_all_reduce_out_place"


class TpReductionProbe:
    """Copy BF16 partials, reduce in the selected dtype, and return BF16.

    The original communicator performs the collective. Captures stay on the
    execution device until the caller saves them after forward completes.
    Other input dtypes retain the original method's mutation and return rules.
    """

    def __init__(self, dtype, *, capture_first=3, group=None):
        import torch

        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Reduction dtype must be BF16 or FP32")
        if type(capture_first) is not int or capture_first < 0:
            raise ValueError("capture_first must be a nonnegative integer")
        self.dtype = dtype
        self.capture_first = capture_first
        self.group = group
        self.records = []
        self.captures = []
        self._capture_calls = []
        self._active = False
        self._original = None
        self._saved_attribute = _MISSING

    def __enter__(self):
        if self._active:
            raise RuntimeError("Reduction probe is already active")
        if self.group is None:
            from vllm.distributed import get_tp_group

            self.group = get_tp_group()
        self._original = getattr(self.group, _METHOD)
        self._saved_attribute = vars(self.group).get(_METHOD, _MISSING)
        self.records.clear()
        self.captures.clear()
        self._capture_calls.clear()
        setattr(self.group, _METHOD, self._reduce)
        self._active = True
        return self

    def _reduce(self, input_):
        import torch

        if not self._active:
            raise RuntimeError("Reduction arrived outside the probe context")
        changed = input_.dtype == torch.bfloat16
        entry = {
            "call": len(self.records), "shape": list(input_.shape),
            "input_dtype": str(input_.dtype), "device": str(input_.device),
            "input_data_ptr": input_.data_ptr(), "intervened": changed,
        }
        self.records.append(entry)
        if not changed:
            result = self._original(input_)
            entry["output_dtype"] = str(result.dtype)
            entry["output_data_ptr"] = result.data_ptr()
            return result

        capture = len(self.captures) < self.capture_first
        saved_input = input_.detach().clone() if capture else None
        partial = input_.to(dtype=self.dtype, copy=True)
        entry["reduction_dtype"] = str(partial.dtype)
        entry["partial_data_ptr"] = partial.data_ptr()
        reduced = self._original(partial)
        if (reduced.shape != input_.shape or reduced.dtype != self.dtype
                or reduced.device != input_.device):
            raise ValueError("The original reduction changed tensor metadata")
        result = reduced.to(dtype=input_.dtype)
        entry["reduced_data_ptr"] = reduced.data_ptr()
        entry["output_dtype"] = str(result.dtype)
        entry["output_data_ptr"] = result.data_ptr()
        if result.data_ptr() == input_.data_ptr():
            raise RuntimeError("The diagnostic result aliases the input")
        if capture:
            self.captures.append({
                "input": saved_input,
                "reduced": reduced.detach().clone(),
                "output": result.detach().clone(),
            })
            self._capture_calls.append(entry["call"])
        return result

    def metadata(self):
        """Return tensor-free evidence without synchronizing the device."""
        return {
            "mode": str(self.dtype),
            "group_name": getattr(self.group, "unique_name", None),
            "world_size": getattr(self.group, "world_size", None),
            "original_method": (
                f"{self._original.__module__}."
                f"{self._original.__qualname__}"
                if self._original is not None else None
            ),
            "calls": len(self.records),
            "intervened_calls": sum(row["intervened"]
                                    for row in self.records),
            "capture_call_indices": list(self._capture_calls),
            "records": [dict(row) for row in self.records],
            "device_synchronization_requested": False,
        }

    def __exit__(self, exc_type, exc_value, traceback):
        if self._active:
            if self._saved_attribute is _MISSING:
                delattr(self.group, _METHOD)
            else:
                setattr(self.group, _METHOD, self._saved_attribute)
            self._active = False
        return False


if __name__ == "__main__":
    import json
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch

    from tilert.models.glm_5_2_ascend.cann_model import CannModel

    class InplaceGroup:
        world_size = 16
        unique_name = "cpu-contract-fixture"

        def __init__(self):
            self.dtypes = []

        def _all_reduce_out_place(self, value):
            self.dtypes.append(value.dtype)
            return value.add_(1)

    checks = []
    for dtype in (torch.bfloat16, torch.float32):
        group = InplaceGroup()
        value = torch.tensor([[1, -2]], dtype=torch.bfloat16)
        with TpReductionProbe(dtype, group=group, capture_first=1) as probe:
            actual = group._all_reduce_out_place(value)
            assert torch.equal(value, torch.tensor([[1, -2]]))
            assert torch.equal(actual, torch.tensor([[2, -1]]))
            assert actual.data_ptr() != value.data_ptr()
            assert probe.captures[0]["reduced"].dtype == dtype
            assert probe.metadata()["intervened_calls"] == 1
        assert _METHOD not in vars(group)
        checks.append(str(dtype) + ": copy/dtype/capture/restore")

    group = InplaceGroup()
    override = lambda value: value.add_(2)
    setattr(group, _METHOD, override)
    value = torch.tensor([3], dtype=torch.int32)
    try:
        with TpReductionProbe(torch.float32, group=group) as probe:
            assert group._all_reduce_out_place(value) is value
            assert value.item() == 5 and not probe.captures
            raise RuntimeError("restore on caller exception")
    except RuntimeError as error:
        assert str(error) == "restore on caller exception"
    assert getattr(group, _METHOD) is override
    checks.append("other dtype: original mutation/identity; exception restore")

    for overridden in (False, True):
        group = InplaceGroup()
        if overridden:
            setattr(group, _METHOD, group._all_reduce_out_place)
        original = group._all_reduce_out_place
        stopped = []
        engine = CannModel.__new__(CannModel)
        engine.worker = SimpleNamespace(shutdown=lambda: stopped.append(True))
        module = SimpleNamespace(get_tp_group=lambda: group)
        with patch.dict(sys.modules, {"vllm.distributed": module}):
            engine._install_tp_sum()
        value = torch.tensor([1, -2], dtype=torch.bfloat16)
        result = group._all_reduce_out_place(value)
        assert torch.equal(value, torch.tensor([1, -2]))
        assert result.dtype == torch.bfloat16
        assert torch.equal(result, torch.tensor([2, -1]))
        assert group.dtypes == [torch.float32]
        other = torch.tensor([3], dtype=torch.int32)
        assert group._all_reduce_out_place(other) is other
        assert other.item() == 4 and group.dtypes[-1] == torch.int32
        engine.shutdown()
        assert group._all_reduce_out_place == original and stopped == [True]
        assert (_METHOD in vars(group)) == overridden
        checks.append(f"runtime FP32/copy/restore; override={overridden}")

    backend = sys.modules.get("torch_npu")
    initialized = bool(backend and backend.npu._initialized)
    assert not initialized
    print(json.dumps({"status": "pass", "checks": checks,
                      "npu_initialized": initialized,
                      "scope": "CPU copy contract; no collective executed"}))
