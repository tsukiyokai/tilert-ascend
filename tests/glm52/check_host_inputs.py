"""Check real packet bytes, stable views and retained CPU source slots."""

import gc
import subprocess
import sys
import types
import unittest
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.models.glm_5_2_ascend.cann_model import CannModel
from tilert.models.glm_5_2_ascend.host_inputs import HostInputs, _layout

WIDTHS = (1, 2, 4, 6, 8, 12, 16, 24, 32)


@contextmanager
def vendor_metadata():
    """Only unavailable vendor metadata classes are substituted locally."""
    state = types.ModuleType("vllm_ascend.attention.attention_v1")
    state.AscendAttentionState = SimpleNamespace(
        DecodeOnly="decode", ChunkedPrefill="prefill"
    )
    common = types.ModuleType("vllm_ascend.attention.utils")
    common.AscendCommonAttentionMetadata = SimpleNamespace
    with patch.dict(sys.modules, {state.__name__: state, common.__name__: common}):
        yield


class Builder:
    def build(self, prefix, common):
        return common


def cpu_engine(packed):
    """Allocate real CPU views; execute the real preparation/metadata methods."""
    engine = CannModel.__new__(CannModel)
    engine.device, engine.position, engine.capacity = "cpu", 8192, 9344
    engine.attention_kernel, engine.graph_width = "sfa", 32
    engine.graph, engine.hoist_metadata, engine._hoisted = None, False, {}
    engine._host_inputs = (
        HostInputs("cpu", WIDTHS, pin_memory=False) if packed else None
    )
    engine._input_buffers = {}
    for width in WIDTHS:
        if packed:
            tokens, positions, slots, _ = engine._host_inputs.views(width)
        else:
            tokens = torch.empty(width, dtype=torch.int32)
            positions = torch.empty(width, dtype=torch.int64)
            slots = torch.empty(width, dtype=torch.int32)
        query = torch.tensor([0, width], dtype=torch.int32)
        engine._input_buffers[width] = (tokens, positions, slots, query, query.clone())
    engine.seq_lens = torch.empty(1, dtype=torch.int32)
    engine._lengths, engine._length_slot = torch.zeros(64, dtype=torch.int32), 0
    engine.block_table = torch.arange(73, dtype=torch.int32).reshape(1, -1)
    engine.builders = [(("observed",), Builder())]
    return engine


class HostInputChecks(unittest.TestCase):
    def test_real_prepare_inputs_matches_default_for_host_tensor_and_mixed(self):
        original, packed = cpu_engine(False), cpu_engine(True)
        retained = []
        with vendor_metadata():
            for width in WIDTHS:
                values = tuple(6500 + row for row in range(width))
                for kind in ("host", "tensor", "mixed"):
                    with self.subTest(width=width, kind=kind):
                        tokens = values
                        if kind == "tensor":
                            tokens = torch.tensor(values, dtype=torch.int64)
                        elif kind == "mixed":
                            tokens = (
                                torch.tensor(values[0], dtype=torch.int32),
                                *values[1:],
                            )
                        depths = tuple(row // 2 for row in range(width))
                        position = 8192 + width * 2
                        for engine in (original, packed):
                            engine.position = position
                            actual_width, metadata = engine._prepare_inputs(
                                tokens, depths=depths
                            )
                            common = metadata["observed"]
                            expected_positions = [position + value for value in depths]
                            self.assertEqual(actual_width, width)
                            self.assertEqual(engine.input_ids.tolist(), list(values))
                            self.assertEqual(
                                engine.positions.tolist(), expected_positions
                            )
                            self.assertEqual(
                                engine.slots.tolist(),
                                list(range(position, position + width)),
                            )
                            self.assertEqual(
                                engine.seq_lens.tolist(), [position + width]
                            )
                            self.assertEqual(
                                common.positions_cpu.tolist(), expected_positions
                            )
                            self.assertEqual(
                                common.seq_lens.data_ptr(), engine.seq_lens.data_ptr()
                            )
                            self.assertEqual(
                                common.slot_mapping.data_ptr(), engine.slots.data_ptr()
                            )
                            self.assertIs(
                                common.seq_lens_cpu_upper_bound, common.seq_lens_cpu
                            )
                            retained.append(
                                (
                                    common.seq_lens_cpu,
                                    position + width,
                                    common.positions_cpu,
                                    expected_positions,
                                )
                            )
                        self.assertEqual(packed._host_inputs._next[width], 1)
        for length, expected_length, positions, expected_positions in retained:
            self.assertEqual(length.tolist(), [expected_length])
            self.assertEqual(positions.tolist(), expected_positions)

    def test_real_prepare_width_switch_and_device_fallback_keep_packet_aliases(self):
        engine = cpu_engine(True)
        with vendor_metadata():
            _, metadata = engine._prepare_inputs(tuple(range(32)))
            common32 = metadata["observed"]
            pointer32 = common32.seq_lens.data_ptr()
            source32 = common32.seq_lens_cpu
            engine.position = 8256
            _, metadata = engine._prepare_inputs(torch.tensor([24187]))
            self.assertNotEqual(engine.seq_lens.data_ptr(), pointer32)
            self.assertEqual(
                engine.seq_lens.data_ptr(), engine._host_inputs.views(1)[3].data_ptr()
            )
            self.assertEqual(common32.seq_lens.tolist(), [8224])
            self.assertEqual(source32.tolist(), [8224])
            self.assertEqual(engine._host_inputs._next[1], 0)
            engine.position = 8288
            mixed = (torch.tensor(3), *range(1, 32))
            _, metadata = engine._prepare_inputs(mixed)
            self.assertEqual(engine.seq_lens.data_ptr(), pointer32)
            self.assertEqual(common32.seq_lens.tolist(), [8320])
            self.assertEqual(metadata["observed"].seq_lens.data_ptr(), pointer32)
            self.assertEqual(source32.tolist(), [8224])
            self.assertEqual(engine._host_inputs._next[32], 1)
            engine.position = 8320
            engine._prepare_inputs(tuple(range(32)))
            self.assertEqual(engine.seq_lens.data_ptr(), pointer32)
            self.assertEqual(engine.seq_lens.tolist(), [8352])
            self.assertEqual(engine._host_inputs._next[32], 2)

    def test_all_widths_have_aligned_disjoint_stable_typed_views(self):
        packet = HostInputs("cpu", WIDTHS, pin_memory=False)
        bases = []
        for width in WIDTHS:
            with self.subTest(width=width):
                views = packet.views(width)
                pointers = [value.data_ptr() for value in views]
                intervals = [
                    (
                        value.data_ptr(),
                        value.data_ptr() + value.numel() * value.element_size(),
                    )
                    for value in views
                ]
                self.assertEqual(
                    [value.dtype for value in views],
                    [torch.int32, torch.int64, torch.int32, torch.int32],
                )
                self.assertEqual(
                    [list(value.shape) for value in views],
                    [[width], [width], [width], [1]],
                )
                self.assertTrue(all(pointer % 8 == 0 for pointer in pointers))
                self.assertTrue(
                    all(left[1] <= right[0] for left, right in pairwise(intervals))
                )
                bases.append(views[0].data_ptr())
                length_cpu, positions_cpu = packet.stage(tuple(range(width)), 8192)
                self.assertEqual(
                    [value.tolist() for value in views],
                    [
                        list(range(width)),
                        list(range(8192, 8192 + width)),
                        list(range(8192, 8192 + width)),
                        [8192 + width],
                    ],
                )
                self.assertEqual(length_cpu.dtype, torch.int32)
                self.assertEqual(positions_cpu.dtype, torch.int64)
                self.assertEqual(positions_cpu.device.type, "cpu")
                self.assertNotEqual(positions_cpu.data_ptr(), views[1].data_ptr())
                packet.stage(tuple(range(width)), 17)
                self.assertIs(packet.views(width), views)
                self.assertEqual([value.data_ptr() for value in views], pointers)
                retained = [value.clone() for value in views[1:]]
                views[0].fill_(99)
                self.assertTrue(
                    all(
                        torch.equal(value, saved)
                        for value, saved in zip(views[1:], retained, strict=True)
                    )
                )
        self.assertEqual(len(set(bases)), len(WIDTHS))

    def test_depths_change_positions_only_for_every_width(self):
        packet = HostInputs("cpu", WIDTHS, pin_memory=False)
        for width in WIDTHS:
            with self.subTest(width=width):
                depths = tuple(index // 2 for index in range(width))
                length_cpu, positions_cpu = packet.stage(
                    tuple(range(width)), 8192, depths
                )
                tokens, positions, slots, length = packet.views(width)
                self.assertEqual(tokens.tolist(), list(range(width)))
                self.assertEqual(positions.tolist(), [8192 + value for value in depths])
                self.assertTrue(torch.equal(positions, positions_cpu))
                self.assertEqual(slots.tolist(), list(range(8192, 8192 + width)))
                self.assertEqual(length.tolist(), [8192 + width])
                self.assertTrue(torch.equal(length_cpu, length))

    def test_64_sources_survive_until_their_width_ring_wraps(self):
        packet = HostInputs("cpu", (1, 32), pin_memory=False)
        sources = [packet.stage((index,), 100 + index) for index in range(64)]
        gc.collect()
        self.assertEqual(len({value[0].data_ptr() for value in sources}), 64)
        self.assertEqual(len({value[1].data_ptr() for value in sources}), 64)
        self.assertEqual([value[0].item() for value in sources], list(range(101, 165)))
        self.assertEqual([value[1].item() for value in sources], list(range(100, 164)))
        packet.stage(tuple(range(32)), 8192)
        self.assertEqual(sources[0][0].item(), 101)
        length, positions = packet.stage((7,), 900)
        self.assertEqual(
            (length.data_ptr(), positions.data_ptr()),
            (sources[0][0].data_ptr(), sources[0][1].data_ptr()),
        )
        self.assertEqual((length.item(), positions.item()), (901, 900))
        self.assertEqual(sources[1][0].item(), 102)

    def test_stage_makes_one_real_uint8_copy_and_preserves_padding(self):
        packet = HostInputs("cpu", WIDTHS, pin_memory=False)
        original_copy = torch.Tensor.copy_
        for width in WIDTHS:
            with self.subTest(width=width):
                calls = []
                pointer = packet.views(width)[0].data_ptr()

                def copy(
                    destination, source, *args, _pointer=pointer, _calls=calls, **kwargs
                ):
                    if (
                        destination.dtype == torch.uint8
                        and destination.data_ptr() == _pointer
                    ):
                        _calls.append((source.clone(), kwargs.copy()))
                    return original_copy(destination, source, *args, **kwargs)

                tokens = tuple(6500 + index for index in range(width))
                with (
                    patch.object(torch.Tensor, "copy_", copy),
                    patch.object(
                        torch.Tensor, "cpu", side_effect=AssertionError("Device read")
                    ),
                    patch.object(
                        torch.Tensor,
                        "item",
                        side_effect=AssertionError("Scalar device read"),
                    ),
                ):
                    packet.stage(tokens, 8192)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][1], {"non_blocking": True})
                offsets, size = _layout(width)
                expected = torch.zeros(size, dtype=torch.uint8)
                values = (
                    torch.tensor(tokens, dtype=torch.int32),
                    torch.arange(8192, 8192 + width, dtype=torch.int64),
                    torch.arange(8192, 8192 + width, dtype=torch.int32),
                    torch.tensor([8192 + width], dtype=torch.int32),
                )
                for offset, value in zip(offsets, values, strict=True):
                    expected[
                        offset : offset + value.numel() * value.element_size()
                    ].copy_(value.view(torch.uint8))
                self.assertTrue(torch.equal(calls[0][0], expected))
                self.assertTrue(
                    all(
                        torch.equal(actual, value)
                        for actual, value in zip(
                            packet.views(width), values, strict=True
                        )
                    )
                )

    def test_token_and_width_validation_remain_at_the_actual_model_boundary(self):
        model = SimpleNamespace(_host_inputs=object())
        for token in (-1, 154880, True, 1.5, "3"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                CannModel._prepare_inputs(model, (token,))
        with self.assertRaisesRegex(ValueError, "Unsupported verification width"):
            CannModel._prepare_inputs(model, (1, 2, 3))

    def test_module_import_does_not_import_torch_or_npu(self):
        path = ROOT / "tilert/models/glm_5_2_ascend/host_inputs.py"
        code = (
            "import importlib.util, sys\n"
            "before = set(sys.modules)\n"
            f"spec = importlib.util.spec_from_file_location('host_inputs', {str(path)!r})\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "assert not any(name == 'torch' or name.startswith('torch_npu') for name in set(sys.modules) - before)\n"
        )
        subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
