"""Reject wrong geometry, changed common addresses and timing-window drift."""

import ast
import ctypes
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest

import torch

from record_layout import compare_records, frozen_layout, measure_layout
from tiling_packets import check_tiling


class PairedTilingContract(unittest.TestCase):
    def test_actual_frozen_cpp_layout_and_complete_field_extents(self):
        root = Path(__file__).parent
        actual = measure_layout(root / "record_headers")
        self.assertEqual(actual, frozen_layout())
        self.assertEqual(len(actual["config"]["fields"]), 98)
        self.assertEqual(len(actual["addresses"]["fields"]), 116)
        occupied = set()
        for field in actual["addresses"]["fields"]:
            span = set(range(field["offset"], field["offset"] + field["size"]))
            self.assertFalse(occupied & span)
            occupied |= span
        self.assertEqual(occupied, set(range(928)))

    def test_every_other_record_byte_is_protected(self):
        layout = frozen_layout()
        field = next(item for item in layout["addresses"]["fields"]
                     if item["name"] == "body.scratch[7]")
        self.assertEqual((field["offset"], field["size"]), (624, 8))
        first, second = bytearray(928), bytearray(928)
        struct.pack_into("<Q", first, 624, 0x101010)
        struct.pack_into("<Q", second, 624, 0x202020)
        control = SimpleNamespace(config=bytes(664), addresses=first)
        candidate = SimpleNamespace(config=bytes(664), addresses=second)
        compare_records(control, candidate, 0x101010, 0x202020, layout)
        for index in range(928):
            if 624 <= index < 632:
                continue
            mutated = bytearray(second)
            mutated[index] ^= 1
            with self.assertRaises(RuntimeError):
                compare_records(control, SimpleNamespace(config=bytes(664), addresses=mutated),
                                0x101010, 0x202020, layout)
        for index in range(664):
            mutated = bytearray(664)
            mutated[index] ^= 1
            with self.assertRaisesRegex(RuntimeError, "configurations"):
                compare_records(control, SimpleNamespace(config=mutated, addresses=second),
                                0x101010, 0x202020, layout)
        with self.assertRaisesRegex(RuntimeError, "independent owners"):
            compare_records(control, control, 0x101010, 0x101010, layout)

    def test_full_tiling_guards_reject_swaps_and_tail_changes(self):
        expected = json.loads(Path(__file__).with_name("paired-tilings.json").read_text())
        tensors = {arm: torch.tensor(list(bytes.fromhex(value["hex"])), dtype=torch.uint8)
                   for arm, value in expected.items()}
        for arm, tensor in tensors.items():
            self.assertTrue(check_tiling(torch, tensor, arm)["pass"])
            for index in (0, 36, 40, 199):
                bad = tensor.clone()
                bad[index] ^= 1
                with self.assertRaises(RuntimeError):
                    check_tiling(torch, bad, arm)
            with self.assertRaises(RuntimeError):
                check_tiling(torch, tensor, "candidate" if arm == "control" else "control")

    def test_event_reset_snapshot_and_graph_retirement_match_parent(self):
        root = Path(__file__).parent

        def tree(name):
            return ast.parse((root / name).read_text())

        old, new = tree("parent-layer-pair-timing.py"), tree("layer_pair_timing.py")

        def named(module, name):
            return next(node for node in ast.walk(module)
                        if isinstance(node, ast.FunctionDef) and node.name == name)

        for name in ("distribution", "pair_values", "reset", "sequence", "collect", "compare"):
            self.assertEqual(ast.dump(named(old, name)), ast.dump(named(new, name)), name)

        def event_loop(module):
            return next(node for node in ast.walk(module) if isinstance(node, ast.For)
                        and isinstance(node.target, ast.Name) and node.target.id == "sample")

        self.assertEqual(ast.dump(event_loop(old)), ast.dump(event_loop(new)))
        for module in (old, new):
            final = next(node.finalbody for node in ast.walk(module)
                         if isinstance(node, ast.Try) and node.finalbody)
            if module is old:
                old_final = ast.dump(ast.Module(body=final, type_ignores=[]))
            else:
                self.assertEqual(old_final, ast.dump(ast.Module(body=final, type_ignores=[])))
        native = tree("native_components.py")
        norm_call = next(node for node in ast.walk(native) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute) and node.func.attr == "norm_quant_out")
        self.assertIn("self.norm[5].view(self.rows, 1)", ast.unparse(norm_call))
        pair = ast.unparse(named(new, "pair"))
        self.assertIn("getattr(packet, arm).launch()", pair)
        self.assertEqual(pair.count("publish_topk("), 1)
        self.assertNotIn("separate", pair)

    def test_real_host_preparation_records_without_device_execution(self):
        try:
            import torch_npu
        except ImportError:
            self.skipTest("Actual AArch64 TorchNPU host-record gate runs in the frozen CPU image")
        from local_chain_host import Library, Prepared

        self.assertFalse(torch.npu.is_initialized())
        root = Path(__file__).parent
        library = Library(root / "layer_local_chain.so",
                          "609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f")
        device = SimpleNamespace(type="npu", index=0)

        class TensorAddress:
            def __init__(self, address):
                self.address, self.device, self.shape = address, device, (12, 6144)

            def data_ptr(self):
                return self.address

        next_address = 0x100000

        def group(count):
            nonlocal next_address
            result = [TensorAddress(next_address + i * 512) for i in range(count)]
            next_address += count * 512
            return result

        norm, attention, post, routed, bridge, body, shared = map(group, (8, 41, 3, 6, 12, 21, 4))
        attention[:3] = [norm[3], norm[6], norm[7]]
        alternative = list(body)
        alternative[7] = group(1)[0]
        for indexed in (True, False):
            def packet(scratch):
                return Prepared(library, norm, attention, post, routed, bridge, scratch,
                                shared, None if indexed else group(1)[0],
                                (12, 66, 64, 66, 64, 24, 1, 1, 0),
                                (.125, 1e-5, 1e-5, 1e-5, 1e-5), 1, indexed)

            # The shared selector owner is common to both records.
            if not indexed:
                topk = group(1)[0]

                def packet(scratch):
                    return Prepared(library, norm, attention, post, routed, bridge, scratch,
                                    shared, topk, (12, 66, 64, 66, 64, 24, 1, 1, 0),
                                    (.125, 1e-5, 1e-5, 1e-5, 1e-5), 1, False)
            candidate, control = packet(body), packet(alternative)
            result = compare_records(control, candidate, alternative[7].data_ptr(),
                                     body[7].data_ptr(), frozen_layout())
            self.assertEqual(result["addresses"]["candidate"]["body.shared.input"],
                             bridge[6].data_ptr())
            self.assertEqual(result["addresses"]["candidate"]["body.dispatch.active_list"],
                             body[20].data_ptr())
        self.assertFalse(torch.npu.is_initialized())


if __name__ == "__main__":
    unittest.main()
