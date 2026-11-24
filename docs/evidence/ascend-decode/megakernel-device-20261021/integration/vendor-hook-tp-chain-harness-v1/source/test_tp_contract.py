"""CPU contracts for rank identity, reduction seams and global group ordering."""

import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

import local_chain
import stage_tp
import tp_chain
import tp_device
import tp_groups
import tp_parameters
import tp_parent


class TPContracts(unittest.TestCase):
    def test_sum_preserves_bf16_rounding_and_zero_signs(self):
        left = torch.tensor([-0.0, 1.0, 0.001953125], dtype=torch.bfloat16)
        right = torch.tensor([-0.0, 0.00390625, -0.001953125], dtype=torch.bfloat16)
        actual = tp_chain.ordered_sum(torch, [left, right])
        expected = torch.tensor([-0.0, 1.0, 0.0], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
        for inputs in ([left], [left.float(), right], [left, right[:1]]):
            with self.assertRaises(ValueError):
                tp_chain.ordered_sum(torch, inputs)

    def rank_records(self):
        common = ("input_hidden", "input_residual", "layer6.retained", "layer7.retained",
                  "layer6.ar1", "layer6.ar2", "layer7.ar1", "layer7.ar2")
        partials = ("layer6.attention_partial", "layer6.ffn_partial",
                    "layer7.attention_partial", "layer7.ffn_partial")
        return [{"rank": rank, **{name: "same" for name in common},
                 **{name: str(rank) for name in partials}} for rank in (0, 1)]

    def test_rank_witness_rejects_own_plus_own_and_disagreement(self):
        records = self.rank_records()
        tp_chain.check_rank_witnesses(records)
        for key in records[0]:
            changed = copy.deepcopy(records)
            changed[1][key] = (changed[0][key] if "partial" in key else "different")
            with self.assertRaises(RuntimeError):
                tp_chain.check_rank_witnesses(changed)
        with self.assertRaises(RuntimeError):
            tp_chain.check_rank_witnesses(records[::-1])

    def test_sharded_names_are_actual_recorded_parameters(self):
        path = Path(tp_parameters.__file__).with_name("parameter_geometry.json")
        recorded = json.loads(path.read_text())
        self.assertEqual(recorded["source_sha256"],
                         "620d0e04516d4c8639e6af372a36300e049dad76ae88e9d1261bf3d79cafcd6e")
        self.assertEqual(set(recorded["parameters"]), {"layer6", "layer7"})
        for parameters in recorded["parameters"].values():
            pair = [copy.deepcopy(parameters) for _ in range(2)]
            for rank, values in enumerate(pair):
                for item in values:
                    item["sha256"] = (f"rank{rank}" if item["name"] in
                                      tp_parameters.SHARDED_MATRICES else "replicated")
            tp_parameters.check_pair(pair)
            sharded = next(item for item in pair[1]
                           if item["name"] in tp_parameters.SHARDED_MATRICES)
            sharded["sha256"] = "rank0"
            with self.assertRaisesRegex(RuntimeError, "identity"):
                tp_parameters.check_pair(pair)

    def test_parameter_gate_rejects_replica_or_geometry_changes(self):
        values = [{"name": name, "shape": [2, 4], "dtype": "torch.int8", "bytes": 8,
                   "sha256": "rank0"} for name in tp_parameters.SHARDED_MATRICES]
        values.append({"name": "mlp.gate.weight", "shape": [2, 4],
                       "dtype": "torch.bfloat16", "bytes": 16, "sha256": "replica"})
        right = copy.deepcopy(values)
        for item in right[:-1]:
            item["sha256"] = "rank1"
        tp_parameters.check_pair([values, right])
        right[-1]["sha256"] = "different"
        with self.assertRaisesRegex(RuntimeError, "identity"):
            tp_parameters.check_pair([values, right])
        right[-1]["sha256"] = "replica"
        right[0]["bytes"] = 16
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            tp_parameters.check_pair([values, right])

    def test_parent_math_is_exactly_recoverable(self):
        source = Path(local_chain.__file__).read_text()
        original, changed = stage_tp.phase_source(source)
        self.assertEqual(inspect.getsource(tp_parent.separate).rstrip(), changed.rstrip())
        restored = changed.replace("def separate(self, attention_sum, output, reduce):",
                                   "def separate(self):")
        restored = restored.replace("    reduce(self.partial, attention_sum, 'ar1')\n", "")
        restored = restored.replace("self.ffn.native.forward_out(\n        attention_sum,",
                                    "self.ffn.native.forward_out(\n        self.partial,")
        restored = restored.replace("    reduce(self.bridge[10], output, 'ar2')\n", "")
        self.assertEqual(restored, original)

    def test_both_reduction_results_feed_the_real_seams(self):
        for mode in ("int8", "shared"):
            events = []
            norm = [torch.ones(1, 2) for _ in range(8)]
            norm[6], norm[7] = torch.empty(1, 2, dtype=torch.int8), torch.empty(1, 1)
            bridge = [torch.empty(1, 2) for _ in range(12)]
            local = SimpleNamespace(norm=norm, bridge=bridge, rows=1, mode=mode,
                                    main=[], front=[], tail=[], main_metadata=[0] * 6,
                                    index_metadata=[], caches=[], selected=None,
                                    scratch=None, tiling=None, scalars=(1, 1, 1),
                                    partial=torch.empty(1, 2), body=[], sync=1,
                                    input_norm=lambda x, r: (x + r, x + r))
            def attention(*args):
                events.append("attention")
                local.partial.fill_(3)
            local.torch = SimpleNamespace(ops=SimpleNamespace(tilert=SimpleNamespace(
                indexed_attention_out=attention, shared_full_attention_out=attention)))
            local.torch_npu = SimpleNamespace(npu_dynamic_quant=lambda x: (
                torch.ones_like(x, dtype=torch.int8), torch.ones(x.shape[0])))
            def ffn(hidden, retained, *args):
                events.append("ffn")
                self.assertIs(hidden, summed)
                self.assertIs(retained, norm[4])
                bridge[10].copy_(hidden + retained)
            local.ffn = SimpleNamespace(native=SimpleNamespace(forward_out=ffn),
                                        gamma=None, gate=None, bias=None, weights=[],
                                        shared=[], epsilon=1)
            summed, output = torch.empty(1, 2), torch.empty(1, 2)
            def reduce(partial, target, phase):
                events.append(phase)
                target.copy_(partial * 2)
            tp_parent.separate(local, summed, output, reduce)
            self.assertEqual(events, ["attention", "ar1", "ffn", "ar2"])
            self.assertTrue(torch.equal(output, torch.full((1, 2), 16.0)))
            self.assertTrue(local.parent_validated)

    def test_all_processes_have_identical_group_order(self):
        # A call-order model only; real HCCL/Gloo creation remains a device gate.
        source = inspect.getsource(tp_groups.create_communication_groups)
        source = source.replace("    from vllm_ascend.utils import create_hccl_pg_options\n", "")
        namespace = {"create_hccl_pg_options": lambda name: name}
        exec(compile(source, "group-order-contract", "exec"), namespace)
        orders = []
        for rank in range(16):
            calls = []
            def new_group(ranks, backend, **kwargs):
                calls.append((tuple(ranks), backend))
                return object() if rank in ranks else -100
            def coordinator(groups, local_rank, backend, **kwargs):
                device = new_group(groups[0], backend)
                cpu = new_group(groups[0], "gloo")
                return SimpleNamespace(device_group=device, cpu_group=cpu)
            fake_torch = SimpleNamespace(distributed=SimpleNamespace(
                barrier=lambda **kwargs: None, new_group=new_group,
                GroupMember=SimpleNamespace(NON_GROUP_MEMBER=-100)),
                npu=SimpleNamespace(is_initialized=lambda: rank < 2))
            state = SimpleNamespace(init_model_parallel_group=coordinator,
                                    get_world_group=lambda: SimpleNamespace(cpu_group=object()))
            singleton, _, _ = namespace["create_communication_groups"](fake_torch, state, rank)
            self.assertEqual(singleton is None, rank >= 2)
            orders.append(calls)
        expected = [((0,), "hccl"), ((0,), "gloo"), ((1,), "hccl"),
                    ((1,), "gloo"), ((0, 1), "hccl"), ((0, 1), "gloo")]
        self.assertTrue(all(order == expected for order in orders))

    def test_cpu_peers_do_not_enter_device_constructor_context(self):
        tree = ast.parse(inspect.getsource(tp_device.worker))
        conditional = next(node for node in ast.walk(tree)
                           if isinstance(node, ast.If) and ast.unparse(node.test) == "rank < 2")
        active = ast.unparse(ast.Module(body=conditional.body, type_ignores=[]))
        helpers = ast.unparse(ast.Module(body=conditional.orelse, type_ignores=[]))
        self.assertIn("run_device_rank(", active)
        self.assertNotIn("create_communication_groups(", active)
        self.assertIn("create_communication_groups(", helpers)
        self.assertNotIn("run_device_rank(", helpers)
        self.assertNotIn("device_groups_context(", helpers)

    def test_constructor_transform_preserves_original_functions(self):
        directory = Path(tp_device.__file__).parent
        original = (directory / "device_local_chain.py").read_text()
        transformed = (directory / "tp_constructor.py").read_text()
        self.assertEqual(transformed, stage_tp.constructor_source(original))
        for name in ("tensor_descriptor", "make_configuration"):
            self.assertEqual(stage_tp.function_source(original, name),
                             stage_tp.function_source(transformed, name))
        self.assertIn("torch.npu.set_device(args.device_rank)", transformed)
        self.assertNotIn("torch.npu.set_device(0)", transformed)
        self.assertIn("process_weights_after_loading(layer,", transformed)
        self.assertIn("for index in (6, 7):", transformed)

    def test_eager_window_is_shared_across_widths_and_closed_after_rendezvous(self):
        source = inspect.getsource(tp_chain.run_tp_chain)
        self.assertLess(source.index("window = Window("), source.index("for rows in"))
        self.assertEqual(source.count("window = Window("), 1)
        self.assertIn("with torch.npu.stream(stream), torch.inference_mode():", source)
        self.assertIn("torch.distributed.barrier(group=args.oracle_cpu)\n            window.close()", source)
        self.assertIn("inputs, first.output,", source)
        self.assertIn("first.local.bridge[1]", source)
        self.assertNotIn("torch.npu.graph(", source)
        self.assertNotIn("window.address", inspect.getsource(tp_chain.ordered_sum))


if __name__ == "__main__":
    unittest.main()
