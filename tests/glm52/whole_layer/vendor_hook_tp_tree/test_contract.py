"""Exercise three-rank arithmetic, reference dtype and communicator membership."""

import ast
import copy
import inspect
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import tp_chain
import tp_device
import tp_graph
import tp_groups
import tp_parameters
from tp_tree_contract import check_parameters, check_rank_witnesses, ordered_sum


class TreeContracts(unittest.TestCase):
    def test_cancellation_signed_zero_and_final_ties(self):
        partials = [torch.tensor(values, dtype=torch.bfloat16) for values in (
            [1, -0.0, 1, 1.0078125],
            [2.0 ** -25, -0.0, 2.0 ** -8, 2.0 ** -8],
            [-1, -0.0, 0, 0],
        )]
        expected = torch.tensor([2.0 ** -25, -0.0, 1, 1.015625], dtype=torch.bfloat16)
        actual = ordered_sum(torch, partials)
        self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
        serial = ((partials[0].float() + partials[1].float()) + partials[2].float()).bfloat16()
        self.assertNotEqual(actual[0].item(), serial[0].item())
        for values in (partials[:2], [partials[0].float(), *partials[1:]],
                       [partials[0][:1], *partials[1:]]):
            with self.assertRaises(ValueError):
                ordered_sum(torch, values)

    def witnesses(self):
        common = ("input_hidden", "input_residual", "layer6.retained", "layer7.retained",
                  "layer6.ar1", "layer6.ar2", "layer7.ar1", "layer7.ar2")
        partials = ("layer6.attention_partial", "layer6.ffn_partial",
                    "layer7.attention_partial", "layer7.ffn_partial")
        return [{"rank": rank, **{name: "same" for name in common},
                 **{name: str(rank) for name in partials}} for rank in range(3)]

    def test_third_rank_is_included_in_every_witness(self):
        records = self.witnesses()
        check_rank_witnesses(records)
        for key in records[2]:
            changed = copy.deepcopy(records)
            changed[2][key] = records[0][key] if "partial" in key else "different"
            with self.assertRaises(RuntimeError):
                check_rank_witnesses(changed)
        with self.assertRaises(RuntimeError):
            check_rank_witnesses(records[::-1])

    def test_three_parameter_shards_and_replicas(self):
        records = [[{"name": name, "shape": [2, 4], "dtype": "torch.int8", "bytes": 8,
                     "sha256": str(rank) if name == "shard" else "same"}
                    for name in ("shard", "replica")] for rank in range(3)]
        check_parameters(records, {"shard"})
        for index, field, value in ((0, "sha256", "0"), (1, "sha256", "bad"),
                                    (0, "bytes", 16), (0, "shape", [4, 2])):
            changed = copy.deepcopy(records)
            changed[2][index][field] = value
            with self.assertRaises(RuntimeError):
                check_parameters(changed, {"shard"})

    def test_third_shard_uses_a_distinct_roll(self):
        source = inspect.getsource(tp_parameters.initialize)
        self.assertIn("value.roll(rank, 0)", source)
        self.assertIn("peers = [None, None, None]", source)
        self.assertIn("check_pair(peers)", source)

    def test_reference_collective_receives_fp32(self):
        for function in (tp_chain.run_tp_chain, tp_graph.run_tp_graph):
            tree = ast.parse(inspect.getsource(function))
            definition = next(node for node in ast.walk(tree)
                              if isinstance(node, ast.FunctionDef) and node.name == "reduce")
            partials = [torch.tensor([value], dtype=torch.bfloat16)
                        for value in (1, 2.0 ** -25, -1)]
            calls, reductions = [], []

            def collective(value, **kwargs):
                self.assertEqual(value.dtype, torch.float32)
                calls.append(kwargs)
                value.copy_((partials[0].float() + partials[2].float()) + partials[1].float())

            def metric(torch_module, actual, expected):
                equal = torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                return {"pass": equal, "bitwise_equal": equal}

            namespace = {"torch": torch, "args": SimpleNamespace(oracle_hccl="group"),
                         "gather": lambda value: partials, "snapshot": lambda t, x: x.clone(),
                         "ordered_sum": ordered_sum, "metric": metric, "number": 6,
                         "reductions": reductions,
                         "tensor_witness": lambda t, x: {"value": x.tolist()}}
            exec(compile(ast.Module(body=[definition], type_ignores=[]),
                         "literal-fp32-reference", "exec"), namespace)
            result = torch.empty_like(partials[0])
            with mock.patch.object(torch.distributed, "all_reduce", side_effect=collective), \
                    mock.patch.object(torch, "npu", SimpleNamespace(synchronize=lambda: None), create=True):
                namespace["reduce"](partials[0], result, "ar1")
            self.assertEqual(result.item(), 2.0 ** -25)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(reductions), 1)

    def test_all_sixteen_processes_create_groups_in_one_order(self):
        source = inspect.getsource(tp_groups.create_communication_groups).replace(
            "    from vllm_ascend.utils import create_hccl_pg_options\n", "")
        namespace = {"create_hccl_pg_options": lambda name: name}
        exec(compile(source, "three-rank-group-order", "exec"), namespace)
        expected = [(ranks, backend) for rank in range(3)
                    for ranks, backend in (((rank,), "hccl"), ((rank,), "gloo"))]
        expected += [((0, 1, 2), "hccl"), ((0, 1, 2), "gloo")]
        for rank in range(16):
            calls = []

            def new_group(ranks, backend, **kwargs):
                calls.append((tuple(ranks), backend))
                return object() if rank in ranks else -100

            def coordinator(groups, local_rank, backend, **kwargs):
                device = new_group(groups[0], backend)
                cpu = new_group(groups[0], "gloo")
                return SimpleNamespace(device_group=device, cpu_group=cpu)

            fake = SimpleNamespace(distributed=SimpleNamespace(
                barrier=lambda **kwargs: None, new_group=new_group,
                GroupMember=SimpleNamespace(NON_GROUP_MEMBER=-100)),
                npu=SimpleNamespace(is_initialized=lambda: rank < 3))
            state = SimpleNamespace(init_model_parallel_group=coordinator,
                                    get_world_group=lambda: SimpleNamespace(cpu_group=object()))
            singleton, _, _ = namespace["create_communication_groups"](fake, state, rank)
            self.assertEqual(singleton is None, rank >= 3)
            self.assertEqual(calls, expected)

    @unittest.skipUnless(hasattr(torch, "npu"), "Requires the actual Torch-NPU CPU environment")
    def test_cpu_nonmember_does_not_request_tp_configuration(self):
        from vllm_ascend.utils import create_hccl_pg_options

        state = SimpleNamespace(get_world_group=lambda: SimpleNamespace(cpu_group=object()))
        nonmember = torch.distributed.GroupMember.NON_GROUP_MEMBER
        with mock.patch("torch.distributed.barrier"), mock.patch(
                "torch.distributed.new_group", return_value=nonmember) as new_group, mock.patch(
                "vllm_ascend.utils.create_hccl_pg_options",
                wraps=create_hccl_pg_options) as options:
            result = tp_groups.create_communication_groups(torch, state, 3)
        self.assertEqual(result, (None, nonmember, nonmember))
        self.assertEqual([call.args for call in options.call_args_list], [("mc2",)] * 3)
        self.assertIsNone(new_group.call_args_list[6].kwargs["pg_options"])
        self.assertFalse(torch.npu.is_initialized())

    def test_only_three_processes_enter_the_device_constructor(self):
        tree = ast.parse(inspect.getsource(tp_device.worker))
        conditional = next(node for node in ast.walk(tree)
                           if isinstance(node, ast.If) and ast.unparse(node.test) == "rank < 3")
        active = ast.unparse(ast.Module(body=conditional.body, type_ignores=[]))
        helpers = ast.unparse(ast.Module(body=conditional.orelse, type_ignores=[]))
        self.assertIn("run_device_rank(", active)
        self.assertNotIn("run_device_rank(", helpers)
        self.assertIn("create_communication_groups(", helpers)

    def test_real_seams_and_graph_lifetime_remain_connected(self):
        for function in (tp_chain.run_tp_chain, tp_graph.run_tp_graph):
            source = inspect.getsource(function)
            self.assertIn("inputs, first.output,", source)
            self.assertIn("first.local.bridge[1]", source)
            self.assertIn("if rank == iteration % 3:", source)
            self.assertEqual(source.count("window = Window("), 1)
            self.assertLess(source.index("window = Window("), source.index("for rows in"))
            self.assertIn("torch.distributed.barrier(group=args.oracle_cpu)\n            window.close()", source)
        graph = inspect.getsource(tp_graph.run_tp_graph)
        self.assertLess(graph.index("current_graph.close()"), graph.index("window.close()"))
        self.assertIn("current_graph.replay()", graph)


if __name__ == '__main__':
    unittest.main()
