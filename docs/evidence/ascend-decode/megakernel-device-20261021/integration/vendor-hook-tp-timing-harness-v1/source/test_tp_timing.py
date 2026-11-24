"""Reject mismatched rank samples, hidden work and changed parent mathematics."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

import stage_tp_timing
import tp_graph
import tp_parent
import tp_timing_benchmark as benchmark
import tp_timing_chain
import tp_timing_native as native
from tp_timing_environment import check_environment


def function(module, name):
    tree = ast.parse(inspect.getsource(module))
    return next(node for node in tree.body if getattr(node, "name", None) == name)


class TimingContracts(unittest.TestCase):
    def test_cross_rank_maximum_rejects_unpaired_events(self):
        key = [0, 16, 1, "candidate"]
        records = [{"rank": rank, "key": key, "duration_us": value}
                   for rank, value in enumerate((4.0, 9.0))]
        self.assertEqual(benchmark.paired_maximum(records, key), 9.0)
        for altered in (records[::-1], [records[0]],
                        [records[0], {**records[1], "key": [0, 16, 1, "control"]}],
                        [records[0], {**records[1], "duration_us": float("nan")}],
                        [records[0], {**records[1], "duration_us": 0.0}]):
            with self.assertRaises(RuntimeError):
                benchmark.paired_maximum(altered, key)

    def test_paired_maxima_are_not_maximum_of_rank_savings(self):
        candidate = [{"rank": rank, "key": "c", "duration_us": value}
                     for rank, value in enumerate((10.0, 6.0))]
        control = [{"rank": rank, "key": "p", "duration_us": value}
                   for rank, value in enumerate((9.0, 12.0))]
        saving = (benchmark.paired_maximum(control, "p")
                  - benchmark.paired_maximum(candidate, "c"))
        self.assertEqual(saving, 2.0)
        self.assertNotEqual(saving, max(9 - 10, 12 - 6))

    def test_norm_replacement_preserves_entire_attention_ffn_suffix(self):
        old = function(tp_parent, "separate")
        new = function(native, "separate")

        class Names(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == "self":
                    return ast.Name(id="local", ctx=node.ctx)
                if node.id in ("attention_sum", "output"):
                    return ast.Attribute(value=ast.Name(id="case", ctx=ast.Load()),
                                         attr=node.id, ctx=node.ctx)
                return node

        before = old.body[next(i for i, node in enumerate(old.body) if isinstance(node, ast.If)):-1]
        after = new.body[next(i for i, node in enumerate(new.body) if isinstance(node, ast.If)):]
        expected = Names().visit(ast.Module(body=before, type_ignores=[]))
        self.assertEqual(ast.dump(expected), ast.dump(ast.Module(body=after, type_ignores=[])))

    def test_rstd_view_keeps_original_owner_and_norm_schema(self):
        value = torch.empty(12)
        view = value.view(12, 1)
        self.assertEqual(view.data_ptr(), value.data_ptr())
        self.assertEqual(view.untyped_storage().data_ptr(), value.untyped_storage().data_ptr())
        source = inspect.getsource(native.prepare)
        self.assertIn("view(local.rows, 1)", source)
        self.assertIn("local.sync", source)
        arguments = torch.ops.tilert.norm_quant_out.default._schema.arguments
        self.assertEqual(str(next(value.type for value in arguments if value.name == "rstd")), "Tensor")

    def test_preallocated_casts_keep_both_numeric_boundaries(self):
        left = torch.tensor([0.0, -0.0, 1.0, -2.25, 65536.0], dtype=torch.bfloat16)
        right = torch.tensor([-0.0, -0.0, -1.0, 0.125, -32.0], dtype=torch.bfloat16)
        temporary = torch.empty_like(left, dtype=torch.float32)
        output = torch.empty_like(left)
        temporary.copy_(left)
        self.assertTrue(torch.equal(temporary.view(torch.uint8), left.float().view(torch.uint8)))
        temporary.add_(right.float())
        output.copy_(temporary)
        expected = (left.float() + right.float()).bfloat16()
        self.assertTrue(torch.equal(output.view(torch.uint8), expected.view(torch.uint8)))
        source = inspect.getsource(native.Reductions.sum)
        self.assertLess(source.index("temporary.copy_(partial)"), source.index("all_reduce("))
        self.assertLess(source.index("all_reduce("), source.index("result.copy_(temporary)"))
        self.assertNotIn("synchronize", source)
        self.assertNotIn("async_op=True", source)

    def test_timing_derivation_keeps_original_tp_packet_and_window(self):
        old = Path(tp_graph.__file__).read_text()
        new = Path(tp_timing_chain.__file__).read_text()
        self.assertEqual(new, stage_tp_timing.timing_source(old))
        self.assertEqual(ast.dump(function(tp_graph, "TPCase")),
                         ast.dump(function(tp_timing_chain, "TPCase")))
        self.assertEqual(new.count("window = Window(library, 32)"), 1)
        self.assertLess(new.index("reserve_capture_stream(torch)"), new.index("window = Window"))
        self.assertLess(new.index("arenas.append((inputs, cases))"), new.index("measured = run_timing("))
        self.assertLess(new.index("measured = run_timing("), new.index("window.close()"))
        self.assertIn("private_owners[0] & private_owners[1]", new)

    def test_no_observation_tasks_or_external_work_inside_events(self):
        source = inspect.getsource(benchmark.run_timing)
        timed = source[source.index("start.record(stream)"):source.index("end.record(stream)")]
        for forbidden in ("snapshot", "collect(", "gather(", ".copy_", "barrier", "synchronize", "sleep"):
            self.assertNotIn(forbidden, timed)
        self.assertIn("graphs[calls, arm].replay()", timed)
        tree = ast.parse(source)
        pair = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "pair")
        calls = [node for node in ast.walk(pair) if isinstance(node, ast.Call)]
        self.assertEqual(sum(isinstance(node.func, ast.Name) and node.func.id == "publish_topk"
                             for node in calls), 1)
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr == "copy_"
                             for node in calls))

    def test_control_and_candidate_keep_fresh_writes_and_live_inputs(self):
        source = inspect.getsource(benchmark.run_timing)
        for required in ("case.local.body[20].fill_(1)", "case.local.body[20][0].zero_()",
                         '"cache_reset_between_replays": False', "input_bytes[name].view(torch.uint8)",
                         "check_rank_witnesses(peers)", "if len(values) != 113:",
                         "1 if arm == \"control\" else 2"):
            self.assertIn(required, source)
        self.assertLess(source.index('report["idempotence_checks"].append'), source.index("start.record(stream)"))
        self.assertIn("graph.close()", source[source.index("    finally:"):])

    def test_literal_reference_retains_two_actual_collective_calls(self):
        source = inspect.getsource(native.Reductions.literal)
        self.assertIn("temporary = partial.float()", source)
        self.assertIn("narrowed = temporary.bfloat16()", source)
        self.assertEqual(source.count("torch.distributed.all_reduce("), 2)
        self.assertIn('("ordered_fp32_sum", reduced, ordered)', source)
        self.assertIn('witnesses[0]["sha256"] != witnesses[1]["sha256"]', source)

    def test_project_defaults_and_launcher_values_are_explicit(self):
        base = Path(__file__).parent
        source = (base / "timing-protocol/source/project-collective-engine.py").read_text()
        tree = ast.parse(source[:source.index("class CannModel:")])
        choose = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        env = {}
        namespace = {"os": SimpleNamespace(environ=env)}
        exec(compile(ast.Module(body=[choose], type_ignores=[]), "pinned-source", "exec"), namespace)
        namespace["_choose_collective_engine"](True)
        self.assertEqual(check_environment(env)["HCCL_DETERMINISTIC"], "true")
        env["HCCL_OP_EXPANSION_MODE"] = "AI_CPU"
        namespace["_choose_collective_engine"](True)
        self.assertEqual(env["HCCL_OP_EXPANSION_MODE"], "AI_CPU")
        with self.assertRaises(RuntimeError):
            check_environment(env)
        with self.assertRaises(RuntimeError):
            check_environment({})
        self.assertIn("aiv_collectives=True", source)
        launcher = (base / "device_tp_timing_run.sh").read_text()
        self.assertLess(launcher.index("HCCL_OP_EXPANSION_MODE=AIV HCCL_DETERMINISTIC=true"),
                        launcher.index("python3 /decoder/tp_timing_device.py"))
        device = (base / "tp_timing_device.py").read_text()
        self.assertLess(device.index('report["collective_environment_before_torch_import"]'),
                        device.index("        import torch\n"))


if __name__ == "__main__":
    unittest.main()
