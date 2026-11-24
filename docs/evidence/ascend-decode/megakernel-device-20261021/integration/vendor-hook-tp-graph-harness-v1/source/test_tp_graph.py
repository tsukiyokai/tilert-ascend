"""CPU guards for stream ownership, replay entry and frozen TP arithmetic."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

import stage_tp_graph
import tp_chain
import tp_graph
from tp_graph_runtime import CapturedChain


class GraphContracts(unittest.TestCase):
    def owner(self, capturing=True, stream=11):
        value = CapturedChain.__new__(CapturedChain)
        value.inputs = SimpleNamespace(positions=torch.zeros(1, dtype=torch.int64),
                                       hidden=torch.ones(1, 2), residual=torch.ones(1, 2))
        value.stream = SimpleNamespace(npu_stream=11)
        value.window = SimpleNamespace(handle=3)
        value.torch = SimpleNamespace(npu=SimpleNamespace(
            current_stream=lambda device: SimpleNamespace(npu_stream=stream),
            is_current_stream_capturing=lambda: capturing))
        value.output = (torch.empty(1, 2), torch.empty(1, 2))
        value.calls = 0
        value.invocations = []
        value.execute = lambda combined, reductions: value.invocations.append((combined, reductions))
        return value

    def call(self, value):
        return value(value.inputs.positions, value.inputs.hidden, value.inputs.residual)

    def test_actual_runnable_checks_window_stream_before_submission(self):
        value = self.owner()
        self.assertIs(self.call(value), value.output)
        self.assertEqual(value.calls, 1)
        self.assertEqual(value.capture_stream.npu_stream, 11)
        self.assertEqual(value.invocations, [(True, [])])
        wrong = self.owner(stream=12)
        with self.assertRaisesRegex(RuntimeError, "stream"):
            self.call(wrong)
        self.assertEqual(wrong.calls, 0)
        self.assertEqual(wrong.invocations, [])

    def test_released_window_and_python_fallback_are_rejected(self):
        closed = self.owner()
        closed.window.handle = 0
        with self.assertRaisesRegex(RuntimeError, "window"):
            self.call(closed)
        eager = self.owner(capturing=False)
        with self.assertRaisesRegex(RuntimeError, "Python"):
            self.call(eager)
        self.assertEqual(eager.calls, 0)

    def test_capture_cannot_substitute_equal_valued_inputs(self):
        value = self.owner()
        with self.assertRaisesRegex(RuntimeError, "owner"):
            value(value.inputs.positions, value.inputs.hidden.clone(), value.inputs.residual)
        self.assertEqual(value.calls, 0)

    def test_close_resets_graph_before_releasing_entries(self):
        value = self.owner()
        events = []
        value.closed = False
        value.torch.npu.synchronize = lambda: events.append("sync")
        value.wrapper = SimpleNamespace(concrete_aclgraph_entries={
            1: SimpleNamespace(aclgraph=SimpleNamespace(reset=lambda: events.append("reset")))})
        value.close()
        self.assertEqual(events, ["sync", "reset"])
        self.assertEqual(value.wrapper.concrete_aclgraph_entries, {})
        self.assertTrue(value.closed)
        value.close()
        self.assertEqual(events, ["sync", "reset"])
        self.assertEqual(value.window.handle, 3)

    def test_generated_graph_keeps_frozen_eager_helpers(self):
        eager = Path(tp_chain.__file__).read_text()
        graph = Path(tp_graph.__file__).read_text()
        self.assertEqual(graph, stage_tp_graph.graph_source(eager))
        eager_tree, graph_tree = ast.parse(eager), ast.parse(graph)
        for name in ("ordered_sum", "check_rank_witnesses", "TPCase"):
            before = next(node for node in eager_tree.body if getattr(node, "name", None) == name)
            after = next(node for node in graph_tree.body if getattr(node, "name", None) == name)
            self.assertEqual(ast.dump(before), ast.dump(after))

    def test_window_precedes_capture_and_survives_all_widths(self):
        source = inspect.getsource(tp_graph.run_tp_graph)
        self.assertLess(source.index("reserve_capture_stream(torch)"), source.index("Window(library"))
        self.assertLess(source.index("Window(library"), source.index("for rows in"))
        self.assertEqual(source.count("Window(library"), 1)
        self.assertLess(source.index("current_graph.close()"), source.index("window.close()"))
        self.assertIn("capture_owner", source)
        self.assertNotIn("window.address).zero_", source)

    def test_reference_remains_eager_with_both_real_hccl_calls(self):
        source = inspect.getsource(tp_graph.run_tp_graph)
        self.assertIn("torch.distributed.all_reduce(", source)
        self.assertIn("execute(False, reductions)", source)
        self.assertIn("current_graph.replay()", source)
        self.assertNotIn("execute(True, [])", source)
        self.assertIn("if rank == iteration % 2:", source)
        self.assertIn("time.sleep(0.01)", source)
        self.assertIn("if len(values) != 113:", source)

    def test_capture_and_replay_enforce_fresh_execution_separately(self):
        capture = inspect.getsource(CapturedChain.capture)
        replay = inspect.getsource(CapturedChain.replay)
        self.assertIn("before = self.snapshot()", capture)
        self.assertIn("after = self.snapshot()", capture)
        self.assertIn("self.torch.uint8", capture)
        self.assertIn("Capture executed work", capture)
        self.assertIn("self.calls != 1", replay)
        self.assertIn("num_cudagraph_captured != before", replay)
        self.assertIn("self.check_stream()", replay)


if __name__ == "__main__":
    unittest.main()
