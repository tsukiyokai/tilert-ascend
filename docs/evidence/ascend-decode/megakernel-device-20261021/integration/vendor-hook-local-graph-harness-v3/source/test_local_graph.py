"""CPU ownership and capture-protocol contracts for the local graph fixture."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

import local_chain_graph


class CPUCase:
    """Arithmetic-free owner fixture; no claim of a device kernel implementation."""

    def __init__(self, rows, hidden, residual, source=None):
        self.mode = "int8" if source is None else "shared"
        self.source = source
        self.norm = [hidden, residual, None, None, None, torch.empty(rows)]
        self.bridge = [torch.empty(1) for _ in range(12)]
        self.bridge[10] = torch.empty_like(hidden)
        self.bridge[1] = torch.empty_like(residual)
        self.selected = torch.empty(rows, 1, 2048, dtype=torch.int32)
        self.body = [None] * 20 + [torch.empty(264, dtype=torch.int32)]
        self.values = {"bridge.combined": self.bridge[10],
                       "bridge.retained": self.bridge[1],
                       "body.active_list": self.body[20]}
        self.calls = 0

    def separate(self):
        self.calls += 1
        self.bridge[10].copy_(self.norm[0] + 1)
        self.bridge[1].copy_(self.norm[1] + 2)
        if self.source is None:
            self.selected.fill_(self.calls)
        else:
            self.selected.copy_(self.source[:self.norm[0].shape[0]].view_as(self.selected))

    combined = separate


class GraphContracts(unittest.TestCase):
    def fixture(self):
        rows = 2
        inputs = SimpleNamespace(rows=rows, hidden=torch.zeros(rows, 6144),
                                 residual=torch.zeros(rows, 6144),
                                 positions=torch.arange(rows))
        topk = torch.full((32, 2048), -1, dtype=torch.int32)
        first = CPUCase(rows, inputs.hidden, inputs.residual)
        second = CPUCase(rows, first.bridge[10], first.bridge[1], topk)
        observation = local_chain_graph.Observation(
            torch, (first, second), inputs, topk, {})
        return observation, first, second, topk

    def test_two_layer_dependency_and_topk_copy_precede_successor(self):
        observation, first, second, topk = self.fixture()
        observation.execute(False)
        self.assertIs(second.norm[0], first.bridge[10])
        self.assertIs(second.norm[1], first.bridge[1])
        self.assertTrue(bool((observation.output[0] == 2).all()))
        self.assertTrue(bool((observation.output[1] == 4).all()))
        self.assertTrue(torch.equal(second.selected, topk[:2].view_as(second.selected)))
        self.assertTrue(bool((topk[2:] == -1).all()))

    def test_preallocated_consumers_survive_source_poison(self):
        observation, first, second, topk = self.fixture()
        pointers = [value.data_ptr() for value in observation.output]
        observation.execute(False)
        expected = [value.clone() for value in observation.output]
        second.bridge[10].fill_(float("nan"))
        second.bridge[1].fill_(float("nan"))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(observation.output, expected)))
        observation.poison(2)
        self.assertTrue(all(bool(torch.isnan(value).all()) for value in observation.output))
        observation.execute(False)
        self.assertEqual(pointers, [value.data_ptr() for value in observation.output])
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in observation.output))

    def test_capture_guard_rejects_python_fallback_and_foreign_owner(self):
        observation, _, _, _ = self.fixture()
        observation.torch = SimpleNamespace(npu=SimpleNamespace(
            is_current_stream_capturing=lambda: False))
        inputs = observation.inputs
        with self.assertRaisesRegex(RuntimeError, "substituted"):
            observation(inputs.positions, inputs.hidden.clone(), inputs.residual)
        with self.assertRaisesRegex(RuntimeError, "fell back"):
            observation(inputs.positions, inputs.hidden, inputs.residual)
        self.assertEqual(observation.calls, 0)

    def test_active_table_poison_detects_unwritten_defined_tail(self):
        observation, first, second, _ = self.fixture()
        observation.poison(2)
        for case in (first, second):
            table = case.body[20]
            self.assertEqual(table[0].item(), 0)
            self.assertTrue(bool((table[1:] == 1).all()))
            table[0] = 1
            table[8] = 23
            expected = torch.zeros_like(table)
            expected[0] = 1
            expected[8] = 23
            self.assertFalse(torch.equal(table, expected))

    def test_graph_has_explicit_replay_owner_and_counter_guards(self):
        source = Path(local_chain_graph.__file__).read_text()
        tree = ast.parse(source)
        names = {ast.unparse(node.func) for node in ast.walk(tree)
                 if isinstance(node, ast.Call)}
        self.assertIn("ACLGraphWrapper", names)
        self.assertIn("wrapper", names)
        self.assertNotIn("torch.npu.NPUGraph", names)
        self.assertIn("captured_not_executed", source)
        self.assertIn("observation.calls != calls", source)
        self.assertIn("compilation_counter.num_cudagraph_captured != counter", source)
        self.assertIn("entry.aclgraph.reset()", source)
        self.assertIn("observation.capture_stream =", source.replace(
            "self.capture_stream =", "observation.capture_stream ="))

    def test_eager_component_classes_remain_the_frozen_imports(self):
        import local_chain

        self.assertIs(local_chain_graph.LayerCase, local_chain.LayerCase)
        self.assertIs(local_chain_graph.Inputs, local_chain.Inputs)
        self.assertIs(local_chain_graph.publish_topk, local_chain.publish_topk)


if __name__ == "__main__":
    unittest.main()
