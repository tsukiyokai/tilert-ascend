"""Retain the official graph ownership, replay guards and explicit TopK copy."""

import ast
from pathlib import Path
import unittest


class GraphN1024Contract(unittest.TestCase):
    def test_observation_is_byte_for_byte_the_accepted_graph_protocol(self):
        root = Path(__file__).resolve().parent
        sources = [(root / name).read_text() for name in
                   ("parent-local_chain_graph.py", "local_chain_graph.py")]
        observations = [next(node for node in ast.parse(source).body
                             if isinstance(node, ast.ClassDef)
                             and node.name == "Observation") for source in sources]
        self.assertEqual(ast.dump(observations[0], include_attributes=False),
                         ast.dump(observations[1], include_attributes=False))
        self.assertIn("if number == 6:\n                publish_topk(", sources[1])
        self.assertNotIn("and not combined", sources[1])
        self.assertIn("case.body[20].fill_(1)", sources[1])
        self.assertIn("case.body[20][0].zero_()", sources[1])

    def test_graph_adds_only_tiling_observation_to_accepted_replay(self):
        root = Path(__file__).resolve().parent
        source = (root / "local_chain_graph.py").read_text()
        self.assertEqual(source.count("check_down_tiling(torch, case.body[7])"), 2)
        for token in ("observation.calls != calls", "observation.launch_modes != modes",
                      "compilation_counter.num_cudagraph_captured != counter",
                      "entry.aclgraph.reset()", "down_tiling_before",
                      '"python_calls_during_replay": 0'):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
