"""Exercise the original runner's host protocol without constructing NPU weights.

Registry entries use the real Ascend runner class with only the fields read by
these host methods. This suite does not validate its constructor or collective.
"""

from contextlib import ExitStack
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.runner import moe_runner
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffn_host_adapter import LayerFFNAdapter, PendingFFN, Scratch, lookup_runner


def registry_entry(name):
    runner = object.__new__(AscendMoERunner)
    torch.nn.Module.__init__(runner)
    runner.layer_name = name
    runner.moe_config = SimpleNamespace(is_sequence_parallel=False,
                                        tp_size=16, ep_size=1)
    runner.routed_output_transform = None
    runner.router = torch.nn.Identity()
    return runner


def context_for(runners, ordered=True):
    return ForwardContext(
        no_compile_layers={value.layer_name: value for value in runners},
        attn_metadata={}, slot_mapping={},
        all_moe_layers=[value.layer_name for value in runners] if ordered else None)


class RunnerLookupContracts(unittest.TestCase):
    def setUp(self):
        self.runners = [registry_entry(f"model.layers.{index}.mlp.experts")
                        for index in (6, 7)]

    def recorded_lookup(self, runner, action):
        events = []
        encode = runner._encode_layer_name
        resolve = moe_runner._resolve_layer_name
        lookup = moe_runner.get_layer_from_name

        def traced(name, function):
            def call(*args):
                events.append(name)
                return function(*args)
            return call

        with (patch.object(runner, "_encode_layer_name", traced("encode", encode)),
              patch.object(moe_runner, "_resolve_layer_name", traced("resolve", resolve)),
              patch.object(moe_runner, "get_layer_from_name", traced("lookup", lookup))):
            result = action()
        self.assertEqual(events, ["encode", "resolve", "lookup"])
        return result

    def compare_original_entry(self, use_layername, ordered):
        for original_entry in (moe_runner._moe_forward, moe_runner._moe_forward_shared):
            with self.subTest(entry=original_entry.__name__, ordered=ordered):
                original_context = context_for(self.runners, ordered)
                adapter_context = context_for(self.runners, ordered)
                with patch.object(moe_runner, "_USE_LAYERNAME", use_layername):
                    for runner in self.runners:
                        marker = object()
                        runner._forward_impl = MethodType(lambda self, *args: marker, runner)
                        with override_forward_context(original_context):
                            returned = self.recorded_lookup(runner, lambda: original_entry(
                                None, None, None, None, runner._encode_layer_name(), 0))
                            self.assertIs(returned, marker)
                        with override_forward_context(adapter_context):
                            resolved = self.recorded_lookup(runner, lambda: lookup_runner(runner))
                            self.assertIs(resolved, runner)
                        self.assertEqual(original_context.moe_layer_index,
                                         adapter_context.moe_layer_index)
                    expected = 2 if not use_layername and ordered else 0
                    self.assertEqual(adapter_context.moe_layer_index, expected)

    def test_actual_torch212_opaque_branch_matches_both_original_entries(self):
        self.assertTrue(moe_runner._USE_LAYERNAME)
        self.compare_original_entry(True, True)

    def test_legacy_ordered_branch_advances_exactly_once_per_layer(self):
        self.compare_original_entry(False, True)

    def test_legacy_explicit_name_does_not_advance_cursor(self):
        self.compare_original_entry(False, False)

    def test_lookup_refuses_a_different_registered_runner(self):
        context = context_for(list(reversed(self.runners)))
        with (patch.object(moe_runner, "_USE_LAYERNAME", False),
              override_forward_context(context)):
            with self.assertRaisesRegex(RuntimeError, "another runner"):
                lookup_runner(self.runners[0])
        self.assertEqual(context.moe_layer_index, 1)

    def test_legacy_duplicate_lookup_exhausts_the_original_registry(self):
        context = context_for(self.runners[:1])
        with (patch.object(moe_runner, "_USE_LAYERNAME", False),
              override_forward_context(context)):
            self.assertIs(lookup_runner(self.runners[0]), self.runners[0])
            with self.assertRaises(AssertionError):
                lookup_runner(self.runners[0])
        self.assertEqual(context.moe_layer_index, 1)

    def test_mlp_retains_one_original_reduction_and_zero_expert_helper(self):
        for use_layername in (True, False):
            with self.subTest(use_layername=use_layername), ExitStack() as stack:
                runner = self.runners[0]
                context = context_for(self.runners[:1])
                adapter = object.__new__(LayerFFNAdapter)
                adapter.runner = runner
                adapter.pending = PendingFFN()
                adapter.hits = []
                bridge = [torch.full((2, 6144), i, dtype=torch.bfloat16)
                          for i in range(12)]
                owner = Scratch(bridge, [], 17)
                normalized, _ = adapter.pending.begin(owner, (0, 19, 2))
                calls = []

                def collective_cut(value, name):
                    calls.append((value, name))
                    return value

                stack.enter_context(patch.object(moe_runner, "_USE_LAYERNAME", use_layername))
                stack.enter_context(override_forward_context(context))
                _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
                stack.enter_context(patch("ffn_host_adapter.stream_state",
                                          return_value=(0, 19, 2, False)))
                stack.enter_context(patch.object(torch.ops.vllm,
                    "maybe_all_reduce_tensor_model_parallel", collective_cut, create=True))
                reduce = stack.enter_context(patch.object(runner,
                    "_maybe_reduce_final_output", wraps=runner._maybe_reduce_final_output))
                zero = stack.enter_context(patch.object(runner,
                    "_maybe_add_zero_expert_output", wraps=runner._maybe_add_zero_expert_output))
                result = self.recorded_lookup(runner, lambda: adapter.mlp(None, normalized))
                self.assertIs(result, bridge[10])
                self.assertEqual(len(calls), 1)
                self.assertIs(calls[0][0], bridge[10])
                self.assertEqual(calls[0][1], runner.layer_name)
                reduce.assert_called_once_with(bridge[10], None, False)
                zero.assert_called_once_with(bridge[10])
                adapter.pending.require_empty()
                self.assertEqual(len(adapter.hits), 1)
                self.assertTrue(adapter.hits[0]["lookup_same_runner"])
                self.assertEqual(adapter.hits[0]["moe_layer_index_before"], 0)
                self.assertEqual(adapter.hits[0]["moe_layer_index_after"], int(not use_layername))


if __name__ == "__main__":
    unittest.main()
