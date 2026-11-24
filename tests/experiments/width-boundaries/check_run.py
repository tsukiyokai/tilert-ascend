"""CPU control-flow checks of the actual width-boundary runner and observer."""

import importlib.util
import json
import os
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OBSERVER = load("tested_width_observer", HERE / "observe.py")
with patch.dict(sys.modules, {"observe": OBSERVER}):
    RUN = load("tested_width_runner", HERE / "run.py")


class Sfa:
    scale = 0.0625

    def __init__(self):
        self.W_UV = torch.ones(2, 4, dtype=torch.bfloat16)
        self.calls = []

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table=None,
    ):
        self.calls.append(
            (
                ql_nope,
                q_pe,
                kv_cache,
                topk_indices,
                attn_metadata,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                block_table,
            )
        )
        return topk_indices.float().sum(-1).reshape(-1, 1).expand(-1, 4).bfloat16()


class Norm(torch.nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states.clone(), residual.clone()


class Routed:
    def _select_experts(self, hidden_states, router_logits, force, input_ids=None):
        rows = hidden_states.shape[0]
        return torch.ones(rows, 8), torch.arange(8).expand(rows, -1)


class Attention(torch.nn.Module):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.mla_attn = types.SimpleNamespace(
            mla_attn=types.SimpleNamespace(impl=engine.impl)
        )

    def forward(self, hidden_states):
        engine = self.engine
        output = engine.impl._execute_sparse_flash_attention_process(
            hidden_states,
            engine.qpe,
            engine.kv,
            engine.indices,
            engine.attn_metadata,
            engine.query_cpu[1:],
            engine.seq_lens,
            block_table=engine.table,
        )
        return engine._tp_sum_original(output.float()).bfloat16()


class Mlp(torch.nn.Module):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.experts = types.SimpleNamespace(routed_experts=Routed())

    def forward(self, hidden_states):
        self.experts.routed_experts._select_experts(
            hidden_states, torch.ones(hidden_states.shape[0], 256), False
        )
        return self.engine._tp_sum_original(hidden_states.float()).bfloat16()


class Layer(torch.nn.Module):
    def __init__(self, engine):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = Norm(), Norm()
        self.self_attn, self.mlp = Attention(engine), Mlp(engine)

    def forward(self, hidden_states, residual, positions):
        normalized, retained = self.input_layernorm(hidden_states, residual)
        attention = self.self_attn(normalized)
        normalized, retained = self.post_attention_layernorm(attention, retained)
        return self.mlp(normalized), retained


class Engine:
    """Small CPU math fixture; the real runner/observer own all control flow."""

    def __init__(self, corrupt=None):
        self.position, self.corrupt = 8192, corrupt
        self.impl = Sfa()
        self.layer, self.next_norm = Layer(self), Norm()
        self.model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                layers=[None] * 68
                + [self.layer, types.SimpleNamespace(input_layernorm=self.next_norm)]
            )
        )
        self.table = torch.tensor([[0, 1]], dtype=torch.int32)
        self.attn_metadata = types.SimpleNamespace(block_table=self.table)
        self.kv = (
            torch.ones(2, 4, dtype=torch.bfloat16),
            torch.zeros(2, 2, dtype=torch.bfloat16),
        )
        self.qpe = torch.ones(1, 2, dtype=torch.bfloat16)

    def _tp_sum_original(self, value):
        return value.clone()

    def reset(self):
        self.position = 0

    def inject(self):
        self.position = 8192

    def verify(self, tokens, *, full_logits):
        if not full_logits:
            raise AssertionError("Runner must request complete logits")
        width, initial = len(tokens), self.position
        self.input_ids = torch.tensor(tokens, dtype=torch.int32)
        self.positions = torch.arange(initial, initial + width)
        self.slots = self.positions.clone()
        self.query_cpu = torch.tensor([0, width], dtype=torch.int32)
        self.seq_lens = torch.tensor([initial + width], dtype=torch.int32)
        self.indices = (
            torch.tensor([5, 2, 4, 3], dtype=torch.int32)
            .reshape(1, 1, 4)
            .repeat(width, 1, 1)
        )
        self.original_indices = self.indices.clone()
        hidden, retained = self.layer(
            torch.ones(width, 4, dtype=torch.bfloat16),
            torch.zeros(width, 4, dtype=torch.bfloat16),
            self.positions,
        )
        hidden, _ = self.next_norm(hidden, retained)
        logits = torch.arange(6, dtype=torch.float32).expand(width, -1).clone()
        ids = logits.argmax(-1)
        self.position += width
        if self.corrupt in ("input_ids", "positions", "slots", "query_cpu", "seq_lens"):
            getattr(self, self.corrupt).flatten()[0] += 1
        elif self.corrupt == "committed_position":
            self.position += 1
        elif self.corrupt == "logits":
            logits[0, 0] = float("nan")
        elif self.corrupt == "hidden":
            hidden[0, 0] = float("nan")
        elif self.corrupt == "returned_id":
            ids[0] = 0
        return types.SimpleNamespace(logits=logits, last_hidden=hidden, token_ids=ids)


@contextmanager
def capture_mlp(engine, layer):
    if layer != 68:
        raise AssertionError("Unexpected observed layer")
    stages = {}
    module = engine.model.model.layers[layer].mlp
    before = module.register_forward_pre_hook(
        lambda target, inputs: stages.update(input=inputs[0].detach().clone())
    )
    after = module.register_forward_hook(
        lambda target, inputs, output: stages.update(output=output.detach().clone())
    )
    try:
        yield stages
    finally:
        before.remove()
        after.remove()


def install_routes(layer, samples):
    routed = layer.mlp.experts.routed_experts
    original = routed._select_experts

    def observed(*args, **kwargs):
        weights, ids = original(*args, **kwargs)
        samples.append(
            {
                "ids": ids.clone(),
                "weights": weights.clone(),
                "router_logits": args[1].clone(),
                "metadata": {"fake_cpu": True},
            }
        )
        return weights, ids

    replacement = patch.object(routed, "_select_experts", observed)
    replacement.start()
    return replacement.stop


class RunnerChecks(unittest.TestCase):
    def setUp(self):
        self.npu = patch.object(
            torch, "npu", types.SimpleNamespace(synchronize=lambda: None), create=True
        )
        self.npu.start()
        self.addCleanup(self.npu.stop)
        self.dependencies = patch.dict(
            sys.modules,
            {
                "run_cann_model": types.SimpleNamespace(capture_mlp=capture_mlp),
                "routes": types.SimpleNamespace(install_routes=install_routes),
            },
        )
        self.dependencies.start()
        self.addCleanup(self.dependencies.stop)

    def test_transformed_preserves_full_api_and_input_storage(self):
        for mode in RUN.MODES:
            for style in ("positional", "named_table", "all_named"):
                with self.subTest(mode=mode, style=style):
                    impl = Sfa()
                    source = torch.tensor([[[5, 2, 4, 3]]], dtype=torch.int32)
                    before = source.clone()
                    args = [
                        object(),
                        object(),
                        (object(), object()),
                        source,
                        object(),
                        object(),
                        object(),
                    ]
                    table = object()
                    with RUN.transformed(impl, mode, None):
                        if style == "all_named":
                            names = (
                                "ql_nope",
                                "q_pe",
                                "kv_cache",
                                "topk_indices",
                                "attn_metadata",
                                "actual_seq_lengths_query",
                                "actual_seq_lengths_key",
                            )
                            impl._execute_sparse_flash_attention_process(
                                **dict(zip(names, args, strict=True)), block_table=table
                            )
                        elif style == "named_table":
                            impl._execute_sparse_flash_attention_process(
                                *args, block_table=table
                            )
                        else:
                            impl._execute_sparse_flash_attention_process(*args, table)
                    actual = impl.calls[-1]
                    self.assertTrue(torch.equal(source, before))
                    self.assertTrue(
                        all(
                            actual[index] is args[index] for index in (0, 1, 2, 4, 5, 6)
                        )
                    )
                    self.assertIs(actual[-1], table)
                    self.assertNotIn(
                        "_execute_sparse_flash_attention_process", vars(impl)
                    )
                    if mode == "original":
                        self.assertIs(actual[3], source)
                    else:
                        self.assertNotEqual(actual[3].data_ptr(), source.data_ptr())
                        expected = (
                            before
                            if mode == "copy"
                            else torch.tensor([[[2, 3, 4, 5]]], dtype=torch.int32)
                        )
                        self.assertTrue(torch.equal(actual[3], expected))

    def test_actual_forward_observer_sees_transformed_indices_at_both_widths(self):
        for width in (1, 32):
            controls = []
            for mode in RUN.MODES:
                engine = Engine()
                actual = RUN.forward(engine, list(range(width)), mode, capture=True)
                observed = actual["observed"]["tensors"][
                    "layer68.sfa.call0.topk_indices"
                ]
                expected = (
                    engine.original_indices
                    if mode != "sorted"
                    else engine.original_indices.sort(-1).values
                )
                self.assertTrue(torch.equal(observed, expected))
                self.assertTrue(torch.equal(engine.indices, engine.original_indices))
                if mode != "original":
                    self.assertTrue(
                        torch.equal(
                            actual["observed"]["tensors"][
                                "intervention.original_indices"
                            ],
                            engine.original_indices,
                        )
                    )
                self.assertEqual(engine.position, 8192 + width)
                self.assertNotIn("_tp_sum_original", vars(engine))
                controls.append(actual["outputs"])
            self.assertTrue(RUN.exact(controls[0], controls[1]))
            self.assertTrue(RUN.exact(controls[0], controls[2]))

    def test_actual_forward_rejects_each_corrupted_input_or_committed_position(self):
        for field in (
            "input_ids",
            "positions",
            "slots",
            "query_cpu",
            "seq_lens",
            "committed_position",
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "model inputs/committed position"),
            ):
                RUN.forward(Engine(field), [24187], "original", capture=False)

    def test_actual_forward_rejects_nonfinite_output_or_wrong_returned_id(self):
        for field in ("logits", "hidden", "returned_id"):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "predictions/logits"),
            ):
                RUN.forward(Engine(field), [24187], "original", capture=False)

    def test_coverage_exposes_incomplete_moe_fixture(self):
        actual = RUN.forward(Engine(), [24187], "original", capture=True)
        coverage = actual["observed"]["coverage"]
        self.assertTrue(coverage["modules_once"])
        self.assertTrue(coverage["actual_fp32_tp_both_phases"])
        self.assertTrue(coverage["one_actual_router"])
        self.assertFalse(coverage["complete_moe_stages"])

    def test_reference_control_is_bitwise_not_tolerance_or_argmax_only(self):
        actual = RUN.forward(Engine(), [24187], "original", capture=False)["outputs"]
        expected = RUN.cpu(actual)
        self.assertTrue(RUN.output_matches(actual, expected))
        expected["logits"][0, 0] = 1e-7
        self.assertFalse(RUN.output_matches(actual, expected))
        self.assertTrue(torch.equal(actual["token_ids"], expected["token_ids"]))
        self.assertFalse(
            RUN.exact({"x": torch.tensor([0.0])}, {"x": torch.tensor([-0.0])})
        )
        self.assertFalse(
            RUN.exact({"x": torch.ones(1)}, {"x": torch.ones(1, dtype=torch.bfloat16)})
        )
        self.assertFalse(RUN.exact({"x": torch.ones(1)}, {}))

    def test_actual_reset_frames_compare_all_recorded_tensor_leaves(self):
        for mode in RUN.MODES:
            engine = Engine()
            wide = RUN.forward(engine, list(range(32)), mode, capture=True)
            engine.reset()
            engine.inject()
            reset = RUN.forward(engine, list(range(32)), mode, capture=True)
            self.assertTrue(RUN.exact(reset, wide))
            reset["observed"]["moe_stages"]["output"][0, 0] += 1
            self.assertFalse(RUN.exact(reset, wide))

    def test_global_state_excludes_local_tp_partials_but_keeps_global_and_routes(self):
        frame = RUN.forward(Engine(), [24187], "original", capture=True)
        original = RUN.global_state(frame)
        self.assertTrue(original)
        for key, value in frame["observed"]["tensors"].items():
            if key.startswith("layer68.tp."):
                value.add_(1)
        self.assertEqual(original, RUN.global_state(frame))
        frame["observed"]["tensors"]["layer69.input_norm.call0.out.1"][0, 0] += 1
        self.assertNotEqual(original, RUN.global_state(frame))
        changed = RUN.global_state(frame)
        frame["observed"]["routes"][0]["weights"][0, 0] += 1
        self.assertNotEqual(changed, RUN.global_state(frame))


if __name__ == "__main__":
    torch.set_num_threads(2)
    print(
        json.dumps(
            {
                "scope": "CPU fixtures, no NPU execution",
                "torch": torch.__version__,
                "run_sha256": RUN.digest(HERE / "run.py"),
                "observe_sha256": RUN.digest(HERE / "observe.py"),
            }
        )
    )
    unittest.main(verbosity=2)
