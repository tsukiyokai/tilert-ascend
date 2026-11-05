"""Check independent head arithmetic with real CPU tensors and raw weights."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

import run_hf_reference as runner
from check_hf_reference import make_trace
from tilert.models.glm_5_2_ascend.reference import HfReference, ReferenceLinear
from tilert.models.glm_5_2_ascend.reference_weights import ReferenceWeights


def head_weight():
    return torch.stack((torch.arange(1, 33, dtype=torch.bfloat16),
                        torch.full((32,), 2**-8, dtype=torch.bfloat16)), dim=1)


def tie_weight():
    weight = torch.full((32, 2), -1.0, dtype=torch.bfloat16)
    weight[0] = torch.tensor([1, 0], dtype=torch.bfloat16)
    weight[17] = torch.tensor([1, 2**-8], dtype=torch.bfloat16)
    return weight


@contextmanager
def checkpoint():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "config.json").write_text(json.dumps({
            "vocab_size": 32, "hidden_size": 2,
        }))
        (root / "quant_model_description.json").write_text(json.dumps({
            "group_size": 0, "lm_head.weight": "FLOAT",
            "model.proj.weight": "FLOAT",
        }))
        (root / "quant_model_weights.safetensors.index.json").write_text(
            json.dumps({"weight_map": {
                "lm_head.weight": "weights.safetensors",
                "model.proj.weight": "weights.safetensors",
            }}),
        )
        save_file({"lm_head.weight": head_weight(),
                   "model.proj.weight": head_weight()},
                  root / "weights.safetensors")
        yield root


def single_layer_trace(logits):
    return {
        "position": 0, "committed_position": 1,
        "logits": logits, "layer_logits": (logits.clone(),),
        "hidden_states": (torch.zeros(1, 2), torch.zeros(1, 2)),
        "indices": (torch.tensor([0]),), "routes": (None,),
        "cache_rows": ({"latent": torch.zeros(1), "rope": torch.zeros(1),
                        "index": torch.zeros(1), "index_scale": None},),
        "index_scores": (torch.zeros(1),),
        "attention_weights": (torch.ones(64, 1),),
    }


class ReferenceHeadChecks(unittest.TestCase):
    def setUp(self):
        self.hidden = torch.tensor([[1, 1], [-1, 1]], dtype=torch.bfloat16)
        self.expected = torch.tensor([
            [row + 2**-8 for row in range(1, 33)],
            [-row + 2**-8 for row in range(1, 33)],
        ], dtype=torch.float32)

    def test_default_head_keeps_the_bf16_rounding_boundary(self):
        with checkpoint() as root:
            weights = ReferenceWeights(root)
            output = ReferenceLinear(weights, "lm_head", "output")(self.hidden)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(output, self.expected.to(torch.bfloat16)))
        self.assertEqual(output[0, 0].item(), 1.0)

    def test_fp32_head_preserves_bits_lost_by_bf16_rounding(self):
        with checkpoint() as root:
            weights = ReferenceWeights(root, fp32_head=True)
            output = ReferenceLinear(weights, "lm_head", "output")(self.hidden)
            self.assertEqual(weights.tensor("lm_head.weight").dtype,
                             torch.bfloat16)
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.equal(output, self.expected))
        self.assertEqual(output[0, 0].item(), 1.00390625)
        self.assertFalse(torch.equal(output, output.bfloat16().float()))

    def test_head_option_preserves_non_head_floating_projections(self):
        with checkpoint() as root:
            ordinary = ReferenceWeights(root)
            precise = ReferenceWeights(root, fp32_head=True)
            baseline = ReferenceLinear(ordinary, "model.proj")(self.hidden)
            actual = ReferenceLinear(precise, "model.proj")(self.hidden)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual, baseline))

    def test_head_mode_can_change_greedy_selection_across_vocab_shards(self):
        with checkpoint() as root:
            save_file({"lm_head.weight": tie_weight(),
                       "model.proj.weight": head_weight()},
                      root / "weights.safetensors")
            selected = []
            for mode in (False, True):
                reader = ReferenceWeights(root, fp32_head=mode)
                head = ReferenceLinear(reader, "lm_head", "output")
                selected.append(head(self.hidden[:1]).argmax(-1).item())
        self.assertEqual(selected, [0, 17])

    def test_quantized_head_cannot_claim_the_fp32_float_contract(self):
        with checkpoint() as root:
            path = root / "quant_model_description.json"
            recipe = json.loads(path.read_text())
            recipe["lm_head.weight"] = "W8A8"
            path.write_text(json.dumps(recipe))
            with self.assertRaisesRegex(ValueError, "unquantized head"):
                ReferenceWeights(root, fp32_head=True)

    def test_fp32_head_requires_the_declared_bf16_input(self):
        with checkpoint() as root:
            weights = ReferenceWeights(root, fp32_head=True)
            head = ReferenceLinear(weights, "lm_head", "output")
            with self.assertRaisesRegex(ValueError, "BF16 hidden input"):
                head(self.hidden.float())

    def test_cli_requires_explicit_fp32_selection(self):
        common = ["run", "--checkpoint", "/fixture", "--output", "/fixture",
                  "--tokens", "1", "--source-commit", "fixture"]
        for flags, expected in (([], False), (["--fp32-head"], True)):
            with patch.object(sys, "argv", common + flags):
                self.assertIs(runner.parse_arguments().fp32_head, expected)

    def test_cli_accepts_an_explicit_c8_source_directory(self):
        common = ["run", "--checkpoint", "/fixture", "--output", "/fixture",
                  "--tokens", "1", "--source-commit", "fixture"]
        for flags, expected in (([], None), (
            ["--c8-source-directory", "/verified/c8"], Path("/verified/c8"),
        )):
            with patch.object(sys, "argv", common + flags):
                self.assertEqual(runner.parse_arguments().c8_source_directory,
                                 expected)

    def test_collector_rejects_rounded_diagnostic_layer_logits(self):
        trace = single_layer_trace(self.expected[:1])
        runner.validate_trace(trace, 1, fp32_head=True)
        trace["layer_logits"] = (trace["layer_logits"][0].bfloat16(),)
        with self.assertRaisesRegex(ValueError, "FP32 final and layer"):
            runner.validate_trace(trace, 1, fp32_head=True)

    def test_collector_checks_all_78_head_outputs(self):
        trace = make_trace()
        trace["logits"] = self.expected[:1]
        trace["layer_logits"] = (self.expected[:1],) * 78
        runner.validate_trace(trace, fp32_head=True)
        for layer in range(78):
            values = list(trace["layer_logits"])
            values[layer] = values[layer].bfloat16()
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                runner.validate_trace({**trace, "layer_logits": tuple(values)},
                                      fp32_head=True)

    def test_source_binding_reads_the_loaded_class_files(self):
        # This head-only fixture checks source binding separately from forward.
        engine = HfReference.__new__(HfReference)
        with checkpoint() as root:
            engine.weights = ReferenceWeights(root, fp32_head=True)
            project = Path(__file__).resolve().parents[2]
            names = ("tilert/models/glm_5_2_ascend/reference.py",
                     "tilert/models/glm_5_2_ascend/reference_weights.py")
            report = {"source_sha256": {
                name: runner.digest(project / name) for name in names
            }}
            runner.bind_reference_sources(engine, report)
            self.assertEqual({name: value["sha256"] for name, value in
                              report["loaded_reference_sources"].items()},
                             report["source_sha256"])
            report["source_sha256"][names[0]] = "0" * 64
            with self.assertRaisesRegex(ValueError, "Loaded HF reference"):
                runner.bind_reference_sources(engine, report)


if __name__ == "__main__":
    unittest.main()
