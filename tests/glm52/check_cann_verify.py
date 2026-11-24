"""Exercise verify-runner helpers with CPU tensors and a scalar fake model.

The fixture replaces only the model capture call. It does not load CANN,
initialize an NPU, or establish device numerical acceptance.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import run_cann_verify as runner


class ScalarEngine:
    """CPU fixture with real cache storage and configurable invalid reads."""

    def __init__(self, fault=None, capacity=32):
        self.position = self.high_watermark = 0
        self.fault = fault
        specifications = []
        for layer in range(78):
            fields = [("latent", 4, torch.bfloat16),
                      ("rope", 2, torch.bfloat16)]
            if layer % 4 == 0:
                fields += [("index_keys", 2, torch.int8),
                           ("index_scales", 1, torch.float16)]
            elif layer % 4 == 1:
                fields += [("index_keys", 2, torch.bfloat16)]
            specifications.append(fields)
        total = sum(
            capacity * width * torch.empty((), dtype=dtype).element_size()
            for fields in specifications for _, width, dtype in fields
        )
        self.storage = torch.zeros(total, dtype=torch.uint8)
        self.cache, offset = [], 0
        for fields in specifications:
            values = {}
            for name, width, dtype in fields:
                item_size = torch.empty((), dtype=dtype).element_size()
                size = capacity * width * item_size
                shape = ((capacity,) if name == "index_scales"
                         else (capacity, width))
                raw = self.storage[offset:offset + size]
                values[name] = raw.view(dtype).view(shape)
                offset += size
            self.cache.append(values)

    def reset(self):
        self.position = 0

    def inject(self, prefix):
        self.position = len(prefix[0]["latent"])
        for source, target in zip(prefix, self.cache):
            for field, values in source.items():
                target[field][:self.position].copy_(values)
        self.high_watermark = max(self.high_watermark, self.position)

    def truncate(self, length):
        self.position = length

    def verify(self, tokens):
        start, width = self.position, len(tokens)
        self.input_ids = torch.tensor(tokens)
        self.positions = torch.arange(start, start + width)
        self.slots = self.positions.to(torch.int32)
        self.query_cpu = torch.tensor([0, width], dtype=torch.int32)
        self.seq_lens = torch.tensor([start + width], dtype=torch.int32)
        hidden = []
        for row, token in enumerate(tokens):
            position = start + row
            previous = int(self.cache[0]["latent"][:position, 0].sum())
            if self.fault == "future":
                previous += sum(tokens[row + 1:])
            if self.fault == "stale" and width == 1:
                tail = self.cache[0]["latent"][
                    position + 1:self.high_watermark, 0,
                ]
                previous += int(torch.count_nonzero(tail) > 0)
            value = (token + previous) % 31 + 1
            hidden.append([value + channel for channel in range(4)])
            for layer in self.cache:
                layer["latent"][position].fill_(value)
                layer["rope"][position].fill_(value / 8)
                if "index_keys" in layer:
                    layer["index_keys"][position].fill_(value % 7)
                if "index_scales" in layer:
                    layer["index_scales"][position] = 0.25 + value % 5 / 16
        self.position += width
        self.high_watermark = max(self.high_watermark, self.position)
        hidden = torch.tensor(hidden, dtype=torch.bfloat16)
        logits = hidden[:, :1].float() + torch.arange(8).float() / 8
        return SimpleNamespace(last_hidden=hidden, logits=logits,
                               token_ids=logits.argmax(-1))


def capture_cpu(engine, tokens, layer_trace):
    position = engine.position
    result = engine.verify(tokens)
    saved = runner.snapshot(engine, result, tokens, position)
    if layer_trace:
        saved["hidden_states"] = tuple(
            saved["last_hidden"].clone() for _ in range(79)
        )
        saved["layer_logits"] = tuple(
            saved["logits"].clone() for _ in range(78)
        )
        indices = torch.full((len(tokens), 2048), -1, dtype=torch.int32)
        for row in range(len(tokens)):
            count = min(position + row + 1, 2048)
            indices[row, :count] = torch.arange(count, dtype=torch.int32)
        saved["indices"] = tuple(indices.clone() for _ in range(78))
        saved["routes"] = (None,) * 3 + tuple(
            torch.stack([(torch.arange(8, dtype=torch.int32)
                          + position + row + layer) % 256
                         for row in range(len(tokens))])
            for layer in range(3, 78)
        )
    return saved


def fixture_run(width, prefix_length=0, fault=None):
    engine = ScalarEngine(fault)
    prefixes = []
    if prefix_length:
        engine.verify(tuple(range(2, 2 + prefix_length)))
        prefixes = [{key: value[:prefix_length].clone()
                     for key, value in layer.items()}
                    for layer in engine.cache]
    tokens = (3, 7, 11, 17)[:width]
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(runner, "capture", capture_cpu):
            result = runner.run_checks(
                engine, prefixes, tokens, Path(directory), True,
                lambda label, operation: operation(),
            )
        artifacts = {
            path.stem: torch.load(path, weights_only=True)
            for path in Path(directory).glob("*.pt")
        }
    return result, artifacts


class VerifyRunnerChecks(unittest.TestCase):
    def test_teacher_defaults_and_capture_first_binding(self):
        self.assertEqual(runner.teacher_tokens(4, None, 6500),
                         (6500, 198, 199, 200))
        with self.assertRaises(ValueError):
            runner.teacher_tokens(2, (1, 2), 6500)

    def test_invalid_teacher_ids_and_width(self):
        for width, tokens in ((1, (1,)), (2, (1,)), (2, (True, 2)),
                              (2, (-1, 2)), (2, (154880, 2))):
            with self.subTest(width=width, tokens=tokens):
                with self.assertRaises(ValueError):
                    runner.teacher_tokens(width, tokens)

    def test_future_variants_cover_each_future_position(self):
        tokens = (3, 7, 11, 154879)
        variants = list(runner.future_variants(tokens))
        self.assertEqual([row for row, _ in variants], [1, 2, 3])
        for row, changed in variants:
            self.assertEqual([index for index in range(4)
                              if tokens[index] != changed[index]], [row])

    def test_rollback_leaves_a_distinct_uncovered_tail(self):
        for width in (2, 4):
            tokens = (3, 7, 11, 17)[:width]
            plan = runner.rollback_plan(8192, tokens)
            self.assertLess(plan["new_end"], plan["old_high_watermark"])
            self.assertNotEqual(plan["replacement"],
                                tokens[plan["retained_rows"]])

    def test_control_flow_passes_for_cold_and_frozen_prefixes(self):
        for width in (2, 4):
            for prefix_length in (0, 3):
                with self.subTest(width=width, prefix=prefix_length):
                    (checks, hashes, plan, _, _), artifacts = fixture_run(
                        width, prefix_length,
                    )
                    self.assertTrue(all(item["passed"] for item in checks),
                                    [item for item in checks
                                     if not item["passed"]][:3])
                    self.assertEqual(len(hashes), width + 5)
                    self.assertEqual(artifacts["wide"]["logits"].shape,
                                     (width, 8))
                    self.assertEqual(len(artifacts["wide"]["hidden_states"]),
                                     79)
                    self.assertEqual(len(artifacts["wide"]["layer_logits"]),
                                     78)
                    self.assertLess(plan["new_end"],
                                    plan["old_high_watermark"])

    def test_stale_high_watermark_fixture_is_rejected(self):
        for width in (2, 4):
            (checks, _, _, _, _), _ = fixture_run(width, 3, "stale")
            failures = [item["name"] for item in checks
                        if not item["passed"]]
            self.assertTrue(any(name.startswith("rollback.same_m1")
                                for name in failures), failures)

    def test_future_read_fixture_is_rejected(self):
        (checks, _, _, _, _), _ = fixture_run(4, 3, "future")
        failures = [item["name"] for item in checks if not item["passed"]]
        self.assertTrue(any(name.startswith("future-") for name in failures))

    def test_cpu_snapshots_do_not_alias_live_cache(self):
        engine = ScalarEngine()
        saved = capture_cpu(engine, (3, 7), True)
        expected = saved["cache_rows"][0]["latent"].clone()
        engine.storage.zero_()
        self.assertTrue(torch.equal(
            saved["cache_rows"][0]["latent"], expected,
        ))
        selected = runner.first_rows(saved, 1)
        selected["cache_rows"][0]["latent"].zero_()
        self.assertTrue(torch.equal(
            saved["cache_rows"][0]["latent"], expected,
        ))

    def test_choices_survive_merge_slice_and_exact_reset(self):
        (checks, _, _, _, _), artifacts = fixture_run(4, 3)
        self.assertTrue(all(item["passed"] for item in checks),
                        [item for item in checks if not item["passed"]][:3])
        wide, serial = artifacts["wide"], artifacts["serial"]
        for field in ("indices", "routes"):
            for actual, expected in zip(wide[field], serial[field]):
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    self.assertTrue(torch.equal(actual, expected))
        row = runner.snapshot_rows(wide, 2, 3)
        self.assertTrue(torch.equal(row["indices"][0][0, :6],
                                    torch.arange(6, dtype=torch.int32)))
        self.assertTrue(torch.equal(row["routes"][3][0],
                                    torch.arange(8, 16, dtype=torch.int32)))
        row["indices"][0].zero_()
        row["routes"][3].zero_()
        self.assertTrue(torch.equal(wide["indices"][0],
                                    serial["indices"][0]))
        self.assertTrue(torch.equal(wide["routes"][3],
                                    serial["routes"][3]))

    def test_metadata_and_wrong_returned_token_are_rejected(self):
        saved = capture_cpu(ScalarEngine(), (3, 7), False)
        for name in saved["model_inputs"]:
            with self.subTest(field=name):
                original = saved["model_inputs"][name].clone()
                saved["model_inputs"][name][0] += 1
                checks = []
                runner.check_record("probe", saved, (3, 7), checks)
                self.assertTrue(any(not item["passed"] for item in checks))
                saved["model_inputs"][name] = original
        saved["token_ids"][0] = 0
        checks = []
        runner.check_record("probe", saved, (3, 7), checks)
        self.assertFalse(checks[-1]["passed"])

    def test_row_slice_preserves_absolute_position_and_ownership(self):
        saved = capture_cpu(ScalarEngine(), (3, 7, 11, 17), True)
        row = runner.snapshot_rows(saved, 2, 3)
        self.assertEqual((row["position"], row["committed_position"]), (2, 3))
        self.assertEqual(row["token"], [11])
        self.assertEqual(row["logits"].shape, (1, 8))
        self.assertEqual(row["cache_rows"][0]["latent"].shape, (1, 4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        VerifyRunnerChecks,
    )
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    import torch_npu

    paths = (Path(__file__), Path(runner.__file__),
             Path(__file__).with_name("run_cann_model.py"),
             Path(__file__).with_name("run_native_decode.py"))
    initialized = bool(torch_npu.npu._initialized)
    report = {
        "status": "pass" if result.wasSuccessful() and not initialized
        else "fail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "npu_initialized": initialized, "device_execution": False,
        "scope": "Real CPU tensors and verify-runner control flow; "
        "scalar fixture replaces model capture",
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "log": output.getvalue(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
