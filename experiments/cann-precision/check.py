"""Real CPU tensor checks for the strict entry; no checkpoint or NPU proof."""

import copy
import ctypes
from functools import wraps
from importlib import util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = load("strict_cann_precision", HERE / "run.py")
producer = load("precision_cpu_producer", entry.ROOT / "experiments/graph-evidence/producer_probe.py")


def fixture():
    return producer.scalar_reference(1, 0)


def actual_row():
    return copy.deepcopy(fixture())


def passes(actual, expected=None, *, exact=False):
    checks = []
    entry.strict_row("cpu", actual, fixture() if expected is None else expected,
                     checks, exact=exact)
    return all(item["passed"] for item in checks), checks


class StrictInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_mapped_elf_is_hashed_through_the_path_interface(self):
        source = next(Path(line.split()[-1]) for line in
                      Path("/proc/self/maps").read_text().splitlines()
                      if "/libm.so." in line and Path(line.split()[-1]).is_file())
        with tempfile.TemporaryDirectory(prefix="ascend-mapped-library-") as temporary:
            path = Path(temporary) / source.name
            shutil.copyfile(source, path)
            library = ctypes.CDLL(str(path))
            self.assertIsNotNone(library._handle)
            self.assertEqual(entry.mapped_libraries()[str(path)], entry.digest(path))

    def test_original_comparator_accepts_real_complete_tensors(self):
        result, checks = passes(actual_row())
        self.assertTrue(result)
        self.assertTrue(any(item["name"] == "cpu.ids_vs_reference_logits" for item in checks))
        self.assertEqual(len(actual_row()["hidden_states"]), 79)
        self.assertEqual(len(actual_row()["cache_rows"]), 78)

    def test_independent_numerical_differences_are_hard_failures(self):
        for field in ("logits", "hidden", "cache", "selection", "missing_layer"):
            actual = actual_row()
            if field == "logits":
                actual["logits"] = actual["logits"] + 0.2
            elif field == "hidden":
                actual["hidden_states"][10].add_(100)
            elif field == "cache":
                actual["cache_rows"][6]["index"].add_(1)
            elif field == "selection":
                actual["routes"][3][0] = 250
            else:
                actual["layer_logits"].pop()
            self.assertFalse(passes(actual)[0], field)

    def test_returned_token_ids_cannot_hide_behind_matching_logits(self):
        for ids in (torch.tensor([0]), torch.tensor([16.0]), torch.tensor([16], dtype=torch.int32)):
            actual = actual_row()
            actual["token_ids"] = ids
            self.assertFalse(passes(actual)[0])
        actual = actual_row()
        actual["logits"][0, 0] = float("nan")
        self.assertFalse(passes(actual)[0])

    def test_exact_reset_and_typed_identity(self):
        self.assertTrue(passes(actual_row(), exact=True)[0])
        actual = actual_row()
        actual["hidden_states"][10][0, 0] += 1
        self.assertFalse(passes(actual, exact=True)[0])
        actual = actual_row()
        actual["token"] = True  # Python otherwise equates this with input ID 1.
        self.assertFalse(passes(actual)[0])

    def test_existing_width32_observer_works_through_real_verify_and_packaging(self):
        engine = producer.fixture_engine()
        tokens = list(range(1, 33))
        with producer.cpu_context_adapters(), torch.inference_mode(), producer.PersistentTrace(
            engine, index_dtype=torch.int32, route_dtype=torch.int32,
        ) as observation:
            observation.begin(1)
            result = engine.verify(tokens)
            values = observation.readback(result, tokens, 0)
            wide = entry.wide_trace(values, tokens, 0, engine.position)
            for row in range(32):
                actual = entry.single_row(wide, row, choices=True)
                expected = producer.scalar_reference(tokens[row], row)
                self.assertTrue(passes(actual, expected, exact=True)[0], row)
            owned = wide["hidden_states"][3].clone()
            observation.values["hidden_states.3"].zero_()
            self.assertTrue(torch.equal(wide["hidden_states"][3], owned))
            observation.seen[0] = -1
            with self.assertRaisesRegex(ValueError, "not observed"):
                observation.readback(result, tokens, 0)
        with self.assertRaises(ValueError):
            entry.wide_trace(values, tokens, 0, 31)
        bad = copy.deepcopy(wide)
        bad["cache_rows"][6]["index_scale"] = torch.zeros(32, 2, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "exactly one index scale"):
            entry.single_row(bad, 0, choices=True)
        with self.assertRaises(ValueError):
            producer.CannTrace(engine.model, 32)  # The old guard remains intact.

    def test_actual_verify_keeps_request_dtypes_addresses_and_replay_epochs(self):
        tokens = list(range(1, 33))
        for token_dtype in (torch.int32, torch.int64):
            with self.subTest(token_dtype=token_dtype):
                engine = producer.fixture_engine()
                engine.input_ids = torch.empty(32, dtype=token_dtype)
                with producer.cpu_context_adapters(), torch.inference_mode(), producer.PersistentTrace(
                    engine, index_dtype=torch.int32, route_dtype=torch.int32,
                ) as observation:
                    addresses = observation.addresses()
                    request_addresses = (engine.input_ids.data_ptr(), engine.positions.data_ptr())
                    first = None
                    for epoch, ids in ((1, tokens), (2, list(range(65, 97))), (3, tokens)):
                        if epoch == 3:
                            engine.reset()
                        start = engine.position
                        observation.begin(epoch)
                        self.assertTrue(bool(torch.all(observation.seen == -1)))
                        result = engine.verify(ids)
                        values = observation.readback(result, ids, start)
                        self.assertEqual(observation.values["input_ids"].dtype, token_dtype)
                        self.assertEqual(observation.values["positions"].dtype, torch.int64)
                        self.assertTrue(torch.equal(observation.values["input_ids"], engine.input_ids))
                        self.assertTrue(torch.equal(observation.values["positions"], engine.positions))
                        self.assertTrue(bool(torch.all(observation.seen == epoch)))
                        self.assertEqual(observation.addresses(), addresses)
                        self.assertEqual((engine.input_ids.data_ptr(), engine.positions.data_ptr()),
                                         request_addresses)
                        if epoch == 1:
                            first = values
                        elif epoch == 3:
                            self.assertTrue(all(value is None if first[tag] is None else
                                                torch.equal(value, first[tag])
                                                for tag, value in values.items()))

    def test_request_slots_follow_both_integer_dtypes_and_reject_contract_changes(self):
        for token_dtype in (torch.int32, torch.int64):
            for position_dtype in (torch.int32, torch.int64):
                with self.subTest(token_dtype=token_dtype, position_dtype=position_dtype):
                    engine = producer.fixture_engine()
                    engine.input_ids = torch.empty(1, dtype=token_dtype)
                    engine.positions = torch.empty(1, dtype=position_dtype)
                    observation = producer.PersistentTrace(
                        engine, index_dtype=torch.int32, route_dtype=torch.int32)
                    self.assertEqual(observation.values["input_ids"].shape, (32,))
                    self.assertEqual(observation.values["positions"].shape, (32,))
                    addresses = observation.addresses()
                    engine.input_ids = torch.arange(32, dtype=token_dtype)
                    engine.positions = torch.arange(32, dtype=position_dtype)
                    observation.begin(1)
                    observation._embedding_input(None, (engine.input_ids,))
                    for tag in ("input_ids", "positions"):
                        value = getattr(engine, tag)
                        self.assertEqual(observation.values[tag].dtype, value.dtype)
                        self.assertTrue(torch.equal(observation.values[tag], value))
                        self.assertEqual(int(observation.seen[observation.indices[tag]]), 1)
                    observation.begin(2)
                    for tag in ("input_ids", "positions"):
                        value = getattr(engine, tag)
                        other_dtype = torch.int64 if value.dtype == torch.int32 else torch.int32
                        for wrong in (value[:31], value.to(other_dtype),
                                      torch.empty(32, dtype=value.dtype, device="meta")):
                            with self.assertRaisesRegex(ValueError, "contract differs"):
                                observation.write(tag, wrong)
                            self.assertEqual(int(observation.seen[observation.indices[tag]]), -1)
                    self.assertEqual(observation.addresses(), addresses)

    def test_request_slots_reject_unsupported_engine_buffer_dtypes(self):
        for tag in ("input_ids", "positions"):
            for dtype in (torch.float32, torch.float16, torch.bool, torch.int16):
                with self.subTest(tag=tag, dtype=dtype):
                    engine = producer.fixture_engine()
                    setattr(engine, tag, torch.empty(32, dtype=dtype))
                    with self.assertRaisesRegex(ValueError, "INT32 or INT64"):
                        producer.PersistentTrace(
                            engine, index_dtype=torch.int32, route_dtype=torch.int32)

    def test_teacher_contract_rejects_changed_prompt_and_missing_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prompt = [17] * 8192
            manifest = {"prompt_token_ids": prompt, "seq_len": 8192, "first_token_id": 1}
            (directory / "manifest.json").write_text(json.dumps(manifest))
            import hashlib

            teacher = {"prompt_sha256": hashlib.sha256(
                (json.dumps(prompt) + "\n").encode()).hexdigest(),
                "prompt_tokens": 8192, "first_token_id": 1,
                "teacher_tokens": list(range(1, 33)), "expected_next_tokens": list(range(2, 34))}
            path = directory / "teacher.json"
            path.write_text(json.dumps(teacher))
            self.assertEqual(len(entry.load_teacher(path, directory, 32, 9344)["teacher_tokens"]), 32)
            for change in ({"prompt_sha256": "0" * 64}, {"teacher_tokens": list(range(1, 32))}):
                path.write_text(json.dumps({**teacher, **change}))
                with self.assertRaises(ValueError):
                    entry.load_teacher(path, directory, 32, 9344)

    def test_failed_or_diagnostic_hf_generation_cannot_be_admitted(self):
        for status in ("error", "diagnostic", "running"):
            with self.assertRaisesRegex(ValueError, "generation and exact replay"):
                entry.validate_reference_manifest({
                    "implementation": "transformers.GlmMoeDsaForCausalLM",
                    "status": status,
                }, {}, fp32_head=True)

    def test_capture_only_cannot_certify_precision_even_if_every_check_passes(self):
        for execution in (False, True):
            for hf in (False, True):
                for reference_present in (False, True):
                    result = entry.verdict(True, execution, hf, reference_present)
                    self.assertFalse(result["case_precision_passed"])
                    self.assertFalse(result["precision_verified"])
                    self.assertFalse(result["formal_precision_gate_passed"])
                    self.assertFalse(result["independent_hf_compared"])
                    self.assertEqual(result["entry_passed"], execution)
                    self.assertEqual(result["status"], "captured_unverified" if execution
                                     else "captured_with_failures")
        self.assertFalse(entry.verdict(False, True, True, False)["case_precision_passed"])
        self.assertFalse(entry.verdict(False, True, False, True)["case_precision_passed"])
        self.assertTrue(entry.verdict(False, True, True, True)["case_precision_passed"])

    def test_cli_keeps_reference_mandatory_in_default_mode(self):
        command = [sys.executable, str(HERE / "run.py"), "--checkpoint", "unused",
                   "--capture-case", "unused", "--teacher", "unused", "--output", "unused",
                   "--source-commit", "cpu-interface"]
        for extra in ([], ["--capture-only", "--reference", "unused"]):
            result = subprocess.run(command + extra, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Strict mode requires --reference", result.stderr)

    def test_known_sfa_signatures_preserve_arguments_and_actual_table_values(self):
        engine = producer.fixture_engine()
        observation = producer.PersistentTrace(
            engine, index_dtype=torch.int32, route_dtype=torch.int32)
        observation.begin(1)
        query = torch.tensor([1.0])
        indices = torch.zeros((32, 1, 2048), dtype=torch.int32)
        arguments = (query, object(), object(), indices, object(), object(), object())
        calls = []

        def old(ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
                actual_seq_lengths_query, actual_seq_lengths_key):
            return ql_nope + 1

        def new(ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
                actual_seq_lengths_query, actual_seq_lengths_key, block_table=None):
            return ql_nope + (0 if block_table is None else block_table.sum())

        @wraps(new)
        def recorded(*args, **kwargs):
            calls.append((args, kwargs))
            return new(*args, **kwargs)

        self.assertTrue(torch.equal(observation._attention(0, old)(*arguments), query + 1))
        wrapped = observation._attention(0, recorded)
        self.assertTrue(torch.equal(wrapped(*arguments), query))
        self.assertEqual(calls[-1][1], {})
        self.assertEqual(len(calls[-1][0]), 7)
        table = torch.tensor([2, 4], dtype=torch.int32)
        self.assertTrue(torch.equal(wrapped(*arguments, table), query + 6))
        self.assertIs(calls[-1][0][-1], table)
        self.assertEqual(calls[-1][1], {})
        named = dict(zip(producer.ATTENTION_ARGUMENTS, arguments))
        self.assertTrue(torch.equal(wrapped(**named, block_table=table), query + 6))
        self.assertEqual(calls[-1][0], ())
        self.assertEqual(set(calls[-1][1]), set(named) | {"block_table"})
        self.assertTrue(all(calls[-1][1][key] is value for key, value in named.items()))
        self.assertIs(calls[-1][1]["block_table"], table)
        self.assertTrue(torch.equal(observation.values["indices.0"], indices[:, 0, :]))
        self.assertEqual(int(observation.seen[observation.indices["indices.0"]]), 1)
        count = len(calls)
        with self.assertRaises(TypeError):
            wrapped(*arguments, unexpected=table)
        self.assertEqual(len(calls), count)

    def test_unknown_sfa_signatures_are_rejected_before_any_call(self):
        observation = producer.PersistentTrace(
            producer.fixture_engine(), index_dtype=torch.int32, route_dtype=torch.int32)

        def wrong_name(ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
                       actual_seq_lengths_query, actual_seq_lengths_key, other_table=None):
            raise AssertionError("must not run")

        def wrong_default(ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
                          actual_seq_lengths_query, actual_seq_lengths_key, block_table=0):
            raise AssertionError("must not run")

        def unknown_tail(ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
                         actual_seq_lengths_query, actual_seq_lengths_key,
                         block_table=None, unknown=None):
            raise AssertionError("must not run")

        def unknown_variadic(*args, **kwargs):
            raise AssertionError("must not run")

        for original in (wrong_name, wrong_default, unknown_tail, unknown_variadic):
            with self.assertRaisesRegex(ValueError, "SFA consumer signature differs"):
                observation._attention(0, original)


if __name__ == "__main__":
    unittest.main()
