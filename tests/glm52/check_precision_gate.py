"""Exercise saved-report validation with synthetic stdlib metadata only.

Fixtures contain no tensors or model results. Passing this checker does
not establish numerical precision, NPU execution, HIXL behavior or TPOT.
"""

import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import precision_gate as gate  # noqa: E402
from prefix_io import (  # noqa: E402
    mtp_prefix_layer, prefix_prompt_identity, prompt_identity,
)
from tilert.models.glm_5_2_ascend.ops.indexer import (  # noqa: E402
    cann_arch22_scoring_contract,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@contextmanager
def changed(path, update):
    original = path.read_bytes()
    value = json.loads(original)
    update(value)
    write_json(path, value)
    try:
        yield
    finally:
        path.write_bytes(original)


def make_case(directory, metadata, sources, prompt):
    prefixes = {f"rank-{rank}.pt": f"{rank + 100:064x}"
                for rank in range(16)}
    scoring = cann_arch22_scoring_contract()
    scoring["source_verified"] = True
    reference = {
        "implementation": "transformers.GlmMoeDsaForCausalLM",
        "status": "pass", "numerical_execution": True, "tp_size": 16,
        "metadata_sha256": metadata, "source_sha256": sources,
        "installed_hf_source_sha256": {"fixture.py": "f" * 64},
        "trace_sha256": {"step-08192.pt": "e" * 64},
        "weight_arithmetic": {"c8_index_scoring": scoring},
        "rms_norm_eps": {"ordinary": 1e-5, "q_a": 1e-5, "kv_a": 1e-5},
        "index_rope_basis": "interleaved", "initial_position": 8192,
        "teacher_tokens": list(range(32)),
        "prefix_files_sha256": prefixes, **prompt_identity(prompt),
    }
    for rank in range(16):
        common = {
            "rank": rank, "world_size": 16, "device": f"npu:{rank}",
            "checks": [{"name": "synthetic_metadata", "passed": True}],
            "metadata_sha256": metadata, "source_sha256": sources,
            "versions": {"torch": "fixture", "torch_npu": "fixture"},
            "timed": False, "profiled": False,
            "expert_backend": "per_expert",
        }
        main = dict(
            common, status="pass" if rank == 0 else "rank_forward_complete",
            implementation="NativeDecodeEngine", initial_position=8192,
            teacher_tokens=list(range(32)), reference_manifest=reference,
            prefix_file_sha256=prefixes[f"rank-{rank}.pt"],
        )
        verify = dict(
            common, status="pass", implementation="NativeDecodeEngine.verify",
            initial_position=8192, teacher_tokens=list(range(4)),
            reference_manifest=reference,
        )
        write_json(directory / "main" / f"rank-{rank}.json", main)
        write_json(directory / "verify" / f"rank-{rank}.json", verify)
        mtp_prefix = f"{rank + 200:064x}"
        for stage, parent, position, width, external in (
            ("mtp_seed", "target_final_norm", 8191, 1, False),
            ("mtp_chain", "mtp_recycle_norm", 8192, 1, True),
            ("mtp_verify", "target_final_norm", 8192, 4, False),
        ):
            tensor = directory / stage / f"rank-{rank}.pt"
            tensor.parent.mkdir(parents=True, exist_ok=True)
            tensor.write_bytes(b"opaque synthetic fixture, not a tensor")
            oracle = {
                "status": "pass", "numerical_execution": True,
                "implementation": "transformers.GlmMoeDsaDecoderLayer.MTP",
                "checkpoint_sha256": metadata, "input_sha256": "d" * 64,
                "source_sha256": sources, "position_convention": "parent",
                "prefix_files_sha256": {f"rank-{rank}.pt": mtp_prefix},
            }
            report = dict(
                common, status="parity", passed=True, parent_kind=parent,
                position=position, width=width, external_indices=external,
                input_sha256="d" * 64, prefix_sha256=mtp_prefix,
                oracle=oracle, trace_sha256=gate.digest(tensor),
            )
            write_json(tensor.with_suffix(".json"), report)
        session = dict(
            common, status="capture_replay", initial_position=8192,
            mtp_width=4, ignore_eos=True, serial_tokens=list(range(16)),
            rounds=[{}, {}], final_rank_states=[[0, 0, 40]] * 16,
            last_prompt_token=prompt[-1],
            prefix_sha256={f"main/rank-{rank}.pt": prefixes[f"rank-{rank}.pt"],
                           f"mtp/rank-{rank}.pt": mtp_prefix},
        )
        write_json(directory / "session" / f"rank-{rank}.json", session)


class PrecisionGateChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.checkpoint = cls.root / "checkpoint"
        for name in gate.CHECKPOINT_FILES:
            write_json(cls.checkpoint / name, {"synthetic": name})
        cls.metadata = {name: gate.digest(cls.checkpoint / name)
                        for name in gate.CHECKPOINT_FILES}
        paths = [*(ROOT / "tilert/models/glm_5_2_ascend").rglob("*.py"),
                 *(ROOT / "tilert/pd_vllm").glob("*.py"),
                 *(ROOT / "tests/glm52").glob("*.py")]
        cls.sources = {str(path.relative_to(ROOT)): gate.digest(path)
                       for path in paths}
        cls.workload = cls.root / "workload"
        for case in range(8):
            make_case(cls.workload / f"case-{case:02d}", cls.metadata,
                      cls.sources, [case + 7] * 8192)
        cls.case = cls.workload / "case-00"

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def validate(self, width=4, backend="per_expert"):
        return gate.validate_precision(self.case, self.checkpoint,
                                       mtp_width=width, expert_backend=backend)

    def reject_change(self, stage, update, rank=0, width=4):
        path = self.case / stage / f"rank-{rank}.json"
        with changed(path, update), self.assertRaises(ValueError):
            self.validate(width)

    def test_m1_binds_original_prompt(self):
        result = self.validate(1)
        self.assertEqual(result["last_prompt_token"], 7)
        self.assertEqual(result["prompt_length"], 8192)
        self.assertEqual(result["session_prefix_sha256"], {})
        self.assertTrue(all(name.startswith("tilert/")
                            for name in result["source_sha256"]))

    def test_m4_binds_paired_prefixes(self):
        result = self.validate()
        self.assertEqual(len(result["prefix_sha256"]), 16)
        self.assertEqual(len(result["session_prefix_sha256"]), 16)

    def test_m2(self):
        with ExitStack() as stack:
            for rank in range(16):
                for stage, field in (("mtp_verify", "width"),
                                     ("session", "mtp_width")):
                    stack.enter_context(changed(
                        self.case / stage / f"rank-{rank}.json",
                        lambda report, field=field: report.update({field: 2}),
                    ))
            self.assertEqual(self.validate(2)["mtp_width"], 2)

    def test_bank_configuration(self):
        with ExitStack() as stack:
            for stage in ("verify", "mtp_seed", "mtp_chain", "mtp_verify",
                          "session"):
                for path in (self.case / stage).glob("*.json"):
                    stack.enter_context(changed(
                        path, lambda report: report.update(
                            expert_backend="bank",
                        ),
                    ))
            self.assertEqual(self.validate(backend="bank")["expert_backend"],
                             "bank")

    def test_missing_rank(self):
        path = self.case / "main/rank-15.json"
        original = path.read_bytes()
        path.unlink()
        try:
            with self.assertRaises(ValueError):
                self.validate()
        finally:
            path.write_bytes(original)

    def test_cpu_or_wrong_rank(self):
        for values in ({"device": "cpu"}, {"world_size": 8}, {"rank": 3}):
            with self.subTest(values=values):
                self.reject_change("main", lambda item: item.update(values))

    def test_failed_or_empty_checks(self):
        for checks in ([], [{"passed": False}], ["not a numerical check"]):
            with self.subTest(checks=checks):
                self.reject_change("verify", lambda item: item.update(
                    checks=checks,
                ))

    def test_wrong_shape_or_teacher(self):
        for values in ({"initial_position": 128}, {"teacher_tokens": [1]},
                       {"teacher_tokens": list(range(1, 33))}):
            with self.subTest(values=values):
                self.reject_change("main", lambda item: item.update(values))

    def test_stale_checkpoint(self):
        self.reject_change("main", lambda item: item["metadata_sha256"].update(
            {"config.json": "0" * 64},
        ))

    def test_stale_math_and_runner_sources(self):
        for name in (gate.MODEL + "native_engine.py",
                     "tests/glm52/run_native_decode.py"):
            with self.subTest(name=name):
                self.reject_change("main", lambda item: item["source_sha256"]
                                   .update({name: "0" * 64}))

    def test_changed_packages(self):
        self.reject_change("verify", lambda item: item["versions"].update(
            torch_npu="another build",
        ))

    def test_profiled_precision(self):
        for field in ("timed", "profiled", "performance_measured"):
            with self.subTest(field=field):
                self.reject_change("main", lambda item: item.update(
                    {field: True},
                ))

    def test_hf_metadata_is_not_numerical(self):
        self.reject_change("main", lambda item: item["reference_manifest"]
                           .update(numerical_execution=False))

    def test_mtp_parent_roles(self):
        for values in ({"parent_kind": "mtp_recycle_norm"}, {"position": 8192},
                       {"width": 4}, {"external_indices": True}):
            with self.subTest(values=values):
                self.reject_change("mtp_seed", lambda item: item.update(
                    values,
                ))

    def test_mtp_oracle_identity(self):
        for values in ({"input_sha256": "a" * 64},
                       {"numerical_execution": False},
                       {"prefix_files_sha256": {}}):
            with self.subTest(values=values):
                self.reject_change("mtp_verify", lambda item: item["oracle"]
                                   .update(values))

    def test_corrupt_device_trace(self):
        path = self.case / "mtp_seed/rank-0.pt"
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        try:
            with self.assertRaises(ValueError):
                self.validate()
        finally:
            path.write_bytes(original)

    def test_session_shape_replay_and_reset(self):
        for values in ({"last_prompt_token": 8}, {"ignore_eos": False},
                       {"rounds": [{}]}, {"serial_tokens": [1]},
                       {"final_rank_states": [[1, 0, 40]] * 16},
                       {"prefix_sha256": {"wrong.pt": "a" * 64}}):
            with self.subTest(values=values):
                self.reject_change("session", lambda item: item.update(values))

    def test_workload_has_eight_original_prompts(self):
        result = gate.validate_workload_precision(
            self.workload, self.checkpoint, mtp_width=4,
        )
        self.assertEqual(len(set(result["prompt_sha256s"])), 8)

    def test_duplicate_workload_prompts(self):
        other = self.workload / "case-01"
        identity = prompt_identity([7] * 8192)
        with ExitStack() as stack:
            for stage in ("main", "verify", "session"):
                for path in (other / stage).glob("*.json"):
                    def update(report, stage=stage):
                        if stage == "session":
                            report["last_prompt_token"] = 7
                        else:
                            report["reference_manifest"].update(identity)
                    stack.enter_context(changed(path, update))
            with self.assertRaisesRegex(ValueError, "eight different"):
                gate.validate_workload_precision(
                    self.workload, self.checkpoint, mtp_width=4,
                )

    def test_canonical_mtp_bundle(self):
        layer = {"latent": "rows", "rope": "rows", "index_keys": "rows"}
        bundle = {"metadata_sha256": self.metadata, "layer": layer}
        self.assertEqual(mtp_prefix_layer(bundle, self.metadata), layer)
        for invalid in (layer, dict(bundle, metadata_sha256={}),
                        dict(bundle, layer=dict(layer, index_scale="scale"))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                mtp_prefix_layer(invalid, self.metadata)

    def test_prompt_identity_and_prefix_length(self):
        saved = {"prompt_token_ids": [1, 2, 3]}
        expected = hashlib.sha256(b"[1,2,3]").hexdigest()
        self.assertEqual(prefix_prompt_identity(saved, 3)["prompt_sha256"],
                         expected)
        for tokens in (None, [], [True], [-1], [154880]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                prompt_identity(tokens)
        with self.assertRaises(ValueError):
            prefix_prompt_identity(saved, 2)

    def test_hf_prefix_checks_prompt_ids_on_all_ranks(self):
        import run_hf_reference as runner

        directory = self.root / "hf-prefix"
        directory.mkdir(exist_ok=True)
        row = SimpleNamespace(shape=(3, 512), dtype="BF16")
        saved = {"metadata_sha256": self.metadata,
                 "layers": [{"latent": row}] * 78,
                 "prompt_token_ids": [1, 2, 3]}
        bundles = {}
        for rank in range(16):
            path = directory / f"rank-{rank}.pt"
            path.write_bytes(b"opaque metadata fixture")
            bundles[path] = deepcopy(saved)
        fake_torch = SimpleNamespace(load=lambda path, **kwargs: bundles[path],
                                     equal=lambda left, right: left == right)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            report = {"metadata_sha256": self.metadata}
            runner.load_prefix(SimpleNamespace(prefix=directory), report)
            self.assertEqual(report["last_prompt_token"], 3)
            bundles[directory / "rank-15.pt"]["prompt_token_ids"] = [1, 2, 4]
            with self.assertRaisesRegex(ValueError, "prompt IDs differ"):
                runner.load_prefix(SimpleNamespace(prefix=directory),
                                   {"metadata_sha256": self.metadata})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        PrecisionGateChecks,
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    torch_imported = "torch" in sys.modules or "torch_npu" in sys.modules
    files = (Path(__file__), Path(gate.__file__),
             Path(__file__).with_name("prefix_io.py"),
             Path(__file__).with_name("run_hf_reference.py"))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Synthetic stdlib report metadata; no numerical execution",
        "status": "pass" if result.wasSuccessful() and not torch_imported
        else "fail", "checks_total": result.testsRun,
        "errors": [message for _, message in result.errors],
        "failures": [message for _, message in result.failures],
        "torch_imported": torch_imported, "python": platform.python_version(),
        "source_sha256": {str(path.relative_to(ROOT)): gate.digest(path)
                          for path in files},
    }
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "report": str(args.output)}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
