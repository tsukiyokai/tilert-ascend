"""Exercise resident pairing decisions with stdlib CPU collaborators only."""

import argparse
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from importlib import util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = util.spec_from_file_location("resident_controlflow", HERE / "run_resident.py")
resident = util.module_from_spec(SPEC)
SPEC.loader.exec_module(resident)


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


class Tensor:
    def __init__(self, values=(7,)):
        self.values = tuple(values)

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.values)

    def cpu(self):
        return self

    def argmax(self, _dimension):
        return Tensor()

    def tolist(self):
        return list(self.values)


class Bank:
    def __init__(self, owner):
        self.owner = owner
        self.shape, self.dtype = (78, 4, 8192, 256), "torch.bfloat16"
        self.device = SimpleNamespace(type="cpu")
        self.raw = b"CPU bank tensor bytes"

    def detach(self):
        return self

    def contiguous(self):
        return self

    def view(self, _dtype):
        return self

    def reshape(self, *_shape):
        return self

    def numpy(self):
        return self.raw

    def __getitem__(self, layer):
        owner = self.owner

        def move(device):
            assert device == "CPU mock device"
            owner.bank_transfers.append(layer)
            return SimpleNamespace(layer=layer)

        return SimpleNamespace(to=move)


class Model:
    scenario = None

    def __init__(self, *, fp32_head=False):
        self.owner = type(self).scenario
        self.owner.engine = self
        self.rank, self.position, self.graph = 0, 0, None
        self.device = "CPU mock device"
        self.fp32_head = fp32_head
        self.storage = SimpleNamespace(zero_=lambda: None)
        layers = [SimpleNamespace(
            self_attn=SimpleNamespace(mla_attn=SimpleNamespace(
                mla_attn=SimpleNamespace(impl=SimpleNamespace(
                    expanded=False, cached=False, owner=self.owner, number=number,
                )),
            )), route_samples=None,
        ) for number in range(78)]
        self.model = SimpleNamespace(model=SimpleNamespace(layers=layers))

    def source_metadata(self):
        return {str(Path(__file__).resolve()): resident.digest(__file__)}

    def reset(self):
        self.position = 0

    def inject(self, _prefixes):
        self.position = 8192

    def shutdown(self):
        self.owner.shutdowns += 1
        assert all(layer.route_samples is None for layer in self.model.model.layers)
        assert all(not layer.self_attn.mla_attn.mla_attn.impl.expanded
                   for layer in self.model.model.layers)
        assert all(not layer.self_attn.mla_attn.mla_attn.impl.cached
                   for layer in self.model.model.layers)


class Scenario:
    def __init__(self, directory, local_status="diagnostic",
                 global_status="diagnostic", failed_rank=None,
                 raise_before_report=False, control_attention="original",
                 cached=False, bank_error=None):
        self.directory = directory
        self.local_status, self.global_status = local_status, global_status
        self.failed_rank, self.raise_before_report = failed_rank, raise_before_report
        self.barriers, self.shutdowns, self.route_restores = 0, 0, 0
        self.attention_installs, self.attention_restores = 0, 0
        self.forwards = []
        self.frame_recipes = []
        self.control_attention, self.cached = control_attention, cached
        self.cached_installs, self.cached_restores, self.bank_transfers = 0, 0, []
        self.inputs = {"initial_position": 8192, "prefix_sha256": "a" * 64}
        self.bank = Bank(self)
        self.matmul = SimpleNamespace(allow_hf32=True)
        self.args = SimpleNamespace(
            output=directory / "control", reference=directory / "hf",
            capture_case=directory / "capture", checkpoint=directory / "checkpoint",
            steps=1, layer_trace=True, tp_reduction="default", device_offset=0,
            trace_mlp_layer=6,
        )
        self.args.reference.mkdir()
        trace = self.args.reference / "step-08192.pt"
        trace.write_bytes(b"CPU oracle placeholder\n")
        (self.args.reference / "manifest.json").write_text(json.dumps({
            "trace_sha256": {trace.name: resident.digest(trace)},
        }))
        self.cached_helper = directory / "cached_attention.py"
        self.cached_helper.write_text(
            "def install(impl, prefix_values):\n"
            "    return impl.owner.install_cached_attention(impl, prefix_values)\n"
        )
        self.helper_sha = resident.digest(self.cached_helper)
        self.bank_directory = directory / "bank"
        if cached:
            self.prepare_bank(bank_error)
        self.legacy = module(
            "run_cann_model", parse_arguments=lambda: self.args,
            validate_reference_manifest=self.validate_reference,
            capture_forward=self.forward, capture_step=self.capture_step,
            execute=self.execute, load_capture=self.load_capture,
            digest=resident.digest, check_prefix=self.check_prefix,
            compare_step=self.compare_step,
        )

    def validate_reference(self, _reference, _metadata, *, fp32_head=False):
        assert fp32_head

    def load_capture(self, *_args):
        return [{} for _ in range(78)], [], dict(self.inputs)

    def prepare_bank(self, error):
        self.bank_directory.mkdir()
        path = self.bank_directory / "rank-0.pt"
        path.write_bytes(b"CPU bank file")
        receipt = {"rank": 0, "checkpoint": {"metadata_sha256": {}},
                   "prefix": dict(self.inputs), "shape": [78, 4, 8192, 256],
                   "dtype": "torch.bfloat16", "layers": [{"layer": n} for n in range(78)],
                   "file_sha256": resident.digest(path),
                   "tensor_sha256": hashlib.sha256(self.bank.raw).hexdigest()}
        if error == "checkpoint":
            receipt["checkpoint"]["metadata_sha256"] = {"wrong": "checkpoint"}
        if error == "prefix":
            receipt["prefix"]["prefix_sha256"] = "b" * 64
        if error == "rank":
            receipt["rank"] = 1
        receipt_path = self.bank_directory / "rank-0.json"
        receipt_path.write_text(json.dumps(receipt))
        summary = {"schema": "cpu-prefix-values-v1", "status": "prepared",
                   "layers": list(range(78)), "requested_ranks": list(range(16)),
                   "checkpoint": receipt["checkpoint"],
                   "ranks": [{"rank": rank, "receipt_sha256": resident.digest(receipt_path),
                              "file_sha256": receipt["file_sha256"],
                              "tensor_sha256": receipt["tensor_sha256"]}
                             for rank in range(16)]}
        summary_path = self.bank_directory / "summary.json"
        summary_path.write_text(json.dumps(summary))
        self.bank_summary_sha = resident.digest(summary_path)
        if error == "file":
            path.write_bytes(b"changed bank file")
        elif error == "receipt":
            receipt_path.write_text(receipt_path.read_text() + " ")
        elif error == "summary":
            summary_path.write_text(summary_path.read_text() + " ")
        elif error == "shape":
            self.bank.shape = (77, 4, 8192, 256)
        elif error == "dtype":
            self.bank.dtype = "torch.float32"
        elif error == "tensor":
            self.bank.raw = b"changed tensor bytes"

    def forward(self, engine, _token, _position, _layer_trace, mlp_layer=None):
        assert mlp_layer == 6
        expanded = engine.model.model.layers[0].self_attn.mla_attn.mla_attn.impl.expanded
        cached = engine.model.model.layers[0].self_attn.mla_attn.mla_attn.impl.cached
        self.forwards.append("cached_prefix" if cached else "expanded" if expanded else "control")
        for layer in engine.model.model.layers[3:]:
            layer.route_samples.append({
                "ids": Tensor(range(8)), "weights": Tensor(range(8)),
                "router_logits": Tensor(range(256)), "metadata": {},
            })
        engine.position = 8193
        return {"token_ids": Tensor(), "logits": Tensor(),
                "choice_source_sha256": {}}

    def capture_step(self, engine, token, position, layer_trace,
                     tp_reduction="default", mlp_layer=None):
        assert tp_reduction == "default"
        actual = self.legacy.capture_forward(
            engine, token, position, layer_trace, mlp_layer,
        )
        self.frame_recipes.append(actual["attention_recipe"])
        return actual

    def check_prefix(self, _engine, _prefixes, checks, label):
        checks.append({"name": label + ".prefix", "passed": True})

    def compare_step(self, name, _actual, _expected, checks, **_kwargs):
        checks.append({"name": name, "passed": True})

    def compare_tensor(self, name, actual, expected, checks, **_kwargs):
        checks.append({"name": name, "passed": actual.values == expected.values})

    def execute(self, args):
        engine = Model()
        self.legacy.validate_reference_manifest({}, {})
        if self.raise_before_report:
            raise RuntimeError("control failed before its report")
        for _ in range(2):
            engine.reset()
            engine.inject([])
            self.legacy.capture_step(engine, 7, 8192, True, mlp_layer=6)
        (args.output / "rank-0.json").write_text(json.dumps({
            "status": self.local_status, "checkpoint_sha256": {},
            "inputs": dict(self.inputs),
        }))
        (args.output / "summary.json").write_text(json.dumps({
            "status": self.global_status,
        }))
        engine.shutdown()
        return int(self.global_status == "fail")

    def install_attention(self, impl):
        previous = impl.expanded
        impl.expanded = True
        self.attention_installs += 1

        def restore():
            impl.expanded = previous
            self.attention_restores += 1

        return restore

    def install_routes(self, layer, samples):
        previous = layer.route_samples
        layer.route_samples = samples

        def restore():
            layer.route_samples = previous
            self.route_restores += 1

        return restore

    def install_cached_attention(self, impl, prefix_values):
        assert prefix_values.layer == impl.number
        previous = impl.cached
        impl.cached = True
        self.cached_installs += 1

        def restore():
            impl.cached = previous
            self.cached_restores += 1

        return restore

    def barrier(self, *, group):
        assert group == "CPU fixture group"
        self.barriers += 1

    def all_gather_object(self, ranks, local, *, group):
        assert group == "CPU fixture group" and len(ranks) == 16
        for rank in range(16):
            ranks[rank] = {**local, "rank": rank}
            if "passed" in local:
                ranks[rank]["passed"] = local["passed"] and rank != self.failed_rank

    def load_tensor(self, path, **_kwargs):
        return self.bank if Path(path).name == "rank-0.pt" else {"token": 7}

    def run(self):
        collective = module("torch.distributed", barrier=self.barrier,
                            all_gather_object=self.all_gather_object)
        modules = {
            "run_cann_model": self.legacy,
            "run_native_decode": module("run_native_decode",
                                        compare_tensor=self.compare_tensor),
            "expanded_attention": module("expanded_attention",
                                         install=self.install_attention),
            "routes": module("routes", install_routes=self.install_routes),
            "torch_npu": module("torch_npu", npu=SimpleNamespace(matmul=self.matmul)),
            "torch": module("torch", distributed=collective,
                            load=self.load_tensor, Tensor=(Tensor, Bank),
                            bfloat16="torch.bfloat16", uint8="torch.uint8",
                            save=lambda _value, stream: stream.write(b"CPU trace\n")),
            "torch.distributed": collective,
            "vllm.distributed": module(
                "vllm.distributed",
                get_world_group=lambda: SimpleNamespace(cpu_group="CPU fixture group"),
            ),
            "tilert.models.glm_5_2_ascend.cann_model": module(
                "tilert.models.glm_5_2_ascend.cann_model", CannModel=Model,
            ),
        }
        argv = [str(HERE / "run_resident.py"), "--source", str(HERE),
                "--recipe", "paired"]
        if self.control_attention != "original":
            argv.extend(["--control-attention", self.control_attention])
        if self.cached:
            argv.extend(["--candidate-prefix-values", str(self.bank_directory)])
        with (patch.dict(sys.modules, modules), patch.object(Model, "scenario", self),
              patch.object(sys, "argv", argv), patch.object(sys, "path", list(sys.path)),
              patch.dict("os.environ"), redirect_stdout(io.StringIO()),
              patch.object(resident, "__file__", str(self.directory / "run_resident.py")),
              patch.object(resident, "CACHED_ATTENTION_SHA", self.helper_sha),
              patch.object(resident, "PREFIX_SUMMARY_SHA", getattr(
                  self, "bank_summary_sha", resident.PREFIX_SUMMARY_SHA))):
            return resident.main()


class ResidentControlFlow(unittest.TestCase):
    def test_expanded_control_and_cached_candidate_use_the_requested_implementations(self):
        with tempfile.TemporaryDirectory() as path:
            scenario = Scenario(Path(path), control_attention="expanded", cached=True)
            self.assertEqual(scenario.run(), 0)
            self.assertEqual(scenario.forwards,
                             ["expanded", "expanded", "cached_prefix", "cached_prefix"])
            self.assertEqual(scenario.bank_transfers, list(range(78)))
            self.assertEqual(scenario.cached_restores, 78)
            self.assertEqual(scenario.attention_restores, 78)
            recipe = json.loads((Path(path) / "control/recipe-rank-0.json").read_text())
            report = json.loads((Path(path) / "cached_prefix/rank-0.json").read_text())
            self.assertEqual(recipe["control_attention"], "expanded")
            self.assertEqual(recipe["candidate_attention"], "cached_prefix")
            self.assertEqual(report["attention_implementation"], "cached_prefix")
            self.assertEqual(report["prefix_values"]["summary_sha256"], scenario.bank_summary_sha)
            self.assertEqual([row["recipe"] for row in scenario.frame_recipes],
                             ["control", "control", "cached_prefix", "cached_prefix"])
            self.assertEqual([row["implementation"] for row in scenario.frame_recipes],
                             ["expanded", "expanded", "cached_prefix", "cached_prefix"])

    def test_cached_candidate_keeps_the_default_original_control(self):
        with tempfile.TemporaryDirectory() as path:
            scenario = Scenario(Path(path), cached=True)
            self.assertEqual(scenario.run(), 0)
            self.assertEqual(scenario.forwards,
                             ["control", "control", "cached_prefix", "cached_prefix"])
            self.assertEqual(scenario.attention_installs, 0)
            self.assertEqual(scenario.cached_restores, 78)

    def test_cached_remote_failure_keeps_the_failed_top_level_exit(self):
        with tempfile.TemporaryDirectory() as path:
            scenario = Scenario(Path(path), control_attention="expanded",
                                cached=True, failed_rank=9)
            self.assertEqual(scenario.run(), 1)
            summary = json.loads((Path(path) / "cached_prefix/summary.json").read_text())
            self.assertEqual(summary["status"], "fail")
            self.assertEqual(scenario.shutdowns, 1)

    def test_unverified_cached_helper_is_rejected_before_model_construction(self):
        with tempfile.TemporaryDirectory() as path:
            scenario = Scenario(Path(path), cached=True)
            scenario.cached_helper.write_text(scenario.cached_helper.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "measured source"):
                scenario.run()
            self.assertEqual(scenario.forwards, [])
            self.assertEqual(scenario.attention_installs, 0)

    def test_expanded_control_without_bank_keeps_the_expanded_candidate(self):
        with tempfile.TemporaryDirectory() as path:
            scenario = Scenario(Path(path), control_attention="expanded")
            self.assertEqual(scenario.run(), 0)
            self.assertEqual(scenario.forwards, ["expanded"] * 4)
            self.assertEqual(scenario.attention_restores, 156)
            self.assertEqual(scenario.cached_installs, 0)

    def test_bank_identity_failures_precede_candidate_installation(self):
        for error in ("checkpoint", "prefix", "rank", "file", "receipt",
                      "summary", "shape", "dtype", "tensor"):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as path:
                scenario = Scenario(Path(path), control_attention="expanded",
                                    cached=True, bank_error=error)
                with self.assertRaisesRegex(ValueError, "Prefix Value admission"):
                    scenario.run()
                self.assertEqual(scenario.cached_installs, 0)
                self.assertEqual(scenario.bank_transfers, [])
                self.assertEqual(scenario.shutdowns, 1)
                self.assertEqual(scenario.attention_restores, 78)

    def test_failed_local_control_still_reaches_common_decision(self):
        with tempfile.TemporaryDirectory() as path:
            scenario = Scenario(Path(path), local_status="fail", global_status="fail")
            self.assertEqual(scenario.run(), 1)
            self.assertEqual(scenario.barriers, 1)
            self.assertEqual(scenario.attention_installs, 0)
            self.assertEqual(scenario.shutdowns, 1)

    def test_remote_expanded_failure_propagates_to_top_level_exit(self):
        with tempfile.TemporaryDirectory() as path:
            scenario = Scenario(Path(path), failed_rank=9)
            self.assertEqual(scenario.run(), 1)
            report = json.loads((Path(path) / "expanded/rank-0.json").read_text())
            summary = json.loads((Path(path) / "expanded/summary.json").read_text())
            self.assertEqual(report["status"], "captured")
            self.assertEqual(summary["status"], "fail")
            self.assertFalse(summary["ranks"][9]["passed"])
            self.assertEqual(scenario.shutdowns, 1)

    def test_exception_without_control_report_only_cleans_the_engine(self):
        with tempfile.TemporaryDirectory() as path:
            scenario = Scenario(Path(path), raise_before_report=True)
            with self.assertRaisesRegex(RuntimeError, "before its report"):
                scenario.run()
            self.assertEqual(scenario.barriers, 0)
            self.assertEqual(scenario.forwards, [])
            self.assertEqual(scenario.shutdowns, 1)
            self.assertTrue(scenario.matmul.allow_hf32)

    def test_success_uses_one_model_and_restores_all_per_frame_routes(self):
        with tempfile.TemporaryDirectory() as path:
            scenario = Scenario(Path(path))
            self.assertEqual(scenario.run(), 0)
            self.assertTrue(scenario.engine.fp32_head)
            self.assertEqual(scenario.forwards,
                             ["control", "control", "expanded", "expanded"])
            self.assertEqual(scenario.route_restores, 75 * 4)
            self.assertEqual(scenario.attention_restores, 78)
            self.assertEqual(scenario.shutdowns, 1)
            self.assertTrue(scenario.matmul.allow_hf32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = (HERE / "run_resident.py", Path(__file__))
    before = {str(path): resident.digest(path) for path in sources}
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ResidentControlFlow),
    )
    after = {str(path): resident.digest(path) for path in sources}
    passed = result.wasSuccessful() and before == after and "torch" not in sys.modules
    report = {"status": "pass" if passed else "fail", "tests": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "scope": "CPU mocked collaborators through the real resident main and capture",
              "torch_imported": "torch" in sys.modules, "source_stable": before == after,
              "source_sha256": after, "test_output": stream.getvalue()}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(stream.getvalue())
    print(json.dumps({key: report[key] for key in ("status", "tests", "failures", "errors")}))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
