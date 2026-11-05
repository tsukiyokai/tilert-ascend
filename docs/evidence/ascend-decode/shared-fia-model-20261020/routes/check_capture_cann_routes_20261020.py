"""CPU logic checks for the actual capture wrapper, with fake graph replays."""

from contextlib import nullcontext
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "capture_cann_routes.py"
ACTIVE = None


class Router:
    def __init__(self, owner, layer):
        self.owner, self.layer = owner, layer

    def _select_experts(self):
        return None, self.owner.ids(self.layer)


def layer(owner, number):
    router = Router(owner, number)
    return SimpleNamespace(mlp=SimpleNamespace(
        experts=SimpleNamespace(routed_experts=router)))


class Target:
    def __init__(self):
        self.device = torch.device("cpu")
        self.position, self.version, self.width = 8192, 0, 1
        self.mode, self.failure = "eager", None
        self.model = SimpleNamespace(model=SimpleNamespace(
            layers=[None] * 3 + [layer(self, n) for n in range(3, 78)]))

    def ids(self, number):
        return (torch.arange(self.width)[:, None] * 7
                + torch.arange(8)[None, :] + number * 11
                + self.version) % 256

    def inject(self, prefix=None):
        self.position = 8191

    def verify(self, tokens, *, full_logits=False):
        self.width = len(tokens)
        routers = [item.mlp.experts.routed_experts
                   for item in self.model.model.layers[3:]]
        for index, router in enumerate(routers):
            if self.mode == "replay":
                # Replay runs the captured copy directly, without entering
                # the Python route_copy wrapper a second time.
                destination = inspect.getclosurevars(
                    router._select_experts).nonlocals["destination"]
                if self.failure == "stale_all":
                    continue
                if self.failure == "missing_last" and index == 74:
                    continue
                destination[:self.width].copy_(self.ids(index + 3))
            else:
                router._select_experts()
        self.position += self.width
        if isinstance(tokens, torch.Tensor):
            tokens.fill_(999)
        return object()


class Draft:
    def __init__(self):
        self.device = torch.device("cpu")
        self.position, self.version, self.width = 8192, 0, 1
        self.model = SimpleNamespace(model=SimpleNamespace(
            layers={"78": SimpleNamespace(mtp_block=layer(self, 78))}))

    ids = Target.ids

    def forward(self, tokens, parent=None, *, external_indices=None):
        self.width = len(tokens)
        self.model.model.layers["78"].mtp_block.mlp.experts.routed_experts._select_experts()
        self.position += self.width
        return object()


runner = ModuleType("bench_cann_mtp")
runner.__file__ = __file__
runner.digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
runner.arguments = lambda: ACTIVE


def atomic_json(path, value):
    Path(path).write_text(json.dumps(value) + "\n")


runner.atomic_json = atomic_json


def run_request():
    target, draft = Target(), Draft()
    target.inject()
    target.verify((9249,))
    target.verify(list(range(500, 500 + ACTIVE.width)))
    target.position = 8193
    target.mode = "replay" if ACTIVE.replay else "eager"
    target.version, target.failure = 37, ACTIVE.failure
    target.verify(tokens=torch.arange(700, 700 + ACTIVE.width))
    target.inject()
    target.version, target.failure = 0, None
    target.mode = "eager"
    target.verify((9249,))
    target.mode = "replay" if ACTIVE.replay else "eager"
    target.verify(tuple(range(500, 500 + ACTIVE.width)))
    if not ACTIVE.lookup:
        draft.forward((1,))
        if ACTIVE.width > 2:
            draft.forward((2,), external_indices=object())
        draft.forward(tuple(range(ACTIVE.width)))
    canonical = {"status": "measured", "tpot_ms": {"median": 12.34}}
    runner.atomic_json(ACTIVE.output / "summary.json", canonical)
    assert canonical == {"status": "measured", "tpot_ms": {"median": 12.34}}
    runner.atomic_json(ACTIVE.output / f"rank-{os.environ['RANK']}.json",
                       {"status": "completed", "rank_requests": []})
    return 0


runner.main = run_request
sys.modules["bench_cann_mtp"] = runner
for name, cls in (("cann_model", Target), ("cann_mtp", Draft)):
    module = ModuleType("tilert.models.glm_5_2_ascend." + name)
    setattr(module, "CannModel" if name == "cann_model" else "CannMtp", cls)
    sys.modules[module.__name__] = module
spec = importlib.util.spec_from_file_location("capture_under_test", SOURCE)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def check_case(width, *, rank=0, lookup=True, replay=True, failure=None):
    global ACTIVE
    name = f"w{width}-r{rank}-lookup{int(lookup)}-replay{int(replay)}-{failure}"
    ACTIVE = SimpleNamespace(width=width, lookup=lookup, replay=replay,
                             failure=failure, target_graph=replay,
                             draft_graph=False, output=ROOT / name)
    os.environ["RANK"] = str(rank)
    try:
        status = capture.main()
    except ValueError as error:
        assert failure and "invalid expert IDs" in str(error)
        status = 1
    receipt = json.loads((ACTIVE.output / f"routes-rank-{rank}.json").read_text())
    assert status == (1 if failure else 0)
    assert receipt["status"] == ("fail" if failure else "captured")
    assert receipt["buffers"]["target"]["shape"] == [75, width, 8]
    if not failure:
        records = [json.loads(line) for line in
                   (ACTIVE.output / f"routes-rank-{rank}.jsonl").read_text().splitlines()]
        main = [row for row in records if row["kind"] == "target"]
        first, second, reset = [row for row in main if row["width"] == width]
        assert first["shape"] == [width, 75, 8]
        assert first["layout"] == "query,layer,topk"
        assert first["input_ids"] == list(range(500, 500 + width))
        assert second["input_ids"] == list(range(700, 700 + width))
        assert first["positions"] == list(range(8192, 8192 + width))
        assert second["positions"] == list(range(8193, 8193 + width))
        assert first["expert_ids"] != second["expert_ids"]
        assert first["expert_ids"] == reset["expert_ids"]
        assert first["request"] == second["request"] == 0
        assert reset["request"] == 1
        summary = json.loads((ACTIVE.output / "summary.json").read_text())
        assert summary["status"] == "diagnostic" and summary["tpot_ms"] is None
        assert summary["route_capture"] and not summary["performance_measured"]
        if lookup:
            assert receipt["counts"] == {"target": 5}
    return {"case": name, "status": "pass", "negative": bool(failure),
            "records_sha256": receipt["records_sha256"]}


torch.set_num_threads(1)
results = []
with patch.object(torch, "npu", SimpleNamespace(device=lambda _: nullcontext()),
                  create=True):
    for width in (24, 32):
        left = check_case(width, rank=0)
        right = check_case(width, rank=1)
        assert left["records_sha256"] == right["records_sha256"]
        results.extend((left, right))
    results.append(check_case(16, replay=False))
    for failure in ("missing_last", "stale_all"):
        results.append(check_case(32, failure=failure))
    results.append(check_case(2, lookup=False, replay=False))
    results.append(check_case(4, lookup=False, replay=False))
assert "torch_npu" not in sys.modules
result = {"scope": "CPU fake graph replay checks actual wrapper logic only; no NPU Graph, CANN, or performance validation",
          "npu_launched": False, "torch_npu_imported": False,
          "case_count": len(results), "cases": results,
          "source_sha256": runner.digest(SOURCE),
          "script_sha256": runner.digest(Path(__file__))}
(ROOT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
