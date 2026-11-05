"""Check runner metadata gates without importing Torch or accessing an NPU."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def load_runner():
    path = Path(__file__).with_name("run_native_decode.py")
    spec = importlib.util.spec_from_file_location("native_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def is_indexer(layer):
    return layer < 3 or (layer - 2) % 4 == 0


def make_engine(position):
    layers = []
    for layer in range(78):
        index = SimpleNamespace(length=position) if is_indexer(layer) else None
        layers.append(SimpleNamespace(
            mla=SimpleNamespace(length=position), indexer=index,
        ))
    return SimpleNamespace(
        layers=layers, position=position,
        config=SimpleNamespace(has_indexer=is_indexer),
    )


class Rows:
    """Represent row layout/content without device tensor arithmetic."""

    def __init__(self, values, dtype="BF16"):
        self.values = list(values)
        self.dtype = dtype
        self.shape = (len(self.values), 1)

    def __getitem__(self, selection):
        return Rows(self.values[selection], self.dtype)

    def numel(self):
        return len(self.values)

    def is_floating_point(self):
        return self.dtype != "INT8"


def check_mtp_cache_collection(checks):
    import run_native_mtp_session as runner

    def compare(name, left, right, result, **kwargs):
        result.append({"name": name, "passed": left.values == right.values,
                       "compared_rows": len(left.values),
                       "dtype": left.dtype,
                       "exact": kwargs["exact"]})

    runner.compare_tensor = compare
    expected = {"main": [{"latent": Rows(range(6)), "rope": Rows(range(6))}
                         for _ in range(78)],
                "mtp": [{"latent": Rows(range(6)), "rope": Rows(range(6))}]}
    expected["main"][0].update(index=Rows(range(6), "INT8"),
                               index_scale=Rows(range(6), "FP16"))
    expected["mtp"][0]["index"] = Rows(range(6))
    actual = {model: [{name: value[:4] for name, value in layer.items()}
                      for layer in layers]
              for model, layers in expected.items()}

    def verify(name, left, right, should_pass):
        result = []
        runner.compare_caches(left, right, (4, 4), 3, name, result)
        accepted = bool(result) and all(item["passed"] for item in result)
        checks.append({"name": name, "passed": accepted == should_pass,
                       "expected_acceptance": should_pass, "checks": result})
        return result

    valid = verify("mtp_cache.valid", actual, expected, True)
    scopes = [item for item in valid if "compared_rows" in item]
    checks.append({
        "name": "mtp_cache.prefix_exact_and_new_scope",
        "passed": len(scopes) == 79 * 2 * 2 + 6
        and all(item["compared_rows"] == 2
                and item["exact"] == (item["name"].endswith(".prefix")
                                      or item["dtype"] == "INT8"
                                      or ".index_scale." in item["name"])
                for item in scopes),
    })
    for model in ("main", "mtp"):
        wrong = copy.deepcopy(actual)
        wrong[model] = []
        verify("mtp_cache.empty." + model, wrong, expected, False)
    wrong = copy.deepcopy(actual)
    wrong["main"].pop()
    verify("mtp_cache.missing_layer", wrong, expected, False)
    wrong, reference = copy.deepcopy(actual), copy.deepcopy(expected)
    del wrong["main"][0]["rope"]
    del reference["main"][0]["rope"]
    verify("mtp_cache.missing_required_field", wrong, reference, False)
    wrong = copy.deepcopy(actual)
    del wrong["mtp"][0]["rope"]
    verify("mtp_cache.field_mismatch", wrong, expected, False)
    wrong = copy.deepcopy(actual)
    wrong["main"][0]["latent"].values[0] = -1
    verify("mtp_cache.prompt_corruption", wrong, expected, False)
    wrong = copy.deepcopy(actual)
    wrong["mtp"][0]["latent"].values[3] = -1
    verify("mtp_cache.new_row_corruption", wrong, expected, False)
    wrong = copy.deepcopy(actual)
    wrong["mtp"][0]["latent"] = Rows(range(3))
    verify("mtp_cache.wrong_retained_rows", wrong, expected, False)
    return Path(runner.__file__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runner, source = load_runner()
    checks = []

    def verify(name, engine, expected, should_pass):
        result = {"checks": []}
        runner.check_lengths(result, name, engine, expected)
        actual = result["checks"][0]
        checks.append({"name": name, "passed": actual["passed"] == should_pass,
                       "expected_acceptance": should_pass, "gate": actual})

    for position in (0, 1, 8192, 8193):
        engine = make_engine(position)
        verify(f"valid.{position}", engine, position, True)
        engine.position += 1
        verify(f"wrong_position.{position}", engine, position, False)
        engine.position -= 1
        engine.layers[77].mla.length += 1
        verify(f"wrong_mla.{position}", engine, position, False)
        engine.layers[77].mla.length -= 1
        engine.layers[74].indexer.length += 1
        verify(f"wrong_index_length.{position}", engine, position, False)
    engine = make_engine(0)
    engine.layers[74].indexer = None
    verify("missing_indexer", engine, 0, False)
    engine = make_engine(0)
    engine.layers[3].indexer = SimpleNamespace(length=0)
    verify("shared_layer_owns_indexer", engine, 0, False)

    # Numerical comparisons are replaced only to isolate collection-length
    # checks. This program makes no statement about tensor arithmetic.
    runner.compare_tensor = lambda *unused, **keywords: None
    trace = {
        "token": 1, "position": 0, "committed_position": 1, "logits": None,
        "hidden_states": (None,) * 79, "layer_logits": (None,) * 78,
        "indices": (None,) * 78, "routes": (None,) * 78,
        "cache_rows": [{"latent": None, "rope": None, "index": None,
                        "index_scale": None} for _ in range(78)],
    }
    for field in ("hidden_states", "layer_logits", "indices", "routes",
                  "cache_rows"):
        reference = copy.deepcopy(trace)
        reference[field] = []
        result = []
        runner.compare_trace("missing", trace, reference, result, exact=True)
        checks.append({
            "name": "reject_empty." + field,
            "passed": any(not item["passed"] for item in result),
        })
    mtp_source = check_mtp_cache_collection(checks)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "stdlib metadata gates; no Torch, numerical decode or NPU",
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), source, mtp_source)
        },
        "checks": checks, "checks_total": len(checks),
        "checks_failed": sum(not item["passed"] for item in checks),
    }
    report["status"] = "pass" if not report["checks_failed"] else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "checks_total", "checks_failed",
    )}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
