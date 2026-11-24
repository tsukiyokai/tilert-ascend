"""Independent exact proposals and every-prefix latest for local variants."""

import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import random
import traceback


ROOT = Path("/Users/shanshan/repo/tile-ai/TileRT-Ascend")
VARIANTS = Path("/tmp/tilert-lookup-local-variants-20261028.py")
SCAN = Path("/tmp/tilert-lookup-arena-prototype-20261028.py")
PRODUCTION = ROOT / "tilert/models/glm_5_2_ascend/lookup_draft.py"
REFERENCE = Path("/tmp/tilert-lookup-rfind-probe-20261028.json")
NGRAM = ROOT / "tests/glm52/ngram_acceptance.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def every_prefix(drafter):
    """Build exact prefix sets independently, then consume each trie prefix."""
    context, expected = drafter.context, {}
    for end in range(drafter.shortest, len(context)):
        window = tuple(context[max(0, end - drafter.longest):end][::-1])
        for length in range(1, len(window) + 1):
            expected[window[:length]] = end
    stack = [(drafter._root, ())]
    visited, prefixes, nodes, child_dicts = set(), 0, 0, 0
    while stack:
        node, prefix = stack.pop()
        assert id(node) not in visited, "Cycle or shared node"
        visited.add(id(node))
        nodes += 1
        children = node.children
        if children is not None:
            child_dicts += 1
            if hasattr(node, "label"):
                assert children or node is drafter._root, "Allocated empty lazy leaf table"
        for token, child in (children or {}).items():
            if hasattr(child, "label"):
                label = child.label
            else:
                assert 0 < child.length <= drafter.longest
                assert 0 <= child.hi - child.length + 1 <= child.hi < len(context)
                label = tuple(context[child.hi - child.length + 1:child.hi + 1][::-1])
            assert label and token == label[0]
            assert len(prefix) + len(label) <= drafter.longest
            for length in range(1, len(label) + 1):
                wanted = prefix + label[:length]
                assert wanted in expected, ("Unknown or duplicate prefix", wanted)
                assert child.latest == expected.pop(wanted), (
                    "Wrong latest", wanted, child.latest,
                )
                prefixes += 1
            stack.append((child, prefix + label))
    assert not expected, ("Missing prefixes", len(expected))
    return {"prefixes": prefixes, "nodes": nodes, "child_dicts": child_dicts}


def run():
    variants = load("local_variants", VARIANTS)
    independent_scan = load("independent_arena_scan", SCAN).oracle
    production = load("current_production", PRODUCTION).LookupDrafter
    classes = (variants.LocalCoordinateLookup, variants.LazyChildrenLookup)
    counters = {}
    transcript = hashlib.sha256()
    for cls in classes:
        counts = {"contexts": 0, "proposals": 0, "prefixes": 0,
                  "invariant_states": 0, "large_rounds": 0,
                  "large_scan_crosschecks": 0}

        def compare(actual, baseline, invariant=False):
            assert actual.context == baseline.context
            for count in (-1, 0, 1, 7, 31):
                expected = independent_scan(actual.context, count,
                                            actual.longest, actual.shortest)
                found = actual.propose(count)
                assert found == expected == baseline.propose(count), {
                    "class": cls.__name__, "context": actual.context,
                    "longest": actual.longest, "shortest": actual.shortest,
                    "count": count, "actual": found, "expected": expected,
                }
                assert actual.report() == baseline.report()
                transcript.update(json.dumps([cls.__name__, actual.context,
                                              actual.shortest, actual.longest,
                                              count, found],
                                             separators=(",", ":")).encode())
                counts["proposals"] += 1
            counts["contexts"] += 1
            if invariant:
                counts["prefixes"] += every_prefix(actual)["prefixes"]
                counts["invariant_states"] += 1

        configurations = ((1, 1), (1, 8), (1, 32), (2, 5), (2, 32),
                          (4, 32), (32, 32))
        for length in range(9):
            for context in product(range(2), repeat=length):
                for shortest, longest in configurations:
                    actual = cls(context[:-2], longest, shortest)
                    baseline = production(context[:-2], longest, shortest)
                    for added in ((), context[-2:]):
                        actual.commit(value for value in added)
                        baseline.commit(value for value in added)
                    compare(actual, baseline, invariant=True)

        left = [30000, 30712, 30000, 32048, 30503]
        right = [30306, 30000, 30004, 30000, 30000]
        collision = left + [111] + right + [222] + left[-4:] + [333] + left
        specials = [([], 1, 32), ([1], 1, 32), ([1, 2, 3], 2, 32),
                    ([1, 9, 2, 1], 1, 32), ([7] * 130, 1, 96),
                    ([1, 2, 3, 4, 7, 8, 9, 2, 3, 4], 2, 4),
                    (collision, 2, 5),
                    ([2**80, -1, 3, 2**80, -1], 1, 64)]
        for context, shortest, longest in specials:
            actual = cls(context, longest, shortest)
            baseline = production(context, longest, shortest)
            compare(actual, baseline, invariant=True)
        assert cls(collision, longest=5).propose(1) == [111]

        rng = random.Random(20261028)
        for shortest, longest in (*configurations, (1, 64), (2, 96)):
            for alphabet in (2, 7, 251, 154880):
                initial = [rng.randrange(alphabet) for _ in range(31)]
                actual = cls(initial, longest, shortest)
                baseline = production(initial, longest, shortest)
                for iteration in range(32):
                    compare(actual, baseline, invariant=iteration % 8 == 0)
                    added = [rng.randrange(alphabet)
                             for _ in range(rng.randrange(9))]
                    actual.commit(value for value in added)
                    baseline.commit(value for value in added)

        reference = json.loads(REFERENCE.read_text())
        assert reference["status"] == "pass"
        assert reference["source_sha256"][str(NGRAM)] == digest(NGRAM)
        large = []
        for case in reference["results"]:
            prompt, tokens = case["prompt_token_ids"], case["output_token_ids"]
            actual, baseline = cls(prompt), production(prompt)
            initial = every_prefix(actual)
            actual.commit(tokens[:1])
            baseline.commit(tokens[:1])
            rows = []
            for index, row in enumerate(case["oracle"]["rounds"]):
                proposal = actual.propose(31)
                assert proposal == row["proposal"] == baseline.propose(31)
                if index % 64 == 0:
                    assert proposal == independent_scan(actual.context, 31, 32, 2)
                    counts["large_scan_crosschecks"] += 1
                rows.append((row["position"], proposal, row["accepted"]))
                added = tokens[row["position"]:row["position"] + row["accepted"]]
                actual.commit(value for value in added)
                baseline.commit(value for value in added)
                assert actual.report() == baseline.report()
                counts["large_rounds"] += 1
            assert actual.context == prompt + tokens == baseline.context
            signature = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
            assert signature == case["oracle"]["signature_sha256"]
            final = every_prefix(actual)
            counts["prefixes"] += initial["prefixes"] + final["prefixes"]
            counts["invariant_states"] += 2
            large.append({"name": case["name"], "signature_sha256": signature,
                          "initial": initial, "final": final})
        counters[cls.__name__] = {"coverage": counts, "large": large}
    return {"status": "pass", "classes": counters,
            "signature_sha256": transcript.hexdigest(),
            "performance_measured": False, "npu_executed": False}


def main():
    sources = (Path(__file__).resolve(), VARIANTS, SCAN, PRODUCTION, REFERENCE, NGRAM)
    hashes = {str(path): digest(path) for path in sources}
    try:
        result = run()
    except Exception as error:
        result = {"status": "fail", "error": repr(error),
                  "traceback": traceback.format_exc()}
    result["source_sha256"] = hashes
    result["source_stable"] = hashes == {str(path): digest(path) for path in sources}
    if not result["source_stable"]:
        result["status"] = "fail"
    path = Path("/tmp/tilert-lookup-local-audit-20261028.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
