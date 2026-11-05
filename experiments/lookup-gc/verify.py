"""Check native node ownership, collection flags and constructor release."""

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tracemalloc
import weakref


ROOT = Path(__file__).resolve().parents[2]
MODULE = "tilert.models.glm_5_2_ascend.lookup_draft"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_native(path, builder):
    build = json.loads(path.read_text())
    if build["status"] != "pass":
        raise ValueError("Expected a passing native build receipt")
    binary = Path(build["binary"])
    assert digest(binary) == build["binary_sha256"]
    module = load(MODULE, binary)
    expected = {**build["native_identity"],
                "loaded_binary_sha256": digest(binary)}
    assert module._NATIVE_BUILD == expected == build["isolated_load"]
    original = binary.with_name("lookup_draft.py").read_text()
    source = original
    if build["native_identity"].get("acyclic_nodes", False):
        assert original.count("class _Node:") == 1
        source = original.replace(
            "class _Node:",
            "import cython\n\n\n@cython.no_gc\n@cython.cclass\nclass _Node:",
            1,
        )
    generated = Path(build["generated_c"]).with_suffix(".py")
    assert digest(generated) == build["generated_input_sha256"]
    assert generated.read_text() == source + builder.native_guard(
        build["native_identity"])
    assert digest(Path(build["generated_c"])) == build["generated_c_sha256"]
    for name, expected in build["native_identity"]["source_sha256"].items():
        assert digest(binary.with_name(name)) == expected
    return module, build


def census(drafter):
    pending = [(drafter._root, 0)]
    seen = set()
    result = {"nodes": 0, "tracked_nodes": 0,
              "tracked_child_maps": 0, "maximum_depth": 0}
    while pending:
        node, depth = pending.pop()
        assert id(node) not in seen
        seen.add(id(node))
        result["nodes"] += 1
        result["tracked_nodes"] += gc.is_tracked(node)
        result["tracked_child_maps"] += gc.is_tracked(node.children)
        result["maximum_depth"] = max(result["maximum_depth"], depth)
        for token, child in node.children.items():
            assert type(token) is int and type(child) is type(node)
            assert child.length > 0 and child.hi < len(drafter.context)
            assert depth + child.length <= drafter.longest
            pending.append((child, depth + child.length))
    return result


def construct(cls, prompt, longest):
    return cls(prompt, longest=longest)


def lifetime(cls, prompt, longest, cycle):
    warm = construct(cls, prompt, longest)
    del warm
    gc.collect()
    tracemalloc.start(1)
    try:
        drafter = construct(cls, prompt, longest)
        assert gc.is_tracked(drafter)
        heap = census(drafter)
        line = construct.__code__.co_firstlineno + 1
        selected = (tracemalloc.Filter(True, __file__, lineno=line),)
        live = tracemalloc.take_snapshot().filter_traces(selected)
        assert len(live.traces) >= heap["nodes"]
        reference = weakref.ref(drafter)
        if cycle:
            drafter.cycle = drafter
        del drafter
        gc.collect()
        assert reference() is None
        released = tracemalloc.take_snapshot().filter_traces(selected)
        assert len(released.traces) == 0
        return {**heap, "cycle_on_drafter": cycle,
                "constructor_live_bytes": sum(item.size for item in
                                              live.traces),
                "constructor_remaining_bytes": 0, "weakref_cleared": True}
    finally:
        tracemalloc.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--baseline-build", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    builder = load("lookup_native_builder",
                   root / "experiments/lookup-native/build.py")
    sampler = load("lookup_gc_sampler",
                   root / "experiments/lookup-index/coordinate_probe.py")
    modules, builds = {}, {}
    for arm in ("baseline", "candidate"):
        modules[arm], builds[arm] = load_native(
            getattr(args, arm + "_build"), builder)
    assert modules["baseline"] is not modules["candidate"]
    assert modules["baseline"]._Node is not modules["candidate"]._Node
    assert builds["baseline"]["native_identity"]["source_sha256"] == (
        builds["candidate"]["native_identity"]["source_sha256"])
    reference, decoded_hash = sampler.read_reference(args.reference)
    cases = [(item["name"], item["prompt_token_ids"], 32)
             for item in reference["results"]]
    cases.append(("deep_repeated_tail", [1] * 512, 2**100))
    del reference
    threshold, enabled = gc.get_threshold(), gc.isenabled()
    report = {
        "status": "running", "command": [sys.executable, *sys.argv],
        "scope": "Owned lookup trees; no timing or NPU execution",
        "verifier_sha256": digest(Path(__file__)),
        "reference_decoded_sha256": decoded_hash,
        "build_receipts_sha256": {
            arm: digest(getattr(args, arm + "_build")) for arm in builds},
        "binary_sha256": {arm: build["binary_sha256"] for arm, build in
                          builds.items()}, "cases": [],
    }
    for arm, module in modules.items():
        tracked = arm == "baseline"
        assert gc.is_tracked(module._Node(0, 0, -1)) is tracked
        for name, prompt, longest in cases:
            for cycle in (False, True):
                row = lifetime(module.LookupDrafter, prompt, longest, cycle)
                assert row["tracked_nodes"] == (row["nodes"] if tracked
                                                else 0)
                if not tracked:
                    assert row["tracked_child_maps"] == 0
                report["cases"].append({"arm": arm, "workload": name,
                                        **row})
    assert gc.get_threshold() == threshold and gc.isenabled() == enabled
    assert "torch" not in sys.modules and "Cython" not in sys.modules
    report["status"] = "pass"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"],
                      "cases": len(report["cases"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
