"""Exact lookup via integer radix arenas and context-backed edge labels.

This is an independent /tmp prototype, not production code. The context is
append-only, as in LookupDrafter. The global transition table contains only
integer keys/values for vocabulary-sized tokens. Other Python integers use
tuple keys and keep exact behavior. No hash is used to identify token tails.

Run correctness explicitly with --check --output PATH. This file contains no
performance benchmark. A head-only or scalar check cannot certify NPU decode.
"""

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import random


ROOT = Path("/Users/shanshan/repo/tile-ai/TileRT-Ascend")
PRODUCTION = ROOT / "tilert/models/glm_5_2_ascend/lookup_draft.py"
TOKEN_BITS = 18
TOKEN_LIMIT = 1 << TOKEN_BITS


def transition_key(parent, token):
    if 0 <= token < TOKEN_LIMIT:
        return (parent << TOKEN_BITS) | token
    return parent, token


class ArenaLookupDrafter:
    """Keep the public lookup API; replace nodes, dictionaries and labels.

    Each non-root node stores (hi, length, latest) in integer arrays. Its
    reversed label is context[hi], context[hi-1], ... . Compressed prefixes
    share the same latest eligible continuation end. Node IDs survive arena
    growth; no tensor, view, or Python label object retains a buffer address.
    """

    def __init__(self, prompt, longest=32, shortest=2):
        if shortest < 1 or longest < shortest:
            raise ValueError("Tail lengths must satisfy 1 <= shortest <= longest")
        self.context = []
        self.longest = longest
        self.shortest = shortest
        self.hits = 0
        self.rounds = 0
        self._hi = array("q", [0]) * 16
        self._length = array("q", [0]) * 16
        self._latest = array("q", [0]) * 16
        self._latest[0] = -1
        self._size = 1
        self._children = {}
        self.commit(prompt)

    def _reserve(self, minimum):
        capacity = len(self._hi)
        if minimum <= capacity:
            return
        new_capacity = max(minimum, capacity * 2)
        extra = array("q", [0]) * (new_capacity - capacity)
        self._hi.extend(extra)
        self._length.extend(extra)
        self._latest.extend(extra)

    def _new(self, hi, length, end):
        node = self._size
        self._reserve(node + 1)
        self._hi[node] = hi
        self._length[node] = length
        self._latest[node] = end
        self._size += 1
        return node

    def _shared(self, child, query_hi, remaining):
        hi = self._hi[child]
        length = min(self._length[child], remaining)
        context = self.context
        # Equal forward intervals are also equal when both are reversed.
        # Keep the common whole-edge case in C instead of looping in Python.
        if (context[hi - length + 1:hi + 1]
                == context[query_hi - length + 1:query_hi + 1]):
            return length
        for offset in range(length):
            if context[hi - offset] != context[query_hi - offset]:
                return offset
        raise AssertionError("Unequal edge intervals have no differing token")

    def _insert(self, end):
        window_length = min(self.longest, end)
        node, depth = 0, 0
        context, children = self.context, self._children
        while depth < window_length:
            query_hi = end - 1 - depth
            key = transition_key(node, context[query_hi])
            child = children.get(key)
            if child is None:
                children[key] = self._new(
                    query_hi, window_length - depth, end,
                )
                return
            shared = self._shared(child, query_hi, window_length - depth)
            if shared == 0:
                raise AssertionError("Transition key differs from its edge")
            if shared == self._length[child]:
                self._latest[child] = end
                depth += shared
                node = child
                continue
            replacement = self._new(self._hi[child], shared, end)
            self._hi[child] -= shared
            self._length[child] -= shared
            children[transition_key(
                replacement, context[self._hi[child]],
            )] = child
            children[key] = replacement
            depth += shared
            if depth < window_length:
                query_hi = end - 1 - depth
                children[transition_key(
                    replacement, context[query_hi],
                )] = self._new(query_hi, window_length - depth, end)
            return

    def commit(self, tokens):
        previous = len(self.context)
        self.context.extend(int(token) for token in tokens)
        if len(self.context) == previous:
            return
        if self._size == 1:
            self._reserve(1 + 2 * max(0, len(self.context) - self.shortest))
        for end in range(max(previous, self.shortest), len(self.context)):
            self._insert(end)

    def propose(self, count):
        self.rounds += 1
        maximum = min(self.longest, len(self.context) - 1)
        if maximum < self.shortest or count <= 0:
            return []
        node, matched, end = 0, 0, -1
        context = self.context
        while matched < maximum:
            query_hi = len(context) - 1 - matched
            child = self._children.get(transition_key(node, context[query_hi]))
            if child is None:
                break
            shared = self._shared(child, query_hi, maximum - matched)
            matched += shared
            end = self._latest[child]
            if shared < self._length[child]:
                break
            node = child
        if matched < self.shortest:
            return []
        proposed = context[end:end + count]
        self.hits += bool(proposed)
        return proposed

    def report(self):
        return {
            "drafter": "lookup", "longest_tail": self.longest,
            "shortest_tail": self.shortest, "context_tokens": len(self.context),
            "rounds": self.rounds, "tail_hits": self.hits,
        }

    def storage_stats(self):
        return {
            "nodes": self._size, "transitions": len(self._children),
            "arena_capacity": len(self._hi),
            "arena_bytes": sum(values.buffer_info()[1] * values.itemsize
                               for values in (self._hi, self._length,
                                              self._latest)),
            "tuple_fallback_keys": sum(isinstance(key, tuple)
                                       for key in self._children),
        }


def oracle(context, count, longest, shortest):
    """Independent longest-first, latest-first scan; no radix or hashes."""
    if count <= 0:
        return []
    maximum = min(longest, len(context) - 1)
    for length in range(maximum, shortest - 1, -1):
        for end in range(len(context) - 1, length - 1, -1):
            equal = True
            for offset in range(length):
                if context[end - length + offset] != context[-length + offset]:
                    equal = False
                    break
            if equal:
                return context[end:end + count]
    return []


def check_invariants(drafter):
    context = drafter.context
    adjacency = {}
    incoming = [0] * drafter._size
    for key, child in drafter._children.items():
        if isinstance(key, tuple):
            parent, token = key
        else:
            parent, token = key >> TOKEN_BITS, key & (TOKEN_LIMIT - 1)
        assert 0 <= parent < drafter._size
        assert 0 < child < drafter._size
        assert context[drafter._hi[child]] == token
        incoming[child] += 1
        adjacency.setdefault(parent, []).append(child)
    assert incoming == [0] + [1] * (drafter._size - 1)
    stack, visited, prefixes = [(0, ())], set(), 0
    while stack:
        node, prefix = stack.pop()
        assert node not in visited
        visited.add(node)
        for child in adjacency.get(node, ()):
            hi, length = drafter._hi[child], drafter._length[child]
            assert 0 < length <= drafter.longest
            assert 0 <= hi - length + 1 <= hi < len(context)
            label = tuple(context[hi - length + 1:hi + 1][::-1])
            assert len(prefix) + length <= drafter.longest
            for partial in range(1, length + 1):
                wanted = prefix + label[:partial]
                ends = []
                for end in range(drafter.shortest, len(context)):
                    window = context[max(0, end - drafter.longest):end][::-1]
                    if tuple(window[:len(wanted)]) == wanted:
                        ends.append(end)
                assert ends and drafter._latest[child] == max(ends), (
                    wanted, ends, drafter._latest[child],
                )
                prefixes += 1
            stack.append((child, prefix + label))
    assert len(visited) == drafter._size
    return prefixes


def load_production():
    spec = importlib.util.spec_from_file_location("production_lookup", PRODUCTION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LookupDrafter


def run_checks(comparison_factory=None):
    ProductionLookup = comparison_factory or load_production()
    counters = {"contexts": 0, "proposals": 0, "invariant_prefixes": 0,
                "incremental_contexts": 0, "large_contexts": 0}
    transcript = hashlib.sha256()
    storage = []

    def compare(arena, production, *, invariant=False):
        context = list(arena.context)
        assert context == production.context
        for count in (0, 1, 7, 31):
            expected = oracle(context, count, arena.longest, arena.shortest)
            actual = arena.propose(count)
            baseline = production.propose(count)
            assert actual == expected == baseline, {
                "context": context, "shortest": arena.shortest,
                "longest": arena.longest, "count": count,
                "arena": actual, "oracle": expected, "production": baseline,
            }
            transcript.update(json.dumps([context, arena.shortest,
                                          arena.longest, count, actual],
                                         separators=(",", ":")).encode())
            assert arena.report() == production.report()
            counters["proposals"] += 1
        if invariant:
            counters["invariant_prefixes"] += check_invariants(arena)
        counters["contexts"] += 1

    configurations = ((1, 1), (1, 8), (1, 32), (2, 32), (4, 32), (32, 32))
    for length in range(9):
        for context in product(range(2), repeat=length):
            for shortest, longest in configurations:
                arena = ArenaLookupDrafter(context[:-2], longest, shortest)
                production = ProductionLookup(context[:-2], longest, shortest)
                for value in ((), iter(context[-2:])):
                    added = list(value)
                    arena.commit(iter(added))
                    production.commit(iter(added))
                compare(arena, production, invariant=True)

    left = [30000, 30712, 30000, 32048, 30503]
    right = [30306, 30000, 30004, 30000, 30000]
    collision = left + [111] + right + [222] + left[-4:] + [333] + left
    special = [([], 1, 32), ([1], 1, 32), ([1, 2, 3], 2, 32),
               ([1, 9, 2, 1], 1, 32),
               ([1, 2, 3, 4, 7, 8, 9, 2, 3, 4], 2, 4),
               (collision, 2, 5), ([7] * 130, 1, 96),
               ([2**80, -1, 3, 2**80, -1], 1, 64),
               ([TOKEN_LIMIT, 0, -TOKEN_LIMIT, TOKEN_LIMIT, 0], 1, 64)]
    for context, shortest, longest in special:
        arena = ArenaLookupDrafter(context, longest, shortest)
        production = ProductionLookup(context, longest, shortest)
        compare(arena, production, invariant=True)
        storage.append(arena.storage_stats())
    assert ArenaLookupDrafter(collision, longest=5).propose(1) == [111]

    rng = random.Random(20261028)
    for shortest, longest in (*configurations, (1, 64), (2, 96)):
        for alphabet in (2, 7, 251, 154880):
            initial = [rng.randrange(alphabet) for _ in range(31)]
            arena = ArenaLookupDrafter(initial, longest, shortest)
            production = ProductionLookup(initial, longest, shortest)
            for iteration in range(32):
                compare(arena, production, invariant=iteration < 2)
                added = [rng.randrange(alphabet)
                         for _ in range(rng.randrange(9))]
                arena.commit(iter(added))
                production.commit(iter(added))
                counters["incremental_contexts"] += 1

    for alphabet in (251, 154880):
        initial = [rng.randrange(alphabet) for _ in range(8192)]
        arena = ArenaLookupDrafter(initial)
        production = ProductionLookup(initial)
        for _ in range(8):
            compare(arena, production)
            added = [rng.randrange(alphabet) for _ in range(7)]
            arena.commit(iter(added))
            production.commit(iter(added))
            counters["large_contexts"] += 1
        storage.append(arena.storage_stats())
    return {"status": "pass", "coverage": counters,
            "signature_sha256": transcript.hexdigest(), "storage": storage,
            "timed": False, "performance_measured": False,
            "npu_executed": False}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tuple-arm", action="store_true")
    args = parser.parse_args()
    sources = [Path(__file__).resolve(), PRODUCTION]
    comparison_factory = None
    if args.tuple_arm:
        path = Path(__file__).with_name(
            "tilert-lookup-arena-tuple-prototype-20261028.py",
        ).resolve()
        sources.append(path)
        spec = importlib.util.spec_from_file_location("tuple_arena", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        comparison_factory = module.TupleArenaLookupDrafter
    hashes = {str(path): digest(path) for path in sources}
    try:
        result = run_checks(comparison_factory)
    except Exception as error:
        import traceback

        result = {"status": "fail", "error": repr(error),
                  "traceback": traceback.format_exc()}
    stable = hashes == {str(path): digest(path) for path in sources}
    result.update(created_at=datetime.now(timezone.utc).isoformat(),
                  source_sha256=hashes, source_stable=stable,
                  scope="CPU exact proposals and radix representation only",
                  comparison="tuple_arena" if args.tuple_arm else "production")
    if not stable:
        result["status"] = "fail"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
