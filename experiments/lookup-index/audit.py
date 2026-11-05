"""Audit exact lookup prefixes, public counters and scalar session tails."""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import random
import sys
import traceback


def oracle(context, count, longest, shortest):
    for size in range(min(longest, len(context) - 1), shortest - 1, -1):
        for end in range(len(context) - 1, size - 1, -1):
            if context[end - size:end] == context[-size:]:
                return context[end:end + count]
    return []


def verify_nodes(drafter):
    context = drafter.context
    expected = {}
    for end in range(drafter.shortest, len(context)):
        window = tuple(reversed(
            context[max(0, end - drafter.longest):end],
        ))
        for length in range(1, len(window) + 1):
            expected[window[:length]] = end
    stack = [(drafter._root, ())]
    visited, prefixes = set(), 0
    while stack:
        node, prefix = stack.pop()
        assert id(node) not in visited
        visited.add(id(node))
        for token, child in node.children.items():
            assert 0 < child.length <= drafter.longest
            assert 0 <= child.hi - child.length + 1 <= child.hi < len(context)
            label = tuple(reversed(
                context[child.hi - child.length + 1:child.hi + 1],
            ))
            assert label[0] == token
            assert len(prefix) + len(label) <= drafter.longest
            for length in range(1, len(label) + 1):
                wanted = prefix + label[:length]
                assert child.latest == expected.pop(wanted), (
                    wanted, child.latest,
                )
                prefixes += 1
            stack.append((child, prefix + label))
    assert not expected
    assert len(visited) <= 1 + 2 * max(0, len(context) - drafter.shortest)
    return prefixes


def check_bad_compression(drafter_type):
    drafter = drafter_type([1, 2, 7, 9, 2, 8, 2], longest=3, shortest=1)
    verify_nodes(drafter)
    expected = oracle(drafter.context, 1, 3, 1)
    assert drafter.propose(1) == expected == [8]
    branch = drafter._root.children[2]
    edge = branch.children[1]
    edge.hi += branch.length
    edge.length += branch.length
    drafter._root.children[2] = edge
    endpoint_latest = max(
        end for end in range(2, len(drafter.context))
        if drafter.context[end - 2:end] == [1, 2]
    )
    assert edge.latest == endpoint_latest == 2
    corrupted = drafter.propose(1)
    assert corrupted == [7] and corrupted != expected
    try:
        verify_nodes(drafter)
    except AssertionError:
        return {"valid_proposal": expected, "corrupted_proposal": corrupted,
                "full_edge_endpoint_latest": endpoint_latest,
                "inner_prefix_error_rejected": True}
    raise AssertionError("A wrong compressed-edge prefix was accepted")


def check_proposals(drafter_type):
    checks = contexts = prefix_checks = 0
    transcript = hashlib.sha256()
    for length in range(7):
        for values in product(range(3), repeat=length):
            context = list(values)
            for shortest, longest in ((1, 1), (1, 2), (1, 8), (2, 2),
                                      (2, 8), (4, 32), (32, 32)):
                drafter = drafter_type(iter(context[:-2]), longest, shortest)
                prior = drafter.report()
                drafter.commit(iter(()))
                assert drafter.report() == prior
                drafter.commit(iter(context[-2:]))
                hits = 0
                for calls, count in enumerate((0, 1, 7), 1):
                    expected = oracle(context, count, longest, shortest)
                    actual = drafter.propose(count)
                    assert actual == expected
                    hits += bool(expected)
                    assert drafter.report() == {
                        "drafter": "lookup", "longest_tail": longest,
                        "shortest_tail": shortest,
                        "context_tokens": len(context), "rounds": calls,
                        "tail_hits": hits,
                    }
                    transcript.update(json.dumps([
                        context, shortest, longest, count, expected, actual,
                    ]).encode())
                    checks += 1
                prefix_checks += verify_nodes(drafter)
                contexts += 1
    rng = random.Random(319)
    for shortest, longest in ((1, 32), (2, 32), (32, 32), (1, 64), (1, 96)):
        context = [rng.randrange(154880) for _ in range(110)]
        drafter = drafter_type(context, longest, shortest)
        for step in range(80):
            for count in (0, 1, 31):
                expected = oracle(context, count, longest, shortest)
                actual = drafter.propose(count)
                assert actual == expected
                transcript.update(json.dumps([
                    context, shortest, longest, count, expected, actual,
                ]).encode())
                checks += 1
            if step % 20 == 0:
                prefix_checks += verify_nodes(drafter)
            added = [rng.choice((0, 1, 65536, 154879))
                     for _ in range(rng.randrange(8))]
            context.extend(added)
            drafter.commit(iter(added))
    for longest in (1, 2, 8, 32, 64, 96):
        context = [5] * 128
        drafter = drafter_type(context, longest, 1)
        assert drafter.propose(31) == oracle(context, 31, longest, 1)
        prefix_checks += verify_nodes(drafter)
        checks += 1
    left = [30000, 30712, 30000, 32048, 30503]
    right = [30306, 30000, 30004, 30000, 30000]
    context = left + [111] + right + [222] + left[-4:] + [333] + left
    assert drafter_type(context, 5, 2).propose(1) == [111]
    return {"proposal_checks": checks, "invariant_contexts": contexts,
            "invariant_prefix_checks": prefix_checks,
            "max_longest": 96, "transcript_sha256": transcript.hexdigest(),
            "collision_proposal": [111]}


def check_sessions(drafter_type):
    from check_round_invariance import (
        PROMPT, RULES, ScalarTarget, serial_decode,
    )
    from tilert.models.glm_5_2_ascend.mtp_round import MtpRoundExecutor
    from tilert.models.glm_5_2_ascend.mtp_session import AscendMtpPDEngine
    from tilert.models.glm_5_2_ascend.session import PrefixRequest

    results = []
    for rule, width, room, stop in product(
        RULES, (2, 8, 32), (1, 2, 3, 7, 31, 32, 95), (False, True),
    ):
        target = ScalarTarget(rule, PROMPT, len(PROMPT) + room)
        expected = serial_decode(rule, PROMPT, min(96, room + 1))
        stop_ids = (expected[1],) if stop else ()
        stopped = bool(stop_ids and stop_ids[0] in expected)
        if stopped:
            expected = expected[:expected.index(stop_ids[0])]
        first = rule.predict(target.rows[-1][1], len(PROMPT))

        class Backend:
            def __init__(self):
                self.loop = None
                self.drafter = None

            @property
            def position(self):
                return target.position

            @property
            def capacity(self):
                return target.capacity

            def inject(self, request):
                assert request.seq_len == self.position

            def start_mtp(self, opening, budget, actual_width, stops):
                self.drafter = drafter_type(PROMPT)
                self.drafter.commit((opening,))
                self.loop = MtpRoundExecutor(
                    target, None, actual_width, budget, stops,
                    drafter=self.drafter,
                )
                self.loop.start(opening, None)

            def advance_mtp_n(self, count):
                rounds = []
                for _ in range(count):
                    rounds.append(self.loop.advance())
                    assert target.position == self.loop.state.main_length
                    if self.loop.state.finish_reason is not None:
                        break
                return rounds

            def finish_mtp(self, retained):
                target.truncate(retained)

        backend = Backend()
        session = AscendMtpPDEngine(backend, width, stop_ids)
        session.inject(PrefixRequest("lookup-tail", len(PROMPT)))
        output = session.decode(first, 96)
        assert output == expected
        assert session.last_stats["finish_reason"] == (
            "stop" if stopped else "length")
        consumed = output if stopped else output[:-1]
        state, cache = rule.start, []
        for token in (*PROMPT, *consumed):
            state = rule.advance(state, token)
            cache.append((token, state))
        assert target.rows[:target.position] == cache
        assert target.position == len(cache) <= target.capacity
        results.append({
            "rule": rule.__name__, "width": width, "room": room,
            "eos_tokens": list(stop_ids), "output_token_ids": output,
            "expected_token_ids": expected, "valid_cache_rows": len(cache),
            "valid_cache_sha256": hashlib.sha256(
                json.dumps(cache).encode()).hexdigest(),
            "finish_reason": session.last_stats["finish_reason"],
            "lookup_report": (backend.drafter.report()
                              if backend.drafter is not None else None),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root), str(root / "tests/glm52")]
    from tilert.models.glm_5_2_ascend.lookup_draft import LookupDrafter

    paths = [Path(__file__), *(root / name for name in (
        "tilert/models/glm_5_2_ascend/lookup_draft.py",
        "tests/glm52/check_lookup.py", "tests/glm52/check_round_invariance.py",
        "tilert/models/glm_5_2_ascend/mtp_round.py",
        "tilert/models/glm_5_2_ascend/mtp_state.py",
        "tilert/models/glm_5_2_ascend/mtp_session.py",
        "tilert/models/glm_5_2_ascend/session.py",
    ))]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    report = {"status": "running", "command": [sys.executable, *sys.argv],
              "scope": "CPU proposer invariants and real scalar session "
                       "capacity/EOS checks; no performance or NPU execution",
              "python": platform.python_version(), "source_sha256": hashes}
    try:
        report["bad_compression"] = check_bad_compression(LookupDrafter)
        report["proposals"] = check_proposals(LookupDrafter)
        report["sessions"] = check_sessions(LookupDrafter)
        report["source_stable"] = all(
            hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha
            for name, sha in hashes.items())
        assert report["source_stable"]
        report["status"] = "pass"
    except Exception:
        report.update(status="fail", error=traceback.format_exc())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"],
                      "proposal_checks": report.get("proposals", {}).get(
                          "proposal_checks"),
                      "sessions": len(report.get("sessions", [])),
                      "output": str(args.output)}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
