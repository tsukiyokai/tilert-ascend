"""One fixed-budget copy-tree CPU replay; no model or device imports."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns


ROOT = Path("/workspace/offline/lookup-cpu-20261019/copy-tree-20261020")
SOURCE = ROOT / "source"
RECORD = Path("/workspace/experiments/lookup-20261019/chain-w8/rank-0.json")
MANIFEST = Path("/workspace/offline/prefix8k-allocation-20261017/capture/case-00/manifest.json")
sys.path.insert(0, str(SOURCE / "tests/glm52"))
from ngram_acceptance import lookup, matched_prefix

spec = importlib.util.spec_from_file_location(
    "actual_lookup", SOURCE / "tilert/models/glm_5_2_ascend/lookup_draft.py",
)
actual_lookup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(actual_lookup)
LookupDrafter = actual_lookup.LookupDrafter


class CopyOccurrences(LookupDrafter):
    """Retain all eligible ends using the actual index's eligibility rule."""

    def __init__(self, prompt):
        self.ends = {}
        super().__init__(prompt, longest=8, shortest=1)

    def commit(self, tokens):
        previous = len(self.context)
        super().commit(tokens)
        for end in range(max(previous, self.shortest), len(self.context)):
            for size in range(self.shortest, min(self.longest, end) + 1):
                key = tuple(self.context[end - size:end])
                self.ends.setdefault(key, []).append(end)

    def continuations(self, branches):
        for size in range(self.longest, self.shortest - 1, -1):
            if len(self.context) <= size:
                continue
            ends = self.ends.get(tuple(self.context[-size:]), ())
            found, seen = [], set()
            for end in reversed(ends):
                run = tuple(self.context[end:end + 31])
                if run and run not in seen:
                    found.append({"source_end": end, "tokens": list(run)})
                    seen.add(run)
                    if len(found) == branches:
                        break
            if found:
                return found, size
        return [], 0


def build_tree(runs, pending, budget=31):
    """Allocate by depth, then newest distinct continuation; merge prefixes."""
    padded = [run + [pending] * (31 - len(run)) for run in runs] or [[pending] * 31]
    nodes, children = [[-1, pending, 0]], [{}]
    frontier = [0] * len(padded)
    for depth in range(31):
        for branch, run in enumerate(padded):
            parent, token = frontier[branch], run[depth]
            child = children[parent].get(token)
            if child is None:
                if len(nodes) == budget + 1:
                    return nodes, children
                child = len(nodes)
                nodes.append([parent, token, depth + 1])
                children.append({})
                children[parent][token] = child
            frontier[branch] = child
    return nodes, children


def traverse(children, oracle, budget=None):
    """Follow scalar target predictions only after the whole tree is fixed."""
    node, path = 0, []
    while budget is None or len(path) < budget:
        prediction = oracle(tuple(path))
        child = children[node].get(prediction)
        if child is None:
            return path, prediction
        path.append(prediction)
        node = child
    return path, None


def leaf_match(nodes, children, actual):
    best = 0
    for node, edges in enumerate(children):
        if edges:
            continue
        path = []
        while node:
            parent, token, _ = nodes[node]
            path.append(token)
            node = parent
        best = max(best, matched_prefix(path[::-1], actual))
    return best


def check_scalar_contracts():
    checks = []
    runs = [[10, 11, 12], [10, 11, 13], [10, 11, 12]]
    nodes, children = build_tree(runs, 9, budget=4)
    assert nodes == [[-1, 9, 0], [0, 10, 1], [1, 11, 2], [2, 12, 3], [2, 13, 3]]
    path, bonus = traverse(children, lambda history: 10 + len(history))
    assert path + [bonus] == [10, 11, 12, 13]
    checks.extend(["duplicate_prefixes_coalesce", "scalar_tree_equals_serial_successors"])
    nodes, children = build_tree([[10, 11], [20, 21]], 9, budget=3)
    assert nodes[-1] == [1, 11, 2]
    assert 21 not in children[2]
    checks.append("last_node_tie_uses_newest_branch_first")
    empty = CopyOccurrences([1, 2, 3])
    assert empty.continuations(8) == ([], 0)
    nodes, children = build_tree([], 3)
    assert len(nodes) == 32 and all(node[1] == 3 for node in nodes)
    checks.append("no_occurrence_uses_pending_padding")
    original = [[10 + i for i in range(31)]]
    nodes, children = build_tree(original, 9)
    actual = [10, 11]
    # This oracle has only the recorded budget. Reaching it clips the bonus.
    path, bonus = traverse(children, lambda history: actual[len(history)],
                           len(actual))
    assert path == actual and bonus is None
    assert min(len(path) + 1, len(actual)) == 2
    checks.append("final_budget_excludes_unobserved_bonus")
    assert build_tree(original, 9) == build_tree(original, 9)
    checks.append("allocation_is_deterministic")
    return checks


def replay(prompt, tokens, branches):
    start = perf_counter_ns()
    drafter = (LookupDrafter(prompt, longest=8, shortest=1)
               if branches == 1 else CopyOccurrences(prompt))
    drafter.commit(tokens[:1])
    setup_ns = perf_counter_ns() - start
    position, rounds = 1, []
    while position < len(tokens):
        assert drafter.context == prompt + tokens[:position]
        pending = tokens[position - 1]
        start = perf_counter_ns()
        if branches == 1:
            run = drafter.propose(31)
            padded = run + [pending] * (31 - len(run))
            proposal_ns = perf_counter_ns() - start
            sources, suffix = [{"tokens": run}], None
            nodes, children = build_tree([padded], pending)
        else:
            sources, suffix = drafter.continuations(branches)
            runs = [source["tokens"] for source in sources]
            nodes, children = build_tree(runs, pending)
            proposal_ns = perf_counter_ns() - start
            assert runs == lookup(drafter.context, 31, 8, 1, branches)
            assert (runs[0] if runs else []) == drafter.propose(31)
        assert len(nodes) == 32
        assert all(0 <= parent < index and depth == nodes[parent][2] + 1
                   for index, (parent, _, depth) in enumerate(nodes[1:], 1))
        remaining = tokens[position:]
        path, _ = traverse(children, lambda history: remaining[len(history)],
                           len(remaining))
        matched = len(path)
        assert matched == leaf_match(nodes, children, remaining)
        accepted = min(matched + 1, len(remaining))
        start = perf_counter_ns()
        drafter.commit(tokens[position:position + accepted])
        commit_ns = perf_counter_ns() - start
        rounds.append({"position": position, "pending": pending,
                       "root_logical_position": len(prompt) + position - 1,
                       "suffix": suffix, "sources": sources, "nodes": nodes,
                       "matched": matched, "accepted": accepted,
                       "proposal_ns": proposal_ns, "commit_ns": commit_ns})
        position += accepted
    assert position == len(tokens)
    assert sum(step["accepted"] for step in rounds) == len(tokens) - 1
    return {"branches": branches, "setup_ns": setup_ns, "rounds": rounds}


os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
checks = check_scalar_contracts()
record = json.loads(RECORD.read_text())
manifest = json.loads(MANIFEST.read_text())
requests = [item["tokens"] for item in record["requests"] if not item["warmup"]]
assert len({tuple(tokens) for tokens in requests}) == 1
prompt, tokens = manifest["prompt_token_ids"], requests[0]
raw = [replay(prompt, tokens, branches) for branches in (1, 2, 4, 8)]
assert len(raw[0]["rounds"]) == 147
results = []
for arm in raw:
    rounds = arm["rounds"]
    accepted = sum(step["accepted"] for step in rounds)
    times = [step["proposal_ns"] for step in rounds]
    results.append({
        "branches": arm["branches"], "rounds": len(rounds),
        "committed_tokens": accepted, "A": accepted / len(rounds),
        "verification_rows": 32 * len(rounds),
        "rows_per_committed_token": 32 * len(rounds) / accepted,
        "strict_win_round_cost_ratio_vs_flat32": len(raw[0]["rounds"]) / len(rounds),
        "setup_ms": arm["setup_ns"] / 1e6,
        "proposal_build_total_ms": sum(times) / 1e6,
        "proposal_build_p50_us": statistics.median(times) / 1e3,
        "proposal_build_max_us": max(times) / 1e3,
        "commit_total_ms": sum(step["commit_ns"] for step in rounds) / 1e6,
        "max_matched_drafts": max(step["matched"] for step in rounds),
        "accepted_histogram": dict(sorted(Counter(step["accepted"] for step in rounds).items())),
        "last_round": {key: rounds[-1][key] for key in ("position", "matched", "accepted")},
    })
files = [RECORD, MANIFEST, Path(__file__),
         SOURCE / "tests/glm52/ngram_acceptance.py",
         SOURCE / "tilert/models/glm_5_2_ascend/lookup_draft.py"]
raw_path = ROOT / "raw.json"
raw_path.write_text(json.dumps(raw) + "\n")
summary = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "CPU copy-only bounded-trie acceptance on one old unaccepted free-running trajectory; no NPU, actual model routes, tree-kernel support, or TPOT claim",
    "input": {"prompt_tokens": len(prompt), "output_tokens": len(tokens),
              "prefill_output_excluded": 1, "committed_tokens": len(tokens) - 1,
              "identical_nonwarmup_requests": len(requests), "independent_inputs": 1},
    "policy": {"shortest": 1, "longest": 8, "nonroot_node_budget": 31,
               "occurrences": "Only the longest suffix length with a known continuation; descending source end; distinct observed continuations truncated to 31 tokens",
               "padding": "Repeat current pending to length 31 separately on each continuation; no occurrence gives one all-pending continuation",
               "allocation": "Ascending depth, then newest distinct continuation first; merge equal parent/token edges; stop after 31 unique non-root nodes",
               "context": "Prompt plus this policy's committed output, including pending",
               "verification": "Freeze all nodes first, then scalar greedy traversal; independent maximum leaf-prefix match agrees at every round; commit min(match+1, remaining output budget)",
               "rows": "Pending root plus 31 non-root nodes, including padding and final clipped commit"},
    "source_and_input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    "checks": checks + ["all_rounds_match_reverse_scan_continuations", "all_tree_rounds_latest_branch_matches_actual_drafter", "all_rounds_scalar_equals_leaf_prefix", "each_policy_owns_commit_boundaries", "1023_total_commits_each", "flat32_reproduces_147_rounds"],
    "cpu_timing_scope": "One pinned-core replay sample per arm. Timed proposal construction excludes oracle, reverse-scan checks and raw serialization. Tree prototype additionally maintains occurrence histories; setup and commit costs are shown separately. Not a stable paired CPU benchmark.",
    "results": results,
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
