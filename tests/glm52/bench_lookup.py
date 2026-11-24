"""Time CPU lookup on a frozen trajectory, including index construction."""

import argparse
from functools import partial
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns

from ngram_acceptance import lookup, matched_prefix


class ReverseScan:
    def __init__(self, prompt, shortest=2, *, longest=32):
        self.context = list(prompt)
        self.shortest = shortest
        self.longest = longest

    def commit(self, tokens):
        self.context.extend(tokens)

    def propose(self, count):
        branches = lookup(self.context, count, self.longest, self.shortest)
        return branches[0] if branches else []


def replay(factory, prompt, tokens, width):
    start = perf_counter_ns()
    drafter = factory(prompt)
    drafter.commit(tokens[:1])
    setup_ns = perf_counter_ns() - start
    position, rounds = 1, []
    while position < len(tokens):
        start = perf_counter_ns()
        proposal = drafter.propose(width - 1)
        propose_ns = perf_counter_ns() - start
        padding = [tokens[position - 1]] * (width - 1 - len(proposal))
        padded = proposal + padding
        matched = matched_prefix(padded, tokens[position:])
        accepted = min(matched + 1, len(tokens) - position)
        start = perf_counter_ns()
        drafter.commit(tokens[position:position + accepted])
        commit_ns = perf_counter_ns() - start
        rounds.append({"position": position, "proposal": proposal,
                       "accepted": accepted, "propose_ns": propose_ns,
                       "commit_ns": commit_ns})
        position += accepted
    signature = [(r["position"], r["proposal"], r["accepted"])
                 for r in rounds]
    digest = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
    return {
        "setup_ns": setup_ns,
        "total_ns": setup_ns + sum(r["propose_ns"] + r["commit_ns"]
                                   for r in rounds),
        "signature_sha256": digest, "rounds": rounds,
    }


def stats(values):
    values = sorted(values)
    return {"p50": statistics.median(values),
            "p90": values[round(0.90 * (len(values) - 1))],
            "p99": values[round(0.99 * (len(values) - 1))],
            "std": statistics.pstdev(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--shortest", type=int, choices=(1, 2), default=2)
    parser.add_argument("--longest", type=int, default=32)
    args = parser.parse_args()
    if args.longest < args.shortest:
        parser.error("--longest must be at least --shortest")
    native = any(str(args.candidate).endswith(suffix)
                 for suffix in importlib.machinery.EXTENSION_SUFFIXES)
    name = args.candidate.name.split(".", 1)[0] if native else "candidate"
    spec = importlib.util.spec_from_file_location(name, args.candidate)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    factories = {
        "scan": partial(ReverseScan, shortest=args.shortest,
                        longest=args.longest),
        "indexed": partial(module.LookupDrafter, shortest=args.shortest,
                           longest=args.longest),
    }
    record = json.loads(args.record.read_text())
    prompt = json.loads(args.manifest.read_text())["prompt_token_ids"]
    requests = [r["tokens"] for r in record["requests"] if not r["warmup"]]
    trajectories = list({tuple(tokens) for tokens in requests})
    report = {
        "scope": "CPU proposer only; no target inference or NPU timing",
        "hostname": platform.node(), "python": platform.python_version(),
        "prompt_tokens": len(prompt), "unique_trajectories": len(trajectories),
        "warmup": args.warmup, "samples": args.samples,
        "shortest_tail": args.shortest,
        "longest_tail": args.longest,
        "source_sha256": {str(path): hashlib.sha256(
                              path.read_bytes()).hexdigest()
                          for path in (args.record, args.manifest,
                                       args.candidate, Path(__file__),
                                       Path(__file__).with_name(
                                           "ngram_acceptance.py"))},
        "results": [],
    }
    for tokens in trajectories:
        for width in args.widths:
            arms = {name: [] for name in factories}
            for sample in range(-args.warmup, args.samples):
                names = (("scan", "indexed") if sample % 2
                         else ("indexed", "scan"))
                pair = {name: replay(factories[name], prompt, tokens, width)
                        for name in names}
                assert pair["scan"]["signature_sha256"] == (
                    pair["indexed"]["signature_sha256"])
                if sample >= 0:
                    for name, result in pair.items():
                        arms[name].append(result)
            summary = {}
            for name, values in arms.items():
                summary[name] = {
                    "request_cpu_ms": stats([v["total_ns"] / 1e6
                                             for v in values]),
                    "setup_ms": stats([v["setup_ns"] / 1e6 for v in values]),
                    "propose_us": stats([
                        r["propose_ns"] / 1e3
                        for v in values for r in v["rounds"]]),
                    "commit_us": stats([r["commit_ns"] / 1e3
                                        for v in values for r in v["rounds"]]),
                }
            item = {"width": width, "output_tokens": len(tokens),
                    "rounds": len(arms["scan"][0]["rounds"]),
                    "proposals_identical": True, "summary": summary,
                    "arms": arms}
            report["results"].append(item)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report) + "\n")
            print(json.dumps({k: v for k, v in item.items() if k != "arms"}),
                  flush=True)


if __name__ == "__main__":
    main()
