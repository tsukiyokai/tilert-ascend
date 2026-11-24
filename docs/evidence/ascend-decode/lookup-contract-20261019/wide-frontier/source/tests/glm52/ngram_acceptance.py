"""Replay lookup proposals against a fixed serial token trajectory.

Include the capture manifest to reproduce the deployed prompt context. This
computes acceptance on that trajectory, not target numerical equivalence or
latency. Multiple branches assume independent causal target verification.
"""
import argparse
import json
import pathlib


def lookup(context, proposal_length, longest, shortest, branches=1):
    """Propose what followed the context's tail, one branch per repeat.

    Longest match first, which is what makes a hit worth trusting: a 4-gram
    repeat predicts its continuation far better than a 1-gram repeat does.

    A tail that repeats several times offers several continuations, and a
    verify wide enough to hold them can take whichever survives. Standard
    prompt lookup keeps only the most recent and discards the rest, which is
    the whole of why it misses two rounds in three.
    """
    found = []
    for size in range(longest, shortest - 1, -1):
        if len(context) <= size:
            continue
        tail = context[-size:]
        for start in range(len(context) - size - 1, -1, -1):
            if context[start:start + size] != tail:
                continue
            branch = context[start + size:start + size + proposal_length]
            if branch and branch not in found:
                found.append(branch)
                if len(found) == branches:
                    return found
        if found:
            return found
    return found


def matched_prefix(proposal, actual):
    count = 0
    for proposed, emitted in zip(proposal, actual):
        if proposed != emitted:
            break
        count += 1
    return count


def simulate(tokens, proposal_length, longest, shortest, branches=1,
             prompt=()):
    """Walk deployed padding and budget-clipped commits on this trajectory."""
    rounds, position = [], 1
    while position < len(tokens):
        proposals = lookup(list(prompt) + tokens[:position], proposal_length,
                           longest, shortest, branches) or [[]]
        pending = tokens[position - 1]
        padded = [branch + [pending] * (proposal_length - len(branch))
                  for branch in proposals]
        count = max(matched_prefix(branch, tokens[position:])
                    for branch in padded)
        accepted = min(count + 1, len(tokens) - position)
        rounds.append({"matched_drafts": count, "accepted": accepted})
        position += accepted
    return rounds


def report(name, rounds, width):
    accepted = sum(item["accepted"] for item in rounds)
    matched = [item["matched_drafts"] for item in rounds]
    hits = sum(count > 0 for count in matched)
    print(f"  {name:<26} width {width:3d}  rounds {len(rounds):4d}  "
          f"acceptance {accepted / len(rounds):6.3f}  "
          f"hit {hits / len(rounds):5.1%}  "
          f"max {max(matched):3d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=pathlib.Path)
    parser.add_argument("--proposal", type=int, nargs="+",
                        default=[7, 15, 23, 31])
    parser.add_argument("--branches", type=int, nargs="+", default=[1])
    parser.add_argument("--manifest", type=pathlib.Path,
                        help="capture manifest carrying prompt_token_ids")
    parser.add_argument("--longest", type=int, default=8)
    parser.add_argument("--shortest", type=int, default=2)
    args = parser.parse_args()

    record = json.loads(args.record.read_text())
    prompt = ()
    if args.manifest:
        prompt = json.loads(args.manifest.read_text())["prompt_token_ids"]
    requests = [request for request in record["requests"]
                if not request.get("warmup")]
    for index, request in enumerate(requests):
        tokens = request["tokens"]
        spans = request["session"]["accepted_spans"]
        print(f"request {index}: {len(tokens)} tokens")
        report("MTP chain, measured", spans, spans[0]["width"])
        for length in args.proposal:
            for branches in args.branches:
                matched = simulate(tokens, length, args.longest,
                                   args.shortest, branches, prompt)
                report(f"lookup k={length} b={branches}", matched,
                       length * branches + 1)
        scope = f"{len(prompt)}-token prompt + generation" if prompt \
            else "generation only"
        print(f"  {'':<26} lookup {args.shortest}..{args.longest}, {scope}")


if __name__ == "__main__":
    main()
