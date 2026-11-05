"""Price a merged chain-and-lookup tree from one measured chain record.

Greedy target decoding makes the emitted stream deterministic: speculation
only decides how many of those tokens each round delivers. So a drafter can
be replayed against the recorded stream, and a merged tree -- both
continuations offered from one point, whichever runs further kept -- can be
priced before its attention mask exists.

Two conventions have to match the engine or the answer is wrong by a token.
A round's pending token is the last one committed, so the drafter's context
ends at it and ``propose`` needs nothing passed; and a round must advance by
what it accepted. Scoring every configuration at the *chain's* round starts
instead counts one long repeat once per boundary it spans, which at the
chain's 4.5 tokens a round inflates a 30-token run sevenfold.

The chain's match length is only on record at the boundaries the measured run
visited, so a merged walk has to treat it as independent of position. That
assumption is checkable: ``--correlate`` reports the pairing.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tilert.models.glm_5_2_ascend.lookup_draft import LookupDrafter

# The width-8 chain this record measured: its round, and the six draft
# forwards plus the conditioned pass that a host drafter removes. Keeping the
# chain costs that share of every round, so a merged tree has to earn it back
# in acceptance. Per-row and whole-round cost extrapolations are withdrawn
# pending a device measurement, and nothing here depends on them.
ROUND_MS = 53.05
CHAIN_MS = 12.35
BREAK_EVEN = ROUND_MS / (ROUND_MS - CHAIN_MS)


def matched_prefix(proposal, actual):
    count = 0
    for proposed, emitted in zip(proposal, actual):
        if proposed != emitted:
            break
        count += 1
    return count


def start(prompt, tokens, shortest):
    """A drafter holding what the engine holds before its first round."""
    drafter = LookupDrafter(prompt, shortest=shortest)
    drafter.commit(tokens[:1])
    return drafter


def walk(prompt, tokens, depth, shortest, chain=(), chain_depth=0):
    """Accept what each round earns, then start the next round after it.

    `chain` is the measured distribution of MTP chain match lengths, dealt in
    recorded order, one per round. Empty means the lookup drafts alone.
    """
    drafter = start(prompt, tokens, shortest)
    rounds = hits = 0
    position = 0
    while position < len(tokens) - 1:
        found = matched_prefix(drafter.propose(depth), tokens[position + 1:])
        hits += found > 0
        matched = found
        if chain:
            matched = max(min(chain[rounds % len(chain)], chain_depth),
                          min(found, depth))
        count = min(matched + 1, len(tokens) - 1 - position)
        drafter.commit(tokens[position + 1:position + 1 + count])
        position += count
        rounds += 1
    return {"rounds": rounds, "acceptance": (len(tokens) - 1) / rounds,
            "hit": hits / rounds}


def report(name, rows, acceptance, reference=None):
    against = ""
    if reference:
        against = (f"  {acceptance / reference:5.3f}x lookup alone, "
                   f"break-even {BREAK_EVEN:5.3f}")
    print(f"  {name:<24} rows {rows:3d}  "
          f"acceptance {acceptance:6.3f}{against}")


def correlate(prompt, tokens, spans, depth, shortest):
    """Is the chain's match length independent of the lookup's?

    Both are read at the chain's own boundaries, the only place the chain's is
    known. Inflated as an acceptance, valid as a pairing.
    """
    drafter = start(prompt, tokens, shortest)
    pairs = []
    position = 0
    for span in spans:
        if position >= len(tokens) - 1:
            break
        found = matched_prefix(drafter.propose(depth), tokens[position + 1:])
        pairs.append((span["matched_drafts"], found))
        count = span["accepted"]
        drafter.commit(tokens[position + 1:position + 1 + count])
        position += count
    chain_mean = sum(one for one, _ in pairs) / len(pairs)
    look_mean = sum(other for _, other in pairs) / len(pairs)
    spread = [(sum((value - mean) ** 2 for value in column) / len(column))
              ** 0.5
              for column, mean in ((
                  [one for one, _ in pairs], chain_mean), (
                  [other for _, other in pairs], look_mean))]
    covariance = sum((one - chain_mean) * (other - look_mean)
                     for one, other in pairs) / len(pairs)
    miss = [one for one, other in pairs if other == 0]
    correlation = covariance / (spread[0] * spread[1])
    print(f"  {len(pairs)} boundaries: chain mean {chain_mean:.3f}, "
          f"lookup mean {look_mean:.3f}, r {correlation:+.3f}")
    print(f"  chain match where the lookup misses: {len(miss)} rounds, "
          f"mean {sum(miss) / len(miss):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=pathlib.Path)
    parser.add_argument("manifest", type=pathlib.Path,
                        help="capture manifest carrying prompt_token_ids")
    parser.add_argument("--lookup-depth", type=int, nargs="+",
                        default=[7, 15, 23, 31, 63])
    parser.add_argument("--shortest", type=int, default=2)
    parser.add_argument("--correlate", type=int,
                        help="check chain-lookup independence at this depth")
    args = parser.parse_args()

    record = json.loads(args.record.read_text())
    prompt = json.loads(args.manifest.read_text())["prompt_token_ids"]
    request = next(request for request in record["requests"]
                   if not request.get("warmup"))
    tokens = request["tokens"]
    spans = request["session"]["accepted_spans"]
    chain_depth = spans[0]["width"] - 1
    chain = [span["matched_drafts"] for span in spans]

    print(f"{len(tokens)} tokens, {len(spans)} measured rounds, "
          f"{len(prompt)}-token prompt, shortest tail {args.shortest}")
    report("chain, measured", 1 + chain_depth, (len(tokens) - 1) / len(spans))
    for depth in args.lookup_depth:
        alone = walk(prompt, tokens, depth, args.shortest)
        merged = walk(prompt, tokens, depth, args.shortest, chain, chain_depth)
        print(f"  k={depth:<3d} tail hit {alone['hit']:5.1%}")
        report(f"lookup alone k={depth}", 1 + depth, alone["acceptance"])
        report(f"merged k={depth}", 1 + chain_depth + depth,
               merged["acceptance"], alone["acceptance"])
    if args.correlate:
        correlate(prompt, tokens, spans, args.correlate, args.shortest)


if __name__ == "__main__":
    main()
