"""How much of an arm's output keeps the prompt's structure.

A corrupted cache does not produce wrong-but-plausible text; it produces
repetition and noise. Two known-bad arms and three that passed their gate
calibrate the measure: the share of the output's two-token contexts that occur
in the prompt, and the output's distinct-token count.
"""
import json
import sys


def contexts_in_prompt(prompt, output):
    pairs = {(prompt[i], prompt[i + 1]) for i in range(len(prompt) - 1)}
    seen = [(output[i], output[i + 1]) for i in range(len(output) - 1)]
    return sum(pair in pairs for pair in seen) / max(1, len(seen))


prompt = json.load(open("/tmp/man.json"))["prompt_token_ids"]
print(f"{'arm':22} {'tpot':>8} {'accepted':>8} {'distinct':>8} {'in prompt':>9}")
for tag in sys.argv[1:]:
    record = json.load(open(f"/root/work/out/{tag}/rank-0.json"))
    if record.get("status") != "completed":
        print(f"{tag:22} no record")
        continue
    measured = [r for r in record["requests"] if not r["warmup"]][0]
    tokens = measured["returned_tokens"]
    session = measured["session"]
    accepted = session["accepted"] / session["completed_rounds"]
    print(f"{tag:22} {measured['tpot_ms']:8.3f} {accepted:8.3f} "
          f"{len(set(tokens)):8d} {contexts_in_prompt(prompt, tokens):9.3f}")
