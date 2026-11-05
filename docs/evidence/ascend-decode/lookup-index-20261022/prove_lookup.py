"""Prove the rolling-hash index proposes exactly what the tuple index did.

Equality is checked round by round on both fixtures -- the engine's own two
78-layer streams and a perturbed source file -- at every width the engine can
capture, and the committed prefix is driven by the naive drafter so both see
the same context.
"""
import importlib.util
import json
import pathlib
import random
import re
import sys

sys.path.insert(0, "/home/shanshan/repo/tile-ai/TileRT-Ascend")

from tilert.models.glm_5_2_ascend.lookup_draft import LookupDrafter as Fast


def load_naive():
    path = pathlib.Path("/home/shanshan/work/probe/lookup_naive.py")
    spec = importlib.util.spec_from_file_location("lookup_naive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LookupDrafter


Naive = load_naive()
WORD = re.compile(r"\w+|\s+|[^\w\s]")


def compare(prompt, truth, width):
    naive = Naive(prompt, 32, 2)
    fast = Fast(prompt, 32, 2)
    position = rounds = 0
    while position < len(truth):
        rounds += 1
        want = naive.propose(width)
        got = fast.propose(width)
        if want != got:
            return rounds, want[:6], got[:6]
        matched = 0
        while (matched < len(want) and position + matched < len(truth)
               and want[matched] == truth[position + matched]):
            matched += 1
        took = matched + 1
        naive.commit(truth[position:position + took])
        fast.commit(truth[position:position + took])
        position += took
    return rounds, None, None


def main():
    base = "/home/shanshan/work/probe"
    prompt = json.load(open(f"{base}/prompt-8k.json"))["prompt"]
    streams = json.load(open(f"{base}/real-streams.json"))
    cases = [(f"{name}[{index}]", prompt, truth)
             for name, group in streams.items()
             for index, truth in enumerate(group)]

    vocabulary = {}
    text = pathlib.Path("/home/shanshan/repo/tile-ai/TileRT-Ascend/tilert/"
                        "models/glm_5_2_ascend/cann_model.py").read_text()
    body = []
    for piece in WORD.findall(text):
        vocabulary.setdefault(piece, len(vocabulary))
        body.append(vocabulary[piece])
    generator = random.Random(20261022)
    for rate in (0.0, 0.02, 0.05):
        edited = [token if generator.random() > rate
                  else generator.randrange(len(vocabulary)) for token in body]
        cases.append((f"file@{rate}", body, edited))

    total = mismatches = 0
    for name, source, truth in cases:
        for width in (2, 4, 6, 8, 12, 16, 24, 32):
            rounds, want, got = compare(source, truth, width)
            total += rounds
            if want is not None:
                mismatches += 1
                print(f"MISMATCH {name} width {width} round {rounds}: "
                      f"naive {want} fast {got}")
    print(f"{len(cases)} streams x 8 widths, {total} rounds compared, "
          f"{mismatches} mismatches")


main()
