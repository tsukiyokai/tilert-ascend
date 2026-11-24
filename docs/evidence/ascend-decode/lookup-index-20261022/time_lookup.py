"""Build and per-round cost, tuple keys against the rolling hash."""
import importlib.util
import json
import pathlib
import statistics
import sys
import time
import tracemalloc

sys.path.insert(0, "/home/shanshan/repo/tile-ai/TileRT-Ascend")

from tilert.models.glm_5_2_ascend.lookup_draft import LookupDrafter as Fast

spec = importlib.util.spec_from_file_location(
    "lookup_naive", "/home/shanshan/work/probe/lookup_naive.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Naive = module.LookupDrafter

prompt = json.load(
    open("/home/shanshan/work/probe/prompt-8k.json"))["prompt"]
stream = json.load(
    open("/home/shanshan/work/probe/real-streams.json"))["tail32"][0]

print(f"{'variant':>8} {'build ms':>9} {'peak MB':>8} {'entries':>9} "
      f"{'round us p50':>13} {'commit us p50':>14}")
for name, Drafter in (("naive", Naive), ("fast", Fast)):
    builds = []
    for _ in range(3):
        start = time.perf_counter()
        Drafter(prompt, 32, 2)
        builds.append((time.perf_counter() - start) * 1e3)
    tracemalloc.start()
    drafter = Drafter(prompt, 32, 2)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    proposes, commits = [], []
    position = 0
    while position < len(stream):
        start = time.perf_counter()
        proposal = drafter.propose(32)
        proposes.append((time.perf_counter() - start) * 1e6)
        matched = 0
        while (matched < len(proposal) and position + matched < len(stream)
               and proposal[matched] == stream[position + matched]):
            matched += 1
        took = matched + 1
        start = time.perf_counter()
        drafter.commit(stream[position:position + took])
        commits.append((time.perf_counter() - start) * 1e6)
        position += took
    print(f"{name:>8} {statistics.median(builds):>9.1f} {peak/1e6:>8.1f} "
          f"{len(drafter._latest):>9} {statistics.median(proposes):>13.2f} "
          f"{statistics.median(commits):>14.2f}")
