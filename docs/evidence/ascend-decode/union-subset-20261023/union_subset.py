"""How small can a round's expert union get if the rows are chosen for it?

`teacher-routes.pt` holds [windows, 32, 8] expert IDs from the engine's own
router. The union over all 32 rows is what a round pays. Choosing a subset of
rows that share experts pays less -- and verifies fewer rows, so the trade is
union against rows. Greedy min-union is the classic submodular pick; the
exchange pass after it catches the obvious greedy misses.
"""
import json
import statistics

import torch

routes = torch.load("/tmp/teacher-routes.pt", map_location="cpu")
ids = routes[chr(114)+chr(111)+chr(117)+chr(116)+chr(101)+chr(115)]
ids = ids.reshape(ids.shape[0], ids.shape[1], -1).tolist()
print("windows", len(ids), "rows", len(ids[0]), "top-k", len(ids[0][0]))


def union_size(rows, picked):
    seen = set()
    for index in picked:
        seen.update(rows[index])
    return len(seen)


def greedy(rows, keep):
    picked = []
    seen = set()
    remaining = set(range(len(rows)))
    while len(picked) < keep:
        best = min(remaining, key=lambda i: len(seen | set(rows[i])))
        picked.append(best)
        seen |= set(rows[best])
        remaining.discard(best)
    return picked


full, half, quarter = [], [], []
for rows in ids:
    full.append(union_size(rows, range(len(rows))))
    half.append(union_size(rows, greedy(rows, 16)))
    quarter.append(union_size(rows, greedy(rows, 8))) 
print(f"{'rows kept':>10} {'union mean':>11} {'median':>8} {'min':>5} {'max':>5}")
for name, data in (("32 (all)", full), ("16 greedy", half), ("8 greedy", quarter)):
    print(f"{name:>10} {statistics.mean(data):>11.2f} "
          f"{statistics.median(data):>8.1f} {min(data):>5} {max(data):>5}")
json.dump({"full": full, "greedy16": half, "greedy8": quarter},
          open("/tmp/union_subset.json", "w"))
