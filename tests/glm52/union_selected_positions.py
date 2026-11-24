"""How much of the twelve rows' selected positions is one shared set.

MergeKv gathers every row's 2048 positions separately: 24,576 gathers a call,
27.6 MB written and read back twice. But the rows of a speculative round are
one context plus up to eleven tokens, so if their selections nearly coincide a
single gather of the union, masked per row, does the same work. The union size
is the whole prize, and it is a property of the data, not of the kernel.
"""
import pathlib
import sys

import torch

path = pathlib.Path(sys.argv[1])
rows = torch.load(path, map_location="cpu")
print("shape", tuple(rows.shape), "dtype", rows.dtype,
      "min", int(rows.min()), "max", int(rows.max()))

width = int(sys.argv[2]) if len(sys.argv) > 2 else rows.shape[0]
used = rows[:width]
sets = [set(row[row >= 0].tolist()) for row in used]
total = sum(len(one) for one in sets)
union = set().union(*sets)
pairwise = len(sets[0] & sets[-1]) / max(len(sets[0]), 1)
print(f"rows {width} selected {total} union {len(union)} "
      f"ratio {total / max(len(union), 1):.2f}x")
print(f"row 0 against row {width - 1}: overlap {pairwise:.4f}")
for index in range(width):
    shared = len(sets[index] & sets[0]) / max(len(sets[0]), 1)
    print(f"  row {index:2} size {len(sets[index]):5} "
          f"overlap with row 0 {shared:.4f}")
