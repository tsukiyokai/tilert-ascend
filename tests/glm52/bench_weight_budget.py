"""How many weight bytes does one rank actually read per decode round?

Everything about reachability hangs on this number: the round's budget at
2 ms/token is acceptance x 2 ms, and whatever share the weight stream takes
is not recoverable by any amount of fusion. Counted from the checkpoint index
rather than from the architecture, so packed int4 and per-expert scales are
counted as they are stored.
"""

import collections
import json
import re

INDEX = "/workspace/models/GLM-5.2-w4a8c8/quant_model_weights.safetensors.index.json"
SAFETENSORS = "/workspace/models/GLM-5.2-w4a8c8/"
TP = 16
ACTIVATED_EXPERTS = 27.121   # measured union over a verify of four rows
TOTAL_EXPERTS = 256
CEILING_GIGABYTES = 1190.77


def tensor_bytes():
    """Read every shard's header once; return name -> stored byte length."""
    import os
    import struct

    sizes = {}
    shards = sorted({name for name in
                     json.load(open(INDEX))["weight_map"].values()})
    for shard in shards:
        path = os.path.join(SAFETENSORS, shard)
        with open(path, "rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            sizes[name] = end - begin
    return sizes


sizes = tensor_bytes()
groups = collections.Counter()
for name, size in sizes.items():
    if ".mlp.experts." in name:
        groups["routed_experts"] += size
    elif re.search(r"\.layers\.\d+\.(self_attn|input_layernorm|post_attention)",
                   name):
        groups["attention"] += size
    elif re.search(r"\.layers\.\d+\.", name):
        groups["dense_and_shared"] += size
    else:
        groups["embedding_head_and_norm"] += size

total = sum(groups.values())
print(f"checkpoint total {total / 2**30:.2f} GiB across {len(sizes)} tensors")
for name, size in groups.most_common():
    print(f"  {name:28s}{size / 2**30:8.2f} GiB")

expert_fraction = ACTIVATED_EXPERTS / TOTAL_EXPERTS
read = (groups["routed_experts"] * expert_fraction
        + groups["attention"] + groups["dense_and_shared"]
        + groups["embedding_head_and_norm"])
per_rank = read / TP
floor_ms = per_rank / (CEILING_GIGABYTES * 1e9) * 1e3
print()
print(f"activated experts {ACTIVATED_EXPERTS} of {TOTAL_EXPERTS} "
      f"=> {expert_fraction:.3f} of the expert bytes")
print(f"read per round, all ranks  {read / 2**30:8.2f} GiB")
print(f"read per round, one rank   {per_rank / 2**30:8.2f} GiB "
      f"({per_rank / 1e9:.2f} GB)")
print(f"streaming floor at {CEILING_GIGABYTES} GB/s: {floor_ms:.2f} ms/round")
for acceptance in (3.433, 5.0, 6.0):
    budget = acceptance * 2.0
    print(f"  at A={acceptance:.3f}: budget {budget:.2f} ms/round, "
          f"floor is {floor_ms / budget * 100:.0f}% of it")
