"""Is the shared expert layout-compatible with the routed expert bank?

The shared expert is dimensionally one more expert: `n_shared_experts = 1` at
`moe_intermediate_size = 2048`, the same size each routed expert carries. If
its quantised tensors also match a routed expert's shapes and dtypes, it can
be appended to the bank as group 256 and the separate dense path -- two
QuantBatchMatmulV3 calls, a DequantSwigluQuant, a DynamicQuant and the
scale-and-add combine, 64.0 us a layer -- disappears.

Reads tensor headers only. Pass the checkpoint directory and a layer.
"""
import json
import struct
import sys

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "weight_scale", "weight_offset", "scale_bias")


def headers(directory, names):
    index = json.load(open(
        f"{directory}/quant_model_weights.safetensors.index.json"))
    weight_map = index["weight_map"]
    cache, found = {}, {}
    for name in names:
        shard = weight_map.get(name)
        if shard is None:
            continue
        if shard not in cache:
            with open(f"{directory}/{shard}", "rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                cache[shard] = json.loads(handle.read(size))
        entry = cache[shard][name]
        found[name] = (tuple(entry["shape"]), entry["dtype"])
    return found


def main():
    directory, layer = sys.argv[1], int(sys.argv[2])
    shared = f"model.layers.{layer}.mlp.shared_experts"
    routed = f"model.layers.{layer}.mlp.experts.0"
    names = [f"{prefix}.{projection}.{part}"
             for prefix in (shared, routed)
             for projection in PROJECTIONS for part in PARTS]
    found = headers(directory, names)

    print(f"{'part':<26}{'shared':<28}{'routed expert 0':<28}same")
    same = True
    for projection in PROJECTIONS:
        for part in PARTS:
            one = found.get(f"{shared}.{projection}.{part}")
            other = found.get(f"{routed}.{projection}.{part}")
            agrees = one == other
            same &= agrees
            print(f"{projection + '.' + part:<26}{str(one):<28}"
                  f"{str(other):<28}{agrees}")
    print(f"\nlayout identical: {same}")


if __name__ == "__main__":
    main()
