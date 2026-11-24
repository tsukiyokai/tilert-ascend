"""Three-rank identities and the selected FP32 HCCL expression."""


def ordered_sum(torch, partials):
    if (len(partials) != 3
            or any(value.dtype != torch.bfloat16 for value in partials)
            or any(value.shape != partials[0].shape for value in partials)):
        raise ValueError("The reduction oracle requires three equal-shape BF16 partials")
    return ((partials[0].float() + partials[2].float())
            + partials[1].float()).bfloat16()


def check_rank_witnesses(records):
    if len(records) != 3 or [row["rank"] for row in records] != [0, 1, 2]:
        raise RuntimeError("The three-rank witness order changed")
    common = ("input_hidden", "input_residual", "layer6.retained", "layer7.retained",
              "layer6.ar1", "layer6.ar2", "layer7.ar1", "layer7.ar2")
    partials = ("layer6.attention_partial", "layer6.ffn_partial",
                "layer7.attention_partial", "layer7.ffn_partial")
    for name in common:
        if len({row[name] for row in records}) != 1:
            raise RuntimeError(f"Replicated or reduced values differ between ranks: {name}")
    for name in partials:
        if len({row[name] for row in records}) != 3:
            raise RuntimeError(f"Rank partials are indistinguishable: {name}")


def check_parameters(records, sharded_matrices):
    if len(records) != 3:
        raise ValueError("The fixture requires three parameter manifests")
    indexed = [{item["name"]: item for item in row} for row in records]
    first = indexed[0]
    if (any(row.keys() != first.keys() for row in indexed)
            or not sharded_matrices <= first.keys()):
        raise RuntimeError("The actual parameter names changed")
    for name in first:
        for field in ("shape", "dtype", "bytes"):
            if any(row[name][field] != first[name][field] for row in indexed):
                raise RuntimeError(f"Different rank-local parameter geometry: {name}")
        expected = 3 if name in sharded_matrices else 1
        if len({row[name]["sha256"] for row in indexed}) != expected:
            raise RuntimeError(f"Wrong replicated/sharded parameter identity: {name}")
