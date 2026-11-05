"""Keep replicated parameters equal while distinguishing two TP16-shaped shards."""

import hashlib
import json
from pathlib import Path


SHARDED_MATRICES = frozenset((
    "self_attn.q_b_proj.weight", "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight", "mlp.shared_experts.gate_up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
    "mlp.experts.routed_experts.w13_weight",
    "mlp.experts.routed_experts.w2_weight",
))


def check_pair(records):
    if len(records) != 2:
        raise ValueError("The fixture requires two parameter manifests")
    left, right = ({item["name"]: item for item in row} for row in records)
    if left.keys() != right.keys() or not SHARDED_MATRICES <= left.keys():
        raise RuntimeError("The actual parameter names changed")
    for name in left:
        if any(left[name][field] != right[name][field]
               for field in ("shape", "dtype", "bytes")):
            raise RuntimeError(f"Different rank-local parameter geometry: {name}")
        same = left[name]["sha256"] == right[name]["sha256"]
        if same == (name in SHARDED_MATRICES):
            raise RuntimeError(f"Wrong replicated/sharded parameter identity: {name}")


def initialize(torch, layer, rank, group):
    from vendor_hook_runtime212_device import initialize_parameters

    if rank not in (0, 1):
        raise ValueError("Only the two device participants construct shards")
    report = initialize_parameters(torch, layer)
    recorded = json.loads(Path(__file__).with_name("parameter_geometry.json").read_text())
    if report not in recorded["parameters"].values():
        raise RuntimeError("The real constructor differs from the measured parameter geometry")
    parameters = dict(layer.named_parameters())
    if not SHARDED_MATRICES <= parameters.keys():
        raise RuntimeError("The actual decoder lost a declared sharded matrix")
    with torch.no_grad():
        if rank:
            for name in SHARDED_MATRICES:
                value = parameters[name]
                value.copy_(value.roll(1, 0))
        for item in report:
            value = parameters[item["name"]]
            if value.device.type != "cpu":
                raise RuntimeError("Parameter sharding must precede device postprocessing")
            raw = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
            item.update(sha256=hashlib.sha256(raw).hexdigest(),
                        replicated=item["name"] not in SHARDED_MATRICES)
    peers = [None, None]
    torch.distributed.all_gather_object(peers, report, group=group)
    check_pair(peers)
    return report
