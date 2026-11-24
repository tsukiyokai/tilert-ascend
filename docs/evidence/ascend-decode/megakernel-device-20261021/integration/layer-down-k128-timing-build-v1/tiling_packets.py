"""Keep per-arm prepare owners while sharing every writable local-layer address."""

import hashlib
import json
from pathlib import Path

from ffn_views import all_values
from local_chain_host import Prepared
from local_chain_views import poison
from record_layout import compare_records, frozen_layout
from vendor_hook_forward import snapshot


CONTROL_SHA = "c90b3666a1697070eda19d5b3e133e9ed58f36fe429abefceae6ffe102a6a032"
CANDIDATE_SHA = "0842cd0f3a946c39aefb3e926e01eb870a61d6d1ad8689aebcf52a1b3daa2b65"


def load_control(torch):
    path = Path(__file__).with_name("control_ffn.so")
    if hashlib.sha256(path.read_bytes()).hexdigest() != CONTROL_SHA:
        raise RuntimeError("The N512 FFN preparation library changed")
    torch.ops.load_library(str(path))
    return torch.ops.tilert_ffn_down_n512


def check_tiling(torch, tensor, arm):
    expected = json.loads(Path(__file__).with_name("paired-tilings.json").read_text())[arm]
    if tensor.dtype != torch.uint8 or tensor.ndim != 1 or tensor.numel() < 200:
        raise ValueError("Down tiling needs a contiguous byte tensor of at least 200 bytes")
    if not tensor.is_contiguous():
        raise ValueError("Down tiling must be contiguous")
    payload = bytes(tensor[:200].detach().cpu().tolist())
    if payload.hex() != expected["hex"]:
        raise RuntimeError(f"The {arm} Down tiling differs from its frozen 200 bytes")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected["sha256"]:
        raise RuntimeError("The frozen tiling byte oracle is inconsistent")
    return {"arm": arm, "address": tensor.data_ptr(), "bytes": 200,
            "sha256": actual, "geometry": expected["geometry"], "pass": True}


class TilingPair:
    """Retain two real preparations; only the control Down pointer enters its graph."""

    def __init__(self, case, native):
        from local_chain import BLOCKS, attention_addresses

        self.case, self.native = case, native
        self.bridge, self.body, sync = native.prepare(case.partial, BLOCKS, 1)
        if sync != case.sync:
            raise RuntimeError("Both preparations must use the same mixed-core sync base")
        self.full_owners = (tuple(self.bridge), tuple(self.body))
        self.full_pointers = tuple(value.data_ptr() for group in self.full_owners for value in group)
        body = list(case.body)
        body[7] = self.body[7]
        self.control_body = tuple(body)
        slots = attention_addresses(
            case.norm, case.main, case.front, case.tail, case.main_metadata,
            case.index_metadata, case.caches, case.scratch, case.tiling, case.partial)
        index_pages, index_columns = (66, 64) if case.mode == "shared" else (
            case.caches[2].shape[0], case.index_metadata[5].shape[1])
        geometry = (case.rows, case.caches[0].shape[0], case.main_metadata[5].shape[1],
                    index_pages, index_columns, BLOCKS, int(case.mode == "int8"),
                    int(bool(case.front) and case.front[1].dtype == case.torch.bfloat16), 0)
        self.control = Prepared(
            case.prepared.library, case.norm, slots,
            [case.ffn.gamma, case.ffn.gate, case.ffn.bias], case.ffn.weights,
            case.bridge, self.control_body, case.ffn.shared,
            case.selected if case.mode == "shared" else None, geometry,
            (*case.scalars, case.input_norm.variance_epsilon, case.ffn.epsilon),
            case.sync, case.mode != "shared")
        self.candidate = case.prepared
        self.layout = frozen_layout()
        self.values = {**all_values(case.torch, self.bridge, self.body, case.rows),
                       "body.active_list": self.body[20]}
        self.check()

    def check(self):
        if self.full_pointers != tuple(
                value.data_ptr() for group in self.full_owners for value in group):
            raise RuntimeError("An owning preparation changed its Tensor storage")
        for packet in (self.control, self.candidate):
            if tuple(value.data_ptr() for value in packet.owners) != packet.owner_pointers:
                raise RuntimeError("A graph record lost its retained Tensor owner")
        expected = tuple(self.body[7].data_ptr() if value is self.case.body[7]
                         else value.data_ptr() for value in self.candidate.owners)
        if self.control.owner_pointers != expected:
            raise RuntimeError("The graph retained an extra or different owner")
        return {
            "records": compare_records(self.control, self.candidate,
                                       self.body[7].data_ptr(), self.case.body[7].data_ptr(),
                                       self.layout),
            "tilings": [check_tiling(self.case.torch, self.body[7], "control"),
                        check_tiling(self.case.torch, self.case.body[7], "candidate")],
        }

    def validate_control(self):
        """Run the N512 Torch validator with its own full preparation outside capture."""
        from local_chain import BLOCKS

        case, torch = self.case, self.case.torch
        poison(self.values, 1)
        self.body[20].fill_(1)
        self.body[20][0].zero_()
        self.native.forward_out(
            case.partial, case.norm[4], case.ffn.gamma, case.ffn.gate, case.ffn.bias,
            case.ffn.weights, case.ffn.shared, self.bridge, self.body,
            case.ffn.epsilon, BLOCKS, 1, case.sync)
        torch.npu.synchronize()
        return {name: snapshot(torch, value) for name, value in self.values.items()}
