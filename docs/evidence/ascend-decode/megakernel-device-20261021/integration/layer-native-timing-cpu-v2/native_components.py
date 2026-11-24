"""Same-owner native stages with a separately validated norm/quant launch."""

import hashlib
from pathlib import Path

from local_chain import BLOCKS


def prepare(torch, cases):
    path = Path(__file__).with_name("native_norm.so")
    if hashlib.sha256(path.read_bytes()).hexdigest() != NORM_SHA:
        raise RuntimeError("The frozen norm library changed")
    torch.ops.load_library(str(path))
    for case in cases:
        blocks, sync = torch.ops.tilert.norm_prepare(case.norm[0], BLOCKS)
        if blocks != BLOCKS or sync != case.sync:
            raise RuntimeError("The native norm and layer mixed-core geometry differs")


NORM_SHA = '40a9e3c04d22355cd51228c1dff7d3dc516d3442253ee3310a6cf95c01182027'


def separate(self):
    self.torch.ops.tilert.norm_quant_out(
        *self.norm[:3], None, *self.norm[3:8],
        self.input_norm.variance_epsilon, BLOCKS, self.sync)
    if self.mode == "shared":
        self.torch.ops.tilert.shared_full_attention_out(
            *self.norm[6:8], *self.main, *self.main_metadata[:3],
            *self.caches, self.selected, *self.main_metadata[3:],
            *self.tail, None, self.scratch, self.tiling, self.partial,
            *self.scalars, BLOCKS, self.sync)
    else:
        self.torch.ops.tilert.indexed_attention_out(
            self.norm[3], *self.norm[6:8], self.main, self.front, self.tail,
            None, self.main_metadata, self.index_metadata, self.caches,
            self.scratch, self.tiling, self.partial, *self.scalars,
            BLOCKS, self.sync)
    self.ffn.native.forward_out(
        self.partial, self.norm[4], self.ffn.gamma, self.ffn.gate,
        self.ffn.bias, self.ffn.weights, self.ffn.shared,
        self.bridge, self.body, self.ffn.epsilon, BLOCKS, 1, self.sync)
    self.parent_validated = True

