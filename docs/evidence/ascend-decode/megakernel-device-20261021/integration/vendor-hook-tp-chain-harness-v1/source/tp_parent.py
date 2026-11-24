from local_chain import BLOCKS


def separate(self, attention_sum, output, reduce):
    normalized, retained = self.input_norm(self.norm[0], self.norm[1])
    self.norm[3].copy_(normalized)
    self.norm[4].copy_(retained)
    codes, scales = self.torch_npu.npu_dynamic_quant(self.norm[3])
    self.norm[6].copy_(codes)
    self.norm[7].copy_(scales.view(self.rows, 1))
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
    reduce(self.partial, attention_sum, 'ar1')
    self.ffn.native.forward_out(
        attention_sum, self.norm[4], self.ffn.gamma, self.ffn.gate,
        self.ffn.bias, self.ffn.weights, self.ffn.shared,
        self.bridge, self.body, self.ffn.epsilon, BLOCKS, 1, self.sync)
    reduce(self.bridge[10], output, 'ar2')
    self.parent_validated = True

