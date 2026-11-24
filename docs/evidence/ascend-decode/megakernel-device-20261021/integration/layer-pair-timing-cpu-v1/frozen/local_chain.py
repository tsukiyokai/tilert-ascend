"""Compare one local layer launch with the accepted separate components."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

from decoder_forward import bind_caches, metadata_for_layers
from down_tiling import check_down_tiling
from ffn_host_adapter import LayerFFNAdapter, load as load_ffn
from ffn_views import all_values as ffn_values
from local_chain_host import Library, Prepared
from local_chain_views import attention_values, poison, strict_checks
from vendor_hook_forward import page_plan, snapshot, source_identity
from vendor_hook_graph import (
    METADATA_FIELDS, copy_fields, signature, tensor_witness, validate_page_plan,
)


CHAIN_SHA = "6a326292287a17254b48376400e1b852093ebfe10fbf228938336693402aa528"
INDEXED_SHA = "8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f"
SHARED_SHA = "64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c"
BLOCKS = 24


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nd(torch_npu, value):
    return torch_npu.npu_format_cast(value.contiguous(), 2)


def attention_addresses(norm, main, front, tail, main_metadata,
                        index_metadata, caches, scratch, tiling, output):
    """Use the frozen 41-field AttentionAddresses order for both variants."""
    if (len(norm) != 8 or len(main) != 9 or len(front) not in (0, 6)
            or len(tail) != 3 or len(main_metadata) != 6
            or index_metadata is not None and len(index_metadata) != 6
            or len(caches) not in (2, 4)):
        raise ValueError("The parent attention argument groups changed")
    return [norm[3], norm[6], norm[7], *main,
            *(front if front else [None] * 6), *tail, None, *main_metadata,
            *(index_metadata if index_metadata else [None] * 6),
            *caches, *([None] * (4 - len(caches))), scratch, tiling, output]


def successor_inputs(bridge):
    if len(bridge) != 12 or bridge[10].data_ptr() == bridge[1].data_ptr():
        raise ValueError("The successor requires distinct output and residual owners")
    return bridge[10], bridge[1]


def publish_topk(selected, owner, rows):
    if tuple(selected.shape) != (rows, 1, 2048) or tuple(owner.shape) != (32, 2048):
        raise ValueError("Publish exactly the active prefix of the persistent top-k owner")
    owner[:rows].copy_(selected.view(rows, 2048))


class Inputs:
    """Own changing GPU metadata and the two real local-chain inputs."""

    def __init__(self, torch, torch_npu, layers, rows, device):
        self.torch, self.torch_npu = torch, torch_npu
        self.layers, self.rows, self.device = layers, rows, device
        self.positions, self.metadata = metadata_for_layers(
            torch, torch_npu, layers, self.plan(0), device)
        self.hidden = nd(torch_npu, torch.empty(
            rows, 6144, dtype=torch.bfloat16, device=device))
        self.residual = nd(torch_npu, torch.empty_like(self.hidden))
        self.main = [self.metadata[layer.self_attn.mla_attn.mla_attn.layer_name]
                     for layer in layers]
        prefix = layers[0].self_attn.mla_attn.mla_attn.impl.indexer.k_cache.prefix
        self.index = self.metadata[prefix]
        self.slot_sources = [item.slot_mapping
                             for item in (*self.main, self.index)]
        if len({value.data_ptr() for value in self.slot_sources}) != 3:
            raise RuntimeError("Main and indexer slots need independent owners")
        self.slots = [nd(torch_npu, torch.empty(
            rows, dtype=torch.int64, device=device)) for _ in self.slot_sources]
        self.signatures = None

    def plan(self, iteration):
        result = page_plan(self.torch, self.rows, iteration, 8192, 66, 66)
        validate_page_plan(self.torch, result, self.rows)
        return result

    def metadata_fields(self, positions=None, metadata=None):
        positions = self.positions if positions is None else positions
        metadata = self.metadata if metadata is None else metadata
        values = {"positions": positions}
        for name, item in metadata.items():
            for field in METADATA_FIELDS:
                value = getattr(item, field, None)
                if isinstance(value, self.torch.Tensor):
                    values[f"{name}:{field}"] = value
        return values

    def fields(self):
        return {**self.metadata_fields(), "hidden": self.hidden,
                "residual": self.residual,
                **{f"slot_i64:{i}": value
                   for i, value in enumerate(self.slots)}}

    def update(self, iteration):
        plan = self.plan(iteration)
        positions, metadata = metadata_for_layers(
            self.torch, self.torch_npu, self.layers, plan, self.device)
        copy_fields(self.metadata_fields(), self.metadata_fields(positions, metadata))
        generator = self.torch.Generator().manual_seed(
            42023 + self.rows * 100 + iteration)
        for value in (self.hidden, self.residual):
            value.copy_(self.torch.randn(
                self.rows, 6144, generator=generator).bfloat16())
        for source, destination in zip(self.slot_sources, self.slots):
            destination.copy_(source)
        self.check_owners()
        return plan

    def lock_owners(self):
        self.signatures = {name: signature(value)
                           for name, value in self.fields().items()}

    def check_owners(self):
        if self.signatures is not None and self.signatures != {
                name: signature(value) for name, value in self.fields().items()}:
            raise RuntimeError("A local-chain producer changed address or shape")

    def witness(self):
        for source, destination in zip(self.slot_sources, self.slots):
            if not self.torch.equal(source.cpu().long(), destination.cpu()):
                raise RuntimeError("A local-chain int64 slot conversion is stale")
        return {"actual_main_lengths": [item.seq_lens.cpu().tolist()
                                        for item in self.main],
                "actual_index_length": self.index.actual_seq_lengths_key.cpu().tolist(),
                "tensors": {name: tensor_witness(self.torch, value)
                            for name, value in self.fields().items()}}


class LayerCase:
    """Retain one layer's parent-validated buffers and local-chain packet."""

    def __init__(self, torch, torch_npu, layer, number, inputs, hidden,
                 residual, topk, adapter, ffn, chain_library):
        self.torch, self.torch_npu = torch, torch_npu
        self.layer, self.number, self.ffn = layer, number, ffn
        self.rows = inputs.rows
        self.parent_validated = False
        self.input_norm = layer.input_layernorm
        if (self.input_norm.bias is not None
                or self.input_norm.weight.dtype != torch.bfloat16
                or tuple(self.input_norm.weight.shape) != (6144,)):
            raise ValueError("Require the original unbiased BF16 input norm")

        def empty(shape, dtype):
            return nd(torch_npu, torch.empty(shape, dtype=dtype,
                                            device=hidden.device))

        self.norm = [hidden, residual, nd(torch_npu, self.input_norm.weight),
                     empty((self.rows, 6144), torch.bfloat16),
                     empty((self.rows, 6144), torch.bfloat16),
                     empty((self.rows,), torch.float32),
                     empty((self.rows, 6144), torch.int8),
                     empty((self.rows, 1), torch.float32)]
        attention = layer.self_attn.mla_attn.mla_attn
        self.impl = attention.impl
        self.mode = adapter._validate_impl(self.impl)
        self.main, self.front, self.tail = adapter._pack_weights(self.impl, self.mode)
        self.caches = adapter._cache_tensors(attention, self.mode)
        if self.mode == "bf16":
            self.caches.append(empty((0,), torch.float16))
        self.main_metadata = adapter._metadata_values(
            inputs.main[number - 6], self.rows)
        self.index_metadata = (None if self.mode == "shared" else
                               adapter._metadata_values(inputs.index, self.rows, True))
        if self.main_metadata is None or (
                self.mode != "shared" and self.index_metadata is None):
            raise ValueError("The actual metadata does not satisfy the native contract")
        for values, slots in ((self.main_metadata, inputs.slots[number - 6]),
                              (self.index_metadata, inputs.slots[2])):
            if values is not None:
                values[:2] = [nd(torch_npu, value) for value in values[:2]]
                values[2] = slots
        pages, columns = self.caches[0].shape[0], self.main_metadata[5].shape[1]
        if self.mode == "shared":
            self.scratch, self.tiling, blocks, self.sync = (
                torch.ops.tilert.shared_full_attention_prepare(
                    self.norm[6], pages, columns, BLOCKS))
            self.offsets = {}
            self.selected = topk[:self.rows].view(self.rows, 1, 2048)
            index_pages, index_columns = 66, 64
        else:
            index_pages = self.caches[2].shape[0]
            index_columns = self.index_metadata[5].shape[1]
            self.scratch, self.tiling, self.offsets, blocks, self.sync = (
                torch.ops.tilert.indexed_attention_prepare(
                    self.norm[6], pages, columns, index_pages, index_columns,
                    self.mode == "int8", BLOCKS))
            start = self.offsets["topk"]
            self.selected = self.scratch[start:start + self.rows * 2048 * 4].view(
                torch.int32).view(self.rows, 1, 2048)
        self.partial = empty((self.rows, 6144), torch.bfloat16)
        self.bridge, self.body, ffn_sync = ffn.native.prepare(
            self.partial, BLOCKS, 1)
        self.down_tiling_before = check_down_tiling(torch, self.body[7])
        if blocks != BLOCKS or self.sync != ffn_sync or not self.sync:
            raise RuntimeError("The parent launches need one physical mixed-core context")
        self.scalars = (self.impl.scale,
                        self.impl.q_a_layernorm.variance_epsilon,
                        self.impl.kv_a_layernorm.variance_epsilon)
        slots = attention_addresses(
            self.norm, self.main, self.front, self.tail, self.main_metadata,
            self.index_metadata, self.caches, self.scratch, self.tiling,
            self.partial)
        geometry = (self.rows, pages, columns, index_pages, index_columns,
                    blocks, int(self.mode == "int8"),
                    int(bool(self.front) and self.front[1].dtype == torch.bfloat16), 0)
        self.prepared = Prepared(
            chain_library, self.norm, slots, [ffn.gamma, ffn.gate, ffn.bias],
            ffn.weights, self.bridge, self.body, ffn.shared,
            topk[:self.rows].view(self.rows, 1, 2048), geometry,
            (*self.scalars, self.input_norm.variance_epsilon, ffn.epsilon),
            self.sync, self.mode != "shared")
        self.values = {
            "input_norm.normalized": self.norm[3],
            "input_norm.retained": self.norm[4],
            "input_norm.codes": self.norm[6], "input_norm.scales": self.norm[7],
            "attention.partial": self.partial,
            **{"attention." + name: value for name, value in attention_values(
                torch, self.scratch, self.offsets, self.rows, self.mode).items()},
            **ffn_values(torch, self.bridge, self.body, self.rows),
            "body.active_list": self.body[20],
        }

        self.persistent_topk = topk[:self.rows].view(self.rows, 1, 2048)
        if self.mode != "shared":
            self.values["attention.topk"] = self.persistent_topk

    def separate(self):
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
        self.ffn.native.forward_out(
            self.partial, self.norm[4], self.ffn.gamma, self.ffn.gate,
            self.ffn.bias, self.ffn.weights, self.ffn.shared,
            self.bridge, self.body, self.ffn.epsilon, BLOCKS, 1, self.sync)
        self.parent_validated = True

    def combined(self):
        if not self.parent_validated:
            raise RuntimeError("Run the real parent host validators before the C ABI")
        self.prepared.launch()

    def owners(self):
        return {"norm": [signature(value) for value in self.norm],
                "attention_scratch": signature(self.scratch),
                "attention_tiling": signature(self.tiling),
                "attention_partial": signature(self.partial),
                "bridge": [signature(value) for value in self.bridge],
                "body": [signature(value) for value in self.body],
                "persistent_topk": signature(self.persistent_topk),
                "internal_topk": (None if self.mode == "shared"
                                  else signature(self.selected)),
                "attention_mode": self.mode, "sync": self.sync}


def run_local_chain(torch, layers, config, state, args):
    import torch_npu
    from attention_check import metric

    if args.forward_sequence != 8192 or len(set(args.forward_widths)) != len(args.forward_widths):
        raise ValueError("Use unique widths and the fixed 8192 upper bound")
    if len(layers) != 2 or args.graph_residual_modes != ["provided"]:
        raise ValueError("The local-chain C ABI requires two layers and a provided residual")
    for path, expected in ((args.indexed_library, INDEXED_SHA),
                           (args.shared_library, SHARED_SHA)):
        if digest(path) != expected:
            raise ValueError("Use the accepted separate attention libraries")
    os.environ["TILERT_INDEXED_ATTENTION_LIBRARY"] = str(args.indexed_library)
    os.environ["TILERT_SHARED_ATTENTION_LIBRARY"] = str(args.shared_library)
    sys.path.insert(0, str(args.adapter.resolve().parents[3]))
    adapter = importlib.import_module("tilert.models.glm_5_2_ascend.attention_native")
    if Path(adapter.__file__).resolve() != args.adapter.resolve():
        raise RuntimeError("The attention packing adapter changed")
    adapter.load()
    native_ffn = load_ffn(args.ffn_library)
    chain_library = Library(args.chain_library, CHAIN_SHA)
    topk = layers[0].self_attn.mla_attn.topk_indices_buffer
    if (topk.device.type != "npu" or topk.dtype != torch.int32
            or tuple(topk.shape) != (32, 2048)
            or any(layer.self_attn.mla_attn.topk_indices_buffer is not topk
                   or layer.self_attn.mla_attn.mla_attn.impl.topk_indices_buffer is not topk
                   for layer in layers)):
        raise RuntimeError("Both actual layers must retain the same NPU top-k owner")
    caches, cache_owners = bind_caches(torch, torch_npu, layers, config, args.mode)
    generator = torch.Generator().manual_seed(42022)
    bases = {}
    for name, cache in caches.items():
        if cache.dtype == torch.int8:
            base = torch.randint(-127, 128, cache.shape, generator=generator, dtype=torch.int8)
        elif name.endswith("index_scale"):
            base = (torch.rand(cache.shape, generator=generator) / 128 + 1 / 64).to(cache.dtype)
        else:
            base = torch.randn(cache.shape, generator=generator).to(cache.dtype)
        bases[name] = nd(torch_npu, base.to(topk.device))
    ffns = [LayerFFNAdapter(torch, torch_npu, layer, config, native_ffn)
            for layer in layers]
    cache_signatures = {name: signature(value) for name, value in caches.items()}
    topk_signature = signature(topk)
    report = {
        "scope": "Actual Decoder6/7 weights and metadata: one launch per local layer versus original input norm and accepted separate native attention/FFN",
        "script": source_identity(run_local_chain),
        "decoder_class": source_identity(type(layers[0])),
        "input_norm": source_identity(type(layers[0].input_layernorm)),
        "attention_packing": source_identity(adapter._pack_weights),
        "ffn_packing": source_identity(LayerFFNAdapter),
        "libraries": {str(path): digest(path) for path in (
            args.chain_library, args.indexed_library, args.shared_library, args.ffn_library)},
        "mode": args.mode, "widths": args.forward_widths,
        "iterations": args.forward_replays, "blocks": BLOCKS,
        "tp_geometry": 16, "tp_collectives_executed": 0,
        "tp_contract": "Both reductions are identity; local attention partial directly feeds FFN",
        "successor_contract": "Layer7 consumes layer6 combined output and separately retained FFN residual",
        "topk_contract": "The indexed candidate writes directly into the persistent owner; only the separate parent copies its selected indices",
        "reference": "Original input norm module and npu_dynamic_quant; indexed8deff/shared64108; completeFFNc90-N512",
        "input_rstd_contract": "Candidate diagnostic only; no invented original-rstd oracle",
        "same_addresses": True, "fresh_output_poison": True,
        "poison_contract": "Floating NaN; non-address integers 1/2; bounded indices/counts zero except active_list fills 1 with E=0",
        "timing": True, "timings": [],
        "timing_scope": "609 plus copy versus direct-owner 6a; N512 in both",
        "topk_write_guards": "Candidate internal slice stays at 2; inactive persistent rows stay at -1",
        "cache_signatures": cache_signatures, "topk_signature": topk_signature,
        "prepared": [], "cases": [], "pass": False,
        "down_tiling_checks": [],
    }
    destination = args.output / "local-chain.json"

    def persist():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    persist()
    try:
        from layer_pair_timing import run_pair_timing
        if args.forward_widths != [12]:
            raise ValueError("The paired timing fixture requires M12")
        retained_arenas = []
        for arena in range(2):
            rows = 12
            inputs = Inputs(torch, torch_npu, layers, rows, topk.device)
            first = LayerCase(torch, torch_npu, layers[0], 6, inputs,
                              inputs.hidden, inputs.residual, topk, adapter,
                              ffns[0], chain_library)
            next_hidden, next_residual = successor_inputs(first.bridge)
            second = LayerCase(torch, torch_npu, layers[1], 7, inputs,
                               next_hidden, next_residual, topk, adapter,
                               ffns[1], chain_library)
            if first.mode != args.mode or second.mode != "shared":
                raise RuntimeError("The actual full/shared layer selection changed")
            cases = first, second
            inputs.lock_owners()
            owners = [value.owners() for value in cases]
            report["prepared"].append({
                "rows": rows, "arena": arena, "owners": owners,
                "down_tiling": [case.down_tiling_before for case in cases]})
            for iteration in range(args.forward_replays):
                plan = inputs.update(iteration)
                torch.npu.synchronize()
                before_inputs = {name: snapshot(torch, value)
                                 for name, value in inputs.fields().items()}
                observed = {}
                for arm in ("separate", "combined"):
                    for name, cache in caches.items():
                        cache.copy_(bases[name].roll(iteration, 0))
                    topk.fill_(-1)
                    for case in cases:
                        poison(case.values, 1 if arm == "separate" else 2)
                        case.norm[5].fill_(float("nan"))
                        case.body[20].fill_(1)
                        case.body[20][0].zero_()
                    if arm == "combined":
                        first.selected.fill_(2)
                    topk_copy_calls = 0
                    topk_copies = {}
                    for number, case in zip((6, 7), cases):
                        getattr(case, arm)()
                        if number == 6 and arm == "separate":
                            publish_topk(case.selected, topk, rows)
                            topk_copy_calls += 1
                        topk_copies[f"layer{number}.persistent_topk"] = topk[:rows].clone()
                    consumers = {"consumer.output": second.bridge[10].clone(),
                                 "consumer.retained": second.bridge[1].clone()}
                    torch.npu.synchronize()
                    inactive_unchanged = bool((topk[rows:] == -1).all())
                    if not inactive_unchanged:
                        raise RuntimeError("The selector overwrote inactive persistent rows")
                    if arm == "combined":
                        internal_unchanged = bool((first.selected == 2).all())
                        if not internal_unchanged:
                            raise RuntimeError("The candidate still writes the internal top-k slice")
                    report["down_tiling_checks"].append({
                        "rows": rows, "iteration": iteration, "arm": arm, "arena": arena,
                        "layers": [check_down_tiling(torch, case.body[7])
                                   for case in cases]})
                    values = {f"layer{number}.{name}": snapshot(torch, value)
                              for number, case in zip((6, 7), cases)
                              for name, value in case.values.items()}
                    values.update({name: snapshot(torch, value) for name, value in
                                   {**caches, **topk_copies, **consumers}.items()})
                    observed[arm] = values
                    if arm == "combined":
                        diagnostics = {f"layer{number}.input_rstd": tensor_witness(torch, case.norm[5])
                                       for number, case in zip((6, 7), cases)}
                    if owners != [value.owners() for value in cases]:
                        raise RuntimeError("A local-chain buffer owner changed")
                    if (signature(topk) != topk_signature or cache_signatures !=
                            {name: signature(value) for name, value in caches.items()}):
                        raise RuntimeError("The actual cache or persistent top-k owner changed")
                    inputs.check_owners()
                    if any(not torch.equal(value.view(torch.uint8),
                                           snapshot(torch, inputs.fields()[name]).view(torch.uint8))
                           for name, value in before_inputs.items()):
                        raise RuntimeError("A local-chain consumer overwrote a live input")
                checks, passed = strict_checks(
                    torch, observed["combined"], observed["separate"], metric)
                record = {"rows": rows, "iteration": iteration, "arena": arena,
                          "checks": checks, "pass": passed,
                          "inputs": inputs.witness(), "candidate_diagnostics": diagnostics,
                          "unused_internal_topk_unchanged": internal_unchanged,
                          "inactive_persistent_rows_unchanged": inactive_unchanged,
                          "candidate_topk_copy_calls": topk_copy_calls,
                          "outputs": {name: tensor_witness(torch, value)
                                      for name, value in observed["combined"].items()
                                      if name.startswith("consumer.")}}
                report["cases"].append(record)
                persist()
                if not passed:
                    failure = args.output / f"local-chain-{args.mode}-m{rows}-r{iteration}.pt"
                    torch.save({"inputs": before_inputs, "plan": plan,
                                "separate": observed["separate"],
                                "combined": observed["combined"]}, failure)
                    raise RuntimeError("The local-chain launch differs from its separate parents")
            retained_arenas.append((first, second, inputs))
            report["timings"].append(run_pair_timing(
                torch, cases, inputs, topk, caches, bases, arena, args.output))
            persist()
            del first, second, cases, inputs
        report["pass"] = True
        persist()
        return report
    except Exception:
        report["error"] = traceback.format_exc()
        persist()
        raise
