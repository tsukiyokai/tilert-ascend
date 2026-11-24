"""Compare Prolog query-norm export and downstream SFA with stock CANN."""

import argparse
import json
from pathlib import Path
import statistics

from attention_check import capture, fixture, metric, provenance
from attention_check import sha256, time_graph


HERE = Path(__file__).resolve().parent
WEIGHTS = (
    "weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr",
    "scale_dq", "scale_uq_qr", "scale_dkv_kr", "gamma_cq", "gamma_ckv",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=(
        HERE / "attention_export_build/whole_layer_export_attention.so"))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--widths", type=int, nargs="+", default=[1])
    parser.add_argument("--blocks", type=int, nargs="+", default=[0])
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--queued-producer", action="store_true")
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timing-pairs", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(row < 1 or row > 32 for row in args.widths):
        parser.error("widths must lie in [1,32]")
    if any(block < 0 or block > 64 for block in args.blocks):
        parser.error("blocks must lie in [0,64]")
    if min(args.replays, args.iterations, args.timing_pairs) < 1:
        parser.error("replay and timing counts must be positive")
    if args.sequence < 1:
        parser.error("sequence must be positive")
    return args


class AttentionInputs:
    """Stable main-attention tensors for export and composition probes."""

    def __init__(self, torch, torch_npu, args, weights, rows):
        self.torch, self.npu, self.args = torch, torch_npu, args
        self.weights, self.rows = weights, rows
        self.table_pages = (args.sequence + rows + args.replays + 127) // 128
        self.pages = self.table_pages + 3
        self.scale, self.epsilon = 1 / 16, 1e-5
        generator = torch.Generator().manual_seed(12345 + rows)
        self.cache_base = self.nd(torch.randn(
            self.pages, 128, 1, 512, generator=generator).bfloat16())
        self.rope_base = self.nd(torch.randn(
            self.pages, 128, 1, 64, generator=generator).bfloat16())
        self.caches = [self.nd(self.cache_base.clone()) for _ in range(2)]
        self.ropes = [self.nd(self.rope_base.clone()) for _ in range(2)]
        self.hidden = self.empty((rows, 6144), torch.bfloat16)
        self.codes = self.empty((rows, 6144), torch.int8)
        self.scales = self.empty((rows, 1), torch.float32)
        self.sine = self.empty((rows, 64), torch.bfloat16)
        self.cosine = self.empty((rows, 64), torch.bfloat16)
        self.slots = self.empty((rows,), torch.int64)
        self.topk = self.empty((rows, 1, 2048), torch.int32)
        self.query_lengths = self.nd(torch.tensor([rows], dtype=torch.int32))
        self.key_lengths = self.empty((1,), torch.int32)
        self.page_table = self.empty((1, self.table_pages), torch.int32)

    def nd(self, value):
        return self.npu.npu_format_cast(
            value.to(self.args.device).contiguous(), 2)

    def empty(self, shape, dtype):
        return self.nd(self.torch.empty(shape, dtype=dtype))

    def input_pair(self):
        if self.args.queued_producer:
            return self.codes.clone(), self.scales.clone()
        return self.codes, self.scales

    def native_arguments(self, codes, scales, arm=0):
        return (codes, scales, *(self.weights[name] for name in WEIGHTS),
                self.sine, self.cosine, self.slots, self.caches[arm],
                self.ropes[arm], self.topk, self.query_lengths,
                self.key_lengths, self.page_table)

    def vendor_prolog(self, arm=1):
        codes, scales = self.input_pair()
        weight = self.weights
        query, rope, _, normalized, norm_scale = self.npu.npu_mla_prolog_v3(
            token_x=codes, weight_dq=weight["weight_dq"],
            weight_uq_qr=weight["weight_uq_qr"], weight_uk=weight["weight_uk"],
            weight_dkv_kr=weight["weight_dkv_kr"],
            rmsnorm_gamma_cq=weight["gamma_cq"],
            rmsnorm_gamma_ckv=weight["gamma_ckv"],
            rope_sin=self.sine, rope_cos=self.cosine,
            cache_index=self.slots, kv_cache=self.caches[arm],
            kr_cache=self.ropes[arm], dequant_scale_x=scales,
            dequant_scale_w_dq=weight["scale_dq"],
            dequant_scale_w_uq_qr=weight["scale_uq_qr"],
            dequant_scale_w_dkv_kr=weight["scale_dkv_kr"],
            rmsnorm_epsilon_cq=self.epsilon, rmsnorm_epsilon_ckv=self.epsilon,
            cache_mode="PA_BSND", query_norm_flag=True,
            weight_quant_mode=2, kv_cache_quant_mode=0, query_quant_mode=0,
            qc_qr_scale=1.0, kc_scale=1.0,
        )
        return (query, rope, normalized.reshape(self.rows, 2048),
                norm_scale.view(-1))

    def vendor_sfa(self, query, rope, topk=None, arm=1):
        return self.npu.npu_sparse_flash_attention(
            query, self.caches[arm], self.caches[arm],
            self.topk if topk is None else topk, self.scale,
            block_table=self.page_table,
            actual_seq_lengths_query=self.query_lengths,
            actual_seq_lengths_kv=self.key_lengths,
            query_rope=rope, key_rope=self.ropes[arm], sparse_block_size=1,
            layout_query="TND", layout_kv="PA_BSND", sparse_mode=3,
            attention_mode=2, return_softmax_lse=False,
        )[0]

    def update(self, replay):
        torch, rows = self.torch, self.rows
        prefix = self.args.sequence + replay
        positions = torch.arange(prefix, prefix + rows, dtype=torch.int64)
        table = torch.arange(self.pages, dtype=torch.int32).roll(replay + 1)
        table = table[:self.table_pages]
        slots = table[positions // 128].long() * 128 + positions % 128
        frequency = torch.pow(10000.0, -torch.arange(32).float() / 32)
        angle = positions.float().unsqueeze(1) * frequency
        self.sine.copy_(self.nd(angle.sin().repeat(1, 2).bfloat16()))
        self.cosine.copy_(self.nd(angle.cos().repeat(1, 2).bfloat16()))
        selected = torch.arange(max(0, prefix + rows - 2048), prefix + rows,
                                dtype=torch.int32)
        if replay % 2:
            selected = selected.flip(0)
        indices = torch.full((rows, 1, 2048), -1, dtype=torch.int32)
        indices[:, 0, :selected.numel()] = selected
        self.topk.copy_(self.nd(indices))
        self.slots.copy_(self.nd(slots))
        self.page_table.copy_(self.nd(table.reshape(1, -1)))
        self.key_lengths.copy_(self.nd(torch.tensor(
            [prefix + rows], dtype=torch.int32)))
        factor = (1.0, 0.0, -1.0, .5)[replay % 4]
        hidden = (self.weights["hidden"][:rows].float() * factor).bfloat16()
        if replay >= 4:
            hidden = hidden.roll(replay, dims=1)
        self.hidden.copy_(self.nd(hidden))
        codes, scales = self.npu.npu_dynamic_quant(self.hidden)
        self.codes.copy_(codes)
        self.scales.copy_(scales.reshape(rows, 1))
        for cache, rope in zip(self.caches, self.ropes):
            cache.copy_(self.cache_base)
            rope.copy_(self.rope_base)

    def retained_inputs(self):
        names = ("hidden", "codes", "scales", "sine", "cosine", "slots",
                 "topk", "query_lengths", "key_lengths", "page_table")
        values = {name: getattr(self, name) for name in names}
        values.update({name: value for name, value in self.weights.items()
                       if name != "hidden"})
        return {name: value.detach().cpu().clone(
            memory_format=self.torch.contiguous_format)
            for name, value in values.items()}


def run_case(torch, torch_npu, args, weights, rows, requested):
    common = AttentionInputs(torch, torch_npu, args, weights, rows)
    scratch, qnorm, qscale, blocks, sync = (
        torch.ops.tilert.export_attention_prepare(
            common.codes, common.pages, common.table_pages, requested))
    output = common.empty((rows, 4, 512), torch.bfloat16)

    def native():
        return torch.ops.tilert.export_attention_out(
            *common.native_arguments(*common.input_pair()), scratch, output,
            qnorm, qscale, common.scale, common.epsilon, common.epsilon,
            blocks, sync,
        )

    def vendor():
        query, rope, normalized, norm_scale = common.vendor_prolog()
        return {"query": query, "query_rope": rope,
                "query_norm": normalized, "query_norm_scale": norm_scale,
                "attention": common.vendor_sfa(query, rope)}

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))
    with torch.npu.stream(stream):
        common.update(0)
    stream.synchronize()
    if args.capture:
        native_graph, _ = capture(torch, native, stream)
        vendor_graph, reference = capture(torch, vendor, stream)
    checks = []
    for replay in range(args.replays):
        with torch.npu.stream(stream):
            common.update(replay)
            if args.capture:
                order = (native_graph.replay, vendor_graph.replay)
                for call in order[::1 if replay % 2 == 0 else -1]:
                    call()
            else:
                native()
                reference = vendor()
        stream.synchronize()
        query_bytes = rows * 4 * 512 * 2
        rope_bytes = rows * 4 * 64 * 2
        prefix = scratch[:query_bytes + rope_bytes].cpu()
        observed = {
            "query": prefix[:query_bytes].view(torch.bfloat16).reshape(
                rows, 4, 512),
            "query_rope": prefix[query_bytes:].view(torch.bfloat16).reshape(
                rows, 4, 64),
            "query_norm": qnorm, "query_norm_scale": qscale,
            "attention": output, "kv_cache": common.caches[0],
            "rope_cache": common.ropes[0],
        }
        reference["kv_cache"] = common.caches[1]
        reference["rope_cache"] = common.ropes[1]
        metrics = {name: metric(torch, observed[name], value)
                   for name, value in reference.items()}
        passed = all(item["pass"] and item["bitwise_equal"]
                     for item in metrics.values())
        checks.append({"replay": replay, "pass": passed,
                       "boundaries": metrics})
        if not passed:
            print(json.dumps(checks[-1]), flush=True)
            dump = args.output.with_name(
                f"{args.output.stem}-m{rows}-r{replay}.pt")
            torch.save({"inputs": common.retained_inputs(),
                        "native": {name: value.detach().cpu().clone(
                            memory_format=torch.contiguous_format)
                            for name, value in observed.items()},
                        "reference": {name: value.detach().cpu().clone(
                            memory_format=torch.contiguous_format)
                            for name, value in reference.items()}}, dump)
            break
    report = {"rows": rows, "blocks": blocks, "checks": checks,
              "scratch_bytes": scratch.nbytes,
              "pass": all(item["pass"] for item in checks)}
    if report["pass"] and args.capture and not args.no_timing:
        pairs = []
        for pair in range(args.timing_pairs):
            order = (("native", native_graph), ("vendor", vendor_graph))
            pairs.append({
                name: time_graph(torch, graph, stream, args.iterations)
                for name, graph in order[::1 if pair % 2 == 0 else -1]})
        report["timing"] = {"pairs_us": pairs, "median_us": {
            name: statistics.median(pair[name] for pair in pairs)
            for name in ("native", "vendor")}}
    return report


def load_weights(torch_npu, args, data):
    weights = {}
    for name, value in data.items():
        if name == "hidden":
            weights[name] = value
            continue
        physical_format = 29 if name in {
            "weight_dq", "weight_uq_qr", "weight_dkv_kr"
        } else 2
        weights[name] = torch_npu.npu_format_cast(
            value.to(args.device), physical_format)
    return weights


def main():
    args = arguments()
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.ops.load_library(str(args.library.resolve()))
    weights = load_weights(torch_npu, args, fixture(torch, args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"scope": "Prolog query-norm export, caches and SFA",
              "provenance": provenance(torch, torch_npu, args),
              "harness_sha256": sha256(Path(__file__)),
              "capture": args.capture, "queued_producer": args.queued_producer,
              "cases": [], "pass": False}
    try:
        with torch.inference_mode():
            for rows in args.widths:
                for blocks in args.blocks:
                    case = run_case(torch, torch_npu, args, weights, rows,
                                    blocks)
                    report["cases"].append(case)
                    print(json.dumps({"rows": rows, "blocks": case["blocks"],
                                      "pass": case["pass"]}), flush=True)
                    if not case["pass"]:
                        raise RuntimeError(
                            "export attention differential failed")
        report["pass"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
