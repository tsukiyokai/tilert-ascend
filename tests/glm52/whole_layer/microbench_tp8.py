import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import torch
import torch_npu


def nd(t):
    return torch_npu.npu_format_cast(t.contiguous(), 2)


def nz(t):
    return torch_npu.npu_format_cast(t.contiguous(), 29)


def make_inputs(device, seq_len=8652):
    g = torch.Generator().manual_seed(42)
    pages = (seq_len + 127) // 128 + 4
    table_pages = (seq_len + 127) // 128

    def r_i8(shape):
        return torch.randint(-32, 32, shape, generator=g, dtype=torch.int8).to(device)

    def r_bf16(shape, scale=0.05):
        return (torch.randn(shape, generator=g) * scale).bfloat16().to(device)

    def r_fp32(shape, scale=0.01):
        return ((torch.rand(shape, generator=g) + 0.1) * scale).float().to(device)

    main_w = [
        nz(r_i8((6144, 2048))),
        nz(r_i8((2048, 2048))),
        nd(r_bf16((8, 192, 512))),
        nz(r_i8((6144, 576))),
        nd(r_fp32((1, 2048))),
        nd(r_fp32((1, 2048))),
        nd(r_fp32((1, 576))),
        nd(r_bf16((2048,), 1.0)),
        nd(r_bf16((512,), 1.0)),
    ]
    q_had = torch.ones((1, 1))
    for _ in range(7):
        q_had = torch.cat((torch.cat((q_had, q_had), 1),
                           torch.cat((q_had, -q_had), 1)), 0)
    q_had = (q_had / 128**0.5).bfloat16().to(device)
    front_w = [
        nd(r_i8((16, 2048, 256))),
        nd(r_fp32((4096,))),
        nd(r_bf16((5, 6144, 32))),
        nd((1 + torch.randn(128, generator=g, dtype=torch.float32) * 0.1)
           .to(device)),
        nd((torch.randn(128, generator=g, dtype=torch.float32) * 0.05)
           .to(device)),
        nd(q_had),
    ]
    tail_w = [
        nd(r_bf16((16, 512, 128))),
        nd(r_i8((24, 2048, 256))),
        nd(r_fp32((6144,))),
    ]
    kv_base = nd(r_bf16((pages, 128, 1, 512)))
    rope_base = nd(r_bf16((pages, 128, 1, 64)))
    idx_base = nd(r_i8((pages, 128, 1, 128)))
    idx_scale_base = nd((torch.rand((pages, 128), generator=g) * 0.01 + 0.005).half().to(device))
    page_table = nd(torch.arange(table_pages, dtype=torch.int32, device=device).unsqueeze(0))
    return {
        "g": g, "pages": pages, "table_pages": table_pages, "seq_len": seq_len,
        "main_w": main_w, "front_w": front_w, "tail_w": tail_w,
        "kv_base": kv_base, "rope_base": rope_base,
        "idx_base": idx_base, "idx_scale_base": idx_scale_base,
        "page_table": page_table,
    }


def bench_graph(stream, fn, calls=1, warmup=5, iters=30, pairs=6):
    with torch.npu.stream(stream):
        for _ in range(warmup):
            fn()
    stream.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g, stream=stream):
        for _ in range(calls):
            fn()
    stream.synchronize()
    samples = []
    for _ in range(pairs):
        with torch.npu.stream(stream):
            ev0 = torch.npu.Event(enable_timing=True)
            ev1 = torch.npu.Event(enable_timing=True)
            ev0.record()
            for _ in range(iters):
                g.replay()
            ev1.record()
        ev1.synchronize()
        samples.append(ev0.elapsed_time(ev1) * 1000.0 / (iters * calls))
    return round(statistics.median(samples), 2), round(min(samples), 2)


def framework_front(front, kproj, qproj, cos, sin, slots, cache,
                    cache_scale, rope_kernel):
    key, _, _ = torch.native_layer_norm(
        kproj[:, :128].float(), [128], front[3], front[4], 1e-6)
    key = key.bfloat16().unsqueeze(1)
    query = qproj.clone()
    for value in (query, key):
        heads = value.shape[1]
        rope_kernel[(32,)](
            value, value.stride(0), cos, cos.stride(0), sin, sin.stride(0),
            None, None, None, 32, heads, 128, 64, heads, 64,
            BLOCK_SIZE=heads, IS_NEOX_STYLE=False, USE_COS_SIN=False)
    query = query @ front[5]
    key = key @ front[5]
    qcodes, qscales = torch_npu.npu_dynamic_quant(query.reshape(-1, 128))
    kcodes, kscales = torch_npu.npu_dynamic_quant(key.reshape(-1, 128))
    torch_npu.npu_scatter_nd_update_(
        cache.view(-1, 128), slots.view(-1, 1), kcodes)
    torch_npu.npu_scatter_nd_update_(
        cache_scale.view(-1, 1), slots.view(-1, 1), kscales.half().view(-1, 1))
    return (qcodes.reshape(32, 32, 128), qscales.half().reshape(32, 32),
            kproj[:, 128:].half())


def checkpoint_front(base, directory, layer, device):
    from tilert.models.glm_5_2_ascend.reference_weights import ReferenceWeights

    reader = ReferenceWeights(directory)
    prefix = f"model.layers.{layer}.self_attn.indexer"
    names = ("wq_b.weight", "wq_b.weight_scale", "wk.weight",
             "weights_proj.weight", "k_norm.weight", "k_norm.bias")
    values = [reader.tensor(f"{prefix}.{name}") for name in names]
    query, scale, key, head_weight, gamma, beta = values
    key = torch.cat((key, head_weight), dim=0).T
    base["front_w"][:5] = [nd(value.to(device)) for value in (
        query.T.reshape(2048, 16, 256).permute(1, 0, 2), scale.flatten(),
        key.reshape(6144, 5, 32).permute(1, 0, 2), gamma.float(), beta.float())]
    return {"layer": layer, "tensors": {
        name: {"dtype": str(value.dtype), "shape": list(value.shape),
               "sha256": hashlib.sha256(value.contiguous().view(torch.uint8)
                                         .numpy().tobytes()).hexdigest()}
        for name, value in zip(names, values)}}


def discrete(base, hidden, token_x, token_scale, sin, cos, slots, q_len, kv_len,
             front_oracle="c8", rope_kernel=None, quant_weight_format="nd"):
    rows = int(q_len.cpu()[0])
    active = 32
    w = base["main_w"]
    kv, rope, cache, cache_scale = (nd(base[name].clone()) for name in
        ("kv_base", "rope_base", "idx_base", "idx_scale_base"))
    query, query_rope, _, qnorm, qscale = torch_npu.npu_mla_prolog_v3(
        token_x=token_x[:active], weight_dq=w[0], weight_uq_qr=w[1],
        weight_uk=w[2], weight_dkv_kr=w[3], rmsnorm_gamma_cq=w[7],
        rmsnorm_gamma_ckv=w[8], rope_sin=sin[:active], rope_cos=cos[:active],
        cache_index=slots[:active], kv_cache=kv, kr_cache=rope,
        dequant_scale_x=token_scale[:active], dequant_scale_w_dq=w[4],
        dequant_scale_w_uq_qr=w[5], dequant_scale_w_dkv_kr=w[6],
        rmsnorm_epsilon_cq=1e-6, rmsnorm_epsilon_ckv=1e-6,
        cache_mode="PA_BSND", query_norm_flag=True, weight_quant_mode=2,
        kv_cache_quant_mode=0, query_quant_mode=0, qc_qr_scale=1., kc_scale=1.)
    front = base["front_w"]
    qw = nd(front[0].permute(1, 0, 2).reshape(2048, 4096))
    kw = nd(front[2].permute(1, 0, 2).reshape(6144, 160))
    qproj = torch_npu.npu_quant_matmul(
        nd(qnorm.reshape(active, 2048)),
        nz(qw) if quant_weight_format == "nz" else qw, front[1],
        pertoken_scale=qscale.reshape(-1), output_dtype=torch.bfloat16)
    product = qnorm.reshape(active, 2048).cpu().int() @ qw.cpu().int()
    scales = qscale.reshape(active, 1).cpu() * front[1].cpu().reshape(1, 4096)
    expected_projection = (product.float() * scales).bfloat16()
    torch.testing.assert_close(qproj.cpu(), expected_projection, rtol=0, atol=0)
    kproj = torch.mm(hidden[:active], kw)
    qpad = nd(torch.zeros((32, 32, 128), dtype=torch.bfloat16, device=hidden.device))
    kpad = nd(torch.zeros((32, 160), dtype=torch.bfloat16, device=hidden.device))
    qpad[:active].copy_(qproj.reshape(active, 32, 128))
    kpad[:active].copy_(kproj)
    if front_oracle == "framework":
        qcodes, qscales, weights = framework_front(
            front, kpad, qpad, cos, sin, slots, cache, cache_scale, rope_kernel)
    else:
        torch.ops.tilert.set_active_tokens_ptr(q_len, False, True)
        qcodes, qscales, weights = torch.ops.tilert.indexer_prep_qk_c8(
            kpad, qpad, cos, sin, front[3], front[4], slots, cache, cache_scale, 1e-6)
    index_rows = 32
    index_qlen = nd(torch.tensor([index_rows], dtype=torch.int32, device=hidden.device))
    index_klen = nd(torch.tensor([base["seq_len"]], dtype=torch.int32,
                                device=hidden.device))
    topk = torch_npu.npu_quant_lightning_indexer(
        query=nd(qcodes[:index_rows]), key=cache, weights=nd(weights[:index_rows]),
        query_dequant_scale=nd(qscales[:index_rows]),
        key_dequant_scale=cache_scale.view(base["pages"], 128, 1),
        actual_seq_lengths_query=index_qlen, actual_seq_lengths_key=index_klen,
        block_table=base["page_table"], query_quant_mode=0, key_quant_mode=0,
        layout_query="TND", layout_key="PA_BSND", sparse_count=2048,
        sparse_mode=3, pre_tokens=2**63 - 1, next_tokens=2**63 - 1)
    latent = torch_npu.npu_sparse_flash_attention(
        query, kv, kv, topk, 0.1352337788608801,
        block_table=base["page_table"], actual_seq_lengths_query=q_len,
        actual_seq_lengths_kv=kv_len, query_rope=query_rope, key_rope=rope,
        sparse_block_size=1, layout_query="TND", layout_kv="PA_BSND",
        sparse_mode=3, attention_mode=2, return_softmax_lse=False)[0]
    tail = base["tail_w"]
    value_weight = nd(tail[0].reshape(8, 2, 512, 128).permute(0, 2, 1, 3)
                      .reshape(8, 512, 256))
    values = torch_npu.npu_transpose_batchmatmul(
        latent, value_weight, perm_x1=(1, 0, 2), perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2)).reshape(active, 2048)
    codes, scales = torch_npu.npu_dynamic_quant(values)
    output_weight = nd(tail[1].permute(1, 0, 2).reshape(2048, 6144))
    output = torch_npu.npu_quant_matmul(
        codes, nz(output_weight) if quant_weight_format == "nz" else output_weight,
        tail[2].bfloat16() if quant_weight_format == "nz" else tail[2],
        pertoken_scale=scales,
        output_dtype=torch.bfloat16)
    return {"out_idx": output[:rows].cpu().view(torch.uint16),
            "topk": topk[:rows].cpu(), "cache": cache.cpu(),
            "cache_scale": cache_scale.cpu(),
            "query": qcodes[:rows].cpu(), "query_scale": qscales[:rows].cpu(),
            "weights": weights[:rows].cpu(),
            "query_norm": qnorm.reshape(active, 2048)[:rows].cpu(),
            "attention_query": query[:rows].cpu(),
            "key_projection": kproj[:rows].cpu().view(torch.uint16)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indexed-so")
    p.add_argument("--shared-so")
    p.add_argument("--shmem-so")
    p.add_argument("--device", default="npu:8")
    p.add_argument("--widths", type=int, nargs="+", default=[1, 6, 8, 12, 16, 24, 32])
    p.add_argument("--save-ref", type=Path)
    p.add_argument("--check-ref", type=Path)
    p.add_argument("--compare-refs", type=Path, nargs="+")
    p.add_argument("--output", type=Path)
    p.add_argument("--scatter-slots", action="store_true")
    p.add_argument("--front-oracle", choices=("c8", "framework"), default="c8",
                   help="C8 helper or framework LayerNorm/RoPE/Hadamard chain")
    p.add_argument("--rope-source", type=Path,
                   help="unmodified vLLM-Ascend ops/triton/rope.py")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--front-scale-dtype", choices=("float32", "bfloat16"),
                   default="float32")
    p.add_argument("--quant-weight-format", choices=("nd", "nz"), default="nd")
    args = p.parse_args()
    if args.front_oracle == "framework" and not args.compare_refs:
        if args.widths != [32]:
            p.error("framework oracle requires --widths 32 (all rows valid)")
        if args.rope_source is None or not args.rope_source.is_file():
            p.error("framework oracle requires --rope-source with the original rope.py")

    if args.compare_refs:
        refs = [torch.load(path) for path in args.compare_refs]
        base_ref = refs[0]
        for M in sorted(base_ref):
            row = [f"M={M:2d}"]
            for path, r in zip(args.compare_refs[1:], refs[1:]):
                ei = bool(torch.equal(base_ref[M]["out_idx"], r[M]["out_idx"]))
                ef = bool(torch.equal(base_ref[M]["out_full"], r[M]["out_full"]))
                et = bool(torch.equal(base_ref[M]["topk"], r[M]["topk"]))
                row.append(f"{path.stem}: idx={ei} full={ef} topk={et}")
            print(" | ".join(row))
        return

    rope_kernel = None
    rope_sha256 = None
    if args.front_oracle == "framework":
        from indexer_front_check import load_rope

        rope_directory = tempfile.TemporaryDirectory(prefix="tilert-tp8-rope-")
        rope_kernel = load_rope(args.rope_source,
                                Path(rope_directory.name) / "rope.py")
        rope_sha256 = hashlib.sha256(args.rope_source.read_bytes()).hexdigest()

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.ops.load_library(args.indexed_so)
    torch.ops.load_library(args.shared_so)
    if args.shmem_so:
        ctypes.CDLL(str(Path(args.shmem_so).with_name("libtp_push_kernel.so")),
                    mode=ctypes.RTLD_GLOBAL)
        torch.ops.load_library(args.shmem_so)

    base = make_inputs(args.device, 8731 if args.scatter_slots else 8652)
    fixture = (checkpoint_front(base, args.checkpoint, args.layer, args.device)
               if args.checkpoint else None)
    base["front_w"][1] = nd(base["front_w"][1].to(
        getattr(torch, args.front_scale_dtype)))
    if args.quant_weight_format == "nz":
        base["tail_w"][2] = nd(base["tail_w"][2].bfloat16().float())
    if args.scatter_slots:
        base["page_table"] = nd(base["page_table"].flip(1))
    pages, table_pages, seq_len = base["pages"], base["table_pages"], base["seq_len"]
    ref_data = torch.load(args.check_ref) if args.check_ref else {}
    saved_ref = {}
    results = {}
    stream = torch.npu.Stream()

    with torch.inference_mode():
        W = 32
        before = seq_len - W
        for M in args.widths:
            g = torch.Generator().manual_seed(1000 + M)
            hidden = nd((torch.randn((W, 6144), generator=g) * 0.1).bfloat16().to(args.device))
            token_x = nd(torch.randint(-32, 32, (W, 6144), generator=g, dtype=torch.int8).to(args.device))
            token_scale = nd(((torch.rand((W, 1), generator=g) + 0.1) * 0.01).float().to(args.device))
            sin = nd(torch.randn((W, 64), generator=g).bfloat16().to(args.device))
            cos = nd(torch.randn((W, 64), generator=g).bfloat16().to(args.device))
            positions = torch.arange(before, before + W, dtype=torch.int64, device=args.device)
            table = base["page_table"][0].long()
            slots_t = table[positions // 128] * 128 + positions % 128
            slots = nd(slots_t)
            q_len = nd(torch.tensor([M], dtype=torch.int32, device=args.device))
            kv_len = nd(torch.tensor([before + M], dtype=torch.int32, device=args.device))
            idx_q_len = nd(torch.tensor([W], dtype=torch.int32, device=args.device))
            idx_kv_len = nd(torch.tensor([before + W], dtype=torch.int32, device=args.device))
            main_meta = [sin, cos, slots, q_len, kv_len, base["page_table"]]
            idx_meta = [sin, cos, slots, idx_q_len, idx_kv_len, base["page_table"]]

            kv_c = nd(base["kv_base"].clone())
            rope_c = nd(base["rope_base"].clone())
            idx_c = nd(base["idx_base"].clone())
            idx_s = nd(base["idx_scale_base"].clone())
            caches = [kv_c, rope_c, idx_c, idx_s]

            scratch, tiling, offsets, blocks, sync = torch.ops.tilert.indexed_attention_prepare(
                token_x, pages, table_pages, pages, table_pages, True, 24
            )
            out_idx = nd(torch.empty((W, 6144), dtype=torch.bfloat16, device=args.device))
            torch.ops.tilert.indexed_attention_out(
                hidden, token_x, token_scale, base["main_w"], base["front_w"], base["tail_w"],
                None, main_meta, idx_meta, caches, scratch, tiling, out_idx,
                0.1352337788608801, 1e-6, 1e-6, blocks, sync
            )
            torch.npu.synchronize()
            topk = scratch.narrow(0, offsets["topk"], W * 2048 * 4).view(torch.int32).view(W, 1, 2048).clone()

            f_scratch, f_tiling, f_blocks, f_sync = torch.ops.tilert.shared_full_attention_prepare(
                token_x, pages, table_pages, 24
            )
            out_full = nd(torch.empty((W, 6144), dtype=torch.bfloat16, device=args.device))
            torch.ops.tilert.shared_full_attention_out(
                token_x, token_scale, *base["main_w"], sin, cos, slots, kv_c, rope_c,
                topk, q_len, kv_len, base["page_table"], *base["tail_w"], None,
                f_scratch, f_tiling, out_full, 0.1352337788608801, 1e-6, 1e-6, f_blocks, f_sync
            )
            torch.npu.synchronize()

            cur_idx_cpu = out_idx[:M].cpu().view(torch.uint16)
            cur_full_cpu = out_full[:M].cpu().view(torch.uint16)
            cur_topk_cpu = topk[:M].cpu()
            exact_idx = None
            exact_full = None
            exact_topk = None
            checks = {}
            if args.shmem_so or args.front_oracle == "framework":
                reference = discrete(base, hidden, token_x, token_scale,
                                     sin, cos, slots, q_len, kv_len,
                                     args.front_oracle, rope_kernel,
                                     args.quant_weight_format)
                checks = {
                    "indexed": torch.equal(cur_idx_cpu, reference["out_idx"]),
                    "shared": torch.equal(cur_full_cpu, reference["out_idx"]),
                    "topk": torch.equal(cur_topk_cpu, reference["topk"]),
                    "cache": torch.equal(idx_c.cpu(), reference["cache"]),
                    "cache_scale": torch.equal(idx_s.cpu(), reference["cache_scale"]),
                }
                capacity = 32
                key_offset = offsets["front_begin"] + 32 * 4096 * 4
                key = scratch[key_offset:key_offset + capacity * 160 * 4].view(torch.float32)
                key = key.reshape(10, capacity, 16).permute(1, 0, 2).reshape(capacity, 160)
                checks["key_projection"] = torch.equal(
                    key[:M].bfloat16().cpu().view(torch.uint16), reference["key_projection"])
                for name, dtype, shape in (
                    ("query", torch.int8, (M, 32, 128)),
                    ("query_scale", torch.float16, (M, 32)),
                    ("weights", torch.float16, (M, 32)),
                    ("query_norm", torch.int8, (M, 2048)),
                    ("attention_query", torch.bfloat16, (M, 8, 512)),
                ):
                    count = reference[name].numel()
                    offset = offsets[name]
                    value = scratch[offset:offset + count * reference[name].element_size()]
                    value = value.view(dtype).reshape(shape).cpu()
                    checks[name] = torch.equal(value, reference[name].reshape(shape))
                    if not checks[name]:
                        delta = value.float() - reference[name].reshape(shape).float()
                        print(f"M={M} {name}: unequal={int((delta != 0).sum())} max={float(delta.abs().max())}", flush=True)
            if args.check_ref:
                exact_idx = bool(torch.equal(cur_idx_cpu, ref_data[M]["out_idx"]))
                exact_full = bool(torch.equal(cur_full_cpu, ref_data[M]["out_full"]))
                exact_topk = bool(torch.equal(cur_topk_cpu, ref_data[M]["topk"]))
            if args.save_ref:
                saved_ref[M] = {"out_idx": cur_idx_cpu, "out_full": cur_full_cpu, "topk": cur_topk_cpu}

            idx_med, idx_min = bench_graph(stream, lambda: torch.ops.tilert.indexed_attention_out(
                hidden, token_x, token_scale, base["main_w"], base["front_w"], base["tail_w"],
                None, main_meta, idx_meta, caches, scratch, tiling, out_idx,
                0.1352337788608801, 1e-6, 1e-6, blocks, sync
            ))
            full_med, full_min = bench_graph(stream, lambda: torch.ops.tilert.shared_full_attention_out(
                token_x, token_scale, *base["main_w"], sin, cos, slots, kv_c, rope_c,
                topk, q_len, kv_len, base["page_table"], *base["tail_w"], None,
                f_scratch, f_tiling, out_full, 0.1352337788608801, 1e-6, 1e-6, f_blocks, f_sync
            ))
            if args.shmem_so or args.front_oracle == "framework":
                checks["graph_indexed"] = torch.equal(out_idx[:M].cpu().view(torch.uint16), reference["out_idx"])
                checks["graph_shared"] = torch.equal(out_full[:M].cpu().view(torch.uint16), reference["out_idx"])
                replay_topk = scratch[offsets["topk"]:offsets["topk"] + W * 2048 * 4].view(torch.int32)
                checks["graph_topk"] = torch.equal(replay_topk.view(W, 1, 2048)[:M].cpu(), reference["topk"])
            results[M] = {
                "indexed_med_us": idx_med, "indexed_min_us": idx_min,
                "shared_med_us": full_med, "shared_min_us": full_min,
                "exact_idx": exact_idx, "exact_full": exact_full, "exact_topk": exact_topk,
                "discrete_checks": checks,
                "sequence": seq_len, "scatter_slots": args.scatter_slots,
                "front_oracle": args.front_oracle,
                "rope_source": str(args.rope_source) if rope_kernel else None,
                "rope_source_sha256": rope_sha256,
                "checkpoint_front": fixture,
                "front_scale_dtype": args.front_scale_dtype,
                "quant_weight_format": args.quant_weight_format,
                "library_sha256": {
                    str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    for path in (args.indexed_so, args.shared_so)},
            }
            print(f"M={M:2d} | indexed: {idx_med:6.2f} us (min {idx_min:6.2f}) | shared: {full_med:6.2f} us (min {full_min:6.2f}) | exact={exact_idx and exact_full} (idx={exact_idx}, topk={exact_topk}, full={exact_full})", flush=True)
            print(f"discrete={checks}", flush=True)

    if args.save_ref:
        torch.save(saved_ref, args.save_ref)
    print(json.dumps(results))
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    if any(value is False for result in results.values()
           for value in (*result["discrete_checks"].values(), result["exact_idx"],
                         result["exact_full"], result["exact_topk"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
