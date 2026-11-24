import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess

import torch
import torch_npu

from tilert.models.glm_5_2_ascend.abi import FFN_NAMESPACE, LocalRecords, Window
from tilert.models.glm_5_2_ascend.ops.build_megakernel import (
    ASC_SUBDIRS,
    ROOT,
    _cann_root,
    _r4_flags,
    _site_root,
    _topology_flags,
    effective_tree_digest,
)
from tilert.models.glm_5_2_ascend.weights import (
    _apply_r4_routed_contig,
    _to_nz,
    hadamard_128,
    pack_indexer_weights,
    pack_output_weights,
    pack_shared_weights,
    pack_value_weights,
)

REPO = Path(__file__).resolve().parents[2]
KERNEL_CPP = Path(__file__).resolve().with_name("microbench_tp16_kernel.cpp")
DEFAULT_OUT = REPO / "tests/.artifacts/builds/microbench_tp16"
DEFAULT_WHOLE_LAYER_BASE = {
    "indexed": "/assets/runs/v3x58/build/indexed/whole_layer_indexed_attention_tp8.so",
    "shared": "/assets/runs/v3x58/build/full/whole_layer_full_attention_tp8.so",
    "ffn": "/sfs/shanshan/dspine-glm52/runtime_bundle_v5/agy_tp8/build/complete_ffn_tp8_ep8.so",
    "metadata": (
        "/sfs/shanshan/dspine-glm52/runtime_bundle_v5/agy_tp8/build/layer_local_chain_tp8_ep8.so"
    ),
}
ATTN_SCALARS = (0.1352337788608801, 1e-6, 1e-6, 1e-6, 1e-6)


def _default_build_dir() -> Path:
    env_dir = os.environ.get("TILERT_BUILD_DIR", "").strip()
    if env_dir:
        return Path(env_dir) / "microbench_tp16"
    return DEFAULT_OUT


def build_microbench_tp16_lib(
    out_dir: Path | None = None,
    ver: int = 112830,
    force: bool = False,
    enable_kernel_prof: bool = False,
) -> tuple[Path, Path]:
    base_dir = Path(out_dir) if out_dir is not None else _default_build_dir()
    out = base_dir / "prof" if enable_kernel_prof else base_dir
    out.mkdir(parents=True, exist_ok=True)
    cann = _cann_root()
    site = _site_root()
    defs = [
        f"-DKERNEL_VER={ver}",
        *(f for f in _topology_flags(16) if f != "-DSKIP_SINGLE_KERNELS=1"),
        *_r4_flags(16),
    ]
    if enable_kernel_prof:
        defs.append("-DENABLE_KERNEL_PROF=1")
    so_path = out / "microbench_tp16.so"
    manifest_path = out / "microbench_tp16.manifest.json"
    eff_sha = effective_tree_digest(ROOT, defs, extra_texts=[KERNEL_CPP.read_text("utf-8")])
    if not force and so_path.is_file() and manifest_path.is_file():
        try:
            cached = json.loads(manifest_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if (
            isinstance(cached, dict)
            and cached.get("ver") == ver
            and bool(cached.get("enable_kernel_prof", False)) == bool(enable_kernel_prof)
            and cached.get("flags") == defs
            and cached.get("effective_source_sha256") == eff_sha
            and cached.get("sha256") == hashlib.sha256(so_path.read_bytes()).hexdigest()
        ):
            return so_path, manifest_path

    inc = [
        "-I", str(ROOT),
        "-I", str(ROOT / "source"),
        "-I", str(ROOT / "local/ffn/body/source/routed_int4"),
        "-I", str(cann / "include"),
        "-I", str(ROOT / "local/ffn/bridge/vendor"),
        "-I", str(ROOT / "local/ffn/source"),
        "-I", str(ROOT / "vendor"),
        "-I", "/assets/shmem-lib/include",
        "-I", "/assets/shmem-lib/src/device",
    ]
    for sub in ASC_SUBDIRS:
        inc.extend(["-I", str(cann / "aarch64-linux" / sub)])
    host_inc = [
        "-I", str(site / "torch/include"),
        "-I", str(site / "torch/include/torch/csrc/api/include"),
        "-I", str(site / "torch_npu/include"),
        "-I", str(cann / "include"),
        "-I", "/assets/shmem-lib/include",
    ]
    libs = [
        "-L", "/assets/shmem-lib/lib", "-lshmem", "-lshmem_utils",
        "-L", str(cann / "aarch64-linux/lib64"), "-L", str(cann / "lib64"),
        "-lascendcl", "-lruntime", "-lunified_dlog", "-lc_sec",
        "-lregister", "-lprofapi", "-ltiling_api", "-lplatform",
        "-L", str(site / "torch/lib"), "-L", str(site / "torch_npu/lib"),
        "-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_npu",
    ]
    b_obj = out / f"bench_{ver}.o"
    h_obj = out / f"host_{ver}.o"
    c_obj = out / f"comm_{ver}.o"

    b_cmd = [
        str(cann / "bin/bisheng"), "-xcce", "--npu-arch=dav-2201",
        "-std=c++17", "-O2", "-w", "-ffp-contract=off", "-fPIC",
        *defs, *inc, "-c", str(KERNEL_CPP), "-o", str(b_obj),
    ]
    h_cmd = [
        "c++", "-std=c++17", "-O2", "-fPIC", "-D_GLIBCXX_USE_CXX11_ABI=1",
        *defs, *host_inc, "-c", str(ROOT / "source/host.cpp"), "-o", str(h_obj),
    ]
    c_cmd = [
        "c++", "-std=c++17", "-O2", "-fPIC", "-D_GLIBCXX_USE_CXX11_ABI=1",
        *defs, *host_inc, "-c", str(ROOT / "communication/communication_host.cpp"),
        "-o", str(c_obj),
    ]
    link_cmd = [
        "c++", "-shared", "-Wl,--no-undefined", "-Wl,-Bsymbolic",
        str(b_obj), str(h_obj), str(c_obj), "-o", str(so_path), *libs,
    ]
    script = (
        f"set -euo pipefail\n"
        f"set +u; source {shlex.quote(str(cann / 'set_env.sh'))}; set -u\n"
        f"{' '.join(shlex.quote(x) for x in b_cmd)} &\n"
        f"{' '.join(shlex.quote(x) for x in h_cmd)} &\n"
        f"{' '.join(shlex.quote(x) for x in c_cmd)} &\n"
        f"wait\n"
        f"{' '.join(shlex.quote(x) for x in link_cmd)}\n"
    )
    try:
        subprocess.run(["bash", "-c", script], check=True)
    finally:
        for obj in (b_obj, h_obj, c_obj):
            if obj.exists():
                obj.unlink()

    sha256 = hashlib.sha256(so_path.read_bytes()).hexdigest()
    manifest = {
        "ver": ver,
        "enable_kernel_prof": bool(enable_kernel_prof),
        "flags": defs,
        "sha256": sha256,
        "effective_source_sha256": eff_sha,
    }
    tmp_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    return so_path, manifest_path


class BenchLibrary:

    def __init__(self, so_path: Path):
        self.path = Path(so_path)
        self.handle = ctypes.CDLL(str(self.path.resolve()))
        u64_ptr = ctypes.POINTER(ctypes.c_uint64)
        self.handle.layer_tp_tree_record_bytes.argtypes = [ctypes.c_int]
        self.handle.layer_tp_tree_record_bytes.restype = ctypes.c_size_t
        self.handle.layer_tp_tree_last_error.argtypes = []
        self.handle.layer_tp_tree_last_error.restype = ctypes.c_char_p
        self.config_bytes = int(self.handle.layer_tp_tree_record_bytes(0))
        self.address_bytes = int(self.handle.layer_tp_tree_record_bytes(1))
        if (self.config_bytes, self.address_bytes) != (664, 928):
            raise RuntimeError(
                f"Unexpected record sizes: {(self.config_bytes, self.address_bytes)}"
            )
        self.prepare_records = self.handle.layer_tp_tree_prepare_records
        self.prepare_records.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_float),
            u64_ptr,
            u64_ptr,
            ctypes.c_uint64,
            u64_ptr,
            u64_ptr,
            u64_ptr,
            u64_ptr,
            u64_ptr,
        ]
        self.prepare_records.restype = ctypes.c_int

        self.launch_indexed = self.handle.bench_tp16_indexed_attn_launch
        self.launch_indexed.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int32,
        ]
        self.launch_indexed.restype = ctypes.c_int

        self.launch_shared = self.handle.bench_tp16_shared_attn_launch
        self.launch_shared.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        self.launch_shared.restype = ctypes.c_int

        self.launch_moe = self.handle.bench_tp16_moe_launch
        self.launch_moe.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        self.launch_moe.restype = ctypes.c_int

        self.handle.layer_tp_tree_initialize.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint64,
        ]
        self.handle.layer_tp_tree_initialize.restype = ctypes.c_int
        self.handle.layer_tp_tree_window_create.argtypes = [
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.handle.layer_tp_tree_window_create.restype = ctypes.c_int
        self.handle.layer_tp_tree_window_info.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        self.handle.layer_tp_tree_window_info.restype = ctypes.c_int
        self.handle.layer_tp_tree_window_destroy.argtypes = [ctypes.c_uint64]
        self.handle.layer_tp_tree_window_destroy.restype = ctypes.c_int
        self.handle.layer_tp_tree_finalize.argtypes = []
        self.handle.layer_tp_tree_finalize.restype = ctypes.c_int
        self.handle.layer_tp_tree_submit.argtypes = [
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_bool,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        self.handle.layer_tp_tree_submit.restype = ctypes.c_int

    def check(self, status: int) -> None:
        if status != 0:
            msg = self.handle.layer_tp_tree_last_error().decode("utf-8")
            raise RuntimeError(f"bench_tp16 status {status}: {msg}")

    def initialize(self, device: int, rank: int, ranks: int, endpoint: str, heap_bytes: int):
        self.check(
            self.handle.layer_tp_tree_initialize(
                device, rank, ranks, endpoint.encode("utf-8"), heap_bytes
            )
        )

    def finalize(self):
        self.check(self.handle.layer_tp_tree_finalize())


def _nd(t: torch.Tensor) -> torch.Tensor:
    return torch_npu.npu_format_cast(t.contiguous(), 2)


def make_tp16_inputs(
    device: torch.device,
    seq_len: int = 8704,
    local_experts: int = 16,
    preload_gate: bool = True,
) -> dict:
    g = torch.Generator().manual_seed(42)
    pages = (seq_len + 127) // 128 + 4
    table_pages = (seq_len + 127) // 128
    heads = 4
    sh_dim = 128

    def r_i8(shape):
        return torch.randint(-32, 32, shape, generator=g, dtype=torch.int8)

    def r_bf16(shape, scale=0.05):
        return (torch.randn(shape, generator=g) * scale).bfloat16()

    def r_fp32(shape, scale=0.01):
        return ((torch.rand(shape, generator=g) + 0.1) * scale).float()

    main_w = [
        _to_nz(r_i8((6144, 2048)), device),
        _to_nz(r_i8((2048, heads * 256)), device),
        _nd(r_bf16((heads, 192, 512)).to(device)),
        _to_nz(r_i8((6144, 576)), device),
        _nd(r_fp32((1, 2048)).to(device)),
        _nd(r_fp32((1, heads * 256)).to(device)),
        _nd(r_fp32((1, 576)).to(device)),
        _nd(r_bf16((2048,), 1.0).to(device)),
        _nd(r_bf16((512,), 1.0).to(device)),
    ]

    fq, fk = pack_indexer_weights(r_i8((2048, 4096)), r_bf16((160, 6144)))
    fq = fq.view(16, 2048, 8, 32).permute(0, 2, 1, 3).contiguous().view(16, 2048, 256)
    front_w = [
        _nd(fq.to(device)),
        _nd(r_fp32((4096,)).bfloat16().to(device)),
        _nd(fk.to(device)),
        _nd((1 + torch.randn(128, generator=g, dtype=torch.float32) * 0.1).to(device)),
        _nd((torch.randn(128, generator=g, dtype=torch.float32) * 0.05).to(device)),
        _nd(hadamard_128(device=device)),
    ]

    w_uv = r_bf16((512, heads, 256))
    w_uv_p = pack_value_weights(w_uv.transpose(0, 1).contiguous())
    uv_tile = (heads * 256) // 16
    uv_n16 = uv_tile // 16
    w_uv_p = (
        w_uv_p.view(16, 512, uv_n16, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(16, 512, uv_tile)
    )
    w_o_p = pack_output_weights(r_i8((heads * 256, 6144)))
    w_o_p = (
        w_o_p.view(24, heads * 256, 8, 32)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(24, heads * 256, 256)
    )
    tail_w = [
        _nd(w_uv_p.to(device)),
        _nd(w_o_p.to(device)),
        _nd(r_fp32((6144,)).to(device)),
    ]

    kv_base = _nd(r_bf16((pages, 128, 1, 512)).to(device))
    rope_base = _nd(r_bf16((pages, 128, 1, 64)).to(device))
    idx_base = _nd(r_i8((pages, 128, 1, 128)).to(device))
    idx_scale_base = _nd((torch.rand((pages, 128), generator=g) * 0.01 + 0.005).half().to(device))
    page_table = _nd(torch.arange(table_pages, dtype=torch.int32, device=device).unsqueeze(0))

    gw = r_bf16((6144, 256), 0.05)
    if preload_gate:
        gw = gw.reshape(6144, 16, 16).permute(1, 0, 2).contiguous().view(6144, 256)
    post = (
        _nd(r_bf16((6144,), 1.0).to(device)),
        _nd(gw.to(device)),
        _nd(r_fp32((256,), 0.01).to(device)),
    )

    sh_gu_w = r_i8((6144, 2 * sh_dim))
    sh_dn_w = r_i8((sh_dim, 6144))
    sh_p_gu, sh_p_dn = pack_shared_weights(sh_gu_w, sh_dn_w)
    sh_gu_tile = sh_dim // 2
    sh_gu_g32 = sh_gu_tile // 32
    sh_dn_k128 = sh_dim // 128
    sh_p_gu = (
        sh_p_gu.view(24, 2, 512, sh_gu_g32, 32)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(24, 1024, sh_gu_tile)
    )
    sh_p_dn = (
        sh_p_dn.view(24, sh_dn_k128, 128, 8, 32)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(24, sh_dim, 256)
    )
    shared = (
        _nd(sh_p_gu.to(device)),
        _nd(r_fp32((2 * sh_dim,)).to(device)),
        _nd(sh_p_dn.to(device)),
        _nd(r_fp32((6144,)).to(device)),
    )

    acl = ctypes.CDLL("libascendcl.so")
    tmp_contig = torch.empty((2 * 12582912,), dtype=torch.uint8, device=device)
    w13_cpu = r_i8((local_experts, 6144, 2048))
    w2_cpu = r_i8((local_experts, 2048, 3072))
    r0_nz = _to_nz(w13_cpu, device).view(torch.int32)
    _apply_r4_routed_contig(r0_nz, tmp_contig, 12582912, (16, 4, 4, 49152), acl)
    r3_nz = _to_nz(w2_cpu, device).view(torch.int32)
    _apply_r4_routed_contig(r3_nz, tmp_contig, 6291456, (24, 4, 2, 32768), acl)
    del tmp_contig, w13_cpu, w2_cpu
    routed = (
        r0_nz,
        _nd(r_fp32((local_experts, 4096), 0.001).to(device)),
        _nd(r_fp32((local_experts, 4096), 0.005).to(device)),
        r3_nz,
        _nd(r_fp32((local_experts, 6144), 0.001).to(device)),
        _nd(r_fp32((local_experts, 6144), 0.005).to(device)),
    )

    return {
        "pages": pages,
        "table_pages": table_pages,
        "seq_len": seq_len,
        "input_gamma": _nd(r_bf16((6144,), 1.0).to(device)),
        "main_w": main_w,
        "front_w": front_w,
        "tail_w": tail_w,
        "kv_base": kv_base,
        "rope_base": rope_base,
        "idx_base": idx_base,
        "idx_scale_base": idx_scale_base,
        "page_table": page_table,
        "post": post,
        "shared": shared,
        "routed": routed,
    }


def make_override_routing(
    width: int, distinct_experts: int, local_experts: int, device: torch.device
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if distinct_experts <= 0:
        return None, None
    count = min(int(distinct_experts), int(local_experts))
    slots = torch.arange(width * 8, dtype=torch.int32) % count
    ids = _nd(slots.view(width, 8).to(device))
    probs = _nd(torch.full((width, 8), 1.0 / 8.0, dtype=torch.float32, device=device))
    return ids, probs


def bench_graph(stream, fn, calls: int = 1, warmup: int = 5, iters: int = 30, pairs: int = 6):
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


def _decode_prof_acc(raw: list[int]) -> dict[str, int]:
    return {
        "norm_cyc": raw[0],
        "idx_total_cyc": raw[1],
        "sh_total_cyc": raw[2],
        "bridge_cyc": raw[4],
        "moe_body_cyc": raw[5],
        "idx_prolog_cyc": raw[8],
        "idx_front_cyc": raw[9],
        "idx_rtk_cyc": raw[10],
        "idx_sfa_cyc": raw[11],
        "idx_tail_cyc": raw[12],
        "sh_prolog_cyc": raw[13],
        "sh_sfa_cyc": raw[14],
        "sh_tail_cyc": raw[15],
        "moe_disp0_cyc": raw[16],
        "moe_disp1_cyc": raw[17],
        "moe_gu_cyc": raw[18],
        "moe_act_dn2_cyc": raw[19],
        "moe_comb_cyc": raw[20],
        "moe_active_cnt": raw[21],
        "moe_max_active": raw[22],
        "moe_local_mask_bits": raw[23],
    }


def _load_prepare_libs(whole_layer_json: Path | None) -> dict[str, str]:
    if whole_layer_json and whole_layer_json.is_file():
        paths = json.loads(whole_layer_json.read_text("utf-8"))
    else:
        paths = dict(DEFAULT_WHOLE_LAYER_BASE)
    for key in ("indexed", "shared", "ffn"):
        torch.ops.load_library(str(paths[key]))
    return paths


def _prepare_width(
    bench_lib: BenchLibrary,
    ffn_ops,
    base: dict,
    width: int,
    device: torch.device,
) -> dict:
    seq_len = base["seq_len"]
    pages = base["pages"]
    table_pages = base["table_pages"]
    g = torch.Generator().manual_seed(1000 + width)
    before = seq_len - width

    norm_in = _nd((torch.randn((width, 6144), generator=g) * 0.1).bfloat16().to(device))
    res_in = _nd((torch.randn((width, 6144), generator=g) * 0.1).bfloat16().to(device))
    norm_out_idx = [
        _nd((torch.randn((width, 6144), generator=g) * 0.1).bfloat16().to(device)),
        _nd((torch.randn((width, 6144), generator=g) * 0.1).bfloat16().to(device)),
        _nd(torch.zeros((width,), dtype=torch.float32, device=device)),
        _nd(torch.randint(-32, 32, (width, 6144), generator=g, dtype=torch.int8).to(device)),
        _nd(((torch.rand((width, 1), generator=g) + 0.1) * 0.01).float().to(device)),
    ]
    norm_out_sh = [t.clone() for t in norm_out_idx]
    sin = _nd(torch.randn((width, 64), generator=g).bfloat16().to(device))
    cos = _nd(torch.randn((width, 64), generator=g).bfloat16().to(device))
    pos = torch.arange(before, before + width, dtype=torch.int64, device=device)
    table = base["page_table"][0].long()
    slots = _nd(table[pos // 128] * 128 + pos % 128)
    q_len = _nd(torch.tensor([width], dtype=torch.int32, device=device))
    kv_len = _nd(torch.tensor([before + width], dtype=torch.int32, device=device))

    kv_c = _nd(base["kv_base"].clone())
    rope_c = _nd(base["rope_base"].clone())
    idx_c = _nd(base["idx_base"].clone())
    idx_s = _nd(base["idx_scale_base"].clone())

    idx_scratch, idx_tiling, idx_offsets, blocks, sync_a = (
        torch.ops.tilert.indexed_attention_prepare(
            norm_out_idx[3], pages, table_pages, pages, table_pages, True, 24
        )
    )
    sh_scratch, sh_tiling, f_blocks, sync_s = torch.ops.tilert.shared_full_attention_prepare(
        norm_out_sh[3], pages, table_pages, 24
    )
    partial_idx = _nd(torch.zeros((width, 6144), dtype=torch.bfloat16, device=device))
    partial_sh = _nd(torch.zeros((width, 6144), dtype=torch.bfloat16, device=device))
    partial_moe = _nd(torch.zeros((width, 6144), dtype=torch.bfloat16, device=device))
    bridge, body, sync = ffn_ops.prepare(partial_moe, 24, 1)
    if blocks != 24 or f_blocks != 24 or sync_a != sync or sync_s != sync:
        raise RuntimeError(f"Workspace mismatch blocks={(blocks, f_blocks)} sync={(sync_a, sync)}")
    selected = _nd(torch.zeros((width, 1, 2048), dtype=torch.int32, device=device))

    main_meta = [sin, cos, slots, q_len, kv_len, base["page_table"]]
    idx_meta = [sin, cos, slots, q_len, kv_len, base["page_table"]]
    dims = (width, pages, table_pages, pages, table_pages, 24, 1, 1, 0)

    norm_idx = [norm_in, res_in, base["input_gamma"], *norm_out_idx]
    addrs_idx = [
        norm_out_idx[0],
        norm_out_idx[3],
        norm_out_idx[4],
        *base["main_w"],
        *base["front_w"],
        *base["tail_w"],
        None,
        *main_meta,
        *idx_meta,
        kv_c,
        rope_c,
        idx_c,
        idx_s,
        idx_scratch,
        idx_tiling,
        partial_idx,
    ]
    rec_idx = LocalRecords(
        bench_lib,
        norm_idx,
        addrs_idx,
        base["post"],
        base["routed"],
        bridge,
        body,
        base["shared"],
        selected,
        dims,
        ATTN_SCALARS,
        sync,
        True,
        prepare_fn=bench_lib.prepare_records,
    )

    norm_sh = [norm_in, res_in, base["input_gamma"], *norm_out_sh]
    addrs_sh = [
        norm_out_sh[0],
        norm_out_sh[3],
        norm_out_sh[4],
        *base["main_w"],
        *[None] * 6,
        *base["tail_w"],
        None,
        *main_meta,
        *[None] * 6,
        kv_c,
        rope_c,
        None,
        None,
        sh_scratch,
        sh_tiling,
        partial_sh,
    ]
    rec_sh = LocalRecords(
        bench_lib,
        norm_sh,
        addrs_sh,
        base["post"],
        base["routed"],
        bridge,
        body,
        base["shared"],
        selected,
        dims,
        ATTN_SCALARS,
        sync,
        False,
        prepare_fn=bench_lib.prepare_records,
    )

    return {
        "width": width,
        "sync": sync,
        "rec_idx": rec_idx,
        "rec_sh": rec_sh,
        "partial_idx": partial_idx,
        "partial_sh": partial_sh,
        "partial_moe": partial_moe,
        "idx_scratch": idx_scratch,
        "idx_offsets": idx_offsets,
        "selected": selected,
        "bridge": bridge,
        "body": body,
    }


def _run_single_die(args, so_path: Path, prof_so_path: Path | None) -> dict:
    dev = torch.device(args.device)
    torch.npu.set_device(dev)
    torch.npu.config.allow_internal_format = True
    _load_prepare_libs(args.whole_layer_json)
    ffn_ops = getattr(torch.ops, FFN_NAMESPACE)
    bench_lib = BenchLibrary(so_path)
    prof_lib = BenchLibrary(prof_so_path) if prof_so_path is not None else None

    base = make_tp16_inputs(dev, seq_len=args.seq_len, local_experts=16)
    ref_data = torch.load(args.check_ref) if args.check_ref else {}
    saved_ref = {}
    results = {}
    stream = torch.npu.Stream()
    prof_acc = _nd(torch.zeros((24,), dtype=torch.int64, device=dev))
    prof_zero = torch.zeros((24,), dtype=torch.int64, device=dev)

    with torch.inference_mode():
        for m in args.widths:
            state = _prepare_width(bench_lib, ffn_ops, base, m, dev)
            rec_idx = state["rec_idx"]
            rec_sh = state["rec_sh"]
            ov_ids, ov_probs = make_override_routing(m, args.distinct_experts, 16, dev)
            ov_ids_ptr = 0 if ov_ids is None else int(ov_ids.data_ptr())
            ov_probs_ptr = 0 if ov_probs is None else int(ov_probs.data_ptr())
            inc_norm = int(args.include_norm)
            run_bridge = int(args.run_bridge)

            def call_idx(lib=bench_lib, p_ptr=0):
                st = int(torch.npu.current_stream(dev).npu_stream)
                lib.check(
                    lib.launch_indexed(
                        st,
                        rec_idx.sync,
                        rec_idx.config,
                        len(rec_idx.config),
                        rec_idx.addresses,
                        len(rec_idx.addresses),
                        m,
                        0,
                        p_ptr,
                        inc_norm,
                    )
                )

            def call_sh(lib=bench_lib, p_ptr=0, resolve_mode=2):
                st = int(torch.npu.current_stream(dev).npu_stream)
                lib.check(
                    lib.launch_shared(
                        st,
                        rec_sh.sync,
                        rec_sh.config,
                        len(rec_sh.config),
                        rec_sh.addresses,
                        len(rec_sh.addresses),
                        m,
                        0,
                        p_ptr,
                        inc_norm,
                        resolve_mode,
                    )
                )

            def call_moe(lib=bench_lib, p_ptr=0):
                st = int(torch.npu.current_stream(dev).npu_stream)
                lib.check(
                    lib.launch_moe(
                        st,
                        rec_idx.sync,
                        rec_idx.config,
                        len(rec_idx.config),
                        rec_idx.addresses,
                        len(rec_idx.addresses),
                        m,
                        0,
                        run_bridge,
                        ov_ids_ptr,
                        ov_probs_ptr,
                        p_ptr,
                    )
                )

            call_idx()
            torch.npu.synchronize()
            topk_bytes = m * 2048 * 4
            topk_src = state["idx_scratch"].narrow(0, state["idx_offsets"]["topk"], topk_bytes)
            state["selected"].view(-1).copy_(topk_src.view(torch.int32))
            call_sh(resolve_mode=1)
            call_moe()
            torch.npu.synchronize()

            cur_idx = state["partial_idx"].cpu().view(torch.uint16)
            cur_topk = state["selected"].cpu()
            cur_sh = state["partial_sh"].cpu().view(torch.uint16)
            cur_moe_routed = state["bridge"][11][:m].cpu().view(torch.uint16)
            cur_moe_shared = state["body"][19][:m].cpu().view(torch.uint16)

            idx_med, idx_min = bench_graph(
                stream, call_idx, warmup=args.warmup, iters=args.iters, pairs=args.pairs
            )
            sh_med, sh_min = bench_graph(
                stream,
                lambda: call_sh(resolve_mode=2),
                warmup=args.warmup,
                iters=args.iters,
                pairs=args.pairs,
            )
            moe_med, moe_min = bench_graph(
                stream, call_moe, warmup=args.warmup, iters=args.iters, pairs=args.pairs
            )

            graph_checks = {
                "idx": bool(torch.equal(state["partial_idx"].cpu().view(torch.uint16), cur_idx)),
                "sh": bool(torch.equal(state["partial_sh"].cpu().view(torch.uint16), cur_sh)),
                "moe_routed": bool(
                    torch.equal(state["bridge"][11][:m].cpu().view(torch.uint16), cur_moe_routed)
                ),
                "moe_shared": bool(
                    torch.equal(state["body"][19][:m].cpu().view(torch.uint16), cur_moe_shared)
                ),
            }

            prof_summary = {}
            if prof_lib is not None:
                p_ptr = int(prof_acc.data_ptr())
                prof_acc.copy_(prof_zero)
                call_idx(lib=prof_lib, p_ptr=p_ptr)
                torch.npu.synchronize()
                idx_raw = prof_acc.cpu().tolist()
                prof_acc.copy_(prof_zero)
                call_sh(lib=prof_lib, p_ptr=p_ptr, resolve_mode=2)
                torch.npu.synchronize()
                sh_raw = prof_acc.cpu().tolist()
                prof_acc.copy_(prof_zero)
                call_moe(lib=prof_lib, p_ptr=p_ptr)
                torch.npu.synchronize()
                moe_raw = prof_acc.cpu().tolist()
                merged = [
                    idx_raw[0],
                    idx_raw[1],
                    sh_raw[2],
                    0,
                    moe_raw[4],
                    moe_raw[5],
                    0,
                    0,
                    *idx_raw[8:13],
                    *sh_raw[13:16],
                    *moe_raw[16:24],
                ]
                prof_summary = _decode_prof_acc(merged)

            ref_checks = {}
            if args.check_ref:
                rm = ref_data[m]
                ref_checks = {
                    "idx": bool(torch.equal(cur_idx, rm["out_idx"])),
                    "topk": bool(torch.equal(cur_topk, rm["topk"])),
                    "sh": bool(torch.equal(cur_sh, rm["out_sh"])),
                    "moe_routed": bool(torch.equal(cur_moe_routed, rm["moe_routed"])),
                    "moe_shared": bool(torch.equal(cur_moe_shared, rm["moe_shared"])),
                }
            if args.save_ref:
                saved_ref[m] = {
                    "out_idx": cur_idx,
                    "topk": cur_topk,
                    "out_sh": cur_sh,
                    "moe_routed": cur_moe_routed,
                    "moe_shared": cur_moe_shared,
                }

            results[str(m)] = {
                "width": m,
                "seq_len": args.seq_len,
                "distinct_experts": args.distinct_experts,
                "indexed_med_us": idx_med,
                "indexed_min_us": idx_min,
                "shared_med_us": sh_med,
                "shared_min_us": sh_min,
                "moe_med_us": moe_med,
                "moe_min_us": moe_min,
                "graph_parity": graph_checks,
                "ref_parity": ref_checks,
                "prof": prof_summary,
                "library_sha256": hashlib.sha256(so_path.read_bytes()).hexdigest(),
            }
            print(
                f"M={m:2d} | idx={idx_med:6.2f}us | sh={sh_med:6.2f}us | "
                f"moe={moe_med:6.2f}us | graph={graph_checks} | "
                f"ref={ref_checks if ref_checks else None}",
                flush=True,
            )

    if args.save_ref:
        torch.save(saved_ref, args.save_ref)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    bad = [
        (k, sub)
        for k, row in results.items()
        for sub in (row["graph_parity"], row["ref_parity"])
        if any(v is False for v in sub.values())
    ]
    if bad:
        raise RuntimeError(f"Bitwise parity check failed: {bad}")
    return results


def _run_multi_die(args, so_path: Path) -> dict:
    rank = int(os.environ.get("RANK", "0"))
    ranks = int(os.environ.get("WORLD_SIZE", str(args.ranks)))
    endpoint = f"127.0.0.1:{args.shmem_port}"
    dev = torch.device(f"npu:{rank}")
    torch.npu.set_device(dev)
    torch.npu.config.allow_internal_format = True
    _load_prepare_libs(args.whole_layer_json)
    ffn_ops = getattr(torch.ops, FFN_NAMESPACE)
    bench_lib = BenchLibrary(so_path)
    bench_lib.initialize(rank, rank, ranks, endpoint, 128 << 20)
    win = Window(bench_lib, 32)
    base = make_tp16_inputs(dev, seq_len=args.seq_len, local_experts=16)
    stream = torch.npu.Stream()
    results = {}
    try:
        with torch.inference_mode():
            for m in args.widths:
                state = _prepare_width(bench_lib, ffn_ops, base, m, dev)
                rec = state["rec_idx"]
                attn_sum = _nd(torch.zeros((m, 6144), dtype=torch.bfloat16, device=dev))
                out = _nd(torch.zeros((m, 6144), dtype=torch.bfloat16, device=dev))

                def call_layer():
                    st = int(torch.npu.current_stream(dev).npu_stream)
                    bench_lib.check(
                        bench_lib.handle.layer_tp_tree_submit(
                            rank,
                            rec.sync,
                            rec.config,
                            len(rec.config),
                            rec.addresses,
                            len(rec.addresses),
                            True,
                            win.handle,
                            int(attn_sum.data_ptr()),
                            int(out.data_ptr()),
                        )
                    )
                    _ = st

                med_us, min_us = bench_graph(
                    stream, call_layer, warmup=args.warmup, iters=args.iters, pairs=args.pairs
                )
                results[str(m)] = {
                    "rank": rank,
                    "ranks": ranks,
                    "width": m,
                    "layer_med_us": med_us,
                    "layer_min_us": min_us,
                }
                if rank == 0:
                    print(
                        f"multi-die ranks={ranks} M={m:2d} | layer={med_us:6.2f}us "
                        f"(min {min_us:6.2f})",
                        flush=True,
                    )
    finally:
        win.close()
        bench_lib.finalize()
    if args.output and rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("single-die", "multi-die"), default="single-die")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--ranks", type=int, default=2)
    p.add_argument("--shmem-port", type=int, default=29666)
    p.add_argument("--widths", type=int, nargs="+", default=[1, 8, 12, 16, 20, 24])
    p.add_argument("--seq-len", type=int, default=8704)
    p.add_argument("--distinct-experts", type=int, default=0)
    p.add_argument("--include-norm", action="store_true")
    p.add_argument("--no-bridge", dest="run_bridge", action="store_false")
    p.set_defaults(run_bridge=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--pairs", type=int, default=6)
    p.add_argument("--build-dir", type=Path, default=None)
    p.add_argument("--force-build", action="store_true")
    p.add_argument("--no-prof", dest="collect_prof", action="store_false")
    p.set_defaults(collect_prof=True)
    p.add_argument("--whole-layer-json", type=Path, default=None)
    p.add_argument("--save-ref", type=Path, default=None)
    p.add_argument("--check-ref", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    so_path, _ = build_microbench_tp16_lib(
        out_dir=args.build_dir, force=args.force_build, enable_kernel_prof=False
    )
    prof_so = None
    if args.mode == "single-die" and args.collect_prof:
        prof_so, _ = build_microbench_tp16_lib(
            out_dir=args.build_dir, force=args.force_build, enable_kernel_prof=True
        )
    if args.mode == "single-die":
        results = _run_single_die(args, so_path, prof_so)
    else:
        results = _run_multi_die(args, so_path)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
