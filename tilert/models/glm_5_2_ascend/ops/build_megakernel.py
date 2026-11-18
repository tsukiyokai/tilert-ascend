import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent / "megakernel"
REPO = Path(__file__).resolve().parents[4]
DEFAULT_OUT = REPO / "tests/.artifacts/builds/megakernel_tp8_ep8"

ASC_SUBDIRS = (
    "asc",
    "asc/include",
    "asc/include/adv_api",
    "asc/include/basic_api",
    "asc/include/c_api",
    "asc/include/simt_api",
    "asc/include/utils",
    "asc/include/basic_api/reg_compute",
    "asc/impl/adv_api",
    "asc/impl/basic_api",
    "asc/impl/c_api",
    "asc/impl/simt_api",
    "asc/impl/utils",
    "asc/impl/basic_api/reg_compute",
    "tikcpp/tikcfw",
    "tikcpp/tikcfw/lib",
    "tikcpp/tikcfw/impl",
    "tikcpp/tikcfw/interface",
)

BASE_FLAGS = (
    "-DSFA_MM2_PAIR_LOAD=1",
    "-DQUERY_GROUPS_NARROW=1",
    "-DBF16_QUERY12_GROUPS=1",
    "-DGU_WIDE=1",
    "-DATTENTION_HEADS=8",
    "-DFFN_OP_NS=tilert_ffn_original_shared_gu_static",
    "-DROUTED_ROW_BALANCED=1",
    "-DROUTED_SHARED_PIPELINE=1",
    "-DROUTED_INPUT_FRONT=1",
    "-DSHARED_W8_DYNAMIC_PERFORMANCE_QUANT=1",
    "-DROUTED_ACTIVE_LIST=1",
    "-DSKIP_SINGLE_KERNELS=1",
)

R4_FLAGS = (
    "-DK11040_F0_NZ=1",
    "-DK11040_TAIL_NZ=1",
    "-DK11040_GU_FAST128=1",
    "-DK11080_DIRECT_W4A8=1",
    "-DK11083_RED2_OPT=1",
    "-DK11088_GATE_CONTIG=1",
    "-DK11090_FAST_SFA_VEC=1",
    "-DK11050_AIC_PREFETCH=1",
    "-DK11100_OPP_PREFETCH=1",
    "-DK11110_SFA_PIPE=1",
    "-DK11110_DW4A8_XTASK=1",
    "-DK11070_SFA_M16_COOP=24",
    "-DK11120_SFA_GATHER_OPT=1",
    "-DK11120_SKIP_DUP_PUB=1",
    "-DK11140_DIRECT_MLA_TAIL=1",
    "-DK11150_MLA_TAIL_M32=1",
    "-DK11150_SHARED_W8_NZ=1",
    "-DK11150_SIP_RELAY_RED2=1",
    "-DK11180_QUAD4_SIP_ALL=1",
    "-DK11180_TOK_OUT_BAL=1",
    "-DK11180_NORM_OPT=1",
    "-DK112110_SIP_RED1=1",
    "-DK112140_SIP_RED1_MIN_ROWS=8",
    "-DK112120_CLEAN_DCCI=1",
    "-DK112140_PIPE_KEY_ONLY=1",
    "-DK112120_SFA_COMPACT_WS=1",
    "-DK112130_VEC_RESOLVED_TOPK=1",
    "-DK112130_SPLIT_HEADS_4_4=1",
    "-DK112140_FAST_RTK=1",
    "-DK112160_AIV_DSB_OPT=1",
    "-DK112162_FUSED_IDX_RTK=1",
    "-DK112162_MERGE_CONTIG=1",
    "-DK112168_MLA_TAIL_OVERLAP=1",
    "-DK112170_DEC_SIP_ATT=1",
    "-DK112174_FAST_MERGE_KV=1",
    "-DK112176_PHASE_DSB_OPT=1",
    "-DK112184_IDX_FRONT_OPT=1",
    "-DK112192_DIRECT_IDX_Q14=1",
    "-DK112194_PROLOG_CUBE_BAL=1",
    "-DK11240_OUTLINE_TOP=1",
    "-DK112166_SFA_QV_L1=1",
    "-DK112188_SFA_QVL1_ALL=1",
    "-DK11220_FAST_MIRROR_V4=1",
    "-DK11220_MULTI_M16=1",
    "-DK11220_XTASK_B=1",
    "-DK11220_EXACT_COST=1",
    "-DK11220_FAST_TOK_OUT=1",
    "-DK11220_FAST_ACT_P3=1",
    "-DK11220_BITMASK_OWN_P1=1",
    "-DK11220_DOWN_CONTIG_256=1",
    "-DK11220_SH_ROW_1D=1",
    "-DK11220_LOCAL_SH_RED2=1",
    "-DK11220_PF_RED2_FULL=1",
    "-DK11220_FAST_P1_V3=1",
    "-DK11220_MULTI_M32=1",
    "-DK112612_BRIDGE_FAST=1",
    "-DK112612_DOWN_PIPE2=1",
    "-DK11220_DENSE78_COLD=1",
    "-DK112510_MLA_TAIL_XCORE_V2=1",
    "-DK112512_PROLOG_48AIV_V2=1",
    "-DK112702_NO_DDR_INTRADIE=1",
    "-DK112722_NO_DDR_EXTRA=1",
    "-DK112726_BRIDGE_PRELOAD_GATE=1",
    "-DK112600_EXACT48_RED2_OPT=1",
    "-DK112514_IDX_PIPE_V2=1",
    "-DK112824_NO_R2_END_DSB=1",
    "-DK112612_TRIM_BARRIERS=1",
    "-DK112824_SPLIT_PROF=1",
)


def default_build_dir() -> Path:
    env_dir = os.environ.get("TILERT_BUILD_DIR")
    return Path(env_dir) if env_dir else DEFAULT_OUT


def _cann_root() -> Path:
    for candidate in (
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        "/opt/ascend/cann-9.2.0",
        "/usr/local/Ascend/ascend-toolkit/latest",
    ):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return Path("/opt/ascend/cann-9.2.0")


def _site_root() -> Path:
    try:
        import torch

        return Path(torch.__file__).resolve().parents[1]
    except Exception:
        return Path("/usr/local/lib/python3.11/site-packages")


def source_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(ROOT.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(ROOT)).encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()


def build_chain(
    out_dir: Path | None = None,
    ver: int = 112830,
    union_sfa: bool = False,
) -> tuple[Path, Path]:
    out = default_build_dir() if out_dir is None else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cann = _cann_root()
    site = _site_root()
    defs = [f"-DKERNEL_VER={ver}", *BASE_FLAGS, *R4_FLAGS]
    if union_sfa:
        defs.append("-DK112840_UNION_SFA=1")

    inc = [
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
    k_obj = out / f"kernel_{ver}.o"
    h_obj = out / f"host_{ver}.o"
    c_obj = out / f"comm_{ver}.o"
    so_path = out / "chain.so"
    manifest_path = out / "chain.manifest.json"

    k_cmd = [
        str(cann / "bin/bisheng"), "-xcce", "--npu-arch=dav-2201",
        "-std=c++17", "-O2", "-w", "-ffp-contract=off", "-fPIC",
        *defs, *inc, "-c", str(ROOT / "source/kernel.cpp"), "-o", str(k_obj),
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
        str(k_obj), str(h_obj), str(c_obj), "-o", str(so_path), *libs,
    ]
    script = (
        f"set -euo pipefail\n"
        f"set +u; source {shlex.quote(str(cann / 'set_env.sh'))}; set -u\n"
        f"{' '.join(shlex.quote(x) for x in k_cmd)} &\n"
        f"{' '.join(shlex.quote(x) for x in h_cmd)} &\n"
        f"{' '.join(shlex.quote(x) for x in c_cmd)} &\n"
        f"wait\n"
        f"{' '.join(shlex.quote(x) for x in link_cmd)}\n"
    )
    try:
        subprocess.run(["bash", "-c", script], check=True)
    finally:
        for obj in (k_obj, h_obj, c_obj):
            if obj.exists():
                obj.unlink()

    sha256 = hashlib.sha256(so_path.read_bytes()).hexdigest()
    manifest = {
        "ver": ver,
        "union_sfa": bool(union_sfa),
        "flags": defs,
        "sha256": sha256,
        "source_sha256": source_digest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return so_path, manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--ver", type=int, default=112830)
    parser.add_argument("--union-sfa", action="store_true")
    args = parser.parse_args(argv)
    out_dir = Path(args.out) if args.out else None
    so_path, manifest_path = build_chain(out_dir=out_dir, ver=args.ver, union_sfa=args.union_sfa)
    print(f"built={so_path} manifest={manifest_path}")


if __name__ == "__main__":
    main()
