import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tilert.models.glm_5_2_ascend.abi as abi_mod
from tilert.models.glm_5_2_ascend.abi import (
    Libraries,
    PreparedChain,
    capture_prof_acc,
    extract_prof_acc,
    kernel_prof_enabled,
)
import tilert.models.glm_5_2_ascend.ops.build_megakernel as bm
from tests.glm52.e2e_tp8_ep8.hygiene_gate import verify_python_source_hygiene


@pytest.fixture
def dspine_mods(monkeypatch: pytest.MonkeyPatch):
    if "torch" not in sys.modules:
        fake_torch = ModuleType("torch")
        fake_torch.__file__ = "/opt/fake_site/torch/__init__.py"
        fake_torch.Tensor = object
        fake_torch.bfloat16 = "bf16"
        fake_torch.float16 = "f16"
        fake_torch.float32 = "f32"
        fake_torch.int8 = "i8"
        fake_torch.int32 = "i32"
        fake_torch.int64 = "i64"
        fake_torch.uint8 = "u8"
        fake_torch.nn = SimpleNamespace(Module=object)
        fake_torch.distributed = SimpleNamespace(
            barrier=lambda **_kw: None,
            is_initialized=lambda: False,
        )
        fake_torch.npu = SimpleNamespace(
            device=lambda *_a, **_kw: None,
            synchronize=lambda *_a, **_kw: None,
        )
        fake_torch.inference_mode = lambda: None
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
    cal = importlib.import_module("tilert.models.glm_5_2_ascend.dspine.calibrated_l78")
    tp_mod = importlib.import_module("tilert.models.glm_5_2_ascend.dspine.tp_drafter")
    return cal, tp_mod


def _make_fake_l78_src(base: Path) -> Path:
    src = base / "mk_src"
    for sub in ("local", "source", "vendor", "communication"):
        (src / sub).mkdir(parents=True, exist_ok=True)
    attn_dir = src / "local/attention/source/tests/glm52/whole_layer"
    attn_dir.mkdir(parents=True, exist_ok=True)
    (attn_dir / "attention_native.h").write_text(
        "\n".join(
            [
                "inline void SfaTiling(int config) {",
                "    base.sparseBlockCount = DRAFT_SFA_TOPK;",
                "}",
                "inline void SfaPhase(int config, bool is_shared_only) {",
                "    auto tiling = SfaTiling(config);",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (src / "source/kernel.cpp").write_text(
        "\n".join(
            [
                "template <bool indexed, int Ver = KERNEL_VER>",
                "void Kernel() {}",
                'extern "C" void layer_tp_tree_set_mask_ptr(uint64_t ptr) {}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (src / "source/host.cpp").write_text("int h = 0;\n", encoding="utf-8")
    (src / "communication/communication_host.cpp").write_text("int c = 0;\n", encoding="utf-8")
    return src


def test_targeted_l78_variant_resolution_and_a2_patch(
    tmp_path: Path,
    dspine_mods: tuple[ModuleType, ModuleType],
) -> None:
    cal, _ = dspine_mods
    assert cal._resolve_l78_ksfa(1024, None) == (1024,)
    assert cal._resolve_l78_ksfa(2048, None) == (2048,)
    assert cal._resolve_l78_ksfa(512, None) == (512,)
    assert cal._resolve_l78_ksfa(None, None) == (2048, 1024, 512)
    assert cal._resolve_l78_ksfa(None, (1024, 512)) == (1024, 512)
    with pytest.raises(ValueError):
        cal._resolve_l78_ksfa(768, None)
    with pytest.raises(ValueError):
        cal._resolve_l78_ksfa(1024, (512,))
    with pytest.raises(ValueError):
        cal._resolve_l78_ksfa(None, [])

    script_1024 = cal._render_l78_build_script(
        tmp_path / "out",
        tmp_path / "cann",
        tmp_path / "site",
        (1024,),
    )
    assert "compile_variant 110419 l78k1024" in script_1024
    assert "compile_variant 110415" not in script_1024
    assert "compile_variant 110420" not in script_1024

    src = _make_fake_l78_src(tmp_path)
    out = tmp_path / "copied"
    for sub in ("local", "source", "vendor", "communication"):
        shutil.copytree(src / sub, out / sub)
    cal._patch_l78_copied_sources(out)
    patched = (
        out / "local/attention/source/tests/glm52/whole_layer/attention_native.h"
    ).read_text(encoding="utf-8")
    assert "if (!is_shared_only)" in patched
    assert "tiling.baseParams.sparseBlockCount = kTopK;" in patched


def test_l78_incremental_manifest_cache_and_partial_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dspine_mods: tuple[ModuleType, ModuleType],
) -> None:
    cal, _ = dspine_mods
    src = _make_fake_l78_src(tmp_path)
    out = tmp_path / "l78_out"
    calls: list[str] = []

    def fake_run(cmd: list[str], check: bool = True) -> SimpleNamespace:
        sh_txt = Path(cmd[1]).read_text(encoding="utf-8")
        calls.append(sh_txt)
        for ver in (110415, 110419, 110420):
            if f"compile_variant {ver} " in sh_txt:
                (out / f"chain_K{ver}.so").write_bytes(f"so-{ver}-{len(calls)}".encode("utf-8"))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cal.subprocess, "run", fake_run)
    so_19, so_20 = cal.build_draft_l78_k110419(src_dir=src, out_dir=out, k_sfa=1024)
    assert so_19.exists()
    assert not so_20.exists()
    assert not (out / "chain_K110415.so").exists()
    assert len(calls) == 1

    cal.build_draft_l78_k110419(src_dir=src, out_dir=out, k_sfa=1024)
    assert len(calls) == 1

    bytes_19 = so_19.read_bytes()
    cal.build_draft_l78_k110419(src_dir=src, out_dir=out, k_sfa=512)
    assert len(calls) == 2
    assert "compile_variant 110420 " in calls[1]
    assert "compile_variant 110419 " not in calls[1]
    assert so_19.read_bytes() == bytes_19
    assert so_20.exists()

    so_19.write_bytes(b"corrupt")
    cal.build_draft_l78_k110419(src_dir=src, out_dir=out, k_sfa=1024)
    assert len(calls) == 3
    assert "compile_variant 110419 " in calls[2]
    assert "compile_variant 110420 " not in calls[2]


def test_tp_drafter_ensure_l78_klibs_single_variant_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dspine_mods: tuple[ModuleType, ModuleType],
) -> None:
    cal, tp_mod = dspine_mods
    l78_dir = tmp_path / "dspine_l78"
    l78_dir.mkdir()
    loaded_paths: list[str] = []

    class _FakeLib:
        def __init__(self, path: str, mode: object = None) -> None:
            loaded_paths.append(path)
            for ver in (110415, 110419, 110420):
                setattr(self, f"layer_tp_tree_set_mask_ptr_K{ver}", lambda _ptr: 0)
                setattr(self, f"layer_tp_tree_indexed_launch_K{ver}", lambda *_a: 0)
                setattr(self, f"layer_tp_tree_shared_launch_K{ver}", lambda *_a: 0)
                setattr(self, f"layer_tp_tree_kv_commit_launch_K{ver}", lambda *_a: 0)

    def fake_build(src_dir=None, out_dir=None, k_sfa=None, variants=None):
        ver = cal.L78_VARIANT_SPECS[k_sfa][0]
        target = Path(out_dir) / f"chain_K{ver}.so"
        target.write_bytes(f"lib-{ver}".encode("utf-8"))
        return Path(out_dir) / "chain_K110419.so", Path(out_dir) / "chain_K110420.so"

    monkeypatch.setattr(cal, "build_draft_l78_k110419", fake_build)
    monkeypatch.setattr(tp_mod, "build_draft_l78_k110419", fake_build)
    monkeypatch.setattr(tp_mod, "torch_dist_ready", lambda: False)
    monkeypatch.setattr(tp_mod.ctypes, "CDLL", _FakeLib)
    runner_stub = SimpleNamespace(
        k_sfa=1024,
        rank=0,
        l78_build_dir=l78_dir,
        channel_group=None,
        wt=SimpleNamespace(),
    )
    by_ver = tp_mod.CalibratedL78V3Runner._ensure_l78_klibs(runner_stub)
    assert list(by_ver.keys()) == [110419]
    assert not (l78_dir / "chain_K110415.so").exists()
    assert len(loaded_paths) == 1

    runner_stub.k_sfa = 512
    by_ver = tp_mod.CalibratedL78V3Runner._ensure_l78_klibs(runner_stub)
    assert set(by_ver.keys()) == {110419, 110420}


def test_chain_decoupling_stripper_and_safety_negatives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mk_copy = tmp_path / "megakernel"
    shutil.copytree(bm.ROOT, mk_copy)
    monkeypatch.setattr(bm, "ROOT", mk_copy)
    defs = [f"-DKERNEL_VER=112830", *bm.BASE_FLAGS, *bm.R4_FLAGS]
    base_raw = bm.source_digest()
    base_eff = bm.effective_source_digest(defs)

    norm_h = mk_copy / "local/ffn/bridge/vendor/add_rms_norm_bias_single_n.h"
    orig_norm = norm_h.read_text(encoding="utf-8")
    norm_h.write_text(
        orig_norm.replace("#ifdef L78_PLAIN_RMSNORM\n", "#ifdef L78_PLAIN_RMSNORM\n        ;\n", 1),
        encoding="utf-8",
    )
    assert bm.source_digest() != base_raw
    assert bm.effective_source_digest(defs) == base_eff

    idx_h = mk_copy / "local/attention/source/tests/glm52/whole_layer/indexer_front_native.h"
    orig_idx = idx_h.read_text(encoding="utf-8")
    idx_h.write_text(
        orig_idx.replace("const bool use_k512 = true;", "const bool use_k512 = false;", 1),
        encoding="utf-8",
    )
    assert bm.effective_source_digest(defs) == base_eff

    attn_h = mk_copy / "local/attention/source/tests/glm52/whole_layer/attention_native.h"
    orig_attn = attn_h.read_text(encoding="utf-8")
    attn_h.write_text(
        orig_attn.replace(
            "#ifdef K11089_L78_SFA_TOPK\n",
            "#ifdef K11089_L78_SFA_TOPK\n            tiling.baseParams.sparseBlockCount = 1;\n",
            1,
        ),
        encoding="utf-8",
    )
    assert bm.effective_source_digest(defs) == base_eff

    kern_cpp = mk_copy / "source/kernel.cpp"
    orig_kern = kern_cpp.read_text(encoding="utf-8")
    kern_cpp.write_text(
        orig_kern.replace(
            "#ifndef SKIP_SINGLE_KERNELS\n",
            "#ifndef SKIP_SINGLE_KERNELS\nint z=0;\n",
            1,
        ),
        encoding="utf-8",
    )
    assert bm.effective_source_digest(defs) == base_eff

    out_dir = tmp_path / "builds"
    out_dir.mkdir()
    build_calls: list[int] = []

    def fake_build_run(cmd: list[str], check: bool = True) -> SimpleNamespace:
        build_calls.append(1)
        (out_dir / "chain.so").write_bytes(b"fake-chain-elf")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bm.subprocess, "run", fake_build_run)
    so_p, man_p = bm.build_chain(out_dir=out_dir)
    assert len(build_calls) == 1

    norm_h.write_text(
        orig_norm.replace(
            "#ifdef L78_PLAIN_RMSNORM\n",
            "#ifdef L78_PLAIN_RMSNORM\n        ;;\n",
            1,
        ),
        encoding="utf-8",
    )
    bm.build_chain(out_dir=out_dir)
    assert len(build_calls) == 1

    def dummy_libs() -> Libraries:
        dummy = tmp_path / "dummy.so"
        dummy.write_bytes(b"x")
        return Libraries(dummy, dummy, dummy, dummy, so_p)

    lib_hashes_backup = abi_mod.LIBRARY_HASHES
    monkeypatch.setattr(
        abi_mod,
        "LIBRARY_HASHES",
        {
            **lib_hashes_backup,
            "indexed": hashlib.sha256(b"x").hexdigest(),
            "shared": hashlib.sha256(b"x").hexdigest(),
            "ffn": hashlib.sha256(b"x").hexdigest(),
            "metadata": hashlib.sha256(b"x").hexdigest(),
        },
    )
    libs = dummy_libs()
    libs.verify()

    norm_h.write_text(
        orig_norm.replace("#else\n", "#else\n        Muls(sqxLocal, sqxLocal, 1.0f, 1);\n", 1),
        encoding="utf-8",
    )
    assert bm.effective_source_digest(defs) != base_eff
    with pytest.raises(ValueError, match="source changed"):
        libs.verify()
    norm_h.write_text(orig_norm, encoding="utf-8")
    libs.verify()

    attn_h.write_text(
        orig_attn.replace(
            "base.sparseBlockCount = DRAFT_SFA_TOPK;",
            "base.sparseBlockCount = 1024;",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source changed"):
        libs.verify()
    attn_h.write_text(orig_attn, encoding="utf-8")

    contract_h = mk_copy / "source/contract.h"
    orig_contract = contract_h.read_text(encoding="utf-8")
    contract_h.write_text(orig_contract + "\nstruct Extra {};\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        libs.verify()
    contract_h.write_text(orig_contract, encoding="utf-8")

    man_data = json.loads(man_p.read_text(encoding="utf-8"))
    man_data["flags"] = [*man_data["flags"], "-DEXTRA=1"]
    man_p.write_text(json.dumps(man_data), encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        libs.verify()
    bm.build_chain(out_dir=out_dir, union_sfa=True)
    assert len(build_calls) == 2

    so_p.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="library changed"):
        libs.verify()


def test_stripper_fail_closed_and_legacy_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defined = bm._extract_defined_macros(bm.BASE_FLAGS)
    active_def = "#define L78_PLAIN_RMSNORM 1\n#ifdef L78_PLAIN_RMSNORM\nx\n#endif\n"
    assert bm._strip_inactive_branches(active_def, defined) == active_def.encode("utf-8")

    unbalanced = "#ifdef L78_PLAIN_RMSNORM\nx\n"
    assert bm._strip_inactive_branches(unbalanced, defined) == unbalanced.encode("utf-8")

    raw_lit = 'const char* s = R"(\n#ifdef L78_PLAIN_RMSNORM\nsecret\n#endif\n)";\n'
    stripped_raw = bm._strip_inactive_branches(raw_lit, defined)
    assert b"secret" in stripped_raw

    str_lit = 'const char* s = "#ifdef L78_PLAIN_RMSNORM";\n'
    assert b"#ifdef L78_PLAIN_RMSNORM" in bm._strip_inactive_branches(str_lit, defined)

    mk_copy = tmp_path / "mk_legacy"
    shutil.copytree(bm.ROOT, mk_copy)
    monkeypatch.setattr(bm, "ROOT", mk_copy)
    so_p = tmp_path / "chain.so"
    so_p.write_bytes(b"chain")
    man_p = tmp_path / "chain.manifest.json"
    man_p.write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(b"chain").hexdigest(),
                "source_sha256": bm.source_digest(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        abi_mod,
        "LIBRARY_HASHES",
        {
            **abi_mod.LIBRARY_HASHES,
            "indexed": hashlib.sha256(b"x").hexdigest(),
            "shared": hashlib.sha256(b"x").hexdigest(),
            "ffn": hashlib.sha256(b"x").hexdigest(),
            "metadata": hashlib.sha256(b"x").hexdigest(),
        },
    )
    dummy = tmp_path / "d.so"
    dummy.write_bytes(b"x")
    libs = Libraries(dummy, dummy, dummy, dummy, so_p)
    libs.verify()


def test_build_chain_enable_kernel_prof_and_libraries_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mk_copy = tmp_path / "mk_prof"
    shutil.copytree(bm.ROOT, mk_copy)
    monkeypatch.setattr(bm, "ROOT", mk_copy)
    monkeypatch.setenv("TILERT_BUILD_DIR", str(tmp_path / "default_builds"))
    calls: list[Path] = []

    def fake_run(cmd: list[str], check: bool = True) -> SimpleNamespace:
        script = cmd[2]
        for tok in script.split():
            if tok.endswith("chain.so"):
                out_so = Path(tok)
                out_so.write_bytes(f"elf-{len(calls)}".encode("utf-8"))
                calls.append(out_so)
                break
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bm.subprocess, "run", fake_run)
    std_so, std_man = bm.build_chain(enable_kernel_prof=False)
    assert std_so == tmp_path / "default_builds" / "chain.so"
    std_data = json.loads(std_man.read_text(encoding="utf-8"))
    assert std_data["enable_kernel_prof"] is False
    assert "-DENABLE_KERNEL_PROF=1" not in std_data["flags"]

    prof_so, prof_man = bm.build_chain(enable_kernel_prof=True)
    assert prof_so == tmp_path / "default_builds" / "prof" / "chain.so"
    prof_data = json.loads(prof_man.read_text(encoding="utf-8"))
    assert prof_data["enable_kernel_prof"] is True
    assert "-DENABLE_KERNEL_PROF=1" in prof_data["flags"]
    assert len(calls) == 2

    bm.build_chain(out_dir=prof_so.parent, enable_kernel_prof=True)
    assert len(calls) == 2

    bm.build_chain(out_dir=prof_so.parent, enable_kernel_prof=False)
    assert len(calls) == 3

    bm.build_chain(out_dir=prof_so.parent, enable_kernel_prof=True)
    assert len(calls) == 4

    monkeypatch.setattr(
        abi_mod,
        "LIBRARY_HASHES",
        {
            **abi_mod.LIBRARY_HASHES,
            "indexed": hashlib.sha256(b"x").hexdigest(),
            "shared": hashlib.sha256(b"x").hexdigest(),
            "ffn": hashlib.sha256(b"x").hexdigest(),
            "metadata": hashlib.sha256(b"x").hexdigest(),
        },
    )
    dummy = tmp_path / "d.so"
    dummy.write_bytes(b"x")
    libs = Libraries(dummy, dummy, dummy, dummy, prof_so)
    libs.verify()

    tampered = json.loads(prof_man.read_text(encoding="utf-8"))
    tampered["enable_kernel_prof"] = False
    prof_man.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="prof flag changed"):
        libs.verify()


def test_prepared_chain_capture_prof_acc_and_kernel_prof_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_calls: list[object] = []

    class _FakeTensor:
        def __init__(
            self,
            values: list[int],
            *,
            dtype: str = "i64",
            dev_type: str = "npu",
            elem_size: int = 8,
        ) -> None:
            self.values = list(values)
            self.dtype = dtype
            self.device = SimpleNamespace(type=dev_type)
            self._elem_size = elem_size
            self.cpu_called = False
            self.cloned = False
            self.detached = False

        def numel(self) -> int:
            return len(self.values)

        def element_size(self) -> int:
            return self._elem_size

        def view(self, target_dtype: object) -> "_FakeTensor":
            assert target_dtype == "i64"
            if self.dtype == "u8":
                words = []
                for idx in range(0, len(self.values), 8):
                    chunk = self.values[idx : idx + 8]
                    word = int.from_bytes(bytes(chunk), "little", signed=True)
                    words.append(word)
                return _FakeTensor(
                    words,
                    dtype="i64",
                    dev_type=self.device.type,
                    elem_size=8,
                )
            return self

        def reshape(self, *_shape: int) -> "_FakeTensor":
            return self

        def __getitem__(self, item: slice) -> "_FakeTensor":
            return _FakeTensor(
                self.values[item],
                dtype=self.dtype,
                dev_type=self.device.type,
                elem_size=self._elem_size,
            )

        def detach(self) -> "_FakeTensor":
            out = _FakeTensor(
                self.values,
                dtype=self.dtype,
                dev_type=self.device.type,
                elem_size=self._elem_size,
            )
            out.detached = True
            return out

        def cpu(self) -> "_FakeTensor":
            out = _FakeTensor(
                self.values,
                dtype=self.dtype,
                dev_type="cpu",
                elem_size=self._elem_size,
            )
            out.detached = self.detached
            out.cpu_called = True
            return out

        def clone(self) -> "_FakeTensor":
            out = _FakeTensor(
                self.values,
                dtype=self.dtype,
                dev_type=self.device.type,
                elem_size=self._elem_size,
            )
            out.detached = self.detached
            out.cpu_called = self.cpu_called
            out.cloned = True
            return out

    fake_torch = ModuleType("torch")
    fake_torch.int64 = "i64"
    fake_torch.npu = SimpleNamespace(synchronize=lambda dev: sync_calls.append(dev))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def make_body6_bytes(prof_vals: list[int], dev_type: str = "npu") -> list[object]:
        buf = bytearray(512)
        for idx, val in enumerate(prof_vals):
            buf[(40 + idx) * 8 : (41 + idx) * 8] = int(val).to_bytes(8, "little", signed=True)
        return [None] * 6 + [_FakeTensor(list(buf), dtype="u8", dev_type=dev_type, elem_size=1)]

    assert kernel_prof_enabled(None) is False
    assert kernel_prof_enabled({"enable_kernel_prof": True, "flags": []}) is True
    assert kernel_prof_enabled({"enable_kernel_prof": False, "flags": ["-DENABLE_KERNEL_PROF=1"]})
    assert kernel_prof_enabled({"enable_kernel_prof": False, "flags": []}) is False

    chain = PreparedChain.__new__(PreparedChain)
    chain.device = 0
    chain.steps = (
        SimpleNamespace(local=SimpleNamespace(body=make_body6_bytes([0] * 24))),
        SimpleNamespace(local=SimpleNamespace(body=make_body6_bytes(list(range(100, 124))))),
    )

    got = capture_prof_acc(chain)
    assert got.values == list(range(100, 124))
    assert got.detached and got.cpu_called and got.cloned
    assert sync_calls == [0]

    got_default = chain.capture_prof_acc(sync=False, cpu=False)
    assert got_default.values == list(range(100, 124))
    assert not got_default.cpu_called
    assert sync_calls == [0]

    with pytest.raises(RuntimeError, match="smaller than 512 bytes"):
        extract_prof_acc(_FakeTensor([0] * 128, dtype="u8", elem_size=1))

    with pytest.raises(RuntimeError, match="8-byte aligned"):
        extract_prof_acc(_FakeTensor([0] * 513, dtype="u8", elem_size=1))

    chain.steps = (SimpleNamespace(local=SimpleNamespace(body=[None] * 6)),)
    with pytest.raises(RuntimeError, match="missing body\\[6\\]"):
        chain.capture_prof_acc()

    chain.steps = (SimpleNamespace(local=SimpleNamespace(body=[None] * 7)),)
    with pytest.raises(RuntimeError, match="missing body\\[6\\]"):
        chain.capture_prof_acc()


def test_stripper_cross_file_define_raw_define_and_ident_r(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defined = bm._extract_defined_macros([*bm.BASE_FLAGS, *bm.R4_FLAGS])

    raw_def = '#define RAW_MACRO R"(\n#ifdef L78_PLAIN_RMSNORM\nKEEP_ME\n#endif\n)"\n'
    assert b"KEEP_ME" in bm._strip_inactive_branches(raw_def, defined)

    ident_r = (
        'MACRO_R"(x";\n#ifdef L78_PLAIN_RMSNORM\nSHOULD_STRIP\n#endif\n'
        'const char* b = ")";\n'
    )
    stripped_ident = bm._strip_inactive_branches(ident_r, defined)
    assert b"SHOULD_STRIP" not in stripped_ident
    assert b'const char* b = ")";' in stripped_ident

    mk_copy = tmp_path / "mk_cross"
    shutil.copytree(bm.ROOT, mk_copy)
    monkeypatch.setattr(bm, "ROOT", mk_copy)
    (mk_copy / "cross_def.h").write_text("#define L78_PLAIN_RMSNORM 1\n", encoding="utf-8")
    cross_use = mk_copy / "cross_use.h"
    cross_use.write_text("#ifdef L78_PLAIN_RMSNORM\nint a = 1;\n#endif\n", encoding="utf-8")
    d1 = bm.effective_source_digest()
    cross_use.write_text("#ifdef L78_PLAIN_RMSNORM\nint a = 2;\n#endif\n", encoding="utf-8")
    d2 = bm.effective_source_digest()
    assert d1 != d2

    (mk_copy / "cross_def.h").write_text("#undef SKIP_SINGLE_KERNELS\n", encoding="utf-8")
    cross_use.write_text("#ifndef SKIP_SINGLE_KERNELS\nint b = 1;\n#endif\n", encoding="utf-8")
    u1 = bm.effective_source_digest()
    cross_use.write_text("#ifndef SKIP_SINGLE_KERNELS\nint b = 2;\n#endif\n", encoding="utf-8")
    u2 = bm.effective_source_digest()
    assert u1 != u2


def test_prepare_whole_layer_json_prof_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "torch" not in sys.modules:
        fake_torch = ModuleType("torch")
        fake_torch.__file__ = "/opt/fake_site/torch/__init__.py"
        fake_torch.Tensor = object
        fake_torch.bfloat16 = "bf16"
        fake_torch.float16 = "f16"
        fake_torch.float32 = "f32"
        fake_torch.int8 = "i8"
        fake_torch.int32 = "i32"
        fake_torch.int64 = "i64"
        fake_torch.uint8 = "u8"
        fake_torch.nn = SimpleNamespace(Module=object)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

    run_m2 = importlib.import_module("tests.glm52.run_m2_case01")
    assert run_m2.loaded_modules_has_vllm() is False
    default_dir = tmp_path / "default_builds"
    monkeypatch.setenv("TILERT_BUILD_DIR", str(default_dir))
    calls: list[tuple[object, bool, bool]] = []

    def fake_build_chain(*, out_dir=None, union_sfa=False, enable_kernel_prof=False):
        calls.append((out_dir, union_sfa, enable_kernel_prof))
        target_dir = (
            default_dir / "prof"
            if (out_dir is None and enable_kernel_prof)
            else (default_dir if out_dir is None else Path(out_dir))
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        so_p = target_dir / "chain.so"
        man_p = target_dir / "chain.manifest.json"
        so_p.write_bytes(b"prof-elf")
        man_p.write_text("{}", encoding="utf-8")
        return so_p, man_p

    dspine_pkg = importlib.import_module("tilert.models.glm_5_2_ascend.dspine")
    ops_mod = importlib.import_module("tilert.models.glm_5_2_ascend.dspine.build_dspine_ops")
    monkeypatch.setattr(bm, "build_chain", fake_build_chain)
    monkeypatch.setattr(ops_mod, "build_dspine_ops", lambda out: (Path(out) / "op.so", None))
    monkeypatch.setattr(
        dspine_pkg,
        "build_draft_l78_k110419",
        lambda out_dir=None, k_sfa=1024: (Path(out_dir) / "k19.so", Path(out_dir) / "k20.so"),
    )

    wl_base = tmp_path / "base.json"
    wl_base.write_text(json.dumps({"tree": "orig"}) + "\n", encoding="utf-8")
    args = SimpleNamespace(
        build_dir=None,
        union_sfa=False,
        enable_kernel_prof=True,
        whole_layer_json=None,
        whole_layer_base=str(wl_base),
        k_sfa=1024,
    )
    build_dir, wl_json, ops_dir, l78_dir = run_m2._prepare_build_artifacts(args, rank=0)
    assert calls == [(None, False, True)]
    assert build_dir == default_dir / "prof"
    assert wl_json == default_dir / "prof" / "whole_layer.json"
    assert ops_dir == default_dir / "prof" / "dspine_ops"
    assert l78_dir == default_dir / "prof" / "dspine_l78"
    assert (build_dir / "whole_layer.ready").exists()
    assert not (default_dir / "chain.so").exists()
    assert json.loads(wl_json.read_text(encoding="utf-8"))["tree"] == str(
        default_dir / "prof" / "chain.so"
    )


def test_m21_build_decoupling_python_hygiene() -> None:
    verify_python_source_hygiene(
        [
            REPO_ROOT / "tilert/models/glm_5_2_ascend/ops/build_megakernel.py",
            REPO_ROOT / "tilert/models/glm_5_2_ascend/abi.py",
            REPO_ROOT / "tilert/models/glm_5_2_ascend/dspine/calibrated_l78.py",
            REPO_ROOT / "tilert/models/glm_5_2_ascend/dspine/tp_drafter.py",
            REPO_ROOT / "tests/glm52/run_m2_case01.py",
            Path(__file__),
        ],
        max_line_length=100,
    )


def test_run_m2_case01_emits_valid_m2_report_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.glm52.e2e_tp8_ep8.a3_runner import (
        _accepted_schedule_for_case,
        _build_round_traces,
        build_synthetic_cache_state,
    )
    from tests.glm52.e2e_tp8_ep8.contracts import GOLDEN_CASES
    from tests.glm52.e2e_tp8_ep8.hygiene_gate import allocate_isolation_env
    from tests.glm52.e2e_tp8_ep8.parity_gate import verify_e2e_run

    run_m2 = importlib.import_module("tests.glm52.run_m2_case01")
    spec = GOLDEN_CASES["case-01-coding"]
    case_dir = REPO_ROOT / "tests/glm52/megakernel_tp8_ep8/case-01-coding"
    bench_tokens = json.loads((case_dir / "bench_result.json").read_text())["output_token_ids"]
    acc = _accepted_schedule_for_case("case-01-coding")
    traces, w_hist = _build_round_traces("case-01-coding", acc, width=16, min_w_act=10)
    main_state = build_synthetic_cache_state("case-01-coding", 78, "main")
    draft_state = build_synthetic_cache_state("case-01-coding", 1, "draft")

    run_id = "m22_unit_case01"
    iso = allocate_isolation_env(run_id, port_base=31300)
    for key, val in iso.items():
        monkeypatch.setenv(key, val)
    monkeypatch.setenv("HCCL_DETERMINISTIC", "true")
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")

    report_dir = tmp_path / "m2_report"
    report_dir.mkdir()
    trials = []
    for idx in range(7):
        rdts = [5.5 * 1023.0 - r * 2.0 for r in range(8)]
        trials.append(
            {
                "trial_idx": idx,
                "is_warmup": idx < 2,
                "arm": "candidate",
                "rank_dt_ms": rdts,
                "max_tpot_ms": max(rdts) / 1023.0,
                "rounds": len(acc),
                "accepted_per_round": list(acc),
                "mean_accepted": 1023.0 / len(acc),
                "first_token_id": spec.first_token_id,
                "output_token_ids": list(bench_tokens),
                "output_sha256": spec.output_sha256,
                "eos_drained_cleanly": True,
            }
        )

    ranks_manifest = []
    last_res = None
    for rank in range(8):
        dt = float(trials[-1]["rank_dt_ms"][rank])
        res = {
            "rank": rank,
            "world_size": 8,
            "case_id": "case-01-coding",
            "prompt_sha256": spec.prompt_sha256,
            "first_token_id": spec.first_token_id,
            "output_token_ids": list(bench_tokens),
            "output_sha256": spec.output_sha256,
            "rounds": len(acc),
            "accepted_per_round": list(acc),
            "horizon_levels": [int(t["horizon_level"]) for t in traces],
            "dt_ms": dt,
            "tpot_ms": dt / 1023.0,
            "main_state": main_state,
            "draft_state": draft_state,
            "serial_draft_match": True,
            "ekv_serial_diffs": {},
            "runtime_invariants": {
                "w_act_min": 10,
                "w_act_max": max(w_hist),
                "w_act_history": w_hist,
                "collective_split_max_rows": 16,
                "l78_gemm_chunk_max_rows": 8,
                "in_flight_depth": 2,
                "draft_sfa_topk": 1024,
                "target_sfa_topk": 2048,
            },
            "round_traces": traces,
            "trials": trials,
            "warmups_ms": [t["max_tpot_ms"] for t in trials[:2]],
            "formal_ms": [t["max_tpot_ms"] for t in trials[2:]],
            "median_tpot_ms": trials[-1]["max_tpot_ms"],
            "min_tpot_ms": trials[-1]["max_tpot_ms"],
            "p90_tpot_ms": trials[-1]["max_tpot_ms"],
            "all_ranks_target_exact": True,
            "all_ranks_cache_parity": True,
            "loaded_modules_has_vllm": False,
            "compiled_so_dir": f"/dev/shm/{run_id}/builds",
            "head_warmed": True,
            "head_execution_before_timing": {
                "wrapped": False,
                "active_graph_width": 16,
                "graph_widths": [16],
            },
            "draft_head_distinct_from_lm_head": True,
        }
        if rank == 0:
            last_res = res
        rf = report_dir / f"rank-{rank}.json"
        rf.write_text(
            json.dumps(run_m2.build_rank_report(res, "case-01-coding")) + "\n",
            encoding="utf-8",
        )
        ranks_manifest.append(
            {
                "rank": rank,
                "report_path": rf.name,
                "report_sha256": hashlib.sha256(rf.read_bytes()).hexdigest(),
            }
        )

    assert last_res is not None
    e2e = run_m2.build_e2e_report(
        last_res,
        case_id="case-01-coding",
        milestone="M2",
        width=16,
        warmups=2,
        formals=5,
        ranks_manifest=ranks_manifest,
        run_id=run_id,
    )
    (report_dir / "e2e_result.json").write_text(json.dumps(e2e, indent=2) + "\n", encoding="utf-8")
    verified = verify_e2e_run(
        report_dir,
        case_dir,
        max_tpot_ms=6.0,
        require_warm_head=True,
        milestone="M2",
    )
    assert verified["milestone"] == "M2"
    assert verified["median_tpot_ms"] < 6.0
