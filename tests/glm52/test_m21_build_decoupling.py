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

import tilert.models.glm_5_2_ascend.ops.build_megakernel as bm
import tilert.models.glm_5_2_ascend.whole_layer as wl_mod
from tilert.models.glm_5_2_ascend.whole_layer import Libraries, WholeLayerManager
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
        fake_torch.distributed = SimpleNamespace(barrier=lambda **_kw: None)
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
    monkeypatch.setattr(tp_mod.ctypes, "CDLL", _FakeLib)
    monkeypatch.setattr(
        tp_mod.torch,
        "distributed",
        SimpleNamespace(barrier=lambda **_kw: None),
        raising=False,
    )
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

    lib_hashes_backup = None
    import tilert.models.glm_5_2_ascend.whole_layer as wl_mod

    lib_hashes_backup = wl_mod.LIBRARY_HASHES
    monkeypatch.setattr(
        wl_mod,
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
    import tilert.models.glm_5_2_ascend.whole_layer as wl_mod

    monkeypatch.setattr(
        wl_mod,
        "LIBRARY_HASHES",
        {
            **wl_mod.LIBRARY_HASHES,
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
        wl_mod,
        "LIBRARY_HASHES",
        {
            **wl_mod.LIBRARY_HASHES,
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


def test_whole_layer_manager_capture_prof_acc_stream_disambiguation(
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

    mgr = WholeLayerManager.__new__(WholeLayerManager)
    mgr.device = "npu:0"
    mgr.target_width = 32
    mgr.capture_stream = SimpleNamespace(npu_stream=2002)
    mgr.capture_history = [{"rows": 16, "layers": ("l0", "l77")}]
    mgr.layer_order = ("l0", "l77")
    mgr.tree_manifest = {"enable_kernel_prof": True, "flags": ["-DENABLE_KERNEL_PROF=1"]}
    caller_arena = SimpleNamespace(body=make_body6_bytes([0] * 24))
    capture_arena = SimpleNamespace(body=make_body6_bytes(list(range(100, 124))))
    w32_arena = SimpleNamespace(body=make_body6_bytes(list(range(200, 224))))
    mgr.layers = {
        "l0": SimpleNamespace(name="l0", entries={}),
        "l77": SimpleNamespace(
            name="l77",
            entries={
                (1001, 16, "g0"): caller_arena,
                (2002, 16, "g0"): capture_arena,
                (2002, 32, "g0"): w32_arena,
            },
        ),
    }
    assert mgr.kernel_prof_enabled is True
    got = mgr.capture_prof_acc(16)
    assert got.values == list(range(100, 124))
    assert got.detached and got.cpu_called and got.cloned
    assert sync_calls == ["npu:0"]

    got_default = mgr.capture_prof_acc(sync=False, cpu=False)
    assert got_default.values == list(range(100, 124))
    assert not got_default.cpu_called
    assert sync_calls == ["npu:0"]

    mgr.tree_manifest = {"enable_kernel_prof": False, "flags": []}
    assert mgr.kernel_prof_enabled is False

    mgr.layers["l77"].entries[(2002, 16, "g0")] = SimpleNamespace(
        body=[None] * 6 + [_FakeTensor([0] * 128, dtype="u8", elem_size=1)]
    )
    with pytest.raises(RuntimeError, match="smaller than 512 bytes"):
        mgr.capture_prof_acc(16)

    with pytest.raises(RuntimeError, match="No capture-stream arena"):
        mgr.capture_prof_acc(8)

    mgr.layers["l77"].entries[(2002, 16, "g0")] = SimpleNamespace(body=[None] * 6)
    with pytest.raises(RuntimeError, match="missing body\\[6\\]"):
        mgr.capture_prof_acc(16)

    mgr.layers["l77"].entries[(2002, 16, "g0")] = SimpleNamespace(body=[None] * 7)
    with pytest.raises(RuntimeError, match="missing body\\[6\\]"):
        mgr.capture_prof_acc(16)

    mgr.layers["l77"].entries[(2002, 16, "g0")] = SimpleNamespace(
        body=[None] * 6 + [_FakeTensor([0] * 513, dtype="u8", elem_size=1)]
    )
    with pytest.raises(RuntimeError, match="8-byte aligned"):
        mgr.capture_prof_acc(16)

    mgr.layers["l77"].entries[(2002, 16, "g0")] = capture_arena
    mgr.layers["l77"].entries[(2002, 16, "g1")] = w32_arena
    with pytest.raises(RuntimeError, match="Ambiguous capture-stream arenas"):
        mgr.capture_prof_acc(16)


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
    for mod_name in (
        "torch",
        "vllm",
        "vllm.distributed",
        "cache_digest",
        "tilert.models.glm_5_2_ascend.cann_model",
        "tilert.models.glm_5_2_ascend.cann_mtp",
        "tilert.models.glm_5_2_ascend.dspine",
        "tilert.models.glm_5_2_ascend.dspine.tp_drafter",
        "tilert.models.glm_5_2_ascend.tp_runtime",
        "tilert.pd_vllm.ascend_cache",
    ):
        mod = ModuleType(mod_name)
        if mod_name == "vllm.distributed":
            mod.get_world_group = lambda: None
        elif mod_name == "cache_digest":
            mod.cache_digest = lambda *_a, **_k: {}
        elif mod_name == "tilert.models.glm_5_2_ascend.cann_model":
            mod.CannModel = object
        elif mod_name == "tilert.models.glm_5_2_ascend.cann_mtp":
            mod.CannMtp = object
            mod.MAIN_CACHE = "main"
        elif mod_name == "tilert.models.glm_5_2_ascend.dspine":
            for attr in (
                "CalibratedL78V3Runner",
                "TpCalibratedL78Weights",
                "ZeroLagHorizonController",
                "build_draft_l78_k110419",
                "load_dspine_ops",
            ):
                setattr(mod, attr, object)
        elif mod_name == "tilert.models.glm_5_2_ascend.dspine.tp_drafter":
            mod.capture_shared_target_for_width = lambda *_a, **_k: None
        elif mod_name == "tilert.models.glm_5_2_ascend.tp_runtime":
            mod.TorchControlChannel = object
        elif mod_name == "tilert.pd_vllm.ascend_cache":
            mod.FullRankPrefix = object
        monkeypatch.setitem(sys.modules, mod_name, mod)

    run_m1 = importlib.import_module("tests.glm52.run_m1_case01")
    run_m1 = importlib.reload(run_m1)
    default_dir = tmp_path / "default_builds"
    monkeypatch.setenv("TILERT_BUILD_DIR", str(default_dir))
    calls: list[tuple[object, bool, bool]] = []

    def fake_build_chain(*, out_dir=None, union_sfa=False, enable_kernel_prof=False):
        calls.append((out_dir, union_sfa, enable_kernel_prof))
        target_dir = default_dir / "prof" if (out_dir is None and enable_kernel_prof) else (
            default_dir if out_dir is None else Path(out_dir)
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        so_p = target_dir / "chain.so"
        man_p = target_dir / "chain.manifest.json"
        so_p.write_bytes(b"prof-elf")
        man_p.write_text("{}", encoding="utf-8")
        return so_p, man_p

    monkeypatch.setattr(run_m1, "build_chain", fake_build_chain)
    wl_base = tmp_path / "base.json"
    wl_base.write_text(json.dumps({"tree": "orig"}) + "\n", encoding="utf-8")
    args = SimpleNamespace(
        build_dir=None,
        union_sfa=False,
        enable_kernel_prof=True,
        whole_layer_base=str(wl_base),
    )
    wl_json, ops_dir, l78_dir, ready = run_m1._prepare_whole_layer_json(args, rank=0)
    assert calls == [(None, False, True)]
    assert wl_json == default_dir / "prof" / "whole_layer.json"
    assert ops_dir == default_dir / "prof" / "dspine_ops"
    assert l78_dir == default_dir / "prof" / "dspine_l78"
    assert ready.exists()
    assert not (default_dir / "chain.so").exists()
    assert json.loads(wl_json.read_text(encoding="utf-8"))["tree"] == str(
        default_dir / "prof" / "chain.so"
    )


def test_m21_build_decoupling_python_hygiene() -> None:
    verify_python_source_hygiene(
        [
            REPO_ROOT / "tilert/models/glm_5_2_ascend/ops/build_megakernel.py",
            REPO_ROOT / "tilert/models/glm_5_2_ascend/whole_layer.py",
            REPO_ROOT / "tilert/models/glm_5_2_ascend/dspine/calibrated_l78.py",
            REPO_ROOT / "tilert/models/glm_5_2_ascend/dspine/tp_drafter.py",
            Path(__file__),
        ],
        max_line_length=100,
    )
