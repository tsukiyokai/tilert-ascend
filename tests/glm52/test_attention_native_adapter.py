"""Check engine ownership, exact geometry, and native attention call boundaries."""

import ast
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from tilert.models.glm_5_2_ascend import attention_native as native


@pytest.fixture
def npu(monkeypatch):
    state = SimpleNamespace(stream=1, capturing=False, allocations=[], calls=[])
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        current_stream=lambda device=None: SimpleNamespace(npu_stream=state.stream),
        is_current_stream_capturing=lambda: state.capturing,
    ), raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(
        npu_format_cast=lambda value, format: value,
        get_npu_format=lambda value: 2,
        npu_dynamic_quant=lambda value: (
            value.to(torch.int8), torch.ones(value.shape[0])),
    ))

    def indexed_prepare(codes, pages, table_pages, index_pages,
                        index_table_pages, quantized, blocks):
        value = (torch.zeros(codes.shape[0] * 8192, dtype=torch.uint8),
                 torch.zeros(1, dtype=torch.uint8), {"topk": 0},
                 blocks or 24, 1234)
        state.allocations.append((codes.shape[0], pages, table_pages,
                                  index_pages, index_table_pages, quantized))
        return value

    def shared_prepare(codes, pages, table_pages, blocks):
        state.allocations.append((codes.shape[0], pages, table_pages))
        return (torch.zeros(32, dtype=torch.uint8),
                torch.zeros(1, dtype=torch.uint8), blocks or 8, 1234)

    def indexed_out(*args):
        state.calls.append(("indexed", args))
        args[12].fill_(5)
        rows = args[0].shape[0]
        args[10].view(torch.int32).view(rows, 1, 2048).copy_(
            torch.arange(2048, dtype=torch.int32).expand(rows, 1, -1),
        )
        return args[12]

    def shared_out(*args):
        state.calls.append(("shared", args))
        args[26].fill_(7)
        return args[26]

    for name, function in {
        "indexed_attention_prepare": indexed_prepare,
        "shared_full_attention_prepare": shared_prepare,
        "indexed_attention_out": indexed_out,
        "shared_full_attention_out": shared_out,
    }.items():
        monkeypatch.setattr(torch.ops.tilert, name, function, raising=False)
    return state


def metadata(rows=2, table_pages=3, slots=None):
    value = SimpleNamespace(
        num_actual_tokens=rows, num_input_tokens=rows,
        sin=torch.zeros(rows, 1, 64, dtype=torch.bfloat16),
        cos=torch.ones(rows, 1, 64, dtype=torch.bfloat16),
        slot_mapping=(torch.arange(rows, dtype=torch.int32)
                      if slots is None else slots),
        cum_query_lens=torch.tensor([rows], dtype=torch.int32),
        seq_lens=torch.tensor([128], dtype=torch.int32),
        block_table=torch.arange(table_pages, dtype=torch.int32).view(1, -1),
    )
    value.actual_seq_lengths_query = value.cum_query_lens
    value.actual_seq_lengths_key = value.seq_lens
    return value


@pytest.fixture
def indexer_types(monkeypatch):
    class Backend(SimpleNamespace):
        pass

    class Wrapper:
        def __init__(self, backend):
            self.impl = backend

        @property
        def k_cache(self):
            return self.impl.k_cache

    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.indexer",
                        SimpleNamespace(AscendSFAIndexerBackend=Backend))
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.mla",
                        SimpleNamespace(IndexerWrapper=Wrapper))
    return Backend, Wrapper


@pytest.fixture
def vendor_wrapper(indexer_types, monkeypatch):
    """Execute the original wrapper class without importing NPU dependencies."""
    root = Path(os.environ.get("VLLM_ASCEND_SOURCE", "/tmp/vllm-ascend-clone/va"))
    source = root / "vllm_ascend/ops/mla.py"
    if not source.is_file():
        pytest.skip("set VLLM_ASCEND_SOURCE to vLLM-Ascend 8237ee for wrapper source replay")
    contents = source.read_bytes()
    assert hashlib.sha256(contents).hexdigest() == (
        "e14080c6c0106f2734f5441254e285e7a39fd8980046febe5232b4c9dba75400"
    )
    definition = next(node for node in ast.parse(contents).body
                      if isinstance(node, ast.ClassDef) and node.name == "IndexerWrapper")
    backend_type, _ = indexer_types
    namespace = {"nn": torch.nn, "torch": torch, "AttentionMetadata": object,
                 "AscendSFAIndexerBackend": backend_type}
    exec(compile(ast.Module(body=[definition], type_ignores=[]),
                 str(source), "exec"), namespace)
    wrapper_type = namespace["IndexerWrapper"]
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.mla",
                        SimpleNamespace(IndexerWrapper=wrapper_type))

    def wrap(backend):
        class Upstream(torch.nn.Module):
            def get_ascend_indexer_backend_cls(self):
                return lambda source, rope_dim: backend

        upstream = Upstream()
        for name in ("n_head", "topk_tokens", "q_lora_rank", "wq_b",
                     "wk_weights_proj", "k_norm", "softmax_scale"):
            setattr(upstream, name, getattr(backend, name))
        return wrapper_type(upstream, 64)

    return wrap


@pytest.fixture
def layer_factory(monkeypatch, npu, indexer_types):
    class Scheme:
        act_quant_type = torch.int8

    class Implementation:
        pass

    prolog = object()
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.sfa_v1", SimpleNamespace(
        AscendSFAImpl=Implementation,
        PreprocessType=SimpleNamespace(PROLOG_V3=prolog),
    ))
    monkeypatch.setitem(sys.modules,
        "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic",
        SimpleNamespace(AscendW8A8DynamicLinearMethod=Scheme))

    def reduce(partial):
        npu.calls.append(("reduce", partial.clone()))
        return partial + 100

    monkeypatch.setitem(sys.modules, "vllm.distributed", SimpleNamespace(
        tensor_model_parallel_all_reduce=reduce,
    ))
    monkeypatch.setattr(native, "_pack_weights", lambda impl, mode: (
        [None] * 9, [None] * 6, [None] * 3,
    ))

    def make(mode="int8", rows=2):
        impl = Implementation()
        for name, value in {
            "tp_size": 16, "local_num_heads": 4, "q_lora_rank": 2048,
            "kv_lora_rank": 512, "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64, "v_head_dim": 256,
            "preprocess_type": prolog, "_quant_type": Scheme,
            "enable_sp": False, "is_pcp_decode_sharded": False,
            "uses_packed_sfa_main_cache": False,
            "kv_sharing_target_layer_name": None,
            "rl_weight_update_enabled": False, "g_proj": None,
            "_is_mtp_layer": False, "layerwise_kv_cache_hook": None,
            "layer_name": "model.layers.0.self_attn.attn",
            "skip_topk": mode == "shared", "use_index_cache": True,
            "scale": 1 / 16,
        }.items():
            setattr(impl, name, value)
        impl.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1,
                                            decode_context_parallel_size=1),
            model_config=SimpleNamespace(enable_sleep_mode=False),
            kv_transfer_config=None, lora_config=None,
        )
        impl.o_proj = SimpleNamespace(
            quant_method=SimpleNamespace(quant_method=Scheme()), tp_size=16,
            input_is_parallel=True, bias=None, reduce_results=True,
        )
        impl.q_a_layernorm = SimpleNamespace(variance_epsilon=1e-5)
        impl.kv_a_layernorm = SimpleNamespace(variance_epsilon=1e-5)
        impl.topk_indices_buffer = torch.zeros(32, 2048, dtype=torch.int32)
        impl._get_indexcache_topk_indices = lambda count: (
            impl.topk_indices_buffer[:count].unsqueeze(1)
        )

        def publish(selected):
            npu.calls.append(("publish", selected.clone()))
            impl.topk_indices_buffer[:selected.shape[0]].copy_(selected.squeeze(1))

        impl._update_indexcache_topk_indices = publish
        main_metadata = metadata(rows)
        index_metadata = metadata(rows, 2, torch.arange(rows, dtype=torch.int32) + 17)
        impl._get_indexer_attn_metadata = lambda: index_metadata
        impl.indexer = None
        if mode != "shared":
            dtype = torch.int8 if mode == "int8" else torch.bfloat16
            cache = [torch.zeros(5, 128, 1, 128, dtype=dtype)]
            if mode == "int8":
                cache.append(torch.ones(5, 128, 1, 1, dtype=torch.float16))
            backend_type, wrapper_type = indexer_types
            impl.indexer = wrapper_type(backend_type(
                wq_b=SimpleNamespace(quant_method=SimpleNamespace(quant_method=Scheme()),
                                     bias=None),
                wk_weights_proj=SimpleNamespace(bias=None),
                n_head=32, head_dim=128, q_lora_rank=2048, topk_tokens=2048,
                qk_rope_head_dim=64, is_rope_neox_style=False,
                k_norm=SimpleNamespace(eps=1e-6), enable_sparse_li_c4=False,
                _pcp_active=False, _dsa_cp_active=False,
                _is_pcp_decode_sharded=False, enable_sparse_li_c8=mode == "int8",
                c8_k_cache_dtype=torch.int8, c8_k_scale_cache_dtype=torch.float16,
                k_cache=SimpleNamespace(kv_cache=tuple(cache)),
                softmax_scale=128 ** -.5,
            ))

        def original(*args, **kwargs):
            npu.calls.append(("original", args))
            return "vendor"

        impl.forward = original
        layer = SimpleNamespace(impl=impl, kv_cache=(
            torch.zeros(4, 128, 1, 512, dtype=torch.bfloat16),
            torch.zeros(4, 128, 1, 64, dtype=torch.bfloat16),
        ))
        return layer, main_metadata, index_metadata

    return make


def test_original_vendor_wrapper_exposes_backend_contract(
        layer_factory, vendor_wrapper):
    layer, _, _ = layer_factory()
    backend = layer.impl.indexer.impl
    backend.q_hadamard = torch.ones(128, 128, dtype=torch.bfloat16)
    wrapper = vendor_wrapper(backend)
    layer.impl.indexer = wrapper
    assert isinstance(wrapper, torch.nn.Module)
    assert wrapper.wq_b is backend.wq_b
    assert wrapper.k_cache is backend.k_cache
    assert not hasattr(wrapper, "qk_rope_head_dim")
    assert not hasattr(wrapper, "q_hadamard")
    assert native._indexer_backend(wrapper) is backend
    assert native._validate_impl(layer.impl) == "int8"
    native.install(layer, native.WorkspacePool())()


def test_indexer_contract_rejects_an_unrecognized_backend(layer_factory):
    layer, _, _ = layer_factory()
    layer.impl.indexer.impl = SimpleNamespace()
    with pytest.raises(ValueError, match="IndexerWrapper with AscendSFAIndexerBackend"):
        native.install(layer, native.WorkspacePool())


def test_indexer_packing_preserves_integer_projections():
    query = torch.randint(-8, 8, (2048, 4096), dtype=torch.int8)
    key = torch.randint(-8, 8, (160, 6144)).to(torch.bfloat16)
    q_tiles, k_tiles = native.pack_indexer_weights(query, key)
    q_input = torch.randint(-8, 8, (2, 2048))
    k_input = torch.randint(-8, 8, (2, 6144))
    q_result = torch.cat([q_input @ tile.long() for tile in q_tiles], dim=1)
    k_result = torch.cat([k_input @ tile.long() for tile in k_tiles], dim=1)
    assert torch.equal(q_result, q_input @ query.long())
    assert torch.equal(k_result, k_input @ key.long().T)


@pytest.mark.parametrize("mode", ["int8", "bf16", "shared"])
def test_forward_preserves_metadata_topk_and_single_tp_sum(layer_factory, npu, mode):
    layer, main, index = layer_factory(mode)
    pool = native.WorkspacePool(blocks=24)
    original = layer.impl.forward
    restore = native.install(layer, pool)
    hidden = torch.ones(2, 6144, dtype=torch.bfloat16)
    output = torch.empty_like(hidden)
    result = layer.impl.forward(layer.impl.layer_name, hidden, layer.kv_cache, main, output)
    assert result is output
    expected = 107 if mode == "shared" else 105
    assert torch.equal(output, torch.full_like(output, expected))
    call = npu.calls[0][1]
    if mode == "shared":
        assert call[16].data_ptr() == layer.impl.topk_indices_buffer.data_ptr()
        assert [kind for kind, _ in npu.calls] == ["shared", "reduce"]
    else:
        assert call[7][5] is main.block_table
        assert call[8][5] is index.block_table
        assert torch.equal(call[7][2], main.slot_mapping.long())
        assert torch.equal(call[8][2], index.slot_mapping.long())
        assert call[7][2].data_ptr() != call[8][2].data_ptr()
        assert call[9][3].shape == ((5, 128) if mode == "int8" else (0,))
        assert [kind for kind, _ in npu.calls] == ["indexed", "publish", "reduce"]
        assert torch.equal(layer.impl.topk_indices_buffer[0], torch.arange(2048))
        prepared = next(iter(pool.entries.values()))
        assert prepared[5].data_ptr() != layer.impl.topk_indices_buffer.data_ptr()
        prepared[5].fill_(-1)
        assert layer.impl.topk_indices_buffer[0, 7] == 7
    restore()
    assert layer.impl.forward is original and not pool.layers


@pytest.mark.parametrize("reason", ["prefill", "padding", "multiple_requests",
                                    "profile", "hook", "cache", "skip_topk"])
def test_unsupported_calls_use_original_before_native(layer_factory, npu, reason):
    rows = 33 if reason == "prefill" else 2
    layer, main, _ = layer_factory(rows=rows)
    native.install(layer, native.WorkspacePool())
    hidden = torch.ones(rows, 6144, dtype=torch.bfloat16)
    cache = layer.kv_cache
    if reason == "padding":
        main.num_actual_tokens = 1
    elif reason == "multiple_requests":
        main.seq_lens = torch.tensor([10, 10], dtype=torch.int32)
    elif reason == "profile":
        main = None
        layer.impl._get_indexer_attn_metadata = lambda: pytest.fail("profiling looked up indexer")
    elif reason == "hook":
        layer.impl.layerwise_kv_cache_hook = object()
    elif reason == "cache":
        cache = tuple(value.clone() for value in cache)
    elif reason == "skip_topk":
        layer.impl.skip_topk = True
    assert layer.impl.forward("layer", hidden, cache, main, torch.empty_like(hidden)) == "vendor"
    assert [kind for kind, _ in npu.calls] == ["original"]


def test_mtp_is_left_unmodified(layer_factory):
    layer, _, _ = layer_factory()
    layer.impl._is_mtp_layer = True
    original = layer.impl.forward
    pool = native.WorkspacePool()
    native.install(layer, pool)()
    assert layer.impl.forward is original and not pool.layers


@pytest.mark.parametrize("field,value", [
    ("tp_size", 8), ("local_num_heads", 8), ("enable_sp", True),
    ("uses_packed_sfa_main_cache", True), ("rl_weight_update_enabled", True),
])
def test_static_contract_rejects_other_execution_modes(layer_factory, field, value):
    layer, _, _ = layer_factory()
    setattr(layer.impl, field, value)
    with pytest.raises(ValueError, match="fixed TP16"):
        native.install(layer, native.WorkspacePool())


def test_workspace_uses_exact_rows_geometry_and_capture_stream(npu):
    pool = native.WorkspacePool()
    first = torch.zeros(2, 6144, dtype=torch.bfloat16)
    second = torch.zeros(3, 6144, dtype=torch.bfloat16)
    geometry = native.Geometry("int8", 4, 3, 5, 2)
    slots = torch.tensor([0, 1], dtype=torch.int32)
    pool.reserve(first, geometry, [slots])
    reserved = pool.capture_reserves[(first.device, 2, geometry)]
    slot_reserved = pool.slot_reserves[pool._slot_signature(slots)]
    warmup = pool.get(first, geometry)
    warmup_slots = pool.live_slots(slots)
    assert warmup is not reserved
    assert warmup_slots.data_ptr() != slot_reserved.data_ptr()
    assert pool.capture_reserves and pool.slot_reserves
    npu.stream, npu.capturing = 2, True
    assert pool.get(first, geometry) is reserved
    assert pool.live_slots(slots) is slot_reserved
    assert not pool.capture_reserves and not pool.slot_reserves
    with pytest.raises(RuntimeError, match="geometry"):
        pool.get(second, geometry)
    with pytest.raises(RuntimeError, match="geometry"):
        pool.get(first, native.Geometry("int8", 4, 3, 5, 3))


def test_captured_slot_sources_are_independent_and_replay_live(npu, monkeypatch):
    pool = native.WorkspacePool()
    hidden = torch.zeros(2, 6144, dtype=torch.bfloat16)
    geometry = native.Geometry("shared", 4, 3)
    first = torch.tensor([3, 4], dtype=torch.int32)
    second = torch.tensor([37, 38], dtype=torch.int32)
    pool.reserve(hidden, geometry, [first, second])
    warmup = pool.live_slots(first)
    first.add_(2)
    assert torch.equal(pool.live_slots(first), first.long())
    operations = []
    copy = torch.Tensor.copy_

    def record(destination, source, *args, **kwargs):
        if npu.capturing:
            operations.append((destination, source))
        return copy(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", record)
    npu.capturing = True
    value_a = pool.live_slots(first)
    value_b = pool.live_slots(second)
    assert pool.live_slots(first) is value_a
    assert value_a is warmup
    assert value_a.data_ptr() != value_b.data_ptr()
    assert len(operations) == 2
    for replay in range(3):
        first.add_(11)
        second.add_(17)
        for destination, source in operations:
            copy(destination, source)
        assert torch.equal(value_a, first.long())
        assert torch.equal(value_b, second.long())
    npu.capturing = False
    pool.reserve(hidden, geometry, [first, second])
    npu.capturing = True
    pool.live_slots(first)
    pool.live_slots(second)
    assert len(operations) == 4


def test_capture_reservation_uses_own_indexer_metadata(layer_factory, monkeypatch, npu):
    layer, main, index = layer_factory()
    pool = native.WorkspacePool()
    native.install(layer, pool)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", SimpleNamespace(
        get_forward_context=lambda: SimpleNamespace(
            attn_metadata={layer.impl.layer_name: main}),
        is_forward_context_available=lambda: True,
    ))
    pool.reserve_capture(torch.zeros(2, 6144, dtype=torch.bfloat16))
    assert npu.allocations == [(2, 4, 3, 5, 2, True)]
    assert len(pool.slot_reserves) == 2
    assert pool._slot_signature(main.slot_mapping) in pool.slot_reserves
    assert pool._slot_signature(index.slot_mapping) in pool.slot_reserves


@pytest.mark.parametrize("mode", ["int8", "bf16", "shared"])
def test_body_normalizes_only_rope_and_retains_live_metadata(
        layer_factory, npu, monkeypatch, mode):
    layer, main, index = layer_factory(mode)
    caches = native._cache_tensors(layer, mode)
    if mode == "bf16":
        caches.append(torch.empty(0, dtype=torch.float16))
    cache_owners = [value.data_ptr() for value in caches]
    main_list = native._metadata_values(main, 2)
    index_list = (None if mode == "shared" else
                  native._metadata_values(index, 2, indexed=True))
    converted = []

    def normalize(value):
        converted.append(value)
        return value.clone()

    monkeypatch.setattr(native, "_nd", normalize)
    pool = native.WorkspacePool()
    native.run_body(torch.ones(2, 6144, dtype=torch.bfloat16),
                    [None] * 9, [None] * 6, [None] * 3, caches,
                    main_list, index_list, pool, mode=mode,
                    topk=layer.impl.topk_indices_buffer[:2].unsqueeze(1),
                    scale=1 / 16, epsilon_cq=1e-5, epsilon_ckv=1e-5)
    sources = [main_list] if mode == "shared" else [main_list, index_list]
    expected = [value for metadata_list in sources for value in metadata_list[:2]]
    assert len(converted) == len(expected)
    assert all(actual is original for actual, original in zip(converted, expected))
    call = npu.calls[0][1]
    passed = ([list(call[11:14]) + list(call[17:20])] if mode == "shared"
              else [call[7], call[8]])
    for actual, original in zip(passed, sources):
        for position in (0, 1):
            assert actual[position].data_ptr() != original[position].data_ptr()
            assert torch.equal(actual[position].view(torch.uint8),
                               original[position].view(torch.uint8))
        assert all(actual[position] is original[position] for position in (3, 4, 5))
        assert pool._slot_signature(original[2]) in {
            key[1:] for key in pool.slot_entries}
    assert main_list[2] is main.slot_mapping
    if index_list is not None:
        assert index_list[2] is index.slot_mapping
    assert [value.data_ptr() for value in caches] == cache_owners


def test_runner_four_dimensional_scale_cache_is_borrowed(layer_factory):
    layer, _, _ = layer_factory()
    owner = layer.impl.indexer.k_cache
    key, scale = owner.kv_cache
    scale = scale.view(key.shape[0], 128, 1, 1)
    owner.kv_cache = (key, scale)
    borrowed = native._cache_tensors(layer, "int8")
    assert borrowed[2] is key
    assert borrowed[3].shape == (key.shape[0], 128)
    assert borrowed[3].data_ptr() == scale.data_ptr()
    borrowed[3][1, 2] = 3.5
    assert scale[1, 2, 0, 0] == 3.5
    assert owner.kv_cache[1] is scale
    assert scale.shape == (key.shape[0], 128, 1, 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_wq_preserves_original_scale_descriptor(
        npu, monkeypatch, indexer_types, vendor_wrapper, dtype):
    import torch_npu

    scale = torch.linspace(.002, .01, 4096).to(dtype)
    impl = SimpleNamespace(
        weight_dq=torch.empty(6144, 2048, dtype=torch.int8),
        weight_uq_qr=torch.empty(2048, 1024, dtype=torch.int8),
        weight_dkv_kr=torch.empty(6144, 576, dtype=torch.int8),
        W_UK_T=torch.empty(4, 192, 512, dtype=torch.bfloat16),
        W_UV=torch.empty(4, 512, 256, dtype=torch.bfloat16),
        dequant_scale_w_dq=torch.ones(1, 2048),
        dequant_scale_w_uq_qr=torch.ones(1, 1024),
        dequant_scale_w_dkv_kr=torch.ones(1, 576),
        q_a_layernorm=SimpleNamespace(weight=torch.ones(2048, dtype=torch.bfloat16)),
        kv_a_layernorm=SimpleNamespace(weight=torch.ones(512, dtype=torch.bfloat16)),
        o_proj=SimpleNamespace(
            weight=torch.empty(1024, 6144, dtype=torch.int8),
            weight_scale=torch.ones(6144, dtype=torch.bfloat16)),
        indexer=indexer_types[0](
            wq_b=SimpleNamespace(
                weight=torch.empty(2048, 4096, dtype=torch.int8), weight_scale=scale),
            wk_weights_proj=SimpleNamespace(
                weight=torch.empty(160, 6144, dtype=torch.bfloat16)),
            k_norm=SimpleNamespace(weight=torch.ones(128), bias=torch.zeros(128)),
            q_hadamard=torch.ones(128, 128, dtype=torch.bfloat16),
            n_head=32, topk_tokens=2048, q_lora_rank=2048,
            softmax_scale=128 ** -.5),
    )
    impl.indexer = vendor_wrapper(impl.indexer)
    nz_addresses = {value.data_ptr() for value in
                    (impl.weight_dq, impl.weight_uq_qr, impl.weight_dkv_kr)}
    monkeypatch.setattr(torch_npu, "get_npu_format", lambda value: (
        29 if value.data_ptr() in nz_addresses else 2))
    _, front, tail = native._pack_weights(impl, "int8")
    assert front[1].dtype == dtype
    assert torch.equal(front[1].view(torch.uint8), scale.view(torch.uint8))
    assert tail[2].dtype == torch.float32
