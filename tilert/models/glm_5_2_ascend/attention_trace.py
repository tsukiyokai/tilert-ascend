import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import MethodType


WIDTH = 32
HIDDEN = 6144


def install(engine):
    path = os.environ.get("TILERT_ATTENTION_TRACE")
    if engine.rank != 0 or not path:
        return None
    return AttentionTrace(engine, Path(path), int(os.environ.get(
        "TILERT_ATTENTION_TRACE_POSITION", "8631")))


class AttentionTrace:
    def __init__(self, engine, path, position):
        import torch
        from vllm.distributed import get_tp_group

        self.engine = engine
        self.path = path
        self.position = position
        self.device = engine.device
        self.request = -1
        self.capture_request = int(os.environ.get("TILERT_ATTENTION_TRACE_REQUEST", "0"))
        self.native = engine._native_attention_enabled
        self.heads = 64 // engine.tp_size
        self.source_commit = os.environ.get("SOURCE_COMMIT")
        self.saved = False
        self.frames = {}
        self.originals = []
        self.functions = []
        self.pending = {}
        self.active = False
        self.phase_seen = set()
        self.detail_layer = int(os.environ.get("TILERT_ATTENTION_TRACE_LAYER", "0"))
        self.detail_name = next(name for name in engine._attention_names
                                if f".layers.{self.detail_layer}." in name)
        self.main_cache = self.main_metadata = self.index_metadata = None
        self.pool = engine._native_attention_workspace_pool
        self.native_offsets = None
        self.reduce = getattr(get_tp_group(), "_tilert_onesided_reduce", None)
        self.reduce_op = None
        self.fuse_attention = os.environ.get("TRT_FUSE_ATTN_NORM", "1") == "1"
        context = engine.config.compilation_config.static_forward_context
        for name in engine._attention_names:
            frame = {key: torch.empty(WIDTH, HIDDEN, dtype=torch.bfloat16,
                                      device=self.device)
                     for key in ("input", "output")}
            self.frames[name] = frame
            impl = context[name].impl
            if name == self.detail_name:
                self.detail_impl = impl
            self.originals.append((impl, impl.forward))
            impl.forward = MethodType(
                self._forward(impl.forward, frame, name == self.detail_name), impl)
        self._prepare_phases()
        if not self.native and self.detail_impl.indexer is not None:
            self._front_hooks()
        if (self.reduce is not None and self.fuse_attention
                and hasattr(torch.ops.tilert, "onesided_all_reduce_rmsnorm_quant")):
            self.reduce_op = torch.ops.tilert.onesided_all_reduce_rmsnorm_quant

            def reduce(partial, output, *args, **kwargs):
                result = self.reduce_op(partial, output, *args, **kwargs)
                destination = self.pending.pop(output.data_ptr(), None)
                if destination is not None:
                    destination.copy_(output)
                return result

            reduce.default = reduce
            torch.ops.tilert.onesided_all_reduce_rmsnorm_quant = reduce

    def _forward(self, original, frame, detail):
        def forward(impl, layer_name, hidden_states, kv_cache, attn_metadata,
                    output=None):
            if hidden_states.shape != (WIDTH, HIDDEN) or output is None:
                return original(layer_name, hidden_states, kv_cache,
                                attn_metadata, output)
            is_target = (self.engine.position == self.position
                         and self.request == self.capture_request)
            if detail and is_target:
                self.main_cache = kv_cache
                self.main_metadata = attn_metadata
                if impl.indexer is not None:
                    self.index_metadata = impl._get_indexer_attn_metadata()
            if is_target:
                frame["input"].copy_(hidden_states)
            self.active = detail and is_target
            result = original(layer_name, hidden_states, kv_cache,
                              attn_metadata, output)
            self.active = False
            if detail and self.native and is_target:
                import torch
                torch.npu.synchronize(self.device)
                self._native_phases(hidden_states)
            if is_target:
                frame["output"].copy_(output)
            if (self.reduce_op is not None and getattr(
                    self.reduce.sum, "__name__", "") == "_wrapped_reduce_sum"):
                self.pending[output.data_ptr()] = frame["output"]
            return result

        return forward

    def _prepare_phases(self):
        import torch

        shapes = {
            "query_norm": (torch.int8, (WIDTH, 2048)),
            "query_norm_scale": (torch.float32, (WIDTH,)),
            "query": (torch.int8, (WIDTH, 32, 128)),
            "query_scale": (torch.float16, (WIDTH, 32)),
            "weights": (torch.float16, (WIDTH, 32)),
            "topk": (torch.int32, (WIDTH, 1, 2048)),
            "latent": (torch.bfloat16, (WIDTH, self.heads, 512)),
            "local_partial": (torch.bfloat16, (WIDTH, HIDDEN)),
            "key": (torch.int8, (WIDTH, 128)),
            "key_scale": (torch.float16, (WIDTH,)),
            "front_key_hadamard": (torch.float32, (WIDTH, 128)),
        }
        if self.native:
            shapes.update(
                front_query_acc=(torch.int32, (16, WIDTH, 256)),
                front_key_acc=(torch.float32, (10, WIDTH, 16)),
                front_query_projection=(torch.bfloat16, (WIDTH, 32, 128)),
                front_key_projection=(torch.bfloat16, (WIDTH, 160)),
                front_key_normalized=(torch.bfloat16, (WIDTH, 128)),
                front_query_rope=(torch.bfloat16, (WIDTH, 32, 128)),
                front_key_rope=(torch.bfloat16, (WIDTH, 128)),
                front_query_hadamard=(torch.float32, (WIDTH, 32, 128)))
        else:
            shapes.update(
                query_projection=(torch.bfloat16, (WIDTH, 32, 128)),
                key_projection=(torch.bfloat16, (WIDTH, 160)),
                key_normalized=(torch.bfloat16, (WIDTH, 128)),
                output_codes=(torch.int8, (WIDTH, self.heads * 256)),
                output_scale=(torch.float32, (WIDTH,)))
        self.phases = {name: torch.empty(shape, dtype=dtype, device=self.device)
                       for name, (dtype, shape) in shapes.items()}

    def _record(self, name, value):
        destination = self.phases[name]
        assert value.dtype == destination.dtype
        destination.copy_(value.view(destination.shape))
        self.phase_seen.add(name)

    def _front_hooks(self):
        import torch_npu

        indexer = self.detail_impl.indexer.impl
        query_weight = indexer.wq_b.weight.data_ptr()
        output_weight = self.detail_impl.o_proj.weight.data_ptr()
        original_mm = torch_npu.npu_quant_matmul
        original_qli = torch_npu.npu_quant_lightning_indexer
        original_sfa = torch_npu.npu_sparse_flash_attention

        def matmul(*args, **kwargs):
            result = original_mm(*args, **kwargs)
            if self.active and args[1].data_ptr() == query_weight:
                self._record("query_norm", args[0])
                self._record("query_norm_scale", kwargs["pertoken_scale"])
                self._record("query_projection", result)
            elif self.active and args[1].data_ptr() == output_weight:
                self._record("output_codes", args[0])
                self._record("output_scale", kwargs["pertoken_scale"])
                self._record("local_partial", result)
            return result

        def qli(*args, **kwargs):
            if self.active:
                self._record("query", kwargs["query"])
                self._record("query_scale", kwargs["query_dequant_scale"])
                self._record("weights", kwargs["weights"])
            result = original_qli(*args, **kwargs)
            if self.active:
                self._record("topk", result)
            return result

        def sfa(*args, **kwargs):
            result = original_sfa(*args, **kwargs)
            if self.active:
                self._record("latent", result[0] if isinstance(result, tuple) else result)
            return result

        for name, wrapped in (("npu_quant_matmul", matmul),
                              ("npu_quant_lightning_indexer", qli),
                              ("npu_sparse_flash_attention", sfa)):
            wrapped.default = wrapped
            self.functions.append((torch_npu, name, getattr(torch_npu, name)))
            setattr(torch_npu, name, wrapped)
        for layer, name in ((indexer.wk_weights_proj, "key_projection"),
                            (indexer.k_norm, "key_normalized")):
            original = layer.forward
            self.originals.append((layer, original))
            layer.forward = MethodType(self._module_output(original, name), layer)
        if hasattr(indexer, "_quantize_li_tensor"):
            import torch
            original_quant = indexer._quantize_li_tensor

            def quantize_wrapper(x):
                if self.active:
                    hadamard = getattr(indexer, "q_hadamard", None)
                    if hadamard is not None:
                        kh = x @ hadamard
                        self._record("front_key_hadamard", kh.float())
                result = original_quant(x)
                if self.active:
                    self._record("key", result[0].to(torch.int8))
                    self._record("key_scale", result[1].half())
                return result

            self.originals.append((indexer, "_quantize_li_tensor", original_quant))
            indexer._quantize_li_tensor = quantize_wrapper

    def _module_output(self, original, name):
        def forward(layer, *args, **kwargs):
            result = original(*args, **kwargs)
            if self.active:
                self._record(name, result[0] if isinstance(result, tuple) else result)
            return result

        return forward

    def _native_phases(self, hidden):
        import torch
        from .attention_native import Geometry

        if torch.npu.is_current_stream_capturing():
            return
        main, index = self.main_metadata, self.index_metadata
        if main is None or self.pool is None:
            return
        mode = "int8"
        if self.detail_name in self.pool.layers:
            mode = self.pool.layers[self.detail_name][1]
        index_cache = (self.detail_impl.indexer.k_cache.kv_cache
                       if self.detail_impl.indexer is not None else None)
        if mode == "shared":
            geometry = Geometry(mode, self.main_cache[0].shape[0], main.block_table.shape[1])
        else:
            geometry = Geometry(mode, self.main_cache[0].shape[0],
                                main.block_table.shape[1],
                                index_cache[0].shape[0] if index_cache else 0,
                                index.block_table.shape[1] if index else 0)
        stream = torch.npu.current_stream(self.device).npu_stream
        key = (hidden.device, stream, WIDTH, geometry)
        prepared = self.pool.entries.get(key)
        if prepared is None:
            for (d, s, w, g), entry in self.pool.entries.items():
                if d == hidden.device and w == WIDTH and g == geometry:
                    prepared = entry
                    break
        if prepared is None:
            return
        scratch = prepared[0]
        offsets = self.pool.layouts.get((WIDTH, geometry))
        if offsets is None:
            for (w, g), entry in self.pool.layouts.items():
                if w == WIDTH and g == geometry:
                    offsets = entry
                    break
        if offsets is None:
            return
        self.native_offsets = dict(offsets)
        regions = dict(offsets)
        regions.update(front_query_acc=offsets["front_begin"],
                       front_key_acc=offsets["front_begin"] + WIDTH * 4096 * 4)
        for name, destination in self.phases.items():
            if name == "local_partial":
                self._record(name, prepared[4])
                continue
            if name not in regions:
                continue
            source = scratch.narrow(0, regions[name],
                                    destination.numel() * destination.element_size())
            self._record(name, source.view(destination.dtype))

    def _cpu(self, value):
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if is_dataclass(value):
            value = {field.name: getattr(value, field.name)
                     for field in fields(value) if not field.name.startswith("_")}
        if isinstance(value, dict):
            return {key: self._cpu(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [self._cpu(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return type(value).__qualname__

    def _detail(self):
        impl = self.detail_impl
        main = {name: getattr(impl, name) for name in (
            "weight_dq", "weight_uq_qr", "W_UK_T", "weight_dkv_kr",
            "dequant_scale_w_dq", "dequant_scale_w_uq_qr",
            "dequant_scale_w_dkv_kr")}
        main.update(q_gamma=impl.q_a_layernorm.weight,
                    kv_gamma=impl.kv_a_layernorm.weight,
                    q_epsilon=impl.q_a_layernorm.variance_epsilon,
                    kv_epsilon=impl.kv_a_layernorm.variance_epsilon)
        front = index_cache = None
        if impl.indexer is not None:
            indexer = impl.indexer.impl
            front = {"query_weight": indexer.wq_b.weight,
                     "query_scale": indexer.wq_b.weight_scale,
                     "key_weight": indexer.wk_weights_proj.weight,
                     "gamma": indexer.k_norm.weight,
                     "beta": indexer.k_norm.bias,
                     "hadamard": indexer.q_hadamard,
                     "epsilon": indexer.k_norm.eps}
            index_cache = indexer.k_cache.kv_cache
        return self._cpu({
            "layer": self.detail_name, "scale": impl.scale,
            "main_weights": main, "front_weights": front,
            "tail_weights": {"value_weight": impl.W_UV,
                             "output_weight": impl.o_proj.weight,
                             "output_scale": impl.o_proj.weight_scale},
            "main_cache": self.main_cache, "index_cache": index_cache,
            "main_metadata": self.main_metadata,
            "index_metadata": self.index_metadata,
            "native_weights": (self.pool.weights[self.detail_name]
                               if self.native else None),
            "phases": {name: value for name, value in self.phases.items()
                       if name in self.phase_seen},
            "phase_missing": sorted(self.phases.keys() - self.phase_seen),
            "native_offsets": self.native_offsets})

    def dump(self, position, width, active_rows, *args, **kwargs):
        if (
            self.saved
            or width != WIDTH
            or position != self.position
            or self.request != self.capture_request
        ):
            return
        import torch

        assert not self.pending
        torch.npu.synchronize(self.device)
        frames = {name: {key: value.cpu() for key, value in frame.items()}
                  for name, frame in self.frames.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"position": position, "width": width,
                    "active_rows": active_rows, "rank": 0,
                    "detail_layer": self.detail_layer,
                    "source_commit": self.source_commit,
                    "native_attention": self.native,
                    "output_boundary": "post_tp", "frames": frames,
                    "detail": self._detail()}, self.path)
        self.saved = True

    def close(self):
        import torch

        for item in reversed(self.originals):
            if len(item) == 2:
                impl, original = item
                impl.forward = original
            elif len(item) == 3:
                holder, attr, original = item
                setattr(holder, attr, original)
        self.originals.clear()
        for holder, name, original in reversed(self.functions):
            setattr(holder, name, original)
        self.functions.clear()
        if self.reduce_op is not None:
            torch.ops.tilert.onesided_all_reduce_rmsnorm_quant = self.reduce_op
            self.reduce_op = None
        self.pending.clear()
