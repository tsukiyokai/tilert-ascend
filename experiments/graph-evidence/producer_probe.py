"""Probe persistent observations at the real CannModel.verify boundary.

The model and device contexts in this experiment are explicitly CPU fixtures.
Dynamo/AOT execute real Torch operations; ACL capture, NPU kernels, SuperKernel
and independent HF precision are not exercised.
"""

import argparse
import ast
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch._dynamo.backends.common import aot_autograd


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests/glm52')]

from tilert.models.glm_5_2_ascend.cann_model import CannModel  # noqa: E402
from tilert.models.glm_5_2_ascend.cann_trace import CannTrace  # noqa: E402
from tilert.pd_vllm.ascend_layout import (  # noqa: E402
    FULL_INDEX_LAYERS, INT8_INDEX_LAYERS,
)
from run_cann_model import compare_step, compare_tensor  # noqa: E402


WIDTH, LAYERS, HIDDEN, VOCAB, PICKS = 32, 78, 4, 17, 2048
CACHE_FIELDS = {'latent': 'latent', 'rope': 'rope', 'index': 'index_keys',
                'index_scale': 'index_scales'}
ATTENTION_ARGUMENTS = (
    'ql_nope', 'q_pe', 'kv_cache', 'topk_indices', 'attn_metadata',
    'actual_seq_lengths_query', 'actual_seq_lengths_key',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_receiver():
    path = Path(__file__).with_name('receiver.py')
    spec = importlib.util.spec_from_file_location('producer_receiver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PersistentTrace:
    """Write typed slots and epoch stamps at existing CANN hook boundaries.

    Install before compilation/capture. Every participating tensor retains
    its address; begin/readback run outside compilation. The caller owns
    stream ordering and must read back before another replay overwrites slots.
    Keep the buffers alive until every graph referencing them is destroyed.
    A complete stamp inventory is necessary, not proof of NPU execution.
    """

    def __init__(self, engine, *, index_dtype, route_dtype):
        backbone = engine.model.model
        require(index_dtype in (torch.int32, torch.int64)
                and route_dtype in (torch.int32, torch.int64),
                'Choices must retain an integer dtype')
        require(backbone.start_layer == 0 and backbone.end_layer == LAYERS
                and len(backbone.layers) == LAYERS,
                'All 78 main layers must be local')
        require(not any(layer.use_sequence_parallel_moe
                        for layer in backbone.layers),
                'Observation requires unsharded rows')
        request_buffers = {'input_ids': engine.input_ids,
                           'positions': engine.positions}
        require(all(isinstance(value, torch.Tensor)
                    and value.dtype in (torch.int32, torch.int64)
                    for value in request_buffers.values()),
                'Request buffers must retain INT32 or INT64 dtype')
        self.engine = engine
        self.epoch = torch.zeros((), dtype=torch.int64, device=engine.device)
        shapes = {tag: ((WIDTH,), value.dtype)
                  for tag, value in request_buffers.items()}
        shapes.update({f'hidden_states.{layer}': (
            (WIDTH, backbone.hidden_size), torch.bfloat16,
        ) for layer in range(LAYERS + 1)})
        shapes.update({f'indices.{layer}': ((WIDTH, PICKS), index_dtype)
                       for layer in range(LAYERS)})
        shapes.update({f'routes.{layer}': ((WIDTH, 8), route_dtype)
                       for layer in range(3, LAYERS)})
        self.values = {tag: torch.empty(shape, dtype=dtype,
                                        device=engine.device)
                       for tag, (shape, dtype) in shapes.items()}
        self.tags = tuple(shapes)
        self.indices = {tag: index for index, tag in enumerate(self.tags)}
        self.seen = torch.full((len(shapes),), -1, dtype=torch.int64,
                               device=engine.device)
        self._addresses = self.addresses()
        self._handles = []
        self._overrides = []
        self.current_epoch = None

    def addresses(self):
        return {tag: (value.data_ptr(), tuple(value.shape),
                      tuple(value.stride()), value.dtype, value.device)
                for tag, value in {**self.values, 'epoch': self.epoch,
                                   'seen': self.seen}.items()}

    def begin(self, epoch):
        require(type(epoch) is int and 0 <= epoch < 2**63,
                'Nonnegative INT64 epoch required')
        require(self.addresses() == self._addresses,
                'Persistent observation storage changed')
        self.current_epoch = epoch
        self.epoch.fill_(epoch)
        self.seen.fill_(-1)

    def write(self, tag, value):
        destination = self.values[tag]
        if (value.shape != destination.shape
                or value.dtype != destination.dtype
                or value.device != destination.device):
            raise ValueError('Observation tensor contract differs: ' + tag)
        destination.copy_(value.detach())
        self.seen[self.indices[tag]].copy_(self.epoch)

    def _embedding_input(self, module, inputs):
        self.write('input_ids', inputs[0])
        self.write('positions', self.engine.positions)

    def _embedding_output(self, module, inputs):
        self.write('hidden_states.0', inputs[0])

    def _residual(self, layer, module, inputs, output):
        self.write(f'hidden_states.{layer}', output[1])

    def _final(self, module, inputs, output):
        if isinstance(output, tuple):
            self.write('hidden_states.78', output[0])

    def _attention(self, layer, original):
        signature = inspect.signature(original)
        parameters = tuple(signature.parameters.values())
        prefix = parameters[:len(ATTENTION_ARGUMENTS)]
        required_prefix = (
            tuple(value.name for value in prefix) == ATTENTION_ARGUMENTS
            and all(value.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                    and value.default is inspect.Parameter.empty
                    for value in prefix)
        )
        optional_table = (
            len(parameters) == len(ATTENTION_ARGUMENTS) + 1
            and parameters[-1].name == 'block_table'
            and parameters[-1].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and parameters[-1].default is None
        )
        require(required_prefix and (len(parameters) == len(ATTENTION_ARGUMENTS)
                                     or optional_table),
                'SFA consumer signature differs')

        @wraps(original)
        def observed(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            selected = arguments['topk_indices']
            if selected.ndim == 3:
                selected = selected[:, 0, :]
            self.write(f'indices.{layer}', selected)
            return original(*args, **kwargs)

        return observed

    def _router(self, layer, original):
        @wraps(original)
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            self.write(f'routes.{layer}', result[1])
            return result

        return observed

    def _replace(self, target, name, layer, wrapper):
        previous = vars(target).get(name)
        existed = name in vars(target)
        original = getattr(target, name)
        setattr(target, name, wrapper(layer, original))
        self._overrides.append((target, name, existed, previous))

    def __enter__(self):
        backbone = self.engine.model.model
        try:
            self._handles.append(
                backbone.embed_tokens.register_forward_pre_hook(
                    self._embedding_input,
                ),
            )
            self._handles.append(
                backbone.layers[0].input_layernorm.register_forward_pre_hook(
                    self._embedding_output,
                ),
            )
            for layer, block in enumerate(backbone.layers):
                if layer:
                    self._handles.append(
                        block.input_layernorm.register_forward_hook(
                            partial(self._residual, layer),
                        ),
                    )
                attention = block.self_attn.mla_attn.mla_attn.impl
                self._replace(attention,
                              '_execute_sparse_flash_attention_process',
                              layer, self._attention)
                if layer >= 3:
                    self._replace(block.mlp.experts.routed_experts,
                                  '_select_experts', layer, self._router)
            self._handles.append(backbone.norm.register_forward_hook(
                self._final,
            ))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def readback(self, step, tokens, start):
        require(self.addresses() == self._addresses,
                'Persistent observation storage changed')
        require(self.current_epoch is not None and bool(torch.all(
            self.seen == self.current_epoch,
        )), 'A field was not observed in this replay')
        positions = list(range(start, start + WIDTH))
        require(self.values['input_ids'].tolist() == tokens
                and self.values['positions'].tolist() == positions,
                'Observed request rows differ')
        hidden = tuple(self.values[f'hidden_states.{layer}'].detach().clone()
                       for layer in range(LAYERS + 1))
        require(torch.equal(hidden[-1], step.last_hidden),
                'Returned hidden differs from final norm observation')
        normalized = self.engine.model.model.norm(torch.cat(hidden[1:-1]))
        projected = self.engine.model.compute_logits(normalized)
        require(projected.dtype == step.logits.dtype == torch.float32,
                'Fixture requires the FP32 head path')
        values = {f'hidden_states.{layer}': value
                  for layer, value in enumerate(hidden)}
        values.update({f'layer_logits.{layer}': value.detach().clone()
                       for layer, value in enumerate((
                           *projected.split(WIDTH), step.logits,
                       ))})
        values.update({f'indices.{layer}': self.values[
            f'indices.{layer}'].detach().clone() for layer in range(LAYERS)})
        values.update({f'routes.{layer}': None if layer < 3 else self.values[
            f'routes.{layer}'].detach().clone() for layer in range(LAYERS)})
        values.update(logits=step.logits.detach().clone(),
                      last_hidden=step.last_hidden.detach().clone(),
                      token_ids=step.token_ids.detach().clone())
        for layer, cache in enumerate(self.engine.cache):
            for field, key in CACHE_FIELDS.items():
                values[f'cache_rows.{layer}.{field}'] = (
                    cache[key][start:start + WIDTH].detach().clone()
                    if key in cache else None
                )
        return {tag: None if value is None else value.detach().cpu().clone()
                for tag, value in values.items()}

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for target, name, existed, previous in reversed(self._overrides):
            if existed:
                setattr(target, name, previous)
            else:
                delattr(target, name)
        self._overrides.clear()
        return False


class FixtureNorm(nn.Module):
    def forward(self, value, residual=None):
        return value if residual is None else (value + residual,
                                               value + residual)


class FixtureEmbedding(nn.Module):
    def forward(self, ids):
        return ids[:, None].to(torch.bfloat16) + torch.arange(
            HIDDEN, dtype=torch.bfloat16,
        )


class FixtureAttention:
    def _execute_sparse_flash_attention_process(
        self, ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
        actual_seq_lengths_query, actual_seq_lengths_key,
    ):
        return (torch.zeros_like(ql_nope)
                + topk_indices[:, 0, :1].to(ql_nope.dtype) % 2)


class FixtureRouter:
    def __init__(self, layer):
        self.layer = layer

    def _select_experts(self, hidden):
        ids = (hidden[:, :1].to(torch.int32) + self.layer
               + torch.arange(8, dtype=torch.int32)) % 256
        return torch.ones_like(ids, dtype=torch.float32) / 8, ids


class FixtureLayer(nn.Module):
    def __init__(self, layer, capacity):
        super().__init__()
        self.layer = layer
        self.use_sequence_parallel_moe = False
        self.input_layernorm = FixtureNorm()
        self.self_attn = SimpleNamespace(mla_attn=SimpleNamespace(
            mla_attn=SimpleNamespace(impl=FixtureAttention()),
        ))
        self.mlp = SimpleNamespace(experts=SimpleNamespace(
            routed_experts=FixtureRouter(layer),
        ))
        self.register_buffer('latent', torch.zeros(capacity, 2,
                                                   dtype=torch.bfloat16))
        self.register_buffer('rope', torch.zeros(capacity, 2,
                                                 dtype=torch.bfloat16))
        if layer in FULL_INDEX_LAYERS:
            dtype = (torch.int8 if layer in INT8_INDEX_LAYERS
                     else torch.bfloat16)
            self.register_buffer('index_keys', torch.zeros(capacity, 4,
                                                           dtype=dtype))
        if layer in INT8_INDEX_LAYERS:
            self.register_buffer('index_scales', torch.zeros(
                capacity, 1, dtype=torch.float16,
            ))

    def forward(self, value, residual, positions):
        output = self.input_layernorm(value, residual)
        merged = output if residual is None else output[1]
        columns = torch.arange(PICKS, dtype=torch.int32)
        lengths = positions[:, None].to(torch.int32) + 1
        indices = torch.where(
            columns < lengths,
            (columns + merged[:, :1].to(torch.int32) + self.layer) % lengths,
            -1,
        )
        attention = self.self_attn.mla_attn.mla_attn.impl
        attention_flag = attention._execute_sparse_flash_attention_process(
            merged, merged, None, indices[:, None, :], None, None, None,
        )
        route_flag = 0
        if self.layer >= 3:
            _, routes = self.mlp.experts.routed_experts._select_experts(merged)
            route_flag = routes[:, :1].to(merged.dtype) % 2
        self.latent.index_copy_(0, positions, merged[:, :2])
        rope = torch.stack((positions, positions * 0 + self.layer), dim=1)
        self.rope.index_copy_(0, positions, rope.to(torch.bfloat16))
        if hasattr(self, 'index_keys'):
            keys = merged % 127 if self.layer in INT8_INDEX_LAYERS else merged
            self.index_keys.index_copy_(
                0, positions, keys.to(self.index_keys.dtype),
            )
        if hasattr(self, 'index_scales'):
            scales = torch.full_like(merged[:, :1], (self.layer + 1) / 128,
                                     dtype=torch.float16)
            self.index_scales.index_copy_(0, positions, scales)
        increment = 1 + (attention_flag + route_flag) % 2
        return increment, merged


class FixtureBackbone(nn.Module):
    def __init__(self, capacity):
        super().__init__()
        self.start_layer, self.end_layer, self.hidden_size = 0, LAYERS, HIDDEN
        self.embed_tokens = FixtureEmbedding()
        self.layers = nn.ModuleList(FixtureLayer(layer, capacity)
                                    for layer in range(LAYERS))
        self.norm = FixtureNorm()
        self.register_buffer('output', torch.empty(WIDTH, HIDDEN,
                                                   dtype=torch.bfloat16))

    def forward(self, input_ids, positions):
        value, residual = self.embed_tokens(input_ids), None
        for layer in self.layers:
            value, residual = layer(value, residual, positions)
        hidden, _ = self.norm(value, residual)
        self.output.copy_(hidden)
        return self.output


class FixtureLM(nn.Module):
    def __init__(self, capacity):
        super().__init__()
        self.model = FixtureBackbone(capacity)
        self.register_buffer('weight', torch.arange(VOCAB).repeat(HIDDEN, 1)
                             .to(torch.bfloat16))

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden):
        return hidden.float() @ self.weight.float()


def fixture_engine():
    engine = CannModel.__new__(CannModel)
    engine.device = torch.device('cpu')
    engine.position, engine.capacity = 0, 128
    engine.attention_kernel, engine.graph_width = 'sfa', WIDTH
    engine.graph, engine.config = None, SimpleNamespace()
    engine.model = FixtureLM(engine.capacity)
    engine.input_ids = torch.empty(WIDTH, dtype=torch.int32)
    engine.positions = torch.empty(WIDTH, dtype=torch.int64)
    engine.cache = [
        {name: getattr(layer, name) for name in CACHE_FIELDS.values()
         if hasattr(layer, name)} for layer in engine.model.model.layers
    ]

    def metadata(width, **kwargs):
        require(width == WIDTH, 'Fixture only executes width32')
        engine.positions.copy_(torch.arange(
            engine.position, engine.position + width,
        ))
        return {}

    engine._metadata = metadata
    return engine


@contextmanager
def cpu_context_adapters():
    """Replace only unavailable context managers, never tensor computation."""
    modules = {name: ModuleType(name) for name in (
        'vllm', 'vllm.config', 'vllm_ascend',
        'vllm_ascend.ascend_forward_context',
    )}
    config = modules['vllm.config']
    context = modules['vllm_ascend.ascend_forward_context']
    config.set_current_vllm_config = lambda config: nullcontext()
    context.set_ascend_forward_context = lambda *args, **kwargs: nullcontext()

    def device_scope(device):
        require(device.type == 'cpu', 'This adapter only accepts CPU')
        return nullcontext()

    with patch.dict(sys.modules, modules), patch.object(
        torch, 'npu', SimpleNamespace(device=device_scope), create=True,
    ):
        yield


def scalar_reference(token, position):
    base = token
    hidden = [torch.tensor([[base + column for column in range(HIDDEN)]],
                           dtype=torch.bfloat16)]
    logits = []
    indices, routes, caches = [], [], []
    for layer in range(LAYERS):
        indices.append(torch.tensor([
            (column + base + layer) % (position + 1)
            if column <= position else -1 for column in range(PICKS)
        ], dtype=torch.int32))
        routes.append(None if layer < 3 else torch.tensor([
            (base + layer + column) % 256 for column in range(8)
        ], dtype=torch.int32))
        cache = {'latent': torch.tensor([base, base + 1],
                                        dtype=torch.bfloat16),
                 'rope': torch.tensor([position, layer], dtype=torch.bfloat16),
                 'index': None, 'index_scale': None}
        if layer in FULL_INDEX_LAYERS:
            values = [base + column for column in range(HIDDEN)]
            quantized = layer in INT8_INDEX_LAYERS
            cache['index'] = torch.tensor(
                [value % 127 for value in values] if quantized else values,
                dtype=torch.int8 if quantized else torch.bfloat16,
            )
        if layer in INT8_INDEX_LAYERS:
            cache['index_scale'] = torch.tensor(
                [(layer + 1) / 128], dtype=torch.float16,
            )
        caches.append(cache)
        attention_flag = ((base + layer) % (position + 1)) % 2
        route_flag = (base + layer) % 2 if layer >= 3 else 0
        base += 1 + (attention_flag + route_flag) % 2
        hidden.append(torch.tensor(
            [[base + column for column in range(HIDDEN)]],
            dtype=torch.bfloat16,
        ))
        logits.append(torch.tensor(
            [[sum(base + column for column in range(HIDDEN)) * output
              for output in range(VOCAB)]], dtype=torch.float32,
        ))
    return {'token': token, 'position': position,
            'committed_position': position + 1, 'hidden_states': hidden,
            'layer_logits': logits, 'logits': logits[-1],
            'last_hidden': hidden[-1], 'token_ids': torch.tensor([VOCAB - 1]),
            'indices': indices, 'routes': routes, 'cache_rows': caches}


def frame_header(epoch, tokens, positions, sources, receiver):
    inputs = {'tokens': tokens, 'positions': positions,
              'committed_positions': [position + 1 for position in positions]}
    encoded = json.dumps(inputs, sort_keys=True, separators=(',', ':'))
    checksum = hashlib.sha256(encoded.encode()).hexdigest()
    header = dict.fromkeys(receiver.HEADER_KEYS)
    header.update(origin='cpu-boundary-fixture', replay_id=epoch, width=WIDTH,
                  layer_count=LAYERS, fp32_head=True, mla_preprocess='fixture',
                  inputs=inputs, input_sha256=checksum, source_sha256=sources,
                  checkpoint_sha256={}, trace_sha256={},
                  weight_arithmetic_sha256={}, loaded_reference_sources={},
                  installed_hf_source_sha256={})
    return header


def package(values, header, receiver):
    require(set(values) == receiver.field_tags(), 'Incomplete producer fields')
    frames = [{'tag': tag, 'replay_id': header['replay_id'],
               'input_sha256': header['input_sha256'],
               'row_ids': list(range(WIDTH)), 'tensor': values[tag],
               'lengths': ([PICKS] * WIDTH
                           if tag.startswith('indices.') else None)}
              for tag in sorted(values)]
    return {'header': header, 'frames': frames}


def validate_frame(name, frame, header, references, receiver):
    decoded = receiver.decode_frame(frame, header, references)
    checks, token_checks = [], []
    for row, (actual, expected) in enumerate(zip(decoded, references)):
        compare_step(f'{name}.{row}', actual, expected, checks, exact=True)
        compare_tensor(f'{name}.{row}.token_ids', actual['token_ids'],
                       expected['token_ids'], token_checks, exact=True)
    require(checks and all(check['passed'] for check in checks + token_checks),
            'Strict scalar or output token comparison failed')
    return decoded, checks, token_checks


def graph_nodes(gm):
    return [{'name': node.name, 'target': str(node.target),
             'op': node.op, 'impure': node.is_impure(),
             'inputs': [value.name for value in node.all_input_nodes]}
            for node in gm.graph.nodes]


def cpu_backend(stages, live_graphs):
    def lowered(gm, example_inputs):
        before = graph_nodes(gm)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        after = graph_nodes(gm)
        stages.append({'stage': 'aot', 'before_dce': before,
                       'after_dce': after, 'code': gm.code})
        live_graphs.append(gm)

        def execute(*args):
            return gm(*args)

        return execute

    aot = aot_autograd(fw_compiler=lowered, inference_compiler=lowered,
                       keep_inference_input_mutations=True)

    def dynamo(gm, example_inputs):
        stages.append({'stage': 'dynamo', 'nodes': graph_nodes(gm),
                       'code': gm.code})
        return aot(gm, example_inputs)

    return dynamo


def rejected(name, operation):
    try:
        operation()
    except (ValueError, RuntimeError, TypeError) as error:
        return {'name': name, 'rejected': True,
                'error': type(error).__name__ + ': ' + str(error)}
    raise AssertionError('Negative was admitted: ' + name)


def source_files(args):
    paths = [Path(__file__), Path(__file__).with_name('receiver.py'),
             ROOT / 'tests/glm52/run_cann_model.py',
             ROOT / 'tests/glm52/run_native_decode.py',
             ROOT / 'tests/glm52/precision_gate.py',
             ROOT / 'tilert/pd_vllm/ascend_layout.py',
             *(ROOT / 'tilert/models/glm_5_2_ascend' / name for name in (
                 'cann_model.py', 'cann_graph.py', 'cann_trace.py',
                 'cann_choices.py',
             ))]
    paths.extend(args.torchair / name for name in (
        'python/torchair/_acl_concrete_graph/fx2acl_converter.py',
        'python/torchair/_acl_concrete_graph/utils.py',
        'npugraph_ex/npugraph_ex/_acl_concrete_graph/fx2acl_converter.py',
    ))
    paths.extend(args.vllm_ascend / name for name in (
        'vllm_ascend/ascend_config.py',
        'vllm_ascend/compilation/compiler_interface.py',
    ))
    paths.append(args.vllm_ascend.parent /
                 'vllm/vllm/compilation/decorators.py')
    paths.extend(args.msprobe / name for name in (
        'python/msprobe/pytorch/aclgraph_dump/__init__.py',
        'python/msprobe/pytorch/aclgraph_dump/_meta.py',
        'ccsrc/aclgraph_dump/aclgraph_dump.cpp',
    ))
    return {str(path.resolve()): digest(path) for path in paths}


def static_counterexamples(args):
    path = args.vllm_ascend / 'vllm_ascend/ascend_config.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'AscendCompilationConfig')
    namespace = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'),
         namespace)
    config = namespace[cls.name](enable_static_kernel=True,
                                 enable_super_kernel=True)
    converter = args.torchair / (
        'python/torchair/_acl_concrete_graph/fx2acl_converter.py'
    )
    text = converter.read_text()
    return {
        'local_config_fields': vars(config),
        'local_super_kernel_request_ignored': not hasattr(
            config, 'enable_super_kernel',
        ),
        'local_sk_dump_explicitly_skipped': all(
            fragment in text for fragment in (
                'if self.config.experimental_config.aclgraph.'
                '_super_kernel_optimize',
                'the dump_config.enable_dump configuration becomes invalid.',
            )
        ),
        'scope': 'These local sources do not attest the deployed NPU version',
    }


def run(args, report):
    torch.set_num_threads(1)
    receiver, engine, stages = load_receiver(), fixture_engine(), []
    live_graphs = []
    negatives, cases = [], []
    report.update(cases=cases, negatives=negatives, graph_stages=stages)
    sources = report['source_sha256']
    with cpu_context_adapters(), torch.inference_mode(), PersistentTrace(
        engine, index_dtype=torch.int32, route_dtype=torch.int32,
    ) as producer:
        negatives.append(rejected('current_trace_width32',
                                  lambda: CannTrace(engine.model, WIDTH)))
        engine.model.model.forward = torch.compile(
            engine.model.model.forward,
            backend=cpu_backend(stages, live_graphs),
            fullgraph=True, dynamic=False,
        )
        first_values, borrowed = None, None
        for epoch, name, tokens in (
            (1, 'A0', list(range(1, WIDTH + 1))),
            (2, 'B', list(range(65, 65 + WIDTH))),
            (3, 'A1', list(range(1, WIDTH + 1))),
        ):
            if name == 'A1':
                engine.reset()
            start = engine.position
            expected_header = frame_header(
                epoch, list(tokens), list(range(start, start + WIDTH)),
                sources, receiver,
            )
            producer.begin(epoch)
            step = engine.verify(tokens)
            if name == 'A0':
                borrowed = step.last_hidden
            if name == 'B':
                require(not torch.equal(borrowed,
                                        first_values['last_hidden']),
                        'Fixture did not expose reused output storage')
            values = producer.readback(step, tokens, start)
            actual_header = frame_header(
                int(producer.epoch), producer.values['input_ids'].tolist(),
                producer.values['positions'].tolist(), sources, receiver,
            )
            frame = package(values, actual_header, receiver)
            references = [scalar_reference(token, start + row)
                          for row, token in enumerate(tokens)]
            decoded, checks, token_checks = validate_frame(
                name, frame, expected_header, references, receiver,
            )
            cases.append({'name': name, 'epoch': epoch, 'start': start,
                          'committed_position': engine.position,
                          'fields': len(frame['frames']), 'rows': len(decoded),
                          'checks': len(checks),
                          'output_token_checks': len(token_checks),
                          'stamps': producer.seen.tolist(),
                          'input_sha256': frame['header']['input_sha256']})
            if name == 'A0':
                first_values = values
            if name == 'A1':
                require(all(left is None if right is None else
                            torch.equal(left, right) for tag, right in
                            first_values.items() for left in (values[tag],)),
                        'A/B/A reset changed an owned observation')
        for name, replacement in (
            ('wrong_output_token', torch.zeros_like(values['token_ids'])),
            ('floating_output_token', values['token_ids'].float()),
        ):
            bad_frame = {
                **frame, 'frames': [
                    {**field, 'tensor': replacement}
                    if field['tag'] == 'token_ids' else field
                    for field in frame['frames']
                ],
            }
            negatives.append(rejected(name, lambda: validate_frame(
                name, bad_frame, expected_header, references, receiver,
            )))
        saved_seen = producer.seen.clone()
        producer.seen[-1] = -1
        negatives.append(rejected('missing_field_stamp', lambda:
                                  producer.readback(step, tokens, start)))
        producer.seen.copy_(saved_seen)
        producer.epoch.fill_(epoch + 1)
        producer.current_epoch = epoch + 1
        negatives.append(rejected('stale_replay_stamps', lambda:
                                  producer.readback(step, tokens, start)))
        producer.current_epoch = epoch
        producer.epoch.fill_(epoch)
        negatives.append(rejected(
            'request_row_permutation',
            lambda: producer.readback(step, tokens[::-1], start),
        ))
        previous = producer.values['positions']
        producer.values['positions'] = previous.clone()
        negatives.append(rejected('persistent_address_replaced', lambda:
                                  producer.readback(step, tokens, start)))
        producer.values['positions'] = previous
        placeholders = [node for node in stages[0]['nodes']
                        if node['op'] == 'placeholder']
        seen_index = [index for index, node in enumerate(placeholders)
                      if node['target'].endswith('_self_seen')]
        require(len(live_graphs) == len(seen_index) == 1,
                'Cannot identify the actual stamp input')
        gm = live_graphs[0]
        inputs = [node for node in gm.graph.nodes if node.op == 'placeholder']
        destination = inputs[seen_index[0]]
        writes = [node for node in gm.graph.nodes
                  if str(node.target) == 'aten.copy_.default'
                  and node.args[0] is destination]
        require(len(writes) == 1 and not writes[0].users,
                'Expected one final writeback to the stamp vector')
        removed = writes[0].name
        gm.graph.erase_node(writes[0])
        gm.graph.eliminate_dead_code()
        gm.recompile()
        producer.begin(4)
        start = engine.position
        step = engine.verify(tokens)
        negatives.append(rejected('actual_aot_stamp_writeback_removed', lambda:
                                  producer.readback(step, tokens, start)))
        report['deliberate_bad_graph'] = {
            'removed_node': removed, 'destination': destination.name,
            'code_sha256': hashlib.sha256(gm.code.encode()).hexdigest(),
            'stamps_after_actual_call': producer.seen.tolist(),
        }
    require(len(stages) == 2 and stages[0]['stage'] == 'dynamo'
            and stages[1]['stage'] == 'aot', 'Unexpected recompilation')
    copies = [node for node in stages[1]['after_dce']
              if node['target'] == 'aten.copy_.default']
    report['aot_copy_nodes_after_dce'] = len(copies)
    report['observed_tensor_slots'] = len(producer.tags)
    accounting = {
        'observation_buffers': len(producer.values), 'stamp_vector': 1,
        'cache_buffers': sum(len(cache) for cache in engine.cache),
        'reused_hidden_output': 1,
    }
    targets = [node['inputs'][0] for node in copies]
    scatters = [node for node in stages[1]['after_dce']
                if node['target'] == 'aten.select_scatter.default']
    require(len(copies) == len(set(targets)) == sum(accounting.values())
            and len(scatters) == len(producer.tags),
            'AOT input mutation accounting differs')
    report.update(cases=cases, negatives=negatives, graph_stages=stages,
                  observed_tensor_slots=len(producer.tags),
                  aot_copy_nodes_after_dce=len(copies),
                  aot_copy_accounting=accounting,
                  aot_stamp_select_scatter_nodes=len(scatters),
                  reused_output_alias_observed=True, owned_reset_exact=True,
                  field_rows_checked=sum(case['fields'] * case['rows']
                                         for case in cases),
                  original_comparator_exact_checks=sum(case['checks']
                                                       for case in cases),
                  output_token_exact_checks=sum(case['output_token_checks']
                                                for case in cases),
                  input_header_built_from_observed_slots=True,
                  hooks_removed=not producer._handles
                  and not producer._overrides,
                  static_counterexamples=static_counterexamples(args))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--torchair', type=Path, default=Path(
        '/Users/shanshan/repo/ascend/torchair',
    ))
    parser.add_argument('--vllm-ascend', type=Path, default=Path(
        '/Users/shanshan/repo/vllm-project/vllm-ascend',
    ))
    parser.add_argument('--msprobe', type=Path, default=Path(
        '/Users/shanshan/repo/ascend/msprobe',
    ))
    args = parser.parse_args()
    report = {
        'status': 'running', 'torch': torch.__version__, 'device': 'cpu',
        'precision_verified': False, 'npu_executed': False,
        'acl_graph_executed': False,
        'super_kernel_executed': False, 'performance_measured': False,
        'hf_executed': False,
        'reference_kind': 'Independent scalar CPU fixture; no HF admission',
        'model_scope': '78-layer width32 fixture; no checkpoint or attention',
        'engine_construction': 'CannModel.__new__ with CPU fixture fields',
        'replay_scope': 'One CPU AOT graph; no ACL capture/replay',
        'context_adapters': ['torch.npu.device', 'set_current_vllm_config',
                             'set_ascend_forward_context'],
        'boundary': {'function': 'CannModel.verify',
                     'file': inspect.getfile(CannModel.verify),
                     'line': inspect.getsourcelines(CannModel.verify)[1]},
        'compiled_boundary': 'FixtureBackbone.forward only; head is outside',
        'source_sha256': source_files(args),
        'command': [sys.executable, *sys.argv],
    }
    try:
        run(args, report)
        report['source_stable'] = all(
            digest(path) == checksum
            for path, checksum in report['source_sha256'].items()
        )
        require(report['source_stable'], 'Probe sources changed')
        report['status'] = 'pass'
    except Exception:
        import traceback

        report.update(status='fail', error=traceback.format_exc())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)
                           + '\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output),
                      'error': report.get('error')}), flush=True)
    return int(report['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
