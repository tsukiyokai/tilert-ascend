"""Measure both shared-attention implementations on identical tensor addresses."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys


MAIN = ('weight_dq', 'weight_uq_qr', 'weight_uk', 'weight_dkv_kr',
        'scale_dq', 'scale_uq_qr', 'scale_dkv_kr', 'gamma_cq', 'gamma_ckv')
CONTROL = '64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c'
CANDIDATE = 'ff36c603205f43420b2181ae7193a3b6b4b8a3a2445db84a3cf74840664c4d64'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_helpers(directory):
    expected = {
        'attention_check.py': '94be7f24a6a548263eacbd9c88e69a8172faf9dad3e80ddfcfa074bb9745af77',
        'attention_full_check.py': '41d909ff3bd2803aa86634b70f7894373311161b299156435775c01028b6d707',
    }
    for name, sha in expected.items():
        if digest(directory / name) != sha:
            raise ValueError(f'Frozen helper changed: {name}')
    modules = []
    for name in ('attention_check', 'attention_full_check'):
        spec = importlib.util.spec_from_file_location(name, directory / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def call_arguments(state, bank):
    return (state.x, state.x_scale, *bank['main'], state.sin, state.cos, state.slots,
            state.cache, state.rope, state.topk, state.query_lengths,
            state.kv_lengths, state.page_table, *bank['tail'], bank['bias'],
            state.scratch, state.tiling, state.output, 1 / 16, 1e-5, 1e-5,
            state.blocks, state.sync)


def bind_calls(arguments, operations):
    return {
        arm: [lambda values=values, op=op: op.shared_full_attention_out(*values)
              for values in arguments]
        for arm, op in operations.items()
    }


def pack_bank(torch, torch_npu, args, data, index):
    def tensor(name, form=2):
        value = data[name]
        if value.dim() >= 2:
            value = value.roll(index, dims=-1)
        else:
            value = value.clone()
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), form)

    main = [tensor(name, 29 if name in ('weight_dq', 'weight_uq_qr', 'weight_dkv_kr') else 2)
            for name in MAIN]
    value = data['value_weight'].roll(index, dims=-1)
    value = value.reshape(4, 512, 4, 64).permute(0, 2, 1, 3).contiguous().reshape(16, 512, 64)
    output = data['output_weight'].roll(index, dims=0)
    output = output.reshape(1024, 24, 256).permute(1, 0, 2).contiguous()
    tail = [torch_npu.npu_format_cast(value.to(args.device), 2),
            torch_npu.npu_format_cast(output.to(args.device), 2), tensor('output_scale')]
    return {'main': main, 'tail': tail, 'bias': None}


class State:
    def __init__(self, torch, torch_npu, args, data, operations, owner):
        self.torch, self.torch_npu, self.args, self.data = torch, torch_npu, args, data
        self.rows = args.rows
        self.table_pages = (args.sequence + args.rows + args.replays + 127) // 128
        self.pages = self.table_pages + 3
        generator = torch.Generator().manual_seed(12345 + args.rows)
        self.cache_base = self.nd(torch.randn(self.pages, 128, 1, 512,
                                             generator=generator).to(torch.bfloat16))
        self.rope_base = self.nd(torch.randn(self.pages, 128, 1, 64,
                                            generator=generator).to(torch.bfloat16))
        self.cache = self.nd(self.cache_base.clone())
        self.rope = self.nd(self.rope_base.clone())
        self.x = self.empty((args.rows, 6144), torch.int8)
        self.x_scale = self.empty((args.rows, 1), torch.float32)
        self.sin = self.empty((args.rows, 64), torch.bfloat16)
        self.cos = self.empty((args.rows, 64), torch.bfloat16)
        self.slots = self.empty((args.rows,), torch.int64)
        self.topk = self.empty((args.rows, 1, 2048), torch.int32)
        self.query_lengths = self.nd(torch.tensor([args.rows], dtype=torch.int32))
        self.kv_lengths = self.empty((1,), torch.int32)
        self.page_table = self.empty((1, self.table_pages), torch.int32)
        self.output = self.empty((args.rows, 6144), torch.bfloat16)
        self.owners = {
            name: op.shared_full_attention_prepare(self.x, self.pages, self.table_pages, args.blocks)
            for name, op in operations.items()
        }
        self.scratch, self.tiling, self.blocks, self.sync = self.owners[owner]
        if self.blocks != args.blocks:
            raise ValueError('Prepared block count differs')
        self.latent = self.scratch[:args.rows * 4 * 512 * 2].view(torch.bfloat16)

    def nd(self, value):
        return self.torch_npu.npu_format_cast(value.to(self.args.device).contiguous(), 2)

    def empty(self, shape, dtype):
        return self.nd(self.torch.empty(shape, dtype=dtype))

    def update(self, replay):
        torch = self.torch
        prefix = self.args.sequence + replay
        positions = torch.arange(prefix, prefix + self.rows, dtype=torch.int64)
        table = torch.arange(self.pages, dtype=torch.int32).roll(replay + 1)[:self.table_pages]
        self.slots.copy_(self.nd(table[positions // 128].long() * 128 + positions % 128))
        self.page_table.copy_(self.nd(table.reshape(1, -1)))
        frequencies = torch.pow(10000.0, -torch.arange(32).float() / 32)
        angles = positions.float().unsqueeze(1) * frequencies
        self.sin.copy_(self.nd(angles.sin().repeat(1, 2).to(torch.bfloat16)))
        self.cos.copy_(self.nd(angles.cos().repeat(1, 2).to(torch.bfloat16)))
        selected = torch.arange(max(0, prefix + self.rows - 2048), prefix + self.rows,
                                dtype=torch.int32)
        indices = torch.full((self.rows, 1, 2048), -1, dtype=torch.int32)
        indices[:, 0, :selected.numel()] = selected if replay % 2 == 0 else selected.flip(0)
        self.topk.copy_(self.nd(indices))
        self.kv_lengths.copy_(self.nd(torch.tensor([prefix + self.rows], dtype=torch.int32)))
        multiplier = (1.0, 0.0, -1.0, 0.5)[replay % 4]
        hidden = (self.data['hidden'][:self.rows].float() * multiplier).to(torch.bfloat16)
        if replay >= 4:
            hidden = hidden.roll(replay, dims=1)
        codes, scales = self.torch_npu.npu_dynamic_quant(self.nd(hidden))
        self.x.copy_(codes)
        self.x_scale.copy_(scales.reshape(self.rows, 1))

    def reset(self):
        self.cache.copy_(self.cache_base)
        self.rope.copy_(self.rope_base)
        self.output.fill_(float('nan'))
        self.latent.fill_(float('nan'))

    def snapshot(self):
        return {name: value.detach().cpu().clone() for name, value in (
            ('output', self.output), ('latent', self.latent),
            ('kv_cache', self.cache), ('rope_cache', self.rope))}


def compare(torch, helper, actual, expected):
    checks = {name: helper.metric(torch, actual[name], expected[name]) for name in expected}
    passed = all(item['pass'] and item['bitwise_equal'] for item in checks.values())
    if not passed:
        raise RuntimeError(json.dumps(checks))
    return checks


def describe(torch, value):
    if not torch.is_tensor(value):
        return value
    return {'address': value.data_ptr(), 'storage': value.untyped_storage().data_ptr(),
            'shape': list(value.shape), 'stride': list(value.stride()),
            'dtype': str(value.dtype), 'device': str(value.device)}


def run_owner(torch, torch_npu, args, data, banks, operations, owner, helper):
    state = State(torch, torch_npu, args, data, operations, owner)
    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))
    arguments = [call_arguments(state, bank) for bank in banks]
    calls = bind_calls(arguments, operations)
    report = {'owner': owner, 'same_arguments_for_both_arms': True,
              'argument_descriptors': [[describe(torch, value) for value in values]
                                       for values in arguments],
              'prechecks': [], 'postchecks': [], 'samples': []}

    def checked_inputs(replays, target):
        for replay in replays:
            with torch.npu.stream(stream):
                state.update(replay)
            for bank in range(len(banks)):
                with torch.npu.stream(stream):
                    state.reset()
                    calls['control'][bank]()
                stream.synchronize()
                expected = state.snapshot()
                with torch.npu.stream(stream):
                    state.reset()
                    calls['candidate'][bank]()
                stream.synchronize()
                target.append({'input': replay, 'bank': bank,
                               'checks': compare(torch, helper, state.snapshot(), expected)})

    checked_inputs(range(args.replays), report['prechecks'])
    with torch.npu.stream(stream):
        state.update(args.timed_input)
        state.reset()
        calls['control'][-1]()
    stream.synchronize()
    expected = state.snapshot()
    python_calls = {arm: 0 for arm in operations}
    graphs = {}
    for arm in operations:
        def sequence(arm=arm):
            python_calls[arm] += 1
            for call in calls[arm]:
                call()
            return state.output

        with torch.npu.stream(stream):
            state.reset()
        graphs[arm], _ = helper.capture(torch, sequence, stream)
        if python_calls[arm] != 4:
            raise RuntimeError('Expected three warmups and one capture')
    for pair in range(args.samples):
        sample = {'pair': pair, 'per_call_us': {}, 'tail_checks': {}}
        order = ['control', 'candidate'] if pair % 2 == 0 else ['candidate', 'control']
        sample['order'] = order
        for arm in order:
            with torch.npu.stream(stream):
                state.reset()
            elapsed = helper.time_graph(torch, graphs[arm], stream, args.iterations)
            sample['per_call_us'][arm] = elapsed / len(banks)
            sample['tail_checks'][arm] = compare(torch, helper, state.snapshot(), expected)
            if python_calls[arm] != 4:
                raise RuntimeError('Python executed during graph replay')
        sample['paired_saving_us'] = sample['per_call_us']['control'] - sample['per_call_us']['candidate']
        report['samples'].append(sample)
    checked_inputs([args.timed_input], report['postchecks'])
    savings = [sample['paired_saving_us'] for sample in report['samples']]
    report['summary'] = {
        'control_median_us': statistics.median(sample['per_call_us']['control']
                                               for sample in report['samples']),
        'candidate_median_us': statistics.median(sample['per_call_us']['candidate']
                                                 for sample in report['samples']),
        'paired_median_saving_us': statistics.median(savings),
        'paired_minimum_saving_us': min(savings), 'paired_maximum_saving_us': max(savings),
        'paired_wins': sum(value > 0 for value in savings),
    }
    report['python_calls_after_capture_and_replays'] = python_calls
    report['pass'] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--control-library', type=Path, required=True)
    parser.add_argument('--helpers', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--rows', type=int, default=12)
    parser.add_argument('--blocks', type=int, default=24)
    parser.add_argument('--sequence', type=int, default=8192)
    parser.add_argument('--banks', type=int, default=16)
    parser.add_argument('--replays', type=int, default=8)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--timed-input', type=int, default=3)
    args = parser.parse_args()
    if (args.rows != 12 or args.blocks != 24 or args.sequence != 8192 or args.banks != 16
            or args.replays != 8 or args.samples < 30 or args.iterations < 1
            or args.timed_input != 3):
        parser.error('Use the M12/24, sequence8192, 16-bank, eight-input paired experiment')
    if digest(args.library) != CANDIDATE or digest(args.control_library) != CONTROL:
        raise ValueError('Unexpected frozen libraries')
    helper, full = load_helpers(args.helpers)
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.ops.load_library(str(args.control_library))
    torch.ops.load_library(str(args.library))
    args.fixture, args.widths = None, [args.rows]
    data = helper.fixture(torch, args)
    full.tail_fixture(torch, args, data)
    banks = [pack_bank(torch, torch_npu, args, data, index) for index in range(args.banks)]
    weights = [value for bank in banks for value in bank['main'] + bank['tail']]
    addresses = [value.data_ptr() for value in weights]
    if len(set(addresses)) != len(addresses):
        raise RuntimeError('Every bank must own distinct weights')
    logical_bytes = sum(value.numel() * value.element_size() for value in weights)
    if logical_bytes <= 192 * 1024 ** 2:
        raise RuntimeError('The logical weight working set must exceed the recorded A3 L2 size')
    operations = {'control': torch.ops.tilert, 'candidate': torch.ops.tilert_shared_descriptor}
    report = {
        'scope': 'Synthetic shared-indexer local attention; no TP reduction or model TPOT',
        'arguments': {name: str(value) if isinstance(value, Path) else value
                      for name, value in vars(args).items()},
        'library_sha256': {'control': CONTROL, 'candidate': CANDIDATE},
        'script_sha256': digest(Path(__file__)),
        'helper_sha256': {name: digest(args.helpers / name) for name in
                         ('attention_check.py', 'attention_full_check.py')},
        'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
        'logical_weight_bytes': logical_bytes,
        'working_set_scope': 'Distinct rolled weight banks; common metadata and cache addresses. Logical bytes do not measure HBM traffic.',
        'input_reset_scope': 'Input and cache copies and NaN poisoning precede the start event on the same stream.',
        'calls_per_graph': args.banks, 'graph_replays_per_event': args.iterations,
        'owners': [], 'pass': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for owner in operations:
        result = run_owner(torch, torch_npu, args, data, banks, operations, owner, helper)
        report['owners'].append(result)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'owner': owner, **result['summary']}), flush=True)
    report['pass'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
