"""CPU selection contract for the installed vocab-parallel argmax method."""

import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


SOURCE = Path('/workspace/src/vllm-unified/vllm/model_executor/layers/logits_processor.py')
OUT = Path('/workspace/offline/native-greedy-head-cpu-20261020')
SHARD, RANKS = 9680, 16
VOCAB = SHARD * RANKS
torch.set_num_threads(1)
assert not any(name.startswith('torch_npu') for name in sys.modules)
tree = ast.parse(SOURCE.read_text())
cls = next(node for node in tree.body
           if isinstance(node, ast.ClassDef) and node.name == 'LogitsProcessor')
method = next(node for node in cls.body
              if isinstance(node, ast.FunctionDef) and node.name == 'get_top_tokens')
namespace = {'torch': torch, 'VocabParallelEmbedding': object}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'),
     namespace)
select = namespace['get_top_tokens']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_case(name, width):
    logits = torch.full((width, VOCAB), -9.0, dtype=torch.float32)
    expected = []
    for row in range(width):
        first = (row % 8 + 2) * SHARD + 13
        if name == 'unique':
            logits[row, first] = 3.0
        elif name == 'cross_rank_tie':
            logits[row, first] = logits[row, first + SHARD + 2] = 3.0
        elif name == 'within_rank_tie':
            logits[row, first] = logits[row, first + 1] = 3.0
        elif name == 'all_negative':
            logits[row].fill_(-1.0)
            first = 0
        elif name == 'signed_zero':
            logits[row].fill_(0.0)
            logits[row, ::2] = -0.0
            first = 0
        elif name == 'positive_inf':
            logits[row, first] = logits[row, first + SHARD] = float('inf')
        elif name == 'negative_inf':
            logits[row].fill_(-float('inf'))
            first = 0
        elif name == 'one_nan':
            logits[row, first] = float('nan')
        elif name == 'many_nan':
            logits[row, first] = logits[row, first + 1] = float('nan')
            logits[row, first + SHARD] = float('nan')
        elif name == 'nan_and_inf':
            logits[row, 0] = float('inf')
            logits[row, first] = float('nan')
        elif name == 'highest_legal_id':
            first = VOCAB - 1
            logits[row, first] = 3.0
        elif name == 'fp32_near_tie':
            logits[row, first] = 1.0
            first += SHARD
            logits[row, first] = 1.0 + 2**-21
        else:
            raise ValueError(name)
        expected.append(first)
    return logits, torch.tensor(expected, dtype=torch.int64)


class Collected(Exception):
    pass


cases = []
for width in (1, 8, 32):
    hidden = torch.zeros((width, 6144), dtype=torch.bfloat16)
    for name in ('unique', 'cross_rank_tie', 'within_rank_tie', 'all_negative',
                 'signed_zero', 'positive_inf', 'negative_inf', 'one_nan',
                 'many_nan', 'nan_and_inf', 'highest_legal_id', 'fp32_near_tie'):
        full, expected = make_case(name, width)
        shards = list(full.split(SHARD, dim=-1))
        reference = torch.cat(shards, dim=-1).argmax(dim=-1)
        assert torch.equal(reference, expected)
        pairs = []

        def collect(pair, dim=-1):
            assert dim == -1 and pair.shape == (width, 2)
            pairs.append(pair.clone())
            raise Collected

        namespace['tensor_model_parallel_all_gather'] = collect
        processors, heads = [], []
        for rank in range(RANKS):
            head = SimpleNamespace(tp_size=RANKS, shard_indices=SimpleNamespace(
                num_org_vocab_padding=0, org_vocab_start_index=rank * SHARD))
            processor = SimpleNamespace(scale=1.0, soft_cap=None)
            processor._apply_head = lambda head, hidden, bias, r=rank: shards[r].clone()
            processors.append(processor)
            heads.append(head)
            try:
                select(processor, head, hidden)
            except Collected:
                pass
        assert len(pairs) == RANKS
        gathered = torch.cat(pairs, dim=-1)
        for rank in range(RANKS):
            def gather(pair, dim=-1, r=rank):
                assert dim == -1
                assert torch.equal(pair.view(torch.int32), pairs[r].view(torch.int32))
                return gathered.clone()

            namespace['tensor_model_parallel_all_gather'] = gather
            result = select(processors[rank], heads[rank], hidden)
            assert result.dtype == torch.int64
            assert torch.equal(result.view(torch.uint8), reference.view(torch.uint8))
        cases.append({'width': width, 'case': name, 'all_16_ranks_bit_equal': True,
                      'expected_ids': expected.tolist(),
                      'fixed_logits_sha256': hashlib.sha256(
                          full.numpy().tobytes()).hexdigest()})

paths = [SOURCE, Path('/workspace/src/vllm-ascend-unified/vllm_ascend/ops/vocab_parallel_embedding.py'),
         Path('/workspace/src/vllm-unified/vllm/model_executor/layers/vocab_parallel_embedding.py'),
         Path('/workspace/src/vllm-unified/vllm/model_executor/models/deepseek_mtp.py'),
         Path('/workspace/experiments/teacher-routes-20261020/source-7747b5c/tilert/models/glm_5_2_ascend/cann_head.py'),
         Path('/workspace/models/GLM-5.2-w4a8c8/config.json')]
report = {'scope': 'CPU execution of the actual installed get_top_tokens method; '
                  'only projection logits and gather are simulated. No NPU reduction, '
                  'GEMM, HCCL, Graph, or performance validation.',
          'npu_launched': False, 'torch_npu_imported': False,
          'method_lines': [method.lineno, method.end_lineno],
          'method_source_sha256': hashlib.sha256(ast.get_source_segment(
              SOURCE.read_text(), method).encode()).hexdigest(),
          'fixed_contract': {'tp_size': RANKS, 'shard_rows': SHARD, 'vocab_size': VOCAB,
                             'local_logits_dtype': 'float32', 'scale': 1.0,
                             'soft_cap': None, 'padding': 0, 'added_vocab': 0,
                             'lora': False, 'parallel_mode': 'STANDARD'},
          'case_count': len(cases), 'all_passed': True, 'cases': cases,
          'source_sha256': {str(path): digest(path) for path in paths},
          'script_sha256': digest(Path(__file__))}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: report[key] for key in
                  ('scope', 'case_count', 'all_passed', 'source_sha256')}, indent=2))
