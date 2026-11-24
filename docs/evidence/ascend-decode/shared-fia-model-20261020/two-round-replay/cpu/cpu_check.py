"""Check actual replay schedule code with frozen inputs and a CPU fake layer."""
import argparse
import ast
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch

parser = argparse.ArgumentParser(description=__doc__)
for name in ("runner", "trace", "teacher", "capture-reports", "output"):
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(1)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
tree = ast.parse(args.runner.read_text())
functions = {node.name: node for node in tree.body
             if isinstance(node, ast.FunctionDef)}
scope = dict(argparse=argparse, Path=Path, hashlib=hashlib, json=json,
             CAPACITY=9344, contextmanager=contextmanager,
             nullcontext=nullcontext, patch=patch)
names = ("arguments", "read_round_trace", "replay_attention", "snapshot_round")
exec(compile(ast.Module([functions[n] for n in names], type_ignores=[]),
             str(args.runner), "exec"), scope)
record = json.loads(args.trace.read_text())
teacher = json.loads(args.teacher.read_text())
inputs = json.loads((args.capture_reports / "rank-0.json").read_text())["inputs"]
checkpoint = record["checkpoint_sha256"]
reference, expected = teacher["teacher_tokens"], teacher["expected_next_tokens"]
tokens, schedule, scored, labels, binding = scope["read_round_trace"](
    args.trace, inputs, checkpoint, reference, expected, 0,
)
checks = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


check("trusted_rows_and_labels", scored == [0, 1, 2, 8]
      and labels == [382, 262, 576, 2524])
check("frozen_commits", [p["position"] for p in schedule] == [8192, 8195]
      and [p["retained"] for p in schedule] == [8195, 8196])
check("input_hash", binding["sha256"] == hashlib.sha256(
    args.trace.read_bytes()).hexdigest())
for rank in range(16):
    capture = json.loads((args.capture_reports / f"rank-{rank}.json").read_text())
    replay = scope["read_round_trace"](
        args.trace, capture["inputs"], capture["checkpoint_sha256"],
        reference, expected, rank,
    )
    check("actual_capture_identity_rank" + str(rank), replay[:4]
          == (tokens, schedule, scored, labels))
with TemporaryDirectory() as temp:
    for field in ("retained", "accepted_tokens"):
        bad = json.loads(args.trace.read_text())
        row = bad["rank_requests"][bad["warmup"]]["rounds"][0]
        row[field] = 8196 if field == "retained" else [6500]
        path = Path(temp) / "bad.json"
        path.write_text(json.dumps(bad))
        try:
            scope["read_round_trace"](path, inputs, checkpoint,
                                      reference, expected, 0)
        except ValueError:
            check("reject_bad_" + field, True)
        else:
            raise AssertionError("Malformed trace accepted")

layer_tree = functions["run_layer"]
position_node = next(n for n in ast.walk(layer_tree)
                     if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id == "positions")
loop = next(n for n in ast.walk(layer_tree) if isinstance(n, ast.For)
            and isinstance(n.target, ast.Name) and n.target.id == "offset")
loop_code = compile(ast.Module([position_node, loop], type_ignores=[]),
                    str(args.runner), "exec")


def run_loop(corrupt=False, clear_tail=False, plans=schedule, width=8,
             length_delta=0, indexer_short=False, wrong_table=False):
    count = 16 if plans else 32
    cache = {"latent": torch.zeros(9344, 512, dtype=torch.bfloat16),
             "rope": torch.zeros(9344, 64, dtype=torch.bfloat16),
             "index_keys": torch.zeros(9344, 128, dtype=torch.int8),
             "index_scales": torch.zeros(9344, dtype=torch.float16)}
    engine = SimpleNamespace(
        device="cpu", config=None, model=None, cache=[cache],
        input_ids=torch.empty(width, dtype=torch.int32),
        norm_units=None, replay_progress={},
        block_table=torch.arange(73, dtype=torch.int32)[None, :],
    )

    def metadata(size):
        engine.positions = torch.arange(engine.position, engine.position + size)
        engine.slots = engine.positions.int()
        engine.seq_lens = torch.tensor([engine.position + size])
        return {"model.layers.3.self_attn.indexer.k_cache": SimpleNamespace(
            actual_seq_lengths_query=torch.tensor([size]),
            actual_seq_lengths_key=torch.tensor([engine.position + size]),
            block_table=engine.block_table)}

    def truncate(retained):
        if clear_tail:
            for value in cache.values():
                value[retained:engine.position].zero_()
        engine.position = retained

    engine._metadata, engine.truncate = metadata, truncate
    hidden = torch.arange(count).bfloat16()[:, None].expand(-1, 6144).clone()
    residual = hidden + 100
    shared = torch.arange(2048, dtype=torch.int32).expand(count, -1).clone()
    shared[8:] += 2048
    buffer = torch.empty(width, 2048, dtype=torch.int32)
    calls = []
    impl = SimpleNamespace(_execute_sparse_flash_attention_process=
                           lambda *a, **k: None)

    def forward(positions, h, r):
        call = len(calls)
        rows = slice(call * width, (call + 1) * width)
        if not torch.equal(buffer, shared[rows]):
            raise AssertionError("Consumer received the other round's indices")
        if plans and call == 1 and not clear_tail:
            if cache["latent"][8195:8200, 0].tolist() != list(range(13, 18)):
                raise AssertionError("Rejected tail was cleared before overwrite")
        calls.append((positions.tolist(), h.clone(), r.clone()))
        for value in cache.values():
            for row in range(width):
                value[engine.position + row].fill_((10 + call * 10 + row) % 100)
            if corrupt and call == 1:
                value[8192].fill_(99)
        indexer = env["metadata"]["model.layers.3.self_attn.indexer.k_cache"]
        if indexer_short and call == 1:
            indexer.actual_seq_lengths_key -= 1
        table = engine.block_table.clone()
        if wrong_table and call == 1:
            table[0, 0] = 1
        md = SimpleNamespace(block_table=table)
        impl._execute_sparse_flash_attention_process(
            None, None, None, buffer, md, torch.tensor([width]),
            torch.tensor([engine.position + width +
                          (length_delta if call == 1 else 0)]),
        )
        return h + 1, r + 2

    forward.self_attn = SimpleNamespace(
        topk_indices_buffer=buffer,
        mla_attn=SimpleNamespace(mla_attn=SimpleNamespace(impl=impl)))
    fake_torch = SimpleNamespace(tensor=torch.tensor, ops=SimpleNamespace(
        _C_ascend=SimpleNamespace(npu_add_rms_norm_bias=
                                 lambda h, r, *a: (h, None, h + r))))
    env = dict(scope, torch=fake_torch, engine=engine, layer=forward,
               teacher=tokens if plans else reference, width=width,
               schedule=plans, number=3, skip_topk=True, shared=shared,
               hidden=hidden, residual=residual, next_hidden=[],
               next_residual=[], merged=[], call_snapshots=[], protected=None,
               checks=[], prefix={}, set_ascend_forward_context=
               lambda *a, **k: nullcontext(), check_prefix=lambda *a: None)
    exec(loop_code, env)
    check("H_R_slices_" + str((corrupt, clear_tail, width)),
          torch.equal(torch.cat(env["next_hidden"]), hidden + 1)
          and torch.equal(torch.cat(env["next_residual"]), residual + 2))
    return env, calls


env, calls = run_loop()
check("overlapping_positions", [c[0] for c in calls] ==
      [list(range(8192, 8200)), list(range(8195, 8203))])
check("all_snapshot_gates", all(c["passed"] for c in env["checks"]))
check("logical_final_position", env["engine"].position == 8196)
check("immutable_call0_snapshot", env["call_snapshots"][0]["cache_rows"]
      ["latent"][:, 0].tolist() == list(range(10, 18)))
check("round1_progress", env["engine"].replay_progress["round"] == 1)
check("actual_length_snapshots", [x["attention"]["sfa"]["key"]
      for x in env["call_snapshots"]] == [[8200], [8203]])
for delta in (-1, 1):
    bad, _ = run_loop(length_delta=delta)
    check("reject_SFA_length_delta_" + str(delta), any(not c["passed"]
          and c["name"] == "round1.sfa.lengths" for c in bad["checks"]))
bad, _ = run_loop(indexer_short=True)
check("reject_indexer_short_read", any(not c["passed"]
      and c["name"] == "round1.indexer.lengths" for c in bad["checks"]))
bad, _ = run_loop(wrong_table=True)
check("reject_wrong_SFA_block_table", any(not c["passed"]
      and c["name"] == "round1.sfa.block_table" for c in bad["checks"]))
bad, _ = run_loop(corrupt=True)
check("reject_accepted_KV_mutation", any(not c["passed"] and
      "accepted_prefix" in c["name"] for c in bad["checks"]))
bad, _ = run_loop(clear_tail=True)
check("reject_tail_clear", any(not c["passed"] and
      "rejected_tail_preserved" in c["name"] for c in bad["checks"]))
for width in (1, 32):
    normal, calls = run_loop(plans=None, width=width)
    check("ordinary_teacher_positions_M" + str(width), normal["positions"]
          == list(range(8192, 8224)) and len(calls) == 32 // width)
causal = next(n for n in ast.walk(layer_tree) if isinstance(n, ast.For)
              and isinstance(n.target, ast.Tuple)
              and isinstance(n.iter, ast.Call)
              and isinstance(n.iter.args[0], ast.Name)
              and n.iter.args[0].id == "choices")
env["torch"], env["checks"] = torch, []
env["choices"] = torch.arange(2048).expand(16, -1).clone()
env["choices"][8, -1] = 8196
exec(compile(ast.Module([causal], type_ignores=[]), str(args.runner), "exec"), env)
check("reject_round1_future_ID", any(c["name"] == "token8.causal_indices"
      and not c["passed"] for c in env["checks"]))
execute_nodes = list(ast.walk(functions["execute"]))
scoring = next(n for n in execute_nodes if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name)
               and n.targets[0].id == "scored_predictions")
label_name = next(n for n in execute_nodes if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id == "label_check")
label_gate = next(n for n in execute_nodes if isinstance(n, ast.Expr)
                  and "label_check" in ast.dump(n)
                  and "scored_predictions" in ast.dump(n))
score_code = compile(ast.Module([scoring, label_name, label_gate],
                               type_ignores=[]), str(args.runner), "exec")
predictions = [-1] * 16
for row, label in zip(scored, labels):
    predictions[row] = label
head = dict(predictions=predictions, scored_rows=scored, expected=labels,
            round_trace=binding, checks=[])
exec(score_code, head)
check("only_trusted_rows_scored", head["checks"][0]["passed"])
predictions[8] = 6500
head["checks"] = []
exec(score_code, head)
check("strict_near_tie_label_gate", not head["checks"][0]["passed"])
check("no_NPU_import", "torch_npu" not in sys.modules)
result = dict(status="cpu_contract_pass", checks=checks,
              runner_sha256=hashlib.sha256(args.runner.read_bytes()).hexdigest(),
              check_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              trace_sha256=binding["sha256"], scored_rows=scored, labels=labels,
              scope="Actual replay loop with CPU fake layer; no CANN numerics or Graph validation")
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
