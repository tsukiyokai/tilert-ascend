"""Check online lookup with CPU receive memory and sixteen scalar ranks."""

import argparse
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from check_cann_pd import FullTarget, full_prefix  # noqa: E402
from check_mtp_tp import FanoutChannel, serial_oracle  # noqa: E402
import serve_cann_pd as serving  # noqa: E402
from tilert.models.glm_5_2_ascend.session import PrefixRequest  # noqa: E402
from tilert.models.glm_5_2_ascend.tp_runtime import (  # noqa: E402
    CoordinatedBackend, RankExecutor,
)
from tilert.pd_vllm.ascend_cache import AscendReceiveBuffer  # noqa: E402
from tilert.pd_vllm.ascend_layout import (  # noqa: E402
    PARENT_HIDDEN_REGION, parent_hidden_metadata,
)
from tilert.pd_vllm.ascend_server import (  # noqa: E402
    AscendServingEngine, create_session,
)


def arguments(*extra):
    return serving.parse_arguments([
        "--checkpoint", "/checkpoint", "--hixl-host", "127.0.0.1",
        "--hixl-library", "/hixl.so", "--output", "/reports", *extra,
    ])


def lookup_prefix(request, rank):
    prompt = request.metadata["prompt_token_ids"]
    prefix = full_prefix(PrefixRequest(request.request_id, request.seq_len,
                                       {"prompt": prompt}), rank)
    return replace(prefix, prompt_token_ids=tuple(prompt))


def backend(width=32):
    args = arguments("--lookup", "--mtp-width", str(width))
    ranks = []
    for rank in range(16):
        engine, draft = serving.create_rank_engine(
            args, FullTarget(128), Mock(),
        )
        assert draft is None
        ranks.append(RankExecutor(engine, rank, lookup_prefix))
    channel = FanoutChannel(ranks[1:])
    return CoordinatedBackend(ranks[0], channel), ranks


class GraphTarget(FullTarget):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.graph = None
        self.graph_widths = []

    def enable_graph(self, width):
        self.graph_widths.append(width)
        self.graph = SimpleNamespace(width=width)


class CannLookupPdChecks(unittest.TestCase):
    def test_cli_and_model_constructor_reach_the_wide_configuration(self):
        args = arguments("--lookup", "--mtp-width", "32", "--target-graph",
                         "--compile-target", "--super-kernel", "--fp32-head")
        with patch(
            "tilert.models.glm_5_2_ascend.cann_model.CannModel",
        ) as model:
            serving.create_model(args, "npu:1", 0)
        self.assertEqual(model.call_args.kwargs, {
            "capacity": 9344, "gpu_memory_utilization": 0.4,
            "compile_target": True, "super_kernel": True,
            "fp32_head": True, "graph_width": 32,
        })
        for width in (2, 4, 6, 8, 12, 16, 24, 32):
            self.assertEqual(arguments("--lookup", "--mtp-width",
                                       str(width)).mtp_width, width)
        for options in (("--lookup",), ("--mtp-width", "32"),
                        ("--super-kernel",),
                        ("--lookup", "--mtp-width", "32", "--super-kernel"),
                        ("--lookup", "--mtp-width", "2", "--target-graph",
                         "--compile-target"),
                        ("--lookup", "--mtp-width", "32",
                         "--load-model-after", "/gate")):
            with (
                self.subTest(options=options),
                patch("sys.stderr", io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    arguments(*options)

    def test_lookup_loads_no_mtp_model_and_legacy_modes_keep_their_model(self):
        target = FullTarget(128)
        with patch("tilert.models.glm_5_2_ascend.cann_mtp.CannMtp") as draft:
            rank, model = serving.create_rank_engine(
                arguments("--lookup", "--mtp-width", "32"), target, Mock(),
            )
            self.assertIsNone(model)
            self.assertIsNone(rank.draft)
            draft.assert_not_called()
            serial, model = serving.create_rank_engine(
                arguments(), target, Mock(),
            )
            self.assertIs(serial, target)
            self.assertIsNone(model)
            serving.create_rank_engine(arguments("--mtp-width", "4"),
                                       target, Mock())
            draft.assert_called_once_with(target)

    def test_receive_requires_valid_request_owned_prompt(self):
        receive = AscendReceiveBuffer(0, "cpu", 128, Mock(), "127.0.0.1",
                                      parent_hidden=True)
        metadata = {
            "generation": 1, "mtp_seq_len": 2, "last_prompt_token": 17,
            PARENT_HIDDEN_REGION: parent_hidden_metadata(3),
        }
        for tokens in (None, [3, 17], [3, True, 17], [3, 154880, 17],
                       [3, 10, 16]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                receive.mtp_prefix(PrefixRequest("r", 3, {
                    **metadata, "prompt_token_ids": tokens,
                }), lookup=True)
            self.assertEqual(receive._last_generation, 0)
        prompt = [3, 10, 17]
        prefix = receive.mtp_prefix(PrefixRequest("r", 3, {
            **metadata, "prompt_token_ids": prompt,
        }), lookup=True)
        prompt[0] = 99
        self.assertEqual(prefix.prompt_token_ids, (3, 10, 17))
        self.assertEqual(prefix.parent_hidden.shape, (1, 6144))

    def test_tp16_changes_prompt_and_clears_the_previous_request(self):
        coordinated, ranks = backend()
        session = create_session(coordinated, 32, ())
        for number, prompt in enumerate(([3, 10, 17], [1, 9, 2, 1], [3]), 1):
            session.inject(PrefixRequest(str(number), len(prompt), {
                "prompt_token_ids": prompt,
            }))
            expected, rows = serial_oracle(prompt, 40, 128)
            self.assertTrue(all(rank.engine._request_prompt == tuple(prompt)
                                for rank in ranks))
            self.assertEqual(session.decode(expected[0], 40), expected)
            for rank in ranks:
                self.assertEqual(
                    rank.engine.target.rows[:rank.engine.position], rows,
                )
                self.assertIsNone(rank.engine.draft)
            session.reset()
            self.assertTrue(all(
                rank.engine._request_prompt is None
                and rank.engine.position == 0 for rank in ranks
            ))

    def test_graph_warmup_restores_prefix_and_capture_is_reused(self):
        args = arguments("--lookup", "--mtp-width", "32", "--target-graph")
        target, barrier = GraphTarget(128), Mock()
        engine, _ = serving.create_rank_engine(args, target, barrier)
        prompt = [3, 10, 17]
        request = PrefixRequest("a", 3, {"prompt_token_ids": prompt})
        prefix = lookup_prefix(request, 0)
        observations = []
        engine.on_inject = lambda: observations.append(
            (target.position, target.graph),
        )
        engine.inject(prefix)
        self.assertEqual(target.verify_calls, 4)
        self.assertEqual(target.graph_widths, [32])
        self.assertEqual(barrier.call_count, 2)
        self.assertEqual(target.position, len(prompt))
        self.assertEqual(target.rows[:target.position], prefix.main)
        engine.reset()
        engine.inject(prefix)
        self.assertEqual(target.verify_calls, 4)
        self.assertEqual(target.graph_widths, [32])
        self.assertEqual(observations, [(3, target.graph)] * 2)

    def test_short_capacity_uses_the_eager_tail_without_capturing(self):
        args = arguments("--lookup", "--mtp-width", "32", "--target-graph")
        target, barrier = GraphTarget(128), Mock()
        engine, _ = serving.create_rank_engine(args, target, barrier)
        prompt = [index % 251 for index in range(127)]
        prefix = lookup_prefix(PrefixRequest("tail", 127, {
            "prompt_token_ids": prompt,
        }), 0)
        engine.inject(prefix)
        expected, _ = serial_oracle(prompt, 2, 128)
        engine.start_mtp(expected[0], 2, 32, ())
        result = engine.advance_mtp()
        self.assertEqual(result.commit.accepted_tokens, (expected[1],))
        self.assertEqual(result.commit.width, 1)
        self.assertIsNone(target.graph)
        barrier.assert_not_called()

    def test_http_stream_checks_first_token_releases_and_reuses_the_slot(self):
        from fastapi.testclient import TestClient

        from check_ascend_pd import endpoints
        from tilert.pd_vllm import decode_server, wire
        from tilert.pd_vllm.ascend_layout import cache_layout
        from tilert.pd_vllm.ascend_receiver import AscendReceiveServer
        from tilert.pd_vllm.decode_timing import (
            TimedBackend, TimedEngine, timing_path,
        )

        coordinated, ranks = backend()
        timed = TimedBackend(coordinated)
        session = create_session(timed, 32, ())
        layout = cache_layout(128, parent_hidden=True)
        server = AscendReceiveServer(layout, endpoints(layout), {},
                                     host="127.0.0.1", ctrl_port=0,
                                     require_prompt_token_ids=True)
        try:
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.object(torch, "npu", SimpleNamespace(
                    device=lambda _: nullcontext()), create=True),
                patch.object(decode_server.logger, "info"),
                patch.object(decode_server.logger, "exception"),
            ):
                engine = TimedEngine(AscendServingEngine(
                    session, ranks[0].engine, "cpu",
                ), timed, directory)
                app = decode_server.build_app(server, engine)
                with TestClient(app) as client:
                    prompts = ([3, 10, 17], [1, 9, 2, 1])
                    for generation, prompt in enumerate(prompts, 1):
                        rid = f"lookup-{generation}"
                        for rank in range(16):
                            request = {
                                "rid": rid, "rank": rank,
                                "seq_len": len(prompt),
                                "mtp_seq_len": len(prompt) - 1,
                                "last_prompt_token": prompt[-1],
                                "prompt_token_ids": prompt,
                                PARENT_HIDDEN_REGION: parent_hidden_metadata(
                                    len(prompt)),
                            }
                            if rank == 0:
                                incomplete = dict(request)
                                incomplete.pop("prompt_token_ids")
                                rejected = server._admit(incomplete)
                                self.assertEqual(rejected["error"],
                                                 "invalid_prompt_token_ids")
                                self.assertFalse(coordinated.failed)
                                self.assertEqual(
                                    server.state_snapshot()["state"], "free",
                                )
                            reply = server._admit(request)
                            self.assertEqual(reply["generation"], generation)
                            server._complete_rank(
                                request, generation,
                                wire.done_msg(rid, rank, generation),
                            )
                            server._writer_left(request, generation)
                        expected, _ = serial_oracle(prompt, 12, 128)
                        mismatch = generation == 1
                        response = client.post("/pd/decode", json={
                            "rid": rid,
                            "first_token_id": expected[0] + int(mismatch),
                            "max_tokens": 12, "stream": True,
                        })
                        self.assertEqual(response.status_code, 200)
                        messages = [json.loads(line)
                                    for line in response.text.splitlines()]
                        tokens = [token for message in messages
                                  for token in message.get("t", [])]
                        saved = json.loads(
                            timing_path(directory, rid).read_text(),
                        )
                        self.assertEqual(tokens, [] if mismatch else expected)
                        self.assertEqual(saved["status"],
                                         "error" if mismatch else "complete")
                        self.assertTrue(saved["reset_complete"])
                        self.assertEqual(server.state_snapshot()["state"],
                                         "free")
        finally:
            server.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CannLookupPdChecks),
    )
    paths = (
        "tilert/pd_vllm/ascend_cache.py", "tilert/pd_vllm/ascend_layout.py",
        "tilert/pd_vllm/ascend_receiver.py", "tilert/pd_vllm/ascend_server.py",
        "tilert/pd_vllm/decode_server.py", "tilert/pd_vllm/decode_timing.py",
        "tilert/models/glm_5_2_ascend/mtp_round.py",
        "tilert/models/glm_5_2_ascend/mtp_session.py",
        "tilert/models/glm_5_2_ascend/tp_runtime.py",
    )
    sources = [Path(__file__), Path(serving.__file__),
               *(ROOT / name for name in paths)]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if result.wasSuccessful() else "fail",
        "scope": "CPU tensors, real HTTP/TP16 control, scalar target, "
                 "synthetic graph and WRITE completions; no NPU execution",
        "tests_run": result.testsRun, "test_output": stream.getvalue(),
        "precision_verified": False, "performance_measured": False,
        "torch": torch.__version__,
        "torch_npu_imported": "torch_npu" in sys.modules,
        "source_sha256": {str(path.relative_to(ROOT)):
                          hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(stream.getvalue())
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())
