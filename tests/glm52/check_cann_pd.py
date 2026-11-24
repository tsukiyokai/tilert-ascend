"""CPU contracts for full-P CANN MTP injection and TP16 delivery."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from check_mtp_tp import (
    FanoutChannel, Hidden, ScalarDraft, ScalarTarget, Token, consume, predict,
    serial_oracle,
)


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.models.glm_5_2_ascend.session import PrefixRequest  # noqa: E402
from tilert.models.glm_5_2_ascend.tp_runtime import (  # noqa: E402
    CoordinatedBackend, RankExecutor,
)
from tilert.pd_vllm.ascend_cache import (  # noqa: E402
    AscendReceiveBuffer, CannRankEngine, FullRankPrefix,
)
from tilert.pd_vllm.ascend_layout import (  # noqa: E402
    PARENT_HIDDEN_REGION, parent_hidden_metadata,
)
from tilert.pd_vllm.ascend_server import (  # noqa: E402
    AscendServingEngine, create_session,
)


class Parents(list):
    def clone(self):
        return Parents(self)


class FullTarget(ScalarTarget):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.predict_calls = 0
        self.verify_calls = 0

    def predict(self, parent):
        expected = Hidden("target", self.rows[-1][1], self.position)
        if parent != [expected]:
            raise ValueError("Transferred parent differs from main prefix")
        self.predict_calls += 1
        return SimpleNamespace(token_ids=[Token(predict(
            expected.digest, expected.length,
        ))])

    def verify(self, tokens, **kwargs):
        self.verify_calls += 1
        return super().verify(tokens, **kwargs)


def full_prefix(request, rank):
    rows, mtp, digest = [], [], 0
    prompt = request.metadata["prompt"]
    for position, token in enumerate(prompt):
        digest = consume(digest, token)
        rows.append((token, digest))
        if position + 1 < len(prompt):
            mtp.append((prompt[position + 1],
                        Hidden("target", digest, position + 1)))
    return FullRankPrefix(
        rows, mtp, Parents([Hidden("target", digest, len(prompt))]),
        len(prompt),
    )


def backend(mode, capacity):
    ranks = [RankExecutor(
        CannRankEngine(FullTarget(capacity), ScalarDraft(capacity, mode)),
        rank, full_prefix,
    ) for rank in range(16)]
    channel = FanoutChannel(ranks[1:])
    return CoordinatedBackend(ranks[0], channel), ranks, channel


class CannPdChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torch_npu

        cls.torch, cls.torch_npu = torch, torch_npu
        torch.set_num_threads(1)
        assert not torch_npu.npu._initialized

    def test_receive_full_main_and_P_minus_one_draft(self):
        torch = self.torch
        receive = AscendReceiveBuffer(0, "cpu", 128, Mock(), "127.0.0.1",
                                      parent_hidden=True)
        receive.storage.fill_(1)
        receive.views[PARENT_HIDDEN_REGION].fill_(7)
        request = PrefixRequest("r", 3, {
            "generation": 1, "mtp_seq_len": 2, "last_prompt_token": 17,
            PARENT_HIDDEN_REGION: parent_hidden_metadata(3),
        })
        prefix = receive.mtp_prefix(request)
        self.assertEqual(prefix.seq_len, 3)
        self.assertEqual(len(prefix.main), 78)
        self.assertTrue(all(layer["latent"].shape == (3, 512)
                            for layer in prefix.main))
        self.assertEqual(prefix.mtp["latent"].shape, (2, 512))
        self.assertEqual(prefix.parent_hidden.shape, (1, 6144))
        self.assertEqual(prefix.parent_hidden.dtype, torch.bfloat16)
        self.assertTrue(torch.all(prefix.parent_hidden == 7))
        with self.assertRaises(ValueError):
            receive.mtp_prefix(request)
        legacy = AscendReceiveBuffer(0, "cpu", 128, Mock(), "127.0.0.1")
        with self.assertRaisesRegex(ValueError, "v3"):
            legacy.mtp_prefix(request)

    def test_injection_owns_parent_and_does_not_replay(self):
        request = PrefixRequest("r", 3, {"prompt": (3, 10, 17)})
        prefix = full_prefix(request, 0)
        engine = CannRankEngine(FullTarget(32), ScalarDraft(32, "accept"))
        engine.inject(prefix)
        self.assertEqual((engine.target.position, engine.draft.position),
                         (3, 2))
        self.assertEqual(engine.target.verify_calls, 0)
        self.assertEqual(engine.target.predict_calls, 1)
        prefix.parent_hidden.clear()
        self.assertEqual(len(engine.parent_hidden), 1)
        engine.reset()
        self.assertIsNone(engine.parent_hidden)
        self.assertEqual((engine.position, engine.draft.position), (0, 0))

    def test_tp16_session_matches_greedy_and_retains_parents(self):
        for mode in ("accept", "reject", "mixed"):
            for width in (2, 4):
                for prompt in ((3,), (3, 10, 17)):
                    with self.subTest(mode=mode, width=width, prompt=prompt):
                        coordinated, ranks, channel = backend(mode, 64)
                        session = create_session(coordinated, width, ())
                        session.inject(PrefixRequest(
                            "r", len(prompt), {"prompt": prompt},
                        ))
                        first = ranks[0].engine.first_token_id
                        self.assertTrue(all(
                            item.engine.target.verify_calls == 0
                            and item.engine.target.predict_calls == 1
                            for item in ranks
                        ))
                        output = session.decode(first, 17)
                        expected, rows = serial_oracle(prompt, 17, 64)
                        self.assertEqual(output, expected)
                        for item in ranks:
                            target, draft = (item.engine.target,
                                             item.engine.draft)
                            self.assertEqual(target.rows[:target.position],
                                             rows)
                            self.assertEqual(target.position, len(rows))
                            self.assertEqual(draft.position, len(rows))
                            for position, (token, parent) in enumerate(
                                draft.rows[:draft.position]
                            ):
                                self.assertEqual(parent, Hidden(
                                    "target", rows[position][1], position + 1,
                                ))
                                next_token = (output[-1]
                                              if position + 1 == len(rows)
                                              else rows[position + 1][0])
                                self.assertEqual(token, next_token)
                        session.reset()
                        self.assertTrue(all(item.engine.parent_hidden is None
                                            for item in ranks))
                        self.assertEqual(channel.commands[-1]["operation"],
                                         "reset")

    def test_first_token_mismatch_precedes_delivery(self):
        coordinated, ranks, _ = backend("accept", 32)
        session = create_session(coordinated, 4, ())
        session.inject(PrefixRequest("r", 1, {"prompt": (3,)}))
        engine = AscendServingEngine(session, ranks[0].engine, "cpu")
        emitted = []
        with self.assertRaisesRegex(ValueError, "decode prefix"):
            engine.decode(ranks[0].engine.first_token_id + 1, max_tokens=4,
                          on_token=emitted.append)
        self.assertEqual(emitted, [])
        self.assertEqual(ranks[0].engine.target.verify_calls, 0)

    def test_actual_predict_only_calls_the_head(self):
        torch = self.torch
        from tilert.models.glm_5_2_ascend.cann_model import CannModel

        model = object.__new__(CannModel)
        model.device, model.position = torch.device("cpu"), 8192
        model.config = SimpleNamespace()
        parent = torch.ones((1, 6144), dtype=torch.bfloat16)
        logits = torch.tensor([[0., -1., 4.]])
        head = Mock(return_value=logits)
        model.model = SimpleNamespace(compute_logits=head)
        with (
            patch.object(torch.npu, "device", return_value=nullcontext()),
            patch("vllm.config.set_current_vllm_config",
                  return_value=nullcontext()) as context,
        ):
            result = model.predict(parent)
            self.assertIsNone(result.logits)
            self.assertEqual(result.token_ids.tolist(), [2])
            self.assertIs(result.last_hidden, parent)
            head.assert_called_once_with(parent)
            context.assert_called_once_with(model.config)
            self.assertEqual(model.position, 8192)
            for value in (parent.float(), parent[:, :1], parent.expand(2, -1)):
                with self.assertRaises(ValueError):
                    model.predict(value)

    def test_http_stream_timing_release_and_next_generation(self):
        from fastapi.testclient import TestClient

        from check_ascend_pd import endpoints
        from tilert.pd_vllm import decode_server, wire
        from tilert.pd_vllm.ascend_layout import cache_layout
        from tilert.pd_vllm.ascend_receiver import AscendReceiveServer
        from tilert.pd_vllm.decode_timing import (
            TimedBackend, TimedEngine, timing_path,
        )

        coordinated, ranks, _ = backend("mixed", 64)
        for item in ranks:
            item.prefix_loader = lambda request, rank: full_prefix(
                PrefixRequest(request.request_id, request.seq_len, {
                    "prompt": request.metadata["prompt_token_ids"],
                }), rank,
            )
        timed = TimedBackend(coordinated)
        session = create_session(timed, 4, ())
        rank_engine = ranks[0].engine
        phases = {"head": [], "verify": []}
        for method, phase in (("predict", "head"), ("verify", "verify")):
            original = getattr(rank_engine.target, method)

            def record(*args, _method=original, _phase=phase, **kwargs):
                phases[_phase].append(threading.current_thread().name)
                return _method(*args, **kwargs)

            setattr(rank_engine.target, method, record)
        layout = cache_layout(128, parent_hidden=True)
        server = AscendReceiveServer(layout, endpoints(layout), {},
                                     host="127.0.0.1", ctrl_port=0)
        prompt = [3, 10, 17]
        parent_contract = parent_hidden_metadata(len(prompt))
        expected, _ = serial_oracle(prompt, 8, 64)
        try:
            with tempfile.TemporaryDirectory() as directory, (
                patch.object(self.torch.npu, "device",
                             return_value=nullcontext())
            ):
                engine = TimedEngine(AscendServingEngine(
                    session, rank_engine, "cpu",
                ), timed, directory)
                with TestClient(decode_server.build_app(server, engine)) as c:
                    for generation, mismatch in enumerate((True, False), 1):
                        rid = f"http-{generation}"
                        for rank in range(16):
                            request = {
                                "rid": rid, "rank": rank, "seq_len": 3,
                                "mtp_seq_len": 2, "last_prompt_token": 17,
                                "prompt_token_ids": prompt,
                                PARENT_HIDDEN_REGION: parent_contract,
                            }
                            reply = server._admit(request)
                            self.assertEqual(reply["generation"], generation)
                            server._complete_rank(request, generation,
                                                  wire.done_msg(
                                                      rid, rank, generation,
                                                  ))
                            server._writer_left(request, generation)
                        response = c.post("/pd/decode", json={
                            "rid": rid, "first_token_id": expected[0]
                            + int(mismatch), "max_tokens": 8, "stream": True,
                        })
                        self.assertEqual(response.status_code, 200)
                        messages = [json.loads(line)
                                    for line in response.text.splitlines()]
                        tokens = [token for item in messages
                                  for token in item.get("t", [])]
                        saved = json.loads(timing_path(directory,
                                                       rid).read_text())
                        if mismatch:
                            self.assertEqual(tokens, [])
                            self.assertIn("decode prefix",
                                          messages[-1]["error"])
                            self.assertEqual(saved["status"], "error")
                            self.assertEqual(phases["verify"], [])
                        else:
                            self.assertEqual(tokens, expected)
                            self.assertTrue(messages[-1]["done"])
                            self.assertEqual(saved["status"], "complete")
                        self.assertEqual(saved["token_ids"], tokens)
                        self.assertTrue(saved["reset_complete"])
                        self.assertEqual(server.state_snapshot()["state"],
                                         "free")
                        self.assertTrue(all(
                            item.engine.parent_hidden is None
                            and item.engine.position == 0
                            and item.engine.draft.position == 0
                            for item in ranks
                        ))
                self.assertTrue(phases["head"])
                self.assertTrue(phases["verify"])
                self.assertNotIn("pd-decode", phases["head"])
                self.assertEqual(set(phases["verify"]), {"pd-decode"})
        finally:
            server.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = unittest.TextTestRunner(stream=io.StringIO()).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CannPdChecks),
    )
    import torch_npu

    sources = [Path(__file__),
               ROOT / "tilert/models/glm_5_2_ascend/cann_model.py",
               ROOT / "tilert/models/glm_5_2_ascend/mtp_round.py",
               ROOT / "tilert/models/glm_5_2_ascend/mtp_session.py",
               ROOT / "tilert/models/glm_5_2_ascend/tp_runtime.py",
               ROOT / "tilert/pd_vllm/ascend_cache.py",
               ROOT / "tilert/pd_vllm/ascend_server.py",
               ROOT / "tilert/pd_vllm/ascend_receiver.py",
               ROOT / "tilert/pd_vllm/decode_server.py",
               ROOT / "tilert/pd_vllm/decode_timing.py",
               Path(__file__).with_name("check_mtp_tp.py")]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Actual full-P injection, TP16 dispatch, MTP session, "
                 "ASGI stream/thread/lease/timing; scalar model arithmetic, "
                 "CPU receive memory and synthetic WRITE completions",
        "tests_run": result.testsRun,
        "status": "pass" if result.wasSuccessful() else "fail",
        "failures": [str(error) for _, error in
                     (*result.failures, *result.errors)],
        "npu_initialized": bool(torch_npu.npu._initialized),
        "precision_verified": False, "performance_measured": False,
        "source_sha256": {str(path.relative_to(ROOT)):
                          hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return int(not result.wasSuccessful() or torch_npu.npu._initialized)


if __name__ == "__main__":
    raise SystemExit(main())
