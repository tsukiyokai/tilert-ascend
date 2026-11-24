"""Check TP16 receive ownership, prompt replay, and token wakeups on CPU."""

import argparse
import ast
import asyncio
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import platform
import socket
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.pd_vllm import wire  # noqa: E402
from tilert.pd_vllm.ascend_cache import (  # noqa: E402
    AscendRankEngine, AscendReceiveBuffer, RankPrefix, receive_device_id,
)
from tilert.pd_vllm.ascend_layout import cache_layout  # noqa: E402
from tilert.pd_vllm.ascend_receiver import (  # noqa: E402
    AscendReceiveServer, validate_endpoints,
)
from tilert.pd_vllm.ascend_server import (  # noqa: E402
    AscendDeviceContext, AscendServingEngine,
)
from tilert.pd_vllm.token_queue import TokenQueue, finish_worker  # noqa: E402
from tilert.models.glm_5_2_ascend.session import PrefixRequest  # noqa: E402


def endpoints(layout):
    return [{
        "rank": rank, "base_ptr": 1048576, "transport": "hixl",
        "hixl_engine": f"127.0.0.1:{26100 + rank}",
        "hixl_device": receive_device_id(rank),
        "hixl_regions": [{"base": 1048576, "bytes": layout["nbytes"]}],
    } for rank in range(16)]


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for receiver state")
        time.sleep(0.001)


class ReceiveChecks(unittest.TestCase):
    def test_receive_devices_form_a_shifted_tp16_permutation(self):
        self.assertEqual([receive_device_id(rank) for rank in range(16)],
                         [*range(1, 16), 0])

    def setUp(self):
        self.layout = cache_layout(128)
        self.server = AscendReceiveServer(
            self.layout, endpoints(self.layout), {"config.json": "test"},
            host="127.0.0.1", ctrl_port=0, request_timeout=1,
        )
        self.connections = []

    def tearDown(self):
        for connection in self.connections:
            connection.close()
        self.server.close()

    def connect(self, rank, rid="request", **overrides):
        connection = socket.create_connection(
            ("127.0.0.1", self.server.ctrl_port), timeout=2,
        )
        self.connections.append(connection)
        hello = wire.recv_msg(connection)
        request = {
            "rid": rid, "rank": rank, "seq_len": 3, "mtp_seq_len": 2,
            "last_prompt_token": 17, "sampling": {"temperature": 0},
            **overrides,
        }
        wire.send_msg(connection, request)
        return connection, hello, wire.recv_msg(connection)

    def finish(self, connection, rank, admission, rid="request"):
        wire.send_msg(connection, wire.done_msg(
            rid, rank, admission["generation"],
        ))
        received = wire.recv_msg(connection)
        self.assertEqual(received, {
            "received": True, "rid": rid, "rank": rank,
            "generation": admission["generation"],
        })
        connection.close()

    def test_sixteen_completions_and_release_own_the_slot(self):
        order = list(range(15, -1, -1))
        admitted = [self.connect(rank) for rank in order]
        self.assertEqual(len(admitted[0][1]["rank_endpoints"]), 16)
        self.assertEqual(len(admitted[0][1]["ascend_layout"]["regions"]), 196)
        for rank, (connection, _, reply) in zip(order[:-1], admitted[:-1]):
            self.finish(connection, rank, reply)
            self.assertTrue(self.server.completed.empty())
        self.finish(admitted[-1][0], order[-1], admitted[-1][2])
        request = self.server.completed.get(timeout=2)
        self.assertEqual(request.done_ranks, set(range(16)))
        descriptor = self.server.profile.convert(None, None, 128, request, 16)
        self.assertEqual((descriptor.seq_len, descriptor.metadata), (3, {
            "generation": 1, "mtp_seq_len": 2, "last_prompt_token": 17,
        }))
        _, _, rejected = self.connect(0, "next")
        self.assertEqual(rejected["error"], "busy")
        self.server.release("request")
        wait_for(lambda: self.server.state_snapshot()["state"] == "free")
        connection, _, second = self.connect(0, "next")
        self.assertEqual(second["generation"], 2)
        self.finish(connection, 0, second, "next")
        self.server.release("next")
        wait_for(lambda: self.server.state_snapshot()["state"] == "free")

    def test_wrong_generation_cannot_publish_or_free_memory(self):
        connection, _, admission = self.connect(0)
        wire.send_msg(connection, wire.done_msg(
            "request", 0, admission["generation"] + 1,
        ))
        with self.assertRaises(ConnectionError):
            wire.recv_msg(connection)
        wait_for(lambda: self.server.state_snapshot()["state"] == "failed")
        self.assertTrue(self.server.completed.empty())
        self.server.release("request")
        _, _, rejected = self.connect(0, "next")
        self.assertEqual(rejected["error"], "busy")

    def test_disconnected_writer_keeps_memory_quarantined(self):
        connection, _, _ = self.connect(0)
        connection.close()
        wait_for(lambda: self.server.state_snapshot()["state"] == "failed")
        self.server.release("request")
        _, _, rejected = self.connect(1, "next")
        self.assertEqual(rejected["busy_state"], "failed")

    def test_cancel_waits_for_admitted_write_completion(self):
        connection, _, admission = self.connect(0)
        self.server.release("request")
        self.assertEqual(self.server.state_snapshot()["state"], "cancelling")
        _, _, rejected = self.connect(1)
        self.assertEqual(rejected["error"], "cancelling")
        _, _, rejected = self.connect(0, "next")
        self.assertEqual(rejected["error"], "busy")
        self.finish(connection, 0, admission)
        wait_for(lambda: self.server.state_snapshot()["state"] == "free")
        self.assertTrue(self.server.completed.empty())
        _, _, rejected = self.connect(1)
        self.assertEqual(rejected["error"], "cancelling")

    def test_rank_metadata_must_agree(self):
        connection, _, admission = self.connect(0)
        _, _, rejected = self.connect(1, seq_len=4, mtp_seq_len=3)
        self.assertEqual(rejected["error"], "rank_metadata_mismatch")
        _, _, rejected = self.connect(1, last_prompt_token=18)
        self.assertEqual(rejected["error"], "rank_metadata_mismatch")
        self.finish(connection, 0, admission)
        self.server.release("request")

    def test_only_one_writer_per_rank(self):
        connection, _, admission = self.connect(0)
        _, _, rejected = self.connect(0)
        self.assertEqual(rejected["error"], "duplicate_rank")
        self.finish(connection, 0, admission)
        self.server.release("request")

    def test_invalid_rank_and_lengths_are_not_admitted(self):
        cases = (
            ({"rank": -1}, "invalid_rank"),
            ({"rank": 16}, "invalid_rank"),
            ({"rank": True}, "invalid_rank"),
            ({"seq_len": 0, "mtp_seq_len": 0}, "invalid_prefix_length"),
            ({"seq_len": 129}, "invalid_prefix_length"),
            ({"mtp_seq_len": 3}, "MTP prefix must contain P-1 parent rows"),
            ({"last_prompt_token": 154880}, "invalid_last_prompt_token"),
        )
        for overrides, error in cases:
            with self.subTest(overrides=overrides):
                rank = overrides.get("rank", 0)
                overrides = dict(overrides)
                overrides.pop("rank", None)
                _, _, rejected = self.connect(rank, **overrides)
                self.assertEqual(rejected["error"], error)
        self.assertEqual(self.server.state_snapshot()["state"], "free")

    def test_virtual_addresses_are_rank_local(self):
        peers = endpoints(self.layout)
        validate_endpoints(peers, self.layout)
        peers[1]["hixl_engine"] = peers[0]["hixl_engine"]
        with self.assertRaises(ValueError):
            validate_endpoints(peers, self.layout)
        peers = endpoints(self.layout)
        peers[1]["hixl_regions"][0]["bytes"] -= 1
        with self.assertRaises(ValueError):
            validate_endpoints(peers, self.layout)

    def test_pre_http_cancel_cannot_be_reannounced(self):
        self.server.release("request")
        self.assertFalse(self.server.expect("request"))
        _, _, rejected = self.connect(0)
        self.assertEqual(rejected["error"], "cancelling")
        self.assertEqual(self.server.state_snapshot()["state"], "free")
        self.assertTrue(self.server.expect("next"))

    def test_complete_prompt_identity_reaches_the_descriptor(self):
        prompt = [3, 10, 17]
        admitted = [self.connect(rank, prompt_token_ids=prompt)
                    for rank in range(16)]
        for rank, (connection, _, reply) in enumerate(admitted):
            self.finish(connection, rank, reply)
        received = self.server.completed.get(timeout=2)
        descriptor = self.server.profile.convert(
            None, None, 128, received, 16,
        )
        self.assertEqual(descriptor.metadata["prompt_token_ids"], prompt)
        self.assertIsNot(descriptor.metadata["prompt_token_ids"],
                         received.prompt_token_ids)
        self.server.release(received.rid)

    def test_invalid_prompt_ids_do_not_reserve_the_slot(self):
        cases = (
            None, [], [3, 17], [3, 10, 17, 17], "3,10,17",
            [True, 10, 17], [3, 10.0, 17], [-1, 10, 17],
            [154880, 10, 17], [3, 10, 18],
        )
        for tokens in cases:
            with self.subTest(tokens=tokens):
                _, _, rejected = self.connect(0, prompt_token_ids=tokens)
                self.assertEqual(rejected["error"], "invalid_prompt_token_ids")
                self.assertEqual(self.server.state_snapshot()["state"],
                                 "free")

    def test_rank_prompt_conflict_is_rejected_before_write(self):
        connection, _, admission = self.connect(
            0, prompt_token_ids=[3, 10, 17],
        )
        _, _, rejected = self.connect(1, prompt_token_ids=[4, 10, 17])
        self.assertEqual(rejected["error"], "rank_metadata_mismatch")
        self.assertEqual(self.server.state_snapshot()["active_writers"], 1)
        self.finish(connection, 0, admission)
        self.server.release("request")


class LayoutView:
    """Track view shapes and aliases without allocating a tensor."""

    def __init__(self, region, shape=None, storage=None):
        self.region = region
        self.shape = tuple(region["shape"] if shape is None else shape)
        self.dtype = region["dtype"]
        self.storage = object() if storage is None else storage

    def view(self, *shape):
        size = math.prod(self.shape)
        known = math.prod(value for value in shape if value != -1)
        if shape.count(-1) > 1 or known == 0 or size % known:
            raise ValueError("Invalid metadata view")
        shape = tuple(size // known if value == -1 else value
                      for value in shape)
        if math.prod(shape) != size:
            raise ValueError("Metadata view changes the element count")
        return LayoutView(self.region, shape, self.storage)

    def __getitem__(self, rows):
        start, stop, step = rows.indices(self.shape[0])
        if start != 0 or step != 1:
            raise ValueError("Prefix metadata requires consecutive rows")
        return LayoutView(self.region, (stop, *self.shape[1:]), self.storage)


class FrozenPrefixChecks(unittest.TestCase):
    def buffer(self):
        buffer = AscendReceiveBuffer.__new__(AscendReceiveBuffer)
        buffer.layout = cache_layout(8192)
        buffer.views = {
            name: LayoutView(region)
            for name, region in buffer.layout["regions"].items()
        }
        buffer._last_generation = 0
        return buffer

    def request(self, length=128, generation=1):
        return PrefixRequest("capture", length, {
            "generation": generation, "mtp_seq_len": length - 1,
            "last_prompt_token": length - 1,
            "prompt_token_ids": list(range(length)),
        })

    def test_frozen_main_and_mtp_keep_exact_lengths_dtypes_and_aliases(self):
        for length in (1, 128, 129, 8192):
            with self.subTest(length=length):
                buffer = self.buffer()
                request = self.request(length)
                hashes = {"config.json": "checkpoint"}
                frozen = buffer.frozen_prefix(request, hashes)
                self.assertEqual(frozen["main"]["metadata_sha256"], hashes)
                self.assertEqual(frozen["mtp"]["metadata_sha256"], hashes)
                self.assertEqual(frozen["main"]["prompt_token_ids"],
                                 request.metadata["prompt_token_ids"])
                self.assertIsNot(frozen["main"]["prompt_token_ids"],
                                 request.metadata["prompt_token_ids"])
                self.assertEqual(len(frozen["main"]["layers"]), 78)
                layers = [*frozen["main"]["layers"],
                          frozen["mtp"]["layer"]]
                for layer, values in enumerate(layers):
                    rows = length if layer < 78 else length - 1
                    expected = {"latent": ("mla_latent", 512),
                                "rope": ("mla_rope", 64)}
                    if layer < 3 or (layer - 2) % 4 == 0:
                        expected["index_keys"] = "index", 128
                        if 6 <= layer <= 66:
                            expected["index_scales"] = "index_scale", 1
                    self.assertEqual(set(values), set(expected))
                    for key, (region, width) in expected.items():
                        original = buffer.views[f"{region}.{layer}"]
                        value = values[key]
                        shape = (rows,) if width == 1 else (rows, width)
                        self.assertEqual(value.shape, shape)
                        self.assertEqual(value.dtype, original.dtype)
                        self.assertIs(value.storage, original.storage)
                self.assertEqual(buffer._last_generation, 1)

    def test_decode_and_freeze_share_one_generation_consumption(self):
        for first in ("prefix", "main_prefix", "frozen_prefix"):
            with self.subTest(first=first):
                buffer = self.buffer()
                request = self.request()
                if first == "prefix":
                    prefix = buffer.prefix(request)
                    self.assertEqual(prefix.main[0]["latent"].shape[0], 127)
                    self.assertEqual(prefix.mtp["latent"].shape[0], 127)
                elif first == "main_prefix":
                    prefix = buffer.main_prefix(request)
                    self.assertEqual(len(prefix), 78)
                    self.assertEqual(prefix[0]["latent"].shape[0], 128)
                else:
                    buffer.frozen_prefix(request, {})
                with self.assertRaises(ValueError):
                    buffer.prefix(request)
                with self.assertRaises(ValueError):
                    buffer.main_prefix(request)
                with self.assertRaises(ValueError):
                    buffer.frozen_prefix(request, {})
                buffer.frozen_prefix(self.request(generation=2), {})
                self.assertEqual(buffer._last_generation, 2)

    def test_both_consumers_reject_invalid_prefix_metadata(self):
        cases = (
            ("generation", 0), ("generation", True),
            ("seq_len", True), ("seq_len", 0), ("seq_len", 8193),
            ("mtp_seq_len", 128), ("mtp_seq_len", False),
            ("last_prompt_token", True), ("last_prompt_token", 154880),
        )
        for consumer in ("prefix", "main_prefix", "frozen_prefix"):
            for field, value in cases:
                with self.subTest(consumer=consumer, field=field, value=value):
                    buffer = self.buffer()
                    request = self.request()
                    if field == "seq_len":
                        request = PrefixRequest("capture", value,
                                                request.metadata)
                    else:
                        request.metadata[field] = value
                    arguments = (
                        (request, {}) if consumer == "frozen_prefix"
                        else (request,)
                    )
                    with self.assertRaises(ValueError):
                        getattr(buffer, consumer)(*arguments)
                    self.assertEqual(buffer._last_generation, 0)

    def test_main_consumer_keeps_all_rows_and_registered_storage(self):
        for length in (1, 128, 129, 8192):
            with self.subTest(length=length):
                buffer = self.buffer()
                layers = buffer.main_prefix(self.request(length))
                self.assertEqual(len(layers), 78)
                for layer, entry in enumerate(layers):
                    self.assertEqual(entry["latent"].shape, (length, 512))
                    self.assertEqual(entry["rope"].shape, (length, 64))
                    self.assertIs(
                        entry["latent"].storage,
                        buffer.views[f"mla_latent.{layer}"].storage,
                    )
                self.assertEqual(buffer._last_generation, 1)

    def test_invalid_prompt_or_failed_view_does_not_consume_generation(self):
        buffer = self.buffer()
        request = self.request()
        tokens = request.metadata.pop("prompt_token_ids")
        with self.assertRaises(ValueError):
            buffer.frozen_prefix(request, {})
        request.metadata["prompt_token_ids"] = tokens[:-1]
        with self.assertRaises(ValueError):
            buffer.frozen_prefix(request, {})
        request.metadata["prompt_token_ids"] = tokens
        original = buffer.views.pop("mla_rope.78")
        with self.assertRaises(KeyError):
            buffer.frozen_prefix(request, {})
        self.assertEqual(buffer._last_generation, 0)
        buffer.views["mla_rope.78"] = original
        buffer.frozen_prefix(request, {})
        self.assertEqual(buffer._last_generation, 1)


class DeviceContextChecks(unittest.TestCase):
    def test_full_prefix_session_uses_supplied_prefill_token(self):
        context = mock.MagicMock()
        set_device = mock.Mock(return_value=context)
        torch = SimpleNamespace(npu=SimpleNamespace(device=set_device))
        session = mock.Mock()
        session.decode.return_value = [6500, 198]
        engine = AscendDeviceContext(session, "npu:1")
        request = PrefixRequest("full", 8192)
        with mock.patch.dict(sys.modules, {"torch": torch}):
            engine.inject(request)
            output = engine.decode(6500, max_tokens=2)
            engine.reset()
        self.assertEqual(output, [6500, 198])
        self.assertEqual(session.method_calls, [
            mock.call.inject(request),
            mock.call.decode(6500, max_tokens=2), mock.call.reset(),
        ])
        self.assertEqual(set_device.call_args_list, [mock.call("npu:1")] * 3)
        self.assertEqual(context.__enter__.call_count, 3)
        self.assertEqual(context.__exit__.call_count, 3)

    def test_native_wrapper_keeps_replay_token_check(self):
        session = mock.Mock()
        engine = AscendServingEngine(
            session, SimpleNamespace(first_token_id=6500), "npu:1",
        )
        with self.assertRaisesRegex(ValueError, "decode prefix"):
            engine.decode(6501, max_tokens=2)
        session.decode.assert_not_called()


class TokenQueueChecks(unittest.IsolatedAsyncioTestCase):
    async def test_each_token_arrives_before_producer_can_finish(self):
        mailbox = TokenQueue()
        mailbox.bind_loop()
        consumed = [threading.Event() for _ in range(32)]
        failures = []

        def produce():
            for token, acknowledgement in enumerate(consumed):
                mailbox.put(token)
                if not acknowledgement.wait(2):
                    failures.append(token)
                    return

        producer = threading.Thread(target=produce)
        producer.start()
        try:
            for token, acknowledgement in enumerate(consumed):
                self.assertEqual(await mailbox.get(timeout=1), token)
                acknowledgement.set()
        finally:
            for acknowledgement in consumed:
                acknowledgement.set()
            await asyncio.to_thread(producer.join, 2)
        self.assertEqual(failures, [])

    async def test_bursts_preserve_order_and_terminal_message(self):
        mailbox = TokenQueue()
        mailbox.bind_loop()

        def produce():
            for token in range(1024):
                mailbox.put(token)
            mailbox.put(("done", list(range(1024))))

        producer = threading.Thread(target=produce)
        producer.start()
        values = [await mailbox.get(timeout=2) for _ in range(1025)]
        await asyncio.to_thread(producer.join, 2)
        self.assertEqual(values, [*range(1024), ("done", list(range(1024)))])

    async def test_queued_tokens_survive_event_clear(self):
        mailbox = TokenQueue()
        mailbox.bind_loop()
        for token in range(10):
            mailbox.put(token)
        values = [await mailbox.get(timeout=1) for _ in range(10)]
        self.assertEqual(values, list(range(10)))

    async def test_cancelled_wait_does_not_consume_a_later_token(self):
        mailbox = TokenQueue()
        mailbox.bind_loop()
        with self.assertRaises(TimeoutError):
            await mailbox.get(timeout=0.001)
        mailbox.put(7)
        self.assertEqual(await mailbox.get(timeout=1), 7)


class PromptReplayChecks(unittest.TestCase):
    def test_replay_recovers_target_hidden_and_keeps_mtp_at_p_minus_one(self):
        class Target:
            position = 0
            capacity = 128

            def inject(self, prefixes):
                self.position = prefixes[0]["length"]

            def verify(self, tokens, full_logits=True):
                if full_logits or tokens != (17,):
                    raise ValueError("Replay requires greedy verification")
                parent = ("normalized", self.position, tokens[0])
                self.position += 1
                return SimpleNamespace(
                    token_ids=[SimpleNamespace(item=lambda: 23)],
                    last_hidden=parent,
                )

            def reset(self):
                self.position = 0

        class Draft:
            position = 0

            def inject(self, prefix):
                self.position = prefix["length"]

            def reset(self):
                self.position = 0

        engine = AscendRankEngine(Target(), Draft())
        engine.inject(RankPrefix([{"length": 7}] * 78, {"length": 7}, 17, 8))
        self.assertEqual((engine.position, engine.draft.position), (8, 7))
        self.assertEqual(engine.first_token_id, 23)
        self.assertEqual(engine.parent_hidden, ("normalized", 7, 17))
        engine.reset()
        self.assertEqual((engine.position, engine.draft.position), (0, 0))
        self.assertIsNone(engine.parent_hidden)


class WorkerOwnershipChecks(unittest.TestCase):
    def test_join_timeout_retains_the_slot_until_worker_exits(self):
        finish = threading.Event()
        cleaned = threading.Event()
        lock = threading.Lock()
        lock.acquire()
        worker = threading.Thread(target=finish.wait)
        worker.start()

        def cleanup():
            self.assertFalse(worker.is_alive())
            lock.release()
            cleaned.set()

        waiter = finish_worker(worker, cleanup)
        try:
            self.assertTrue(waiter.is_alive())
            self.assertTrue(lock.locked())
            self.assertFalse(cleaned.is_set())
        finally:
            finish.set()
            waiter.join(2)
        self.assertTrue(cleaned.is_set())
        self.assertFalse(lock.locked())

    def test_terminated_worker_releases_immediately(self):
        cleaned = threading.Event()
        worker = threading.Thread(target=lambda: None)
        worker.start()
        worker.join()
        waiter = finish_worker(worker, cleaned.set)
        self.assertIsNone(waiter)
        self.assertTrue(cleaned.is_set())


class LeaseChecks(unittest.TestCase):
    def test_known_prefill_request_is_cancelled_before_decode_dispatch(self):
        with mock.patch.dict(sys.modules, {"requests": SimpleNamespace()}):
            from tilert.pd_vllm import decode_pool

        node = decode_pool.DecodeNode("127.0.0.1", 5556, 18080)
        pool = decode_pool.Pool([node])
        lease = decode_pool.NodeLease(pool, pool.acquire())
        lease.rid = "prefill-only"
        cancelled = threading.Event()
        calls = []

        def cancel(peer, rid):
            calls.append((peer, rid))
            cancelled.set()

        with mock.patch.object(decode_pool, "cancel_decode", cancel):
            lease.release()
            self.assertTrue(cancelled.wait(1))
            lease.release()
        self.assertEqual(calls, [(node, "prefill-only")])
        self.assertFalse(node.busy)

    def test_terminated_decode_does_not_issue_another_cancel(self):
        with mock.patch.dict(sys.modules, {"requests": SimpleNamespace()}):
            from tilert.pd_vllm import decode_pool

        node = decode_pool.DecodeNode("127.0.0.1", 5556, 18080)
        pool = decode_pool.Pool([node])
        lease = decode_pool.NodeLease(pool, pool.acquire())
        lease.rid = "complete"
        with mock.patch.object(decode_pool, "cancel_decode") as cancel:
            lease.release(terminated=True)
        cancel.assert_not_called()


def http_handlers(engine, server, state, lock, state_lock):
    """Call the real small handlers without importing HTTP or Torch."""
    path = ROOT / "tilert/pd_vllm/decode_server.py"
    module = ast.parse(path.read_text())
    app = next(node for node in module.body
               if isinstance(node, ast.FunctionDef)
               and node.name == "build_app")
    functions = [node for node in app.body
                 if isinstance(node, ast.FunctionDef)
                 and node.name in ("pd_cancel", "_cleanup")]
    for function in functions:
        function.decorator_list = []
    namespace = {
        "engine": engine, "server": server, "state": state, "lock": lock,
        "state_lock": state_lock, "logger": logging.getLogger("pd-check"),
        "JSONResponse": lambda *args, **kwargs: (args, kwargs),
    }
    tree = ast.Module(body=functions, type_ignores=[])
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class HttpOwnershipChecks(unittest.TestCase):
    def setUp(self):
        self.lock = threading.Lock()
        self.lock.acquire()
        self.state_lock = threading.Lock()
        self.event = threading.Event()
        self.state = {"current_rid": "active", "cancel_event": self.event}
        self.engine = SimpleNamespace(reset=mock.Mock())
        self.server = SimpleNamespace(release=mock.Mock())
        self.handlers = http_handlers(
            self.engine, self.server, self.state, self.lock, self.state_lock,
        )

    def test_failed_reset_retains_request_and_http_lock(self):
        self.engine.reset.side_effect = RuntimeError("device reset failed")
        with self.assertLogs("pd-check", level="ERROR"):
            self.handlers["_cleanup"]()
        self.assertTrue(self.lock.locked())
        self.assertEqual(self.state["current_rid"], "active")
        self.server.release.assert_not_called()

    def test_successful_reset_releases_ownership_after_reset(self):
        order = []
        self.engine.reset.side_effect = lambda: order.append("reset")

        def release(rid):
            self.assertTrue(self.state_lock.locked())
            self.assertTrue(self.lock.locked())
            order.append(rid)

        self.server.release.side_effect = release
        self.handlers["_cleanup"]()
        self.assertEqual(order, ["reset", "active"])
        self.assertFalse(self.lock.locked())
        self.assertEqual(self.state, {"current_rid": None,
                                      "cancel_event": None})

    def test_active_cancel_sets_only_the_matching_event_under_lock(self):
        def set_event():
            self.assertTrue(self.state_lock.locked())
            self.event.set()

        self.state["cancel_event"] = SimpleNamespace(set=set_event)
        result = self.handlers["pd_cancel"]({"rid": "active"})
        self.assertEqual(result, {"cancelled": "active"})
        self.assertTrue(self.event.is_set())
        self.server.release.assert_not_called()

    def test_pending_cancel_does_not_reset_the_active_engine(self):
        def release(rid):
            self.assertTrue(self.state_lock.locked())

        self.server.release.side_effect = release
        result = self.handlers["pd_cancel"]({"rid": "pending"})
        self.assertEqual(result, {"cancelled": "pending", "pending": True})
        self.server.release.assert_called_once_with("pending")
        self.engine.reset.assert_not_called()
        self.assertFalse(self.event.is_set())
        self.assertTrue(self.lock.locked())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.getLogger("pd_vllm.ascend_receive").setLevel(logging.CRITICAL)
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(case)
        for case in (
            ReceiveChecks, FrozenPrefixChecks, DeviceContextChecks,
            TokenQueueChecks,
            PromptReplayChecks,
            WorkerOwnershipChecks, LeaseChecks, HttpOwnershipChecks,
        )
    )
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    files = (
        Path(__file__), ROOT / "tilert/pd_vllm/ascend_receiver.py",
        ROOT / "tilert/pd_vllm/ascend_cache.py",
        ROOT / "tilert/pd_vllm/ascend_layout.py",
        ROOT / "tilert/pd_vllm/ascend_server.py",
        ROOT / "tilert/pd_vllm/token_queue.py",
        ROOT / "tilert/pd_vllm/decode_server.py",
        ROOT / "tilert/pd_vllm/decode_pool.py",
    )
    report = {
        "status": "pass" if result.wasSuccessful() else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "python": platform.python_version(),
        "command": [sys.executable, *sys.argv], "tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "test_output": output.getvalue(),
        "scope": ("CPU TCP receiver, full-P/frozen-prefix view metadata, "
                  "device context dispatch, ownership, replay, token wakeups"),
        "torch_imported": "torch" in sys.modules,
        "not_executed": ["HIXL DMA", "NPU arithmetic", "NPU-to-CPU copy",
                         "frozen tensor values", "HTTP delivery timing"],
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest() for path in files
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(output.getvalue(), end="")
    print(json.dumps({"status": report["status"], "report": str(args.output)}))
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())
