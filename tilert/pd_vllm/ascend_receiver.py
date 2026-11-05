"""One request slot and sixteen HIXL destinations for A3 decode."""

from dataclasses import dataclass, field
import logging
import queue
import socket
import threading
import time

from . import wire
from .ascend_layout import (
    LAYOUT_NAME, LAYOUT_VERSION, NUM_RANKS, PARENT_HIDDEN_LAYOUT_VERSION,
    PARENT_HIDDEN_REGION, validate_layout, validate_parent_hidden_metadata,
    validate_prompt_token_ids,
)


logger = logging.getLogger("pd_vllm.ascend_receive")


@dataclass
class AscendReceivedRequest:
    rid: str
    seq_len: int
    mtp_seq_len: int
    last_prompt_token: int
    sampling: dict | None
    generation: int
    first_token_id: int | None = None
    prompt_token_ids: list = field(default_factory=list)
    parent_hidden: dict | None = None
    admitted_ranks: set = field(default_factory=set)
    active_ranks: set = field(default_factory=set)
    done_ranks: set = field(default_factory=set)
    state: str = "transferring"
    t_first_conn: float = field(default_factory=time.time)
    t_complete: float = 0.0


class AscendPDProfile:
    name = LAYOUT_NAME
    num_ranks = NUM_RANKS
    layout_version = LAYOUT_VERSION
    sender_ranks = frozenset(range(NUM_RANKS))

    def __init__(self, *, parent_hidden=False):
        self.parent_hidden = parent_hidden
        if parent_hidden:
            self.layout_version = PARENT_HIDDEN_LAYOUT_VERSION

    def convert(self, buffer, base_ptr, max_seq_len, received, num_devices):
        """Publish a descriptor only after all sixteen WRITE completions."""
        from tilert.models.glm_5_2_ascend.session import PrefixRequest

        if (
            received.state != "complete"
            or received.done_ranks != self.sender_ranks
            or num_devices != NUM_RANKS
        ):
            raise ValueError("Ascend prefix is not complete on sixteen ranks")
        metadata = {
            "generation": received.generation,
            "mtp_seq_len": received.mtp_seq_len,
            "last_prompt_token": received.last_prompt_token,
        }
        if self.parent_hidden:
            validate_parent_hidden_metadata(received.parent_hidden,
                                            received.seq_len)
            metadata[PARENT_HIDDEN_REGION] = dict(received.parent_hidden)
        elif received.parent_hidden is not None:
            raise ValueError("Parent hidden was not enabled on this receiver")
        if received.prompt_token_ids:
            validate_prompt_token_ids(
                received.prompt_token_ids, received.seq_len,
                received.last_prompt_token,
            )
            metadata["prompt_token_ids"] = list(received.prompt_token_ids)
        return PrefixRequest(received.rid, received.seq_len, metadata)


def validate_endpoints(endpoints, layout):
    """Keep rank routing separate from process-local NPU virtual addresses."""
    validate_layout(layout)
    if len(endpoints) != NUM_RANKS:
        raise ValueError("Publish exactly sixteen HIXL rank endpoints")
    engines = set()
    for rank, endpoint in enumerate(endpoints):
        if endpoint["rank"] != rank or endpoint["transport"] != "hixl":
            raise ValueError("HIXL endpoints must be ordered by TP rank")
        address = endpoint["base_ptr"]
        if type(address) is not int or address <= 0:
            raise ValueError("A receive buffer needs a positive base pointer")
        if not any(
            region["base"] <= address
            and address + layout["nbytes"]
            <= region["base"] + region["bytes"]
            for region in endpoint["hixl_regions"]
        ):
            raise ValueError("Receive buffer exceeds registered HIXL memory")
        engine = endpoint["hixl_engine"]
        if not engine or engine in engines:
            raise ValueError("Each rank needs its own HIXL listening endpoint")
        engines.add(engine)


class AscendReceiveServer:
    """Coordinate WRITE admission without owning another rank's NPU memory.

    A completed prefix keeps the slot until release(), including throughout
    decode. An admitted writer that disconnects without a matching done
    quarantines the slot: TCP EOF does not prove that RDMA has stopped.
    Stop senders and drain requests before close() and HIXL deregistration.
    Lookup serving requires complete prompt IDs at admission, before WRITE.
    """

    def __init__(
        self, layout, endpoints, checkpoint_sha256, *, host="0.0.0.0",
        ctrl_port=5556, request_timeout=120.0, require_prompt_token_ids=False,
    ):
        validate_endpoints(endpoints, layout)
        if request_timeout <= 0:
            raise ValueError("Control timeout must be positive")
        self.layout = layout
        self.endpoints = endpoints
        self.checkpoint_sha256 = checkpoint_sha256
        self.max_seq_len = layout["capacity_tokens"]
        self.request_timeout = request_timeout
        self.require_prompt_token_ids = require_prompt_token_ids
        self.profile = AscendPDProfile(
            parent_hidden=PARENT_HIDDEN_REGION in layout["regions"],
        )
        self.buffer = None
        self.base_ptr = [endpoint["base_ptr"] for endpoint in endpoints]
        self.completed = queue.Queue()
        self._lock = threading.Lock()
        self._current = None
        self._generation = 0
        self._cancelled = {}
        self._stop = threading.Event()
        self._handlers = set()
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, ctrl_port))
        self._srv.listen(NUM_RANKS * 2)
        self._srv.settimeout(0.25)
        self.ctrl_port = self._srv.getsockname()[1]
        self._thread = threading.Thread(
            target=self._accept_loop, name="ascend-pd-receive", daemon=True,
        )
        self._thread.start()

    def _hello(self):
        with self._lock:
            busy = self._current is not None
        return wire.hello_msg(
            "hixl", {}, self.max_seq_len, self.layout["version"], {
                "profile": LAYOUT_NAME, "num_ranks": NUM_RANKS,
                "senders": NUM_RANKS, "ascend_layout": self.layout,
                "rank_endpoints": self.endpoints,
                "checkpoint_sha256": self.checkpoint_sha256,
            }, busy,
        )

    def expect(self, rid):
        """An HTTP announcement cannot revive a cancelled generation."""
        with self._lock:
            return self._cancelled.get(rid, 0) <= time.monotonic()

    def state_snapshot(self):
        with self._lock:
            current = self._current
            if current is None:
                return {"state": "free", "rid": None, "generation": None,
                        "active_writers": 0}
            return {
                "state": current.state, "rid": current.rid,
                "generation": current.generation,
                "active_writers": len(current.active_ranks),
                "done_ranks": sorted(current.done_ranks),
            }

    def _admit(self, request):
        rid = request["rid"]
        rank = request["rank"]
        length = request["seq_len"]
        mtp_length = request["mtp_seq_len"]
        token = request["last_prompt_token"]
        if not isinstance(rid, str) or not rid:
            return wire.reject_msg("invalid_request_id")
        if type(rank) is not int or not 0 <= rank < NUM_RANKS:
            return wire.reject_msg("invalid_rank")
        if type(length) is not int or not 1 <= length <= self.max_seq_len:
            return wire.reject_msg("invalid_prefix_length")
        if type(mtp_length) is not int or mtp_length != length - 1:
            return wire.reject_msg("MTP prefix must contain P-1 parent rows")
        if type(token) is not int or not 0 <= token < 154880:
            return wire.reject_msg("invalid_last_prompt_token")
        parent_hidden = request.get(PARENT_HIDDEN_REGION)
        if self.profile.parent_hidden:
            try:
                validate_parent_hidden_metadata(parent_hidden, length)
            except ValueError:
                return wire.reject_msg("invalid_mtp_parent_hidden")
            parent_hidden = dict(parent_hidden)
        elif PARENT_HIDDEN_REGION in request:
            return wire.reject_msg("mtp_parent_hidden_not_enabled")
        sampling = request.get("sampling")
        prompt = request.get("prompt_token_ids", [])
        if self.require_prompt_token_ids or "prompt_token_ids" in request:
            try:
                validate_prompt_token_ids(prompt, length, token)
            except ValueError:
                return wire.reject_msg("invalid_prompt_token_ids")
            prompt = list(prompt)
        with self._lock:
            now = time.monotonic()
            self._cancelled = {
                key: expiry for key, expiry in self._cancelled.items()
                if expiry > now
            }
            if rid in self._cancelled:
                return wire.reject_msg("cancelling", rid=rid)
            current = self._current
            if current is None:
                self._generation += 1
                current = AscendReceivedRequest(
                    rid, length, mtp_length, token, sampling, self._generation,
                    prompt_token_ids=prompt, parent_hidden=parent_hidden,
                )
                self._current = current
            elif current.rid != rid:
                return wire.reject_msg("busy", busy_rid=current.rid,
                                       busy_state=current.state)
            if current.state != "transferring":
                return wire.reject_msg(current.state, rid=rid)
            if (
                current.seq_len != length
                or current.mtp_seq_len != mtp_length
                or current.last_prompt_token != token
                or current.sampling != sampling
                or current.parent_hidden != parent_hidden
                or (prompt and current.prompt_token_ids
                    and prompt != current.prompt_token_ids)
            ):
                return wire.reject_msg("rank_metadata_mismatch", rid=rid)
            if rank in current.admitted_ranks:
                return wire.reject_msg("duplicate_rank", rid=rid, rank=rank)
            if prompt and not current.prompt_token_ids:
                current.prompt_token_ids = prompt
            current.admitted_ranks.add(rank)
            current.active_ranks.add(rank)
            return wire.accept_msg(rid, rank, current.generation)

    def _complete_rank(self, request, generation, done):
        rid, rank = request["rid"], request["rank"]
        if (
            done != wire.done_msg(rid, rank, generation)
            or done.get("done") is not True
            or type(done.get("rank")) is not int
            or type(done.get("generation")) is not int
        ):
            raise ValueError("WRITE request/rank/generation mismatch")
        with self._lock:
            current = self._current
            if (
                current is None or current.rid != rid
                or current.generation != generation
                or rank not in current.active_ranks
            ):
                raise ValueError("WRITE completion has no active admission")
            current.done_ranks.add(rank)
            if (
                current.state == "transferring"
                and current.done_ranks == self.profile.sender_ranks
            ):
                current.state = "complete"
                current.t_complete = time.time()
                self.completed.put(current)
        return {"received": True, "rid": rid, "rank": rank,
                "generation": generation}

    def _writer_left(self, request, generation):
        with self._lock:
            current = self._current
            if current is None or current.generation != generation:
                return
            rank = request["rank"]
            current.active_ranks.discard(rank)
            if rank not in current.done_ranks:
                current.state = "failed"
            self._retire_cancelled()

    def _retire_cancelled(self):
        current = self._current
        if (
            current.state == "cancelling" and not current.active_ranks
            and current.done_ranks == current.admitted_ranks
        ):
            self._current = None

    def release(self, rid):
        with self._lock:
            self._cancelled[rid] = time.monotonic() + self.request_timeout
            current = self._current
            if current is None or current.rid != rid:
                return
            if current.state != "failed":
                current.state = "cancelling"
                self._retire_cancelled()

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                connection, address = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            handler = threading.Thread(
                target=self._handle, args=(connection, address), daemon=True,
            )
            with self._lock:
                self._handlers.add(handler)
            handler.start()

    def _handle(self, connection, address):
        admitted = None
        try:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection.settimeout(self.request_timeout)
            wire.send_msg(connection, self._hello())
            request = wire.recv_msg(connection)
            reply = self._admit(request)
            if reply.get("accepted"):
                admitted = request, reply["generation"]
            wire.send_msg(connection, reply)
            if not reply.get("accepted"):
                return
            done = wire.recv_msg(connection)
            reply = self._complete_rank(*admitted, done)
            wire.send_msg(connection, reply)
        except (ConnectionError, OSError, ValueError, KeyError, TypeError):
            logger.exception("Ascend PD connection from %s failed", address)
        finally:
            if admitted is not None:
                self._writer_left(*admitted)
            connection.close()
            with self._lock:
                self._handlers.discard(threading.current_thread())

    def close(self):
        self._stop.set()
        self._srv.close()
        self._thread.join(1.0)
        deadline = time.monotonic() + self.request_timeout + 1.0
        with self._lock:
            handlers = tuple(self._handlers)
        for handler in handlers:
            handler.join(max(0.0, deadline - time.monotonic()))
        if any(handler.is_alive() for handler in handlers):
            raise TimeoutError("Ascend PD writers have not drained")
