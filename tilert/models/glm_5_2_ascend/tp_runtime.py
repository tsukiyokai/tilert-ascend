"""Ordered request commands for sixteen rank-local CANN engines."""

from dataclasses import asdict
from datetime import timedelta

from .session import PrefixRequest


TP_SIZE = 16


class TorchControlChannel:
    """Wait for host commands on Gloo without an outstanding NPU task."""

    def __init__(self, group=None):
        import torch
        import torch.distributed as distributed

        if distributed.get_world_size(group) != TP_SIZE:
            raise ValueError("Ascend request control requires TP16")
        if group is None:
            group = distributed.new_group(
                backend="gloo", timeout=timedelta(hours=24),
            )
        if distributed.get_backend(group) != "gloo":
            raise ValueError("Host request control requires a Gloo group")
        self.distributed = distributed
        self.group = group
        self.device = torch.device("cpu")
        self.rank = distributed.get_rank(group)

    def send(self, command):
        if self.rank != 0:
            raise RuntimeError("Only rank zero publishes decode commands")
        self.distributed.broadcast_object_list(
            [command], src=0, group=self.group, device=self.device
        )

    def receive(self):
        if self.rank == 0:
            raise RuntimeError("Rank zero owns the request loop")
        values = [None]
        self.distributed.broadcast_object_list(
            values, src=0, group=self.group, device=self.device
        )
        return values[0]


class RankExecutor:
    """Apply the same command stream using this rank's own prefix storage."""

    def __init__(self, engine, rank, prefix_loader):
        if not 0 <= rank < TP_SIZE:
            raise ValueError("Rank must be in [0,16)")
        self.engine = engine
        self.rank = rank
        self.prefix_loader = prefix_loader
        self.serial = 0
        self.request_id = None
        self.closed = False

    def execute(self, command):
        if self.closed:
            raise RuntimeError("The rank command stream is closed")
        if command["serial"] != self.serial + 1:
            raise ValueError("Rank command serial is out of order")
        if command["position"] != self.engine.position:
            raise ValueError("Rank cache positions have diverged")
        if command.get("request_id") != self.request_id:
            raise ValueError("Rank command belongs to another request")
        operation = command["operation"]
        if operation in ("step", "start_mtp", "advance_mtp",
                         "advance_mtp_n", "finish_mtp"):
            if self.request_id is None:
                raise RuntimeError("Rank has no injected request")
        result = None
        if operation == "inject":
            request = PrefixRequest(**command["request"])
            prefix = self.prefix_loader(request, self.rank)
            if request.seq_len == 0 and prefix is None:
                self.engine.reset()
            else:
                self.engine.inject(prefix)
            if self.engine.position != request.seq_len:
                raise ValueError("Loaded prefix has the wrong length")
            self.request_id = request.request_id
        elif operation == "step":
            result = self.engine.step(command["token"])
        elif operation == "start_mtp":
            self.engine.start_mtp(
                command["first_token"], command["max_tokens"],
                command["width"], command["eos_tokens"],
            )
        elif operation == "advance_mtp":
            result = self.engine.advance_mtp()
        elif operation == "advance_mtp_n":
            result = self.engine.advance_mtp_n(command["count"])
        elif operation == "finish_mtp":
            self.engine.finish_mtp(command["retained_position"])
        elif operation == "reset":
            self.engine.reset()
            self.request_id = None
        elif operation == "close":
            self.closed = True
        else:
            raise ValueError(f"Unknown rank operation: {operation}")
        self.serial = command["serial"]
        return result

    def serve(self, channel):
        while not self.closed:
            self.execute(channel.receive())


class CoordinatedBackend:
    """Rank-zero facade; workers execute each command before the next one."""

    def __init__(self, executor, channel):
        if executor.rank != 0:
            raise ValueError("The coordinated backend belongs to rank zero")
        self.executor = executor
        self.channel = channel
        self.failed = False

    @property
    def position(self):
        return self.executor.engine.position

    @property
    def capacity(self):
        return self.executor.engine.capacity

    def _execute(self, operation, **payload):
        if self.failed:
            raise RuntimeError(
                "TP command stream failed; restart all sixteen ranks"
            )
        command = {
            "serial": self.executor.serial + 1, "operation": operation,
            "position": self.position,
            "request_id": self.executor.request_id, **payload,
        }
        try:
            self.channel.send(command)
            return self.executor.execute(command)
        except BaseException:
            self.failed = True
            raise

    def inject(self, request):
        self._execute("inject", request=asdict(request))

    def step(self, token):
        return self._execute("step", token=int(token))

    def start_mtp(self, first_token, max_tokens, width, eos_tokens):
        self._execute(
            "start_mtp", first_token=first_token, max_tokens=max_tokens,
            width=width, eos_tokens=tuple(eos_tokens),
        )

    def advance_mtp(self):
        return self._execute("advance_mtp")

    def advance_mtp_n(self, count):
        return self._execute("advance_mtp_n", count=int(count))

    def finish_mtp(self, retained_position):
        self._execute("finish_mtp", retained_position=retained_position)

    def reset(self):
        self._execute("reset")

    def close(self):
        self._execute("close")
