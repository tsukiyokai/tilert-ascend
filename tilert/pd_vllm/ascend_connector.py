"""Synchronous V1 prefill connector for TP16 GLM-5.2 on Ascend.

Requires a fresh batch-one prompt scheduled in one complete step and MTP.
The consumer receives P backbone rows and P-1 target-conditioned MTP rows.
"""

import logging
from pathlib import Path
import socket
import time

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
)

from . import wire
from .ascend_cache import checkpoint_fingerprint
from .ascend_layout import (
    LAYOUT_NAME,
    NUM_RANKS,
    PAGE_SIZE,
    PARENT_HIDDEN_BYTES,
    PARENT_HIDDEN_REGION,
    cache_layout,
    parent_hidden_metadata,
    validate_layout,
)
from .ascend_receiver import validate_endpoints
from .ascend_source import bind_vllm_caches
from .hixl_transport import HixlTransport
from .prefill_connector import TileRTConnector, TileRTMetadata


logger = logging.getLogger("pd_vllm.ascend_connector")


def _require_greedy(parameters):
    get = (
        parameters.get if isinstance(parameters, dict)
        else lambda name, default=None: getattr(parameters, name, default)
    )
    expected = {
        "temperature": 0.0, "n": 1, "repetition_penalty": 1.0,
        "presence_penalty": 0.0, "frequency_penalty": 0.0, "min_tokens": 0,
    }
    for field, value in expected.items():
        if get(field, value) != value:
            raise ValueError(f"Ascend PD requires {field}={value}")
    for field in (
        "logit_bias", "allowed_token_ids", "bad_words", "structured_outputs",
    ):
        if get(field):
            raise ValueError(f"Ascend PD does not apply {field}")


class AscendTileRTConnector(TileRTConnector):
    """Retain vLLM blocks until WRITE and the matching done acknowledgement.

    Scheduler request metadata uses the existing TileRT V1 representation.
    Worker cache binding and transport do not use the GPU staging profile.
    """

    def __init__(self, vllm_config, role, kv_cache_config=None):
        KVConnectorBase_V1.__init__(
            self, vllm_config, role, kv_cache_config,
        )
        self._check_config(vllm_config)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self._transfer_parent_hidden = extra.get(
            "transfer_mtp_parent_hidden", False,
        )
        if type(self._transfer_parent_hidden) is not bool:
            raise ValueError("transfer_mtp_parent_hidden must be boolean")
        if (self._transfer_parent_hidden
                and not vllm_config.model_config.enforce_eager):
            raise ValueError("Parent hidden capture requires an eager target")
        self._parent_capture = None
        self._default_host = extra.get("tilert_host")
        self._default_port = int(extra.get("tilert_ctrl_port", 5556))
        self._max_seq = int(extra.get(
            "tilert_max_seq_len", vllm_config.model_config.max_model_len,
        ))
        self._hixl_host = extra.get("hixl_host")
        self._hixl_library = extra.get("hixl_library")
        self._hixl_port_base = int(extra.get("hixl_port_base", 26000))
        self._timeout = float(extra.get("tilert_timeout_s", 120.0))
        self._admission_attempts = int(extra.get(
            "tilert_admission_attempts", 5,
        ))
        if (
            not 0 < self._hixl_port_base <= 65536 - NUM_RANKS
            or self._timeout <= 0 or self._admission_attempts < 1
        ):
            raise ValueError(
                "Invalid HIXL ports, timeout, or admission attempts"
            )
        self._checkpoint_hashes = checkpoint_fingerprint(
            vllm_config.model_config.model,
        )
        self._pending = {}
        self._binding = None
        self._transport = None
        self._tp_rank = None
        self._written_layers = set()

    @staticmethod
    def _check_config(config):
        if not Path(config.model_config.model).is_dir():
            raise ValueError("Ascend PD requires a local ModelSlim checkpoint")
        for field, expected in (
            ("tensor_parallel_size", NUM_RANKS),
            ("pipeline_parallel_size", 1), ("data_parallel_size", 1),
            ("prefill_context_parallel_size", 1),
            ("decode_context_parallel_size", 1),
        ):
            if getattr(config.parallel_config, field) != expected:
                raise ValueError(f"Ascend PD requires {field}={expected}")
        if config.parallel_config.enable_expert_parallel:
            raise ValueError("Ascend PD requires TP16 expert sharding")
        scheduler = config.scheduler_config
        if scheduler.max_num_seqs != 1:
            raise ValueError("Ascend PD requires batch1")
        if config.cache_config.enable_prefix_caching:
            raise ValueError(
                "Ascend PD requires fresh prompts without prefix hits"
            )
        if config.cache_config.block_size != PAGE_SIZE:
            raise ValueError("Ascend PD requires block_size=128")
        speculative = config.speculative_config
        if (
            speculative is None or speculative.method != "mtp"
            or speculative.num_speculative_tokens < 1
            or speculative.num_speculative_tokens_per_batch_size
        ):
            raise ValueError(
                "Ascend PD requires a fixed positive MTP draft count"
            )
        if speculative.parallel_drafting:
            raise ValueError("Ascend PD requires sequential MTP drafting")
        if Path(speculative.draft_model_config.model).resolve() != Path(
            config.model_config.model,
        ).resolve():
            raise ValueError("MTP must use the same local checkpoint")
        if not config.kv_transfer_config.is_kv_producer:
            raise ValueError("AscendTileRTConnector is a prefill producer")

    def _params_of(self, request):
        params = super()._params_of(request)
        if params is not None:
            _require_greedy(request.sampling_params)
            _require_greedy(params.get("sampling") or {})
            if getattr(request, "lora_request", None) is not None:
                raise ValueError(
                    "Ascend PD requires the unmodified checkpoint"
                )
            params["_wants_prompt_ids"] = True
        return params

    def build_connector_meta(self, scheduler_output):
        metadata = TileRTMetadata()
        config = self._vllm_config
        speculative = config.speculative_config
        for request in scheduler_output.scheduled_new_reqs:
            params = self._params_of(request)
            if params is None:
                continue
            tokens = list(request.prompt_token_ids or [])
            scheduled = scheduler_output.num_scheduled_tokens.get(
                request.req_id, 0,
            )
            if (
                not tokens or request.num_computed_tokens != 0
                or scheduled != len(tokens)
            ):
                raise ValueError(
                    "Ascend PD requires one fresh, complete prefill"
                )
            if (
                len(tokens) > self._max_seq
                or len(tokens) > config.scheduler_config.max_num_batched_tokens
                or len(tokens) + speculative.num_speculative_tokens
                > speculative.draft_model_config.max_model_len
            ):
                raise ValueError(
                    "Prompt leaves no room for the MTP first pass"
                )
            metadata.requests.append(self._emit(
                request.req_id, tokens,
                [list(group) for group in request.block_ids], params,
            ))
        return metadata

    def bind_connector_metadata(self, metadata):
        super().bind_connector_metadata(metadata)
        self._written_layers.clear()
        if self._parent_capture is not None:
            self._parent_capture.clear()
        if self._transfer_parent_hidden and metadata.requests:
            from vllm.forward_context import get_forward_context

            if self._parent_capture is None or len(metadata.requests) != 1:
                raise RuntimeError("Parent capture needs registered batch1 KV")
            context = get_forward_context()
            if getattr(context, "is_draft_model", False):
                raise ValueError("Parent hidden must come from the target")
            self._parent_capture.bind(context.model_instance,
                                      metadata.requests[0])

    def clear_connector_metadata(self):
        if self._parent_capture is not None:
            self._parent_capture.clear()
        super().clear_connector_metadata()

    def register_kv_caches(self, kv_caches):
        from vllm.distributed import get_tensor_model_parallel_rank

        if self._transport is not None:
            raise RuntimeError("Cannot replace registered HIXL source memory")
        self._binding = bind_vllm_caches(
            kv_caches, self._kv_cache_config,
            layout=cache_layout(self._max_seq),
        )
        self._tp_rank = get_tensor_model_parallel_rank()
        if not 0 <= self._tp_rank < NUM_RANKS:
            raise ValueError("Source rank must belong to TP16")
        if self._transfer_parent_hidden:
            from .ascend_parent import ParentHiddenCapture

            device = next(iter(self._binding.regions.values())).tensor.device
            self._parent_capture = ParentHiddenCapture(device)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        self._written_layers.add(layer_name)

    def _ensure_transport(self, remote_host):
        if self._transport is not None:
            return
        transport = HixlTransport(
            library_path=self._hixl_library,
            port=self._hixl_port_base + self._tp_rank,
            timeout_ms=int(self._timeout * 1000),
        )
        transport.init(self._hixl_host or wire.local_ip(remote_host))
        for pointer, size in self._binding.registration_ranges():
            transport.register(pointer, size, self._binding.device_id)
        if self._parent_capture is not None:
            transport.register(self._parent_capture.tensor.data_ptr(),
                               PARENT_HIDDEN_BYTES, self._binding.device_id)
        self._transport = transport

    def wait_for_save(self):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, TileRTMetadata) or not metadata.requests:
            return
        if self._binding is None:
            raise RuntimeError("vLLM has not registered the Ascend caches")
        expected_layers = {
            region.layer_name for name, region in self._binding.regions.items()
            if name.startswith("mla_latent.")
        }
        missing = expected_layers - self._written_layers
        if missing:
            raise RuntimeError(
                f"Unwritten backbone/MTP cache layers: {sorted(missing)}"
            )
        if self._parent_capture is not None:
            for request in metadata.requests:
                self._parent_capture.pointer_for(request)
        import torch

        torch.npu.synchronize(self._binding.device_id)
        for request in metadata.requests:
            self._ensure_transport(request.tilert_host)
            self._send_request(request)

    def _validate_hello(self, hello):
        expected_layout = cache_layout(
            parent_hidden=self._transfer_parent_hidden,
        )
        for field, expected in (
            ("magic", wire.MAGIC), ("protocol_version", wire.PROTOCOL_VERSION),
            ("transport", "hixl"), ("profile", LAYOUT_NAME),
            ("layout_version", expected_layout["version"]),
            ("num_ranks", NUM_RANKS), ("senders", NUM_RANKS),
        ):
            if hello.get(field) != expected:
                raise ValueError(f"Ascend receiver mismatch: {field}")
        if hello.get("checkpoint_sha256") != self._checkpoint_hashes:
            raise ValueError("Prefill and decode checkpoint metadata differ")
        layout = hello["ascend_layout"]
        validate_layout(layout)
        if ((PARENT_HIDDEN_REGION in layout["regions"])
                != self._transfer_parent_hidden):
            raise ValueError("Parent hidden transfer flags differ")
        if hello["max_seq_len"] != layout["capacity_tokens"]:
            raise ValueError("Receiver capacity differs from its layout")
        validate_endpoints(hello["rank_endpoints"], layout)
        return layout, hello["rank_endpoints"][self._tp_rank]

    def _send_request(self, request):
        for attempt in range(self._admission_attempts):
            if self._send_once(request):
                return
            if attempt + 1 < self._admission_attempts:
                time.sleep(0.2 * 2 ** attempt)
        raise RuntimeError(f"Ascend receiver stayed busy for {request.rid}")

    def _send_once(self, request):
        with socket.create_connection(
            (request.tilert_host, request.tilert_ctrl_port), self._timeout,
        ) as connection:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            layout, endpoint = self._validate_hello(wire.recv_msg(connection))
            sources, destinations, lengths = self._binding.transfer_plan(
                request.block_ids_per_group, endpoint["base_ptr"],
                request.num_tokens, mtp_prefix_tokens=request.num_tokens - 1,
                destination_layout=layout,
                parent_hidden_ptr=(
                    self._parent_capture.pointer_for(request)
                    if self._parent_capture is not None else None
                ),
            )
            admission = {
                "rid": request.rid, "rank": self._tp_rank,
                "seq_len": request.num_tokens,
                "mtp_seq_len": request.num_tokens - 1,
                "last_prompt_token": request.last_prompt_token,
                "sampling": request.sampling,
            }
            if request.prompt_token_ids:
                admission["prompt_token_ids"] = request.prompt_token_ids
            if self._transfer_parent_hidden:
                admission[PARENT_HIDDEN_REGION] = parent_hidden_metadata(
                    request.num_tokens,
                )
            wire.send_msg(connection, admission)
            reply = wire.recv_msg(connection)
            if not reply.get("accepted"):
                if reply.get("error") in ("busy", "cancelling"):
                    return False
                raise RuntimeError(
                    f"Ascend receiver refused admission: {reply}"
                )
            generation = reply.get("generation")
            if (
                type(generation) is not int or generation <= 0
                or reply != wire.accept_msg(
                    request.rid, self._tp_rank, generation,
                )
            ):
                raise ValueError("Admission request/rank/generation mismatch")
            self._transport.write(endpoint, sources, destinations, lengths)
            wire.send_msg(connection, wire.done_msg(
                request.rid, self._tp_rank, generation,
            ))
            expected = {
                "received": True, "rid": request.rid,
                "rank": self._tp_rank, "generation": generation,
            }
            if wire.recv_msg(connection) != expected:
                raise ValueError("WRITE acknowledgement mismatch")
            logger.info(
                "sent %s rank=%d main=%d MTP=%d generation=%d bytes=%d",
                request.rid, self._tp_rank, request.num_tokens,
                request.num_tokens - 1, generation, sum(lengths),
            )
            return True
