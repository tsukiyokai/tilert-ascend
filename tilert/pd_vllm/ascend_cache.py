"""Rank-local HIXL receive memory and CANN prefix injection."""

from collections import namedtuple
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path

from .ascend_layout import (
    NUM_RANKS, PARENT_HIDDEN_REGION, cache_layout,
    validate_parent_hidden_metadata, validate_prompt_token_ids,
)


DrafterSpec = namedtuple("DrafterSpec", "module name tail_keyword")

# One name selects the host drafter. `tail_keyword` is how that drafter names
# its shortest matched tail, and None where it has no such bound: the tree's
# expansion is governed by its budget and its branching, not by a tail length.
DRAFTERS = {
    "lookup": DrafterSpec("tilert.models.glm_5_2_ascend.lookup_draft",
                          "LookupDrafter", "shortest"),
    "automaton": DrafterSpec("tilert.models.glm_5_2_ascend.automaton_draft",
                             "AutomatonDrafter", "shortest"),
    "tree": DrafterSpec("tilert.models.glm_5_2_ascend.tree_draft",
                        "TreeDrafter", None),
}

CHECKPOINT_FILES = (
    "config.json", "quant_model_description.json",
    "quant_model_weights.safetensors.index.json",
)


def drafter_metadata(name):
    """Describe the imported host drafter and reject stale native sources."""
    spec = DRAFTERS[name]
    module = importlib.import_module(spec.module)
    path = Path(module.__file__).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    build = getattr(module, "_NATIVE_BUILD", None)
    result = {
        "module": module.__name__, "class": spec.name,
        "kind": "python" if build is None else build["kind"],
        "file": str(path), "file_sha256": actual,
        "source_sha256": {str(path): actual},
    }
    if build is not None:
        if actual != build["loaded_binary_sha256"]:
            raise RuntimeError("Loaded native lookup file changed")
        for filename, expected in build["source_sha256"].items():
            source = path.with_name(filename)
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise RuntimeError("Native lookup sources changed; rebuild")
            result["source_sha256"][str(source)] = expected
        result["build"] = {**build,
                           "source_sha256": dict(build["source_sha256"])}
    return result


def receive_device_id(rank):
    """Move each PD destination off its source die on the single A3 host."""
    return (rank + 1) % NUM_RANKS


def checkpoint_fingerprint(directory):
    directory = Path(directory)
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in CHECKPOINT_FILES
    }


@dataclass(frozen=True)
class RankPrefix:
    """P-1 main/MTP rows plus the final prompt token replayed at injection."""

    main: list
    mtp: dict
    last_prompt_token: int
    seq_len: int


@dataclass(frozen=True)
class FullRankPrefix:
    """Full-P main KV, P-1 MTP KV and normalized main hidden[P-1].

    Optional prompt IDs select request-local host drafting.
    """

    main: list
    mtp: dict
    parent_hidden: object
    seq_len: int
    prompt_token_ids: tuple[int, ...] | None = None


class AscendReceiveBuffer:
    """Retain one contiguous registered allocation until HIXL peers drain."""

    def __init__(self, rank, device, capacity, transport, hostname, *,
                 parent_hidden=False):
        import torch

        if type(rank) is not int or not 0 <= rank < NUM_RANKS:
            raise ValueError("Receive rank must be in [0,16)")
        self.rank = rank
        self.device = torch.device(device)
        self.layout = cache_layout(capacity, parent_hidden=parent_hidden)
        self.transport = transport
        self.storage = torch.empty(
            self.layout["nbytes"], dtype=torch.uint8, device=self.device,
        )
        self.views = {}
        for name, region in self.layout["regions"].items():
            raw = self.storage.narrow(0, region["offset"], region["nbytes"])
            self.views[name] = raw.view(getattr(torch, region["dtype"])).view(
                region["shape"]
            )
        transport.init(hostname)
        transport.register(
            self.storage.data_ptr(), self.storage.numel(), self.device.index,
        )
        self._last_generation = 0

    def endpoint(self):
        return {
            "rank": self.rank, "base_ptr": self.storage.data_ptr(),
            "transport": self.transport.name, **self.transport.local_meta(),
        }

    def _layer_prefix(self, layer, length):
        regions = self.layout["regions"]

        def rows(name):
            width = regions[name]["row_bytes"] // (
                1 if regions[name]["dtype"] == "int8" else 2
            )
            return self.views[name].view(-1, width)[:length]

        result = {"latent": rows(f"mla_latent.{layer}"),
                  "rope": rows(f"mla_rope.{layer}")}
        if f"index.{layer}" in regions:
            result["index_keys"] = rows(f"index.{layer}")
        if f"index_scale.{layer}" in regions:
            result["index_scales"] = rows(f"index_scale.{layer}").view(-1)
        return result

    def _validate_prefix(self, request):
        length = request.seq_len
        generation = request.metadata["generation"]
        if type(generation) is not int or generation <= self._last_generation:
            raise ValueError("Receive generation is stale or out of order")
        if (type(length) is not int
                or not 1 <= length <= self.layout["capacity_tokens"]):
            raise ValueError("Receive prefix exceeds cache capacity")
        mtp_length = request.metadata["mtp_seq_len"]
        if type(mtp_length) is not int or mtp_length != length - 1:
            raise ValueError("MTP prefix must contain P-1 parent rows")
        token = request.metadata["last_prompt_token"]
        if type(token) is not int or not 0 <= token < 154880:
            raise ValueError("Invalid final prompt token")
        if PARENT_HIDDEN_REGION in self.layout["regions"]:
            validate_parent_hidden_metadata(
                request.metadata.get(PARENT_HIDDEN_REGION), length,
            )
        elif PARENT_HIDDEN_REGION in request.metadata:
            raise ValueError("Parent hidden is absent from the receive layout")
        return length, generation

    def prefix(self, request):
        """Consume only the generation released by the central receiver.

        Replaying the last prompt row recovers normalized target hidden
        for the MTP seed. It also avoids transferring that hidden tensor.
        Main row P-1 in the receive buffer is valid but deliberately replayed.
        """
        length, generation = self._validate_prefix(request)
        prefix = RankPrefix(
            [self._layer_prefix(layer, length - 1) for layer in range(78)],
            self._layer_prefix(78, length - 1),
            request.metadata["last_prompt_token"], length,
        )
        self._last_generation = generation
        return prefix

    def main_prefix(self, request):
        """Consume all P main rows for serial decode from prefill's token."""
        length, generation = self._validate_prefix(request)
        prefix = [self._layer_prefix(layer, length) for layer in range(78)]
        self._last_generation = generation
        return prefix

    def mtp_prefix(self, request, *, lookup=False):
        """Consume the v3 parent and both caches without prompt replay."""
        if PARENT_HIDDEN_REGION not in self.views:
            raise ValueError("CANN MTP requires the v3 parent-hidden layout")
        length, generation = self._validate_prefix(request)
        prompt = None
        if lookup:
            prompt = request.metadata.get("prompt_token_ids")
            validate_prompt_token_ids(
                prompt, length, request.metadata["last_prompt_token"],
            )
            prompt = tuple(prompt)
        prefix = FullRankPrefix(
            [self._layer_prefix(layer, length) for layer in range(78)],
            self._layer_prefix(78, length - 1),
            self.views[PARENT_HIDDEN_REGION], length, prompt,
        )
        self._last_generation = generation
        return prefix

    def frozen_prefix(self, request, checkpoint_hashes):
        """Consume P main rows and P-1 MTP rows without prompt replay.

        The returned tensors alias registered receive memory. The caller
        must finish copying them before releasing the receiver slot.
        """
        length, generation = self._validate_prefix(request)
        tokens = request.metadata.get("prompt_token_ids")
        validate_prompt_token_ids(
            tokens, length, request.metadata["last_prompt_token"],
        )
        frozen = {
            "main": {
                "metadata_sha256": dict(checkpoint_hashes),
                "prompt_token_ids": list(tokens),
                "layers": [self._layer_prefix(layer, length)
                           for layer in range(78)],
            },
            "mtp": {
                "metadata_sha256": dict(checkpoint_hashes),
                "layer": self._layer_prefix(78, length - 1),
            },
        }
        if PARENT_HIDDEN_REGION in self.layout["regions"]:
            frozen["mtp"]["parent_hidden"] = {
                **request.metadata[PARENT_HIDDEN_REGION],
                "request_id": request.request_id, "generation": generation,
                "tensor": self.views[PARENT_HIDDEN_REGION],
            }
        self._last_generation = generation
        return frozen


class AscendRankEngine:
    """Bind the receive prefix to the main/MTP caches on one TP rank."""

    def __init__(self, target, draft, lookup_prompt=None, *,
                 lookup_min_ngram=2, drafter="lookup"):
        self.target = target
        self.draft = draft
        # The prompt's token ids, when the round is to draft by looking for
        # where the context repeats rather than by running the MTP engine.
        # Every rank reads them from its own capture, so they cost no
        # broadcast.
        self.lookup_prompt = lookup_prompt
        self.lookup_min_ngram = lookup_min_ngram
        # Set by a caller that wants the round's steps timed; a round measures
        # 50.7 ms where its super kernels account for 16.4, and the record had
        # no account of the rest.
        self.round_phases = None
        if drafter not in DRAFTERS:
            raise ValueError(f"Unknown drafter {drafter!r}")
        self.drafter_name = drafter
        self._clear_request()

    def _clear_request(self):
        self._request_prompt = self.lookup_prompt
        self.parent_hidden = None
        self.first_token_id = None
        self._prompt_length = None
        self._decode_mode = None
        self._mtp = None

    @property
    def position(self):
        return self.target.position

    @property
    def capacity(self):
        return self.target.capacity

    def inject(self, prefix):
        self._clear_request()
        self.target.inject(prefix.main)
        if self.draft is not None:
            self.draft.inject(prefix.mtp)
        if (
            self.target.position != prefix.seq_len - 1
            or (self.draft is not None
                and self.draft.position != prefix.seq_len - 1)
        ):
            raise ValueError("Main/MTP replay prefixes have different lengths")
        result = self.target.verify(
            (prefix.last_prompt_token,), full_logits=False,
        )
        self.parent_hidden = result.last_hidden
        self.first_token_id = int(result.token_ids[0].item())
        self._prompt_length = prefix.seq_len

    def step(self, token):
        if self._decode_mode == "mtp":
            raise RuntimeError("MTP and single-step decode cannot be mixed")
        self._decode_mode = "step"
        return self.target.verify((token,), full_logits=False)

    def start_mtp(self, first_token, max_tokens, width, eos_tokens):
        from tilert.models.glm_5_2_ascend.mtp_round import MtpRoundExecutor

        if self._prompt_length is None or self.parent_hidden is None:
            raise RuntimeError("Inject a prefix before starting MTP")
        if self._decode_mode is not None:
            raise RuntimeError("MTP must start before any decode operation")
        if type(first_token) is not int or first_token != self.first_token_id:
            raise ValueError("First token differs from the injected prefix")
        drafter = None
        if self._request_prompt is not None:
            spec = DRAFTERS[self.drafter_name]
            Drafter = getattr(importlib.import_module(spec.module), spec.name)
            bounds = ({spec.tail_keyword: self.lookup_min_ngram}
                      if spec.tail_keyword else {})
            drafter = Drafter(self._request_prompt, **bounds)
            drafter.commit((first_token,))
        if self.round_phases is not None:
            self.round_phases.reset()
        rounds = MtpRoundExecutor(
            self.target, self.draft, width, max_tokens, eos_tokens, drafter,
            phases=self.round_phases,
        )
        # A tree's off-path node must see its own ancestors rather than the
        # indexer's selection, so the mask is installed when a tree asks.
        if hasattr(drafter, "propose_tree"):
            rounds.tree_mask = self.target.install_tree_mask()
        rounds.start(first_token, self.parent_hidden)
        if rounds.state.finish_reason is not None:
            raise ValueError("MTP start requires another output token")
        self._mtp = rounds
        self._decode_mode = "mtp"

    def advance_mtp(self):
        if self._mtp is None:
            raise RuntimeError("Start MTP before advancing a round")
        return self._mtp.advance()

    def advance_mtp_n(self, count):
        """Run up to ``count`` rounds on one command.

        Every rank computes the same tokens, acceptance and stop decision
        from its own device results, so rounds between commands need no
        coordination beyond the collectives the model already performs.
        """
        results = []
        for _ in range(count):
            results.append(self.advance_mtp())
            if results[-1].commit.finish_reason is not None:
                break
        return results

    def finish_mtp(self, retained_position):
        """Retain delivered-token state after a complete distributed round."""
        if self._mtp is None:
            raise RuntimeError("No MTP request is active")
        if (type(retained_position) is not int
                or not self._prompt_length <= retained_position
                <= self.target.position):
            raise ValueError("Retained position is outside the request")
        if retained_position != self.target.position:
            self.target.truncate(retained_position)
        if self.draft is not None:
            draft_position = min(retained_position, self.draft.position)
            if draft_position != self.draft.position:
                self.draft.truncate(draft_position)
        self._mtp = None

    def reset(self):
        self._clear_request()
        self.target.reset()
        if self.draft is not None:
            self.draft.reset()


class CannRankEngine(AscendRankEngine):
    """Start CANN MTP at P using the transferred parent and full main KV."""

    def inject(self, prefix):
        self._clear_request()
        if prefix.prompt_token_ids is not None:
            self._request_prompt = prefix.prompt_token_ids
        self.target.inject(prefix.main)
        if self.draft is not None:
            self.draft.inject(prefix.mtp)
        if (self.target.position != prefix.seq_len
                or (self.draft is not None
                    and self.draft.position != prefix.seq_len - 1)):
            raise ValueError("CANN main/MTP prefixes require P and P-1 rows")
        self.parent_hidden = prefix.parent_hidden.clone()
        result = self.target.predict(self.parent_hidden)
        self.first_token_id = int(result.token_ids[0].item())
        self._prompt_length = prefix.seq_len
