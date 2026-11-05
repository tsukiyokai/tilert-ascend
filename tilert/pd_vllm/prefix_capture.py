"""Freeze completed TP16 HIXL prefixes without loading decode weights."""

from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile

from tilert.models.glm_5_2_ascend.session import AscendPDEngine
from .ascend_layout import PARENT_HIDDEN_REGION
from .decode_timing import atomic_json


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def prompt_digest(tokens):
    return hashlib.sha256(json.dumps(
        tokens, separators=(",", ":"),
    ).encode("ascii")).hexdigest()


def copy_to_cpu(value):
    # Cloning breaks the receive allocation's ownership and excludes padding
    # from Torch serialization, including an already-contiguous prefix view.
    return value.detach().cpu().clone()


def save_bundle(bundle, path, save=None):
    if save is None:
        import torch

        save = torch.save
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False,
        ) as file:
            temporary = Path(file.name)
            save(bundle, file)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_rank_prefix(buffer, request, checkpoint, directory, *,
                     copy_tensor=copy_to_cpu, save=None):
    """Copy original P/P-1 rows while the receiver still owns the slot."""
    bundles = buffer.frozen_prefix(request, checkpoint)
    filename = f"rank-{buffer.rank}.pt"
    paths = {"main": Path(directory) / "main-prefix" / filename,
             "mtp": Path(directory) / "mtp-prefix" / filename}
    files = {}
    for role, bundle in bundles.items():
        if role == "main":
            bundle["layers"] = [
                {name: copy_tensor(value) for name, value in layer.items()}
                for layer in bundle["layers"]
            ]
        else:
            bundle["layer"] = {name: copy_tensor(value)
                               for name, value in bundle["layer"].items()}
            if "parent_hidden" in bundle:
                parent = bundle["parent_hidden"]
                parent["tensor"] = copy_tensor(parent["tensor"])
        save_bundle(bundle, paths[role], save)
        files[role] = {"path": str(paths[role]),
                       "sha256": digest(paths[role]),
                       "bytes": paths[role].stat().st_size}
    receipt = {
        "status": "saved", "rank": buffer.rank,
        "request_id": request.request_id, "seq_len": request.seq_len,
        "mtp_seq_len": request.metadata["mtp_seq_len"],
        "generation": request.metadata["generation"],
        "prompt_sha256": prompt_digest(request.metadata["prompt_token_ids"]),
        "checkpoint_sha256": checkpoint, "files": files,
    }
    if PARENT_HIDDEN_REGION in request.metadata:
        receipt[PARENT_HIDDEN_REGION] = request.metadata[PARENT_HIDDEN_REGION]
    return receipt


def validate_receipts(receipts, request, checkpoint):
    expected = {
        "status": "saved", "request_id": request.request_id,
        "seq_len": request.seq_len, "mtp_seq_len": request.seq_len - 1,
        "generation": request.metadata["generation"],
        "prompt_sha256": prompt_digest(request.metadata["prompt_token_ids"]),
        "checkpoint_sha256": checkpoint,
    }
    expected[PARENT_HIDDEN_REGION] = request.metadata.get(PARENT_HIDDEN_REGION)
    if len(receipts) != 16:
        raise RuntimeError("Prefix export requires sixteen rank receipts")
    for rank, receipt in enumerate(receipts):
        if (receipt.get("rank") != rank
                or any(receipt.get(name) != value
                       for name, value in expected.items())):
            raise RuntimeError(
                f"Prefix export failed at rank {rank}: {receipt}"
            )


class CaptureRankEngine:
    """Export local buffers and rendezvous before the first-token response."""

    def __init__(self, buffer, checkpoint, output_dir, gather, *, save=None,
                 copy_tensor=copy_to_cpu):
        self.buffer = buffer
        self.checkpoint = checkpoint
        self.output_dir = Path(output_dir)
        self.gather = gather
        self.save = save
        self.copy_tensor = copy_tensor
        self.capacity = buffer.layout["capacity_tokens"]
        self.position = 0
        self.receipts = []

    def inject(self, request):
        try:
            index = request.metadata["capture_index"]
            directory = self.output_dir / f"case-{index:02d}"
            if self.position:
                raise RuntimeError("Reset the capture before another prefix")
            receipt = save_rank_prefix(
                self.buffer, request, self.checkpoint, directory,
                save=self.save, copy_tensor=self.copy_tensor,
            )
        except Exception as error:
            receipt = {"status": "error", "rank": self.buffer.rank,
                       "error": f"{type(error).__name__}: {error}"}
        self.receipts = self.gather(receipt)
        validate_receipts(self.receipts, request, self.checkpoint)
        self.position = request.seq_len

    def reset(self):
        self.position = 0
        self.receipts = []

    def step(self, token):
        raise RuntimeError("Prefix capture records one prefill token only")


class PrefixCaptureEngine:
    """Use the existing PD HTTP lifecycle for export, without inference.

    inject freezes all sixteen rank buffers before decode records the
    client's prefill token. The server retains the HIXL slot through reset.
    """

    def __init__(self, backend, rank_engine, output_dir, metadata, *,
                 device_context=nullcontext):
        self.backend = backend
        self.rank_engine = rank_engine
        self.output_dir = Path(output_dir)
        self.metadata = metadata
        self.device_context = device_context
        self.session = AscendPDEngine(backend, stop_ids=())
        self.next_index = 0
        self.current = None
        self.directory = None

    def __getattr__(self, name):
        return getattr(self.session, name)

    def _save(self):
        atomic_json(self.directory / "manifest.json", self.current)

    def inject(self, request):
        if self.current is not None:
            raise RuntimeError("Reset the previous prefix capture")
        index = self.next_index
        self.directory = self.output_dir / f"case-{index:02d}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.next_index += 1
        request = replace(request, metadata={
            **request.metadata, "capture_index": index,
            "prompt_token_ids": list(request.metadata["prompt_token_ids"]),
        })
        self.current = {
            "status": "exporting", "request_id": request.request_id,
            "capture_index": index, "seq_len": request.seq_len,
            "generation": request.metadata["generation"],
            "prompt_token_ids": list(request.metadata["prompt_token_ids"]),
            "prompt_sha256": prompt_digest(
                request.metadata["prompt_token_ids"],
            ),
            "first_token_id": None, "runtime": self.metadata,
            "scope": "Original HIXL prefix export; no decode inference",
            "numerical_execution": False, "tpot_measured": False,
        }
        if PARENT_HIDDEN_REGION in request.metadata:
            self.current[PARENT_HIDDEN_REGION] = dict(
                request.metadata[PARENT_HIDDEN_REGION],
            )
        self._save()
        try:
            with self.device_context():
                self.session.inject(request)
            self.current.update(status="prefix_saved",
                                ranks=self.rank_engine.receipts)
            self._save()
        except BaseException as error:
            self.current.update(status="error",
                                error=f"{type(error).__name__}: {error}")
            self._save()
            raise

    def decode(self, first_token_id, max_tokens, on_token=None, **kwargs):
        if max_tokens != 1:
            raise ValueError("Prefix capture requires max_tokens=1")
        if type(first_token_id) is not int or not 0 <= first_token_id < 154880:
            raise ValueError("Prefill token is outside the vocabulary")
        if self.current is None:
            raise RuntimeError("Inject a completed prefix before capture")
        tokens = self.session.decode(
            first_token_id, max_tokens=1, on_token=on_token, **kwargs,
        )
        self.session.last_stats.update(
            capture_index=self.current["capture_index"],
            prompt_sha256=self.current["prompt_sha256"],
        )
        self.current.update(
            status="capture_complete" if tokens == [first_token_id]
            else "cancelled", first_token_id=first_token_id,
            first_token_source="vLLM response supplied by the PD client",
            delivered_tokens=tokens,
            engine_stats=dict(self.session.last_stats),
        )
        self._save()
        return tokens

    def reset(self):
        with self.device_context():
            self.session.reset()
        if self.current is not None:
            self.current["reset_complete"] = True
            self._save()
            self.current = None
