"""Split a decode round into its phases, with no profiler and no harness edit.

At width 8 `advance()` is one seed forward when the chain needs reseeding,
seven chained draft forwards, one full-model verify, one target-conditioned
draft forward, then `host_tokens` and the commit. Each returns a value the
host reads, so each is synchronous and host wall per call is a fair split.

The phase names and the way a draft call is classified follow `_Draft` in
`tilert/models/glm_5_2_ascend/profiling.py`: a call carrying
`external_indices` is a chain step, one after a verify is the conditioned
pass, and otherwise it is a seed.

A profiler is deliberately not used: it adds host work to a device the host
feeds, so it changes the durations it reports. This costs two clock reads a
call, about twenty-two against a 57 ms round.

Printed at exit on rank 0. `PHASE_HOOK=off` disables it.
"""

import atexit
import builtins
import collections
import os
import sys
import time

DRAFT_MODULE = "tilert.models.glm_5_2_ascend.cann_mtp"
TARGET_MODULE = "tilert.models.glm_5_2_ascend.cann_model"

samples = collections.defaultdict(list)
verified = [False]


def _record(name, begin):
    samples[name].append((time.perf_counter_ns() - begin) / 1e6)


def _sync():
    """Charge each phase its own device time rather than the next phase's.

    Without this the verify returns as soon as its graph is dispatched and the
    pass after it blocks on the device, so the hook reported 2.196 ms for a
    full-model forward whose weight read alone is 3.886 -- a figure below its
    own floor, which is how the defect showed. The barrier costs nothing it
    was not already paying: the next phase blocks on the same work.
    """
    import torch

    torch.npu.synchronize()


def _wrap_draft(draft_class):
    original = draft_class.forward

    def forward(self, tokens, parent, **kwargs):
        if kwargs.get("external_indices") is not None:
            name = "chain"
        else:
            name = "conditioned" if verified[0] else "seed"
        begin = time.perf_counter_ns()
        try:
            return original(self, tokens, parent, **kwargs)
        finally:
            _sync()
            _record(name, begin)
            if name == "conditioned":
                verified[0] = False

    draft_class.forward = forward


def _wrap_target(model_class):
    for name in ("verify", "host_tokens"):
        original = getattr(model_class, name)

        def timed(self, *args, _name=name, _original=original, **kwargs):
            begin = time.perf_counter_ns()
            try:
                return _original(self, *args, **kwargs)
            finally:
                _sync()
                _record(_name, begin)
                if _name == "verify":
                    verified[0] = True

        setattr(model_class, name, timed)


def _report():
    """Report each phase's distribution, not its mean.

    A mean over every round mixes the three warmup requests -- which compile
    and capture -- with the measured ones, and the first version of this hook
    reported phases summing to 124 ms against a 57 ms round because of it. The
    tenth percentile is the floor of a phase in steady state and is the
    quantity the round's own p10 can be compared against.
    """
    rounds = len(samples["verify"])
    if not rounds or os.environ.get("RANK", "0") != "0":
        return
    print(f"[phase] {rounds} rounds", file=sys.stderr, flush=True)
    floor = 0.0
    for name in sorted(samples, key=lambda key: -len(samples[key])):
        values = sorted(samples[name])
        per_round = len(values) / rounds
        low = values[len(values) // 10]
        floor += low * per_round
        print(f"[phase] {name:12} p10 {low:8.3f}  median "
              f"{values[len(values) // 2]:8.3f}  p90 "
              f"{values[9 * len(values) // 10]:8.3f} ms a call  "
              f"{per_round:5.2f} calls a round",
              file=sys.stderr, flush=True)
    print(f"[phase] floor sums to {floor:8.3f} ms a round",
          file=sys.stderr, flush=True)


def _install_when_ready():
    """Wrap each class once its own module has finished executing.

    A name appears in `sys.modules` before its body runs, so the guard is on
    the attribute rather than on the module.
    """
    original_import = builtins.__import__
    pending = {"draft": True, "target": True}

    def patched(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        draft = sys.modules.get(DRAFT_MODULE)
        if pending["draft"] and draft is not None and hasattr(draft, "CannMtp"):
            pending["draft"] = False
            _wrap_draft(draft.CannMtp)
        target = sys.modules.get(TARGET_MODULE)
        if pending["target"] and target is not None and hasattr(target, "CannModel"):
            pending["target"] = False
            _wrap_target(target.CannModel)
        if not any(pending.values()):
            builtins.__import__ = original_import
            print("[phase] hook installed", file=sys.stderr, flush=True)
        return module

    builtins.__import__ = patched


def _chain_other_hooks():
    """Run any other sitecustomize this run needs, by path.

    Python imports the name `sitecustomize` once, so two directories on
    PYTHONPATH each holding one means only the first is ever executed. The
    static-kernel cache fix is installed by another one, and silently losing it
    would change what the measured binary is -- which is exactly the class of
    error this instrument exists to avoid.
    """
    import importlib.util

    for path in os.environ.get("PHASE_HOOK_CHAIN", "").split(os.pathsep):
        if not path or not os.path.exists(path):
            continue
        spec = importlib.util.spec_from_file_location("_chained_hook", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"[phase] chained {path}", file=sys.stderr, flush=True)


if os.environ.get("PHASE_HOOK", "on") != "off":
    atexit.register(_report)
    _install_when_ready()
_chain_other_hooks()
