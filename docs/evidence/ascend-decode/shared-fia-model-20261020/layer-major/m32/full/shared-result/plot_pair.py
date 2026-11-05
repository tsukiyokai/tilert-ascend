"""Plot the recorded full-model SFA/shared-FIA numerical comparison."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("/workspace/experiments/m32-full-teacher-20261020/shared")
rows = json.loads((root / "paired-diagnostic.json").read_text())["rank0_layer_diagnostics"]
x = [r["layer"] for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
for field, label in (("hidden", "Layer output"), ("merged", "Hidden + residual")):
    axes[0].plot(x, [r[field]["nrmse"] for r in rows], label=label)
    axes[1].plot(x, [r[field]["cosine"] for r in rows], label=label)
for ax, ylabel in zip(axes, ("NRMSE relative to SFA", "Cosine similarity")):
    ax.set(xlabel="Layer (zero-based)", ylabel=ylabel, xlim=(0, 77))
    ax.grid(alpha=0.25)
axes[0].legend(frameon=False)
fig.suptitle("A3 TP16, M32, 8K: SFA 32/32 vs shared-FIA 30/32 teacher labels")
fig.savefig(root / "layer-curves.svg")
