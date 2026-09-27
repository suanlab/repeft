#!/usr/bin/env python3
"""Figure 1: 16-cell adjudication forest plot.

Produces a forest-plot-style figure with one row per claim cell. Each row shows
the paired mean Δ (candidate − LoRA), its 95% CI bar (BC bootstrap where
available, percentile elsewhere), and colored adjudication (supported / unsupported
/ reversed).

Data source (audit C1 fix): loads from analysis/fig1_data.json instead of
hard-coded constants, so corrections to the underlying paired summaries flow
into the figure automatically.

Output: paper/figures/fig1_adjudication.pdf
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "analysis" / "fig1_data.json"
OUT = REPO / "paper" / "figures" / "fig1_adjudication.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

with DATA.open() as _f:
    _payload = json.load(_f)
# Tuple-of-tuples shape preserved for downstream loops.
cells = [
    (c["id"], c["label"], c["delta"], c["ci_lo"], c["ci_hi"], c["adj"])
    for c in _payload["cells"]
]
assert len(cells) == 16, f"Expected 16 cells in {DATA}, got {len(cells)}"

colors = {"sup": "#2ecc71", "unsup": "#bdc3c7", "rev": "#e74c3c"}
adj_label = {"sup": "supported", "unsup": "unsupported", "rev": "reversed"}

# Sized for a single ACL column (~3.0in) so text prints at ~7pt without downscaling.
FS = 7.2
fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.95))

# Compact row labels using the same vocabulary as Table 1.
MODEL = {"roberta-base": "roberta", "bert-base": "bert", "Qwen2.5-3B": "Qwen3B",
         "phi-2": "phi-2", "Mistral-7B": "Mistral-7B", "Gemma-7B": "Gemma-7B"}
TASK = {"SQuADv1": "SQuAD1", "SQuADv2": "SQuAD2"}
REGIME = {"common": "C", "common (MA)": "C", "faithful": "F",
          "faithful (claim-match)": "F", "r=32": "r32"}

# Plot rows top-down (cell 1 at top)
ys = np.arange(len(cells))[::-1]  # invert so cell 1 is top
for y, (cid, label, d, lo, hi, adj) in zip(ys, cells):
    color = colors[adj]
    if d < -10:
        # Off-scale (Cell 5): left-pointing triangle at the axis edge, true value annotated.
        ax.scatter([-7.5], [y], color=color, marker="<", s=28, zorder=3,
                   edgecolor="black", linewidth=0.5)
        ax.annotate(f"{d:.1f}", xy=(-7.5, y), xytext=(-7.0, y), color=color, fontsize=FS - 0.5,
                    va="center", ha="left")
    else:
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.6, zorder=2)
        ax.scatter([d], [y], color=color, marker="o", s=16, edgecolor="black",
                   linewidth=0.5, zorder=3)

ax.axvline(0, color="black", linewidth=0.7, linestyle="-")

def _fmt(cid, label):
    task, method, model, regime = (label.split(None, 3) + ["", "", "", ""])[:4]
    regime = regime.strip()
    return (f"{cid:>2} {MODEL.get(model, model):<10} {method:<7} "
            f"{TASK.get(task, task):<6} {REGIME.get(regime, regime):<3}")
ax.set_yticks(ys)
ax.set_yticklabels([_fmt(cid, label) for cid, label, *_ in cells],
                   family="monospace", fontsize=FS - 0.4)
ax.tick_params(axis="y", length=0, pad=7)  # gap between row labels and the box
ax.set_ylim(-0.7, len(cells) - 0.3)

ax.set_xlim(-8.7, 4)  # left pad for the off-scale Cell 5 marker
ax.set_xticks([-7.5, -5, -2.5, 0, 2.5])
ax.set_xticklabels(["≤−10", "", "−2.5", "0", "2.5"])  # −5 left unlabeled to avoid crowding
ax.tick_params(axis="x", labelsize=FS)
# Right-align the x-label and legend to the box edge so nothing sticks out to the right.
ax.set_xlabel(r"paired $\Delta$ (candidate $-$ LoRA), 95% bootstrap CI", fontsize=FS, loc="right", labelpad=2)
ax.grid(axis="x", linestyle=":", alpha=0.4)

# Cell 16 headline marker
star_y = ys[15]
ax.scatter([-1.58], [star_y], color="#e74c3c", marker="*", s=70, edgecolor="black",
           linewidth=0.6, zorder=4)

# Legend outside the axes, below the x-label
from matplotlib.patches import Patch
legend = [Patch(facecolor=colors[k], label=adj_label[k]) for k in ("sup", "unsup", "rev")]
ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(1.02, -0.17), ncol=3,
          fontsize=FS, frameon=False, handlelength=1.2, columnspacing=1.0, handletextpad=0.4)

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print(f"Wrote {OUT}  (16 cells: sup={sum(1 for c in cells if c[5]=='sup')}, "
      f"unsup={sum(1 for c in cells if c[5]=='unsup')}, "
      f"rev={sum(1 for c in cells if c[5]=='rev')})")
