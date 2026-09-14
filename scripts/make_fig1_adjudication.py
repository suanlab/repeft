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

fig, ax = plt.subplots(1, 1, figsize=(10.0, 7.8))

# Plot rows top-down (cell 1 at top)
ys = np.arange(len(cells))[::-1]  # invert so cell 1 is top
for y, (cid, label, d, lo, hi, adj) in zip(ys, cells):
    color = colors[adj]
    # Special: cell 5 is so far reversed we clip it for visibility.
    # Left-pointing triangle indicates the actual value lies further left
    # (off-scale); annotated value shows the true Δ.
    d_plot, lo_plot, hi_plot = d, lo, hi
    if d < -10:
        ax.annotate(f"{d:.1f}", xy=(-7.5, y), xytext=(-7.3, y), color=color, fontsize=12,
                    va="center", ha="left")
        d_plot = -7.5
        lo_plot = -7.5
        hi_plot = -7.5
        ax.scatter([d_plot], [y], color=color, marker="<", s=90, zorder=3,
                   edgecolor="black", linewidth=0.6)
    else:
        # Draw CI as horizontal bar
        ax.plot([lo_plot, hi_plot], [y, y], color=color, linewidth=2.0, zorder=2)
        ax.scatter([d_plot], [y], color=color, marker="o", s=44, edgecolor="black",
                   linewidth=0.6, zorder=3)

ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

# Row labels (cell number + descriptor)
ax.set_yticks(ys)
ax.set_yticklabels([f"#{cid}  {label}" for cid, label, *_ in cells],
                   family="monospace", fontsize=11.5)

ax.set_xlim(-8, 5)
ax.set_xlabel(r"paired $\Delta$ (candidate $-$ LoRA)  /  BC 95\% CI", fontsize=13)
ax.set_title("16-cell PEFT adjudication: paired multi-seed results", fontsize=14)
ax.tick_params(axis="x", labelsize=12)
ax.grid(axis="x", linestyle=":", alpha=0.4)
ax.set_xticks([-7.5, -5, -3, -1, 0, 1, 3, 5])
ax.set_xticklabels(["$\\leq$ $-10$", "$-5$", "$-3$", "$-1$", "0", "1", "3", "5"])

# Legend (manual)
from matplotlib.patches import Patch
legend = [Patch(facecolor=colors[k], label=adj_label[k]) for k in ("sup", "unsup", "rev")]
ax.legend(handles=legend, loc="upper left", fontsize=12.5, framealpha=0.9)

# Annotate Cell 16 (★ headline)
star_y = ys[15]
ax.scatter([-1.58], [star_y], color="#e74c3c", marker="*", s=260, edgecolor="black",
           linewidth=0.9, zorder=4)
ax.text(-2.0, star_y - 0.55, "Headline 7B reversal", fontsize=11.5, color="#e74c3c",
        ha="center", style="italic")

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print(f"Wrote {OUT}  (16 cells: sup={sum(1 for c in cells if c[5]=='sup')}, "
      f"unsup={sum(1 for c in cells if c[5]=='unsup')}, "
      f"rev={sum(1 for c in cells if c[5]=='rev')})")
