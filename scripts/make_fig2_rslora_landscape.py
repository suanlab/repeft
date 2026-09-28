#!/usr/bin/env python3
"""Figure 2: rsLoRA stability landscape, one panel per rank.

Reads analysis/imported_from_multiagent/rslora_2d_sweep/summary.json
(303 completed trainings across 72 (r, alpha, lr) cells on BERT-base/MNLI 50K).

Cells are split into one panel per rank so that cells sharing (lr, alpha/sqrt r)
but differing in r are not drawn on top of each other: at lr = 1e-3, equal
lr_eff can be stable at one rank and collapsed at another.

Output: paper/figures/fig2_rslora_landscape.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SUMMARY = REPO / "analysis" / "imported_from_multiagent" / "rslora_2d_sweep" / "summary.json"
OUT = REPO / "paper" / "figures" / "fig2_rslora_landscape.pdf"
CRITICAL = 5e-4   # lowest lr_eff at which a collapse was observed (BERT-base, AdamW)
MAJORITY = 0.34   # MNLI majority-class baseline
FS = 9

OUT.parent.mkdir(parents=True, exist_ok=True)
records = json.loads(SUMMARY.read_text())
ranks = sorted({int(r["rank"]) for r in records})

fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0), sharex=True, sharey=True)
lrs = sorted({float(r["lr"]) for r in records})
ratios = sorted({float(r["ratio"]) for r in records})
x_lo, x_hi = min(lrs) * 0.6, max(lrs) * 1.6
y_lo, y_hi = min(ratios) * 0.6, max(ratios) * 1.6
lr_line = np.geomspace(x_lo, x_hi, 200)

sc = None
n_stable = n_coll = 0
for ax, rank in zip(axes.flat, ranks):
    cells = [r for r in records if int(r["rank"]) == rank]
    st = [r for r in cells if r["stable"]]
    co = [r for r in cells if not r["stable"]]
    n_stable += len(st); n_coll += len(co)
    sc = ax.scatter([r["lr"] for r in st], [r["ratio"] for r in st], c=[r["mean"] for r in st],
                    cmap="viridis", vmin=MAJORITY, vmax=0.80, s=70, marker="s",
                    edgecolors="black", linewidths=0.6, zorder=3)
    ax.scatter([r["lr"] for r in co], [r["ratio"] for r in co], c="crimson", s=95, marker="X",
               edgecolors="black", linewidths=0.6, zorder=4)
    ax.plot(lr_line, CRITICAL / lr_line, "--", color="black", linewidth=1.4, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_title(f"$r={rank}$", fontsize=FS + 1)
    ax.grid(True, which="both", linestyle=":", color="gray", alpha=0.4)
    ax.tick_params(labelsize=FS - 1)

for ax in axes[1, :]:
    ax.set_xlabel("learning rate (AdamW)", fontsize=FS)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\alpha/\sqrt{r}$", fontsize=FS + 1)

from matplotlib.lines import Line2D
handles = [
    Line2D([], [], marker="s", linestyle="none", markerfacecolor="#3b528b", markeredgecolor="black",
           markersize=7, label="stable (color = mean acc.)"),
    Line2D([], [], marker="X", linestyle="none", markerfacecolor="crimson", markeredgecolor="black",
           markersize=8, label=r"$\geq$1 seed collapsed"),
    Line2D([], [], linestyle="--", color="black", label=r"$\mathrm{lr}_\mathrm{eff}=5\times10^{-4}$"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=FS, frameon=False,
           bbox_to_anchor=(0.45, -0.01))
fig.tight_layout(rect=(0, 0.05, 0.9, 1))
cax = fig.add_axes([0.915, 0.18, 0.02, 0.72])
cb = fig.colorbar(sc, cax=cax)
cb.set_label("mean accuracy", fontsize=FS)
cb.ax.tick_params(labelsize=FS - 1)

plt.savefig(OUT, bbox_inches="tight")
print(f"Wrote {OUT}  (n_stable={n_stable}, n_collapsed={n_coll})")
