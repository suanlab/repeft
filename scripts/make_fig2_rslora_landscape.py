#!/usr/bin/env python3
"""Figure 2: rsLoRA stability landscape — lr × α/√r product determines collapse.

Reads /projects/REPEFT/analysis/imported_from_multiagent/rslora_2d_sweep/summary.json
which contains 303 completed trainings across 72 cells (r×α×lr grid) on BERT-base/MNLI 50K.

Output: paper/figures/fig2_rslora_landscape.pdf
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent.parent
SUMMARY = REPO / "analysis" / "imported_from_multiagent" / "rslora_2d_sweep" / "summary.json"
OUT = REPO / "paper" / "figures" / "fig2_rslora_landscape.pdf"
CRITICAL = 5e-4  # critical lr_eff product (BERT-base, AdamW)
MAJORITY = 0.34   # MNLI majority-class baseline

OUT.parent.mkdir(parents=True, exist_ok=True)
records = json.loads(SUMMARY.read_text())

# Build (lr_eff, mean acc, stable) per cell
xs, ys, accs, stable = [], [], [], []
for r in records:
    lr = float(r["lr"])
    ratio = float(r["ratio"])    # α/√r
    acc = float(r["mean"])
    xs.append(lr)
    ys.append(ratio)
    accs.append(acc)
    stable.append(bool(r.get("stable", acc > MAJORITY + 0.02)))

xs, ys, accs = np.array(xs), np.array(ys), np.array(accs)

# Single-column width preserved; taller aspect + larger internal elements
# so the figure is visually more prominent when scaled to width=\linewidth.
fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.4))

# Scatter: color = accuracy; marker = stable square / collapsed X
sc_stable = ax.scatter(
    xs[np.array(stable)], ys[np.array(stable)],
    c=accs[np.array(stable)], cmap="viridis",
    vmin=MAJORITY, vmax=0.80, s=140, marker="s", edgecolors="black", linewidths=0.8,
    label="stable",
)
collapsed = ~np.array(stable)
ax.scatter(
    xs[collapsed], ys[collapsed],
    c="crimson", s=180, marker="X", edgecolors="black", linewidths=0.8,
    label="collapsed (≈ majority class)",
)

# Critical iso-product line: lr × ratio = CRITICAL  →  ratio = CRITICAL / lr
lr_line = np.geomspace(min(xs) * 0.7, max(xs) * 1.4, 200)
ratio_line = CRITICAL / lr_line
ax.plot(lr_line, ratio_line, "--", color="black", linewidth=2.0,
        label=rf"$\mathrm{{lr}}_\mathrm{{eff}}={CRITICAL:.0e}$ (critical)")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"learning rate (AdamW)", fontsize=12)
ax.set_ylabel(r"$\alpha/\sqrt{r}$", fontsize=13)
ax.set_title(r"rsLoRA stability landscape ($n{=}303$ across 72 cells, BERT-base / MNLI 50K)",
             fontsize=11)
ax.set_xlim(min(xs) * 0.7, max(xs) * 1.4)
ax.set_ylim(min(ys) * 0.7, max(ys) * 1.4)
ax.tick_params(axis="both", labelsize=10.5)
ax.grid(True, which="both", linestyle=":", color="gray", alpha=0.4)

cbar = plt.colorbar(sc_stable, ax=ax, pad=0.02)
cbar.set_label("mean accuracy (stable cells)", fontsize=11)
cbar.ax.tick_params(labelsize=9.5)

ax.legend(loc="lower left", fontsize=10, framealpha=0.9, borderaxespad=0.4)

# Annotate critical product
ax.text(1.1e-3, CRITICAL / 1.1e-3 * 1.6, r"unsafe region (above curve $\to$ collapse)",
        fontsize=10, color="crimson", ha="right", va="bottom")
ax.text(1.1e-3, CRITICAL / 1.1e-3 / 1.6, r"safe region (below curve $\to$ stable)",
        fontsize=10, color="darkgreen", ha="right", va="top")

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print(f"Wrote {OUT}  (n_stable={int(np.array(stable).sum())}, n_collapsed={int(collapsed.sum())})")
