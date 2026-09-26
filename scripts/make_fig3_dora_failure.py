#!/usr/bin/env python3
"""Figure 3: DoRA failure rate vs α/r at r=64, Wilson 95% CI bars.

Reads /projects/REPEFT/analysis/imported_from_multiagent/dora_alpha_sweep/summary.json
which contains DoRA stability sweeps on BERT-base/MNLI, r=64, varying α.

Output: paper/figures/fig3_dora_failure.pdf
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
SUMMARY = REPO / "analysis" / "imported_from_multiagent" / "dora_alpha_sweep" / "summary.json"
OUT = REPO / "paper" / "figures" / "fig3_dora_failure.pdf"

OUT.parent.mkdir(parents=True, exist_ok=True)
rows = json.loads(SUMMARY.read_text())

# Each row: alpha, alpha_over_r, n, failures, failure_rate, ci_low, ci_high, mean_acc_success
xs = np.array([r["alpha_over_r"] for r in rows], dtype=float)
ns = np.array([r["n"] for r in rows], dtype=int)
fails = np.array([r["failures"] for r in rows], dtype=int)
rates = np.array([r["failure_rate"] for r in rows], dtype=float)
ci_lo = np.array([r["ci_low"] for r in rows], dtype=float)
ci_hi = np.array([r["ci_high"] for r in rows], dtype=float)

# Sort by α/r ascending
order = np.argsort(xs)
xs, ns, fails, rates, ci_lo, ci_hi = (a[order] for a in (xs, ns, fails, rates, ci_lo, ci_hi))

# Wilson bars: lower/upper error (relative to rate)
err_lo = rates - ci_lo
err_hi = ci_hi - rates

fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.6))

colors = ["#2ecc71" if r == 0 else ("#f39c12" if r < 0.2 else "#e74c3c") for r in rates]
bars = ax.bar(xs, rates, width=0.10, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
ax.errorbar(xs, rates, yerr=[err_lo, err_hi], fmt="none", ecolor="black",
            elinewidth=1.2, capsize=4, capthick=1.1)

# Annotate fail/n above each bar
for x, n, f, r, hi in zip(xs, ns, fails, rates, ci_hi):
    ax.text(x, hi + 0.03, f"{f}/{n}", ha="center", va="bottom", fontsize=9)

# Highlight α/r = 1.0 (the standard PEFT convention)
ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
ax.text(1.0, 1.06, r"$\alpha{=}r$ convention", ha="center", va="bottom", fontsize=8, color="gray")

ax.set_xlabel(r"$\alpha / r$  (at $r{=}64$, BERT-base / MNLI)")
ax.set_ylabel("catastrophic failure rate")
ax.set_title(r"DoRA: $\alpha$-driven failure at high rank (Wilson 95% CI)")
ax.set_ylim(0.0, 1.12)
ax.set_xlim(0.40, 1.60)
ax.set_xticks(xs)
ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=8)
ax.grid(True, axis="y", linestyle=":", color="gray", alpha=0.4)

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print(f"Wrote {OUT}  (rows={len(rows)}; failure rate at α/r=1.0: {rates[np.where(xs==1.0)][0]:.1%}, CI=[{ci_lo[np.where(xs==1.0)][0]:.1%}, {ci_hi[np.where(xs==1.0)][0]:.1%}])")
