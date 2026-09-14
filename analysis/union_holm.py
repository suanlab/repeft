#!/usr/bin/env python3
"""Union-family Holm correction over the joint family of confirmatory
(15-cell, Cells 1-14 + Cell 16) and Tier-2 (14-cell + ANLI R1 NaN-dropped
PiSSA/DoRA = 16 cells if we include the NaN-dropped ones; we report 18
because the underlying tier2_paired directory contains all comparisons
including DoRA/PiSSA × {MNLI-matched, MNLI-mismatched, SNLI, ANLI R2, R3,
Yelp, Amazon-CF, SST-2 ID}).

Reviewer R2 raised the concern that running Holm separately on each
family inflates the effective FWER. This script answers that objection
by computing the most conservative Holm correction across the union.

Usage:
    python3 analysis/union_holm.py
    python3 analysis/union_holm.py --alpha 0.05 --out analysis/union_holm_thresholds.csv

Output: CSV with columns
    family, cell, raw_p, holm_rank, holm_threshold, survives_union_holm
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
EXPLORATORY_CELLS = {15}


def collect_confirmatory(csv_path: Path) -> list[tuple[str, float]]:
    """Confirmatory family (Cells 1-14 + Cell 16). Cell 15 is exploratory
    and excluded by convention."""
    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            cid = int(r["cell_id"])
            if cid in EXPLORATORY_CELLS:
                continue
            rows.append((
                f"Conf_Cell{cid:02d}_{r['method']}_{r['model']}_{r['task']}",
                float(r["sign_flip_p"]),
            ))
    return rows


def collect_tier2(dir_path: Path) -> list[tuple[str, float]]:
    """Tier-2 family. Excludes raw anli_r1 (NaN baseline) — replaced by the
    pre-specified NaN-dropped re-analysis (PiSSA + DoRA)."""
    rows = []
    for jf in sorted(dir_path.glob("*.json")):
        if "nan_dropped" in jf.name:
            continue
        if "anli_r1_accuracy" in jf.name:
            continue  # excluded; NaN-dropped variant is added below
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        md = d.get("mean_delta")
        p = d.get("sign_flip_p")
        if p is None or md is None:
            continue
        if isinstance(md, float) and md != md:  # NaN
            continue
        rows.append((f"Tier2_{jf.stem[:50]}", float(p)))
    # Add ANLI R1 NaN-dropped results (PiSSA + DoRA)
    for label, jname in [
        ("Tier2_anli_r1_pissa_nan_dropped",
         "anli_r1_pissa_nan_dropped.json"),
        ("Tier2_anli_r1_dora_nan_dropped",
         "anli_r1_dora_nan_dropped.json"),
    ]:
        jf = dir_path / jname
        if not jf.exists():
            continue
        d = json.loads(jf.read_text())
        p = d.get("sign_flip_p_two_sided")
        if p is not None:
            rows.append((label, float(p)))
    return rows


def union_holm(rows: list[tuple[str, float]], alpha: float) -> list[dict]:
    m = len(rows)
    sorted_rows = sorted(rows, key=lambda x: x[1])
    out = []
    survives = True
    for rank, (name, p) in enumerate(sorted_rows, start=1):
        thr = alpha / (m - rank + 1)
        if survives and p <= thr:
            surv = True
        else:
            survives = False
            surv = False
        out.append({
            "cell": name,
            "raw_p": p,
            "holm_rank": rank,
            "holm_threshold": thr,
            "survives_union_holm": surv,
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=str(REPO / "analysis" / "paired_t_sensitivity.csv"))
    p.add_argument("--tier2-dir", default=str(REPO / "analysis" / "tier2_paired"))
    p.add_argument("--out", default=str(REPO / "analysis" / "union_holm_thresholds.csv"))
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    conf = collect_confirmatory(Path(args.csv))
    tier2 = collect_tier2(Path(args.tier2_dir))
    union = conf + tier2
    m = len(union)

    print(f"Confirmatory family: {len(conf)} cells (excluded Cell 15 exploratory)")
    print(f"Tier-2 family: {len(tier2)} cells (incl. ANLI R1 NaN-dropped)")
    print(f"Union family: {m} cells\n")

    results = union_holm(union, args.alpha)
    survivors = [r for r in results if r["survives_union_holm"]]
    print(f"Union-family Holm survivors at alpha={args.alpha}: "
          f"{len(survivors)}/{m}\n")
    print(f"{'cell':>50s}  {'raw_p':>10s}  {'thr':>10s}  {'surv':>5s}")
    print("-" * 85)
    for r in results:
        nm = r["cell"][:50]
        print(f"{nm:>50s}  {r['raw_p']:>10.4g}  {r['holm_threshold']:>10.4g}  "
              f"{'YES' if r['survives_union_holm'] else 'no':>5s}")

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow({**r,
                         "raw_p": f"{r['raw_p']:.6g}",
                         "holm_threshold": f"{r['holm_threshold']:.6g}"})
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
