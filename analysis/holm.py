#!/usr/bin/env python3
"""Holm-Bonferroni and Benjamini-Hochberg multiplicity correction over the
confirmatory cell family.

Reads `analysis/paired_t_sensitivity.csv`, applies Holm-Bonferroni at
alpha=0.05 across the 15 confirmatory cells (Cell 15 is exploratory and
excluded per PREREGISTRATION.md Section 1.2), and prints / saves the
adjudication-after-correction.

Usage:
    python3 analysis/holm.py
    python3 analysis/holm.py --alpha 0.05 --csv analysis/paired_t_sensitivity.csv \
                             --out analysis/holm_thresholds.csv

Output (stdout + CSV) includes:
    cell_id, method, model, task, raw_p, holm_rank, holm_threshold,
    survives_holm, bh_threshold, survives_bh, adjudication_pre_correction,
    adjudication_post_correction.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


# Cells excluded from the confirmatory Holm family per PREREGISTRATION.md.
EXPLORATORY_CELLS = {15}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parent.parent
    p.add_argument("--csv", default=str(repo / "analysis" / "paired_t_sensitivity.csv"))
    p.add_argument("--out", default=str(repo / "analysis" / "holm_thresholds.csv"))
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--exclude-cells", type=int, nargs="*", default=sorted(EXPLORATORY_CELLS),
                   help="Cell IDs to exclude from the confirmatory family.")
    p.add_argument("--tier2", action="store_true",
                   help="Compute Holm separately for the 14-cell Tier-2 OOD family "
                        "(reads analysis/tier2_paired/*.json instead of confirmatory CSV).")
    return p.parse_args()


def _tier2_holm(repo: Path, alpha: float) -> int:
    """B.2/G.2: separate Holm correction over the 14 Tier-2 OOD cells.

    Reads analysis/tier2_paired/*.json (output of project_a_pairwise_summary.py
    for the Tier-2 family) and applies Holm-Bonferroni at alpha=0.05 within
    this family.
    """
    import json
    tier2_dir = repo / "analysis" / "tier2_paired"
    if not tier2_dir.exists():
        raise SystemExit(f"Tier-2 directory not found: {tier2_dir}")

    family: list[tuple[str, float, dict]] = []
    for jf in sorted(tier2_dir.glob("*.json")):
        try:
            d = json.loads(jf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "sign_flip_p" not in d:
            continue
        # Skip cells with NaN mean_delta (e.g., ANLI R1 with NaN-accuracy seed)
        md = d.get("mean_delta")
        if md is None or (isinstance(md, float) and md != md):  # NaN check
            continue
        family.append((jf.stem, float(d["sign_flip_p"]), d))
    if not family:
        raise SystemExit("No Tier-2 JSONs with sign_flip_p found")

    m = len(family)
    sorted_family = sorted(family, key=lambda kv: kv[1])
    print(f"Tier-2 OOD family: {m} cells")
    print(f"alpha (FWER) = {alpha}")
    print()
    hdr = ("cell", "p", "holm_thr", "survives", "delta", "w/l/t")
    print("  ".join(f"{h:>14s}" for h in hdr))
    print("-" * 90)
    survives_so_far = True
    out_rows = []
    n_surv = 0
    for rank, (name, p, d) in enumerate(sorted_family, start=1):
        thr = alpha / (m - rank + 1)
        if survives_so_far and p <= thr:
            surv = True
            n_surv += 1
        else:
            surv = False
            survives_so_far = False
        delta = d.get("mean_delta", float("nan"))
        wlt = f"{d.get('wins',0)}/{d.get('losses',0)}/{d.get('ties',0)}"
        short_name = name[:40]
        print(f"{short_name:>14s}  {p:>14.4g}  {thr:>14.4g}  "
              f"{'YES' if surv else 'no':>14s}  {delta:>+14.4f}  {wlt:>14s}")
        out_rows.append({"cell_name": name, "p": p, "holm_threshold": thr,
                         "survives": surv, "mean_delta": delta})
    print()
    print(f"Tier-2 Holm survivors: {n_surv}/{m}")
    out_csv = repo / "analysis" / "tier2_holm_thresholds.csv"
    import csv as _csv
    with out_csv.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["cell_name", "p", "holm_threshold",
                                            "survives", "mean_delta"])
        w.writeheader()
        for r in out_rows:
            r["p"] = f"{r['p']:.6g}"
            r["holm_threshold"] = f"{r['holm_threshold']:.6g}"
            r["mean_delta"] = f"{r['mean_delta']:.6f}"
            w.writerow(r)
    print(f"Wrote {out_csv}")
    return 0


def holm_bonferroni(pvalues: list[tuple[Any, float]], alpha: float) -> list[dict[str, Any]]:
    """Apply Holm-Bonferroni step-down at family-wise error rate alpha.

    Returns a list of records keyed by `(id, raw_p)` with `holm_rank`,
    `holm_threshold = alpha / (m - rank + 1)`, and `survives_holm` boolean.
    """
    if not pvalues:
        return []
    m = len(pvalues)
    sorted_p = sorted(pvalues, key=lambda kv: kv[1])
    results = []
    survives_so_far = True
    for rank, (cid, raw_p) in enumerate(sorted_p, start=1):
        threshold = alpha / (m - rank + 1)
        # Step-down: once a hypothesis fails, all subsequent (larger p) also fail.
        if survives_so_far and raw_p <= threshold:
            survives = True
        else:
            survives = False
            survives_so_far = False
        results.append({
            "id": cid,
            "raw_p": raw_p,
            "holm_rank": rank,
            "holm_threshold": threshold,
            "survives_holm": survives,
        })
    return results


def benjamini_hochberg(pvalues: list[tuple[Any, float]], q: float) -> dict[Any, dict[str, Any]]:
    """Apply BH-FDR step-up at false-discovery rate q.

    Returns dict keyed by id with `bh_threshold` and `survives_bh` boolean.
    """
    if not pvalues:
        return {}
    m = len(pvalues)
    sorted_p = sorted(pvalues, key=lambda kv: kv[1])
    # Step-up: find largest rank k such that p_(k) <= (k/m) * q; reject all hypotheses up to k.
    largest_passing_rank = 0
    for rank, (_cid, raw_p) in enumerate(sorted_p, start=1):
        threshold = (rank / m) * q
        if raw_p <= threshold:
            largest_passing_rank = rank
    out: dict[Any, dict[str, Any]] = {}
    for rank, (cid, raw_p) in enumerate(sorted_p, start=1):
        threshold = (rank / m) * q
        out[cid] = {
            "bh_threshold": threshold,
            "survives_bh": (rank <= largest_passing_rank),
        }
    return out


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    if args.tier2:
        return _tier2_holm(repo_root, args.alpha)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        raise SystemExit(f"Input CSV is empty: {csv_path}")

    # Build the confirmatory family.
    excluded = set(args.exclude_cells)
    family_rows = []
    for r in rows:
        try:
            cid = int(r["cell_id"])
        except (KeyError, ValueError):
            continue
        if cid in excluded:
            continue
        try:
            p_raw = float(r["sign_flip_p"])
        except (KeyError, ValueError):
            continue
        family_rows.append((cid, p_raw, r))

    pvalues_for_correction = [(cid, p) for (cid, p, _r) in family_rows]
    holm_results = holm_bonferroni(pvalues_for_correction, args.alpha)
    bh_results = benjamini_hochberg(pvalues_for_correction, args.alpha)
    holm_by_id = {h["id"]: h for h in holm_results}

    # Compose final per-cell records.
    out_rows: list[dict[str, Any]] = []
    for (cid, p_raw, raw_row) in family_rows:
        h = holm_by_id.get(cid, {})
        b = bh_results.get(cid, {})
        rec = {
            "cell_id": cid,
            "method": raw_row.get("method", ""),
            "model": raw_row.get("model", ""),
            "task": raw_row.get("task", ""),
            "regime": raw_row.get("regime", ""),
            "N": raw_row.get("N", ""),
            "mean_delta": raw_row.get("mean_delta", ""),
            "sign_flip_p": f"{p_raw:.6g}",
            "holm_rank": h.get("holm_rank", ""),
            "holm_threshold": f"{h.get('holm_threshold', float('nan')):.6g}",
            "survives_holm": h.get("survives_holm", False),
            "bh_threshold": f"{b.get('bh_threshold', float('nan')):.6g}",
            "survives_bh": b.get("survives_bh", False),
            "adjudication_pre": raw_row.get("adjudication_signflip", raw_row.get("adjudication_t", "")),
        }
        out_rows.append(rec)

    # Sort by raw p ascending for human reading.
    out_rows.sort(key=lambda r: float(r["sign_flip_p"]))

    # Write CSV.
    out_path = Path(args.out)
    fieldnames = list(out_rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    # Print summary.
    m = len(out_rows)
    n_holm = sum(1 for r in out_rows if r["survives_holm"])
    n_bh = sum(1 for r in out_rows if r["survives_bh"])
    print(f"Confirmatory family size: {m} cells (excluded: {sorted(excluded)})")
    print(f"alpha (FWER) = {args.alpha}; alpha (FDR) = {args.alpha}")
    print(f"Holm-Bonferroni: {n_holm}/{m} survive")
    print(f"BH-FDR:          {n_bh}/{m} survive")
    print()
    hdr = ("cell", "method", "model", "task", "raw_p", "holm_thr", "holm?", "bh_thr", "bh?")
    print("  ".join(f"{h:>10s}" for h in hdr))
    for r in out_rows:
        cells = (
            r["cell_id"], r["method"], r["model"], r["task"],
            r["sign_flip_p"], r["holm_threshold"], "YES" if r["survives_holm"] else "no",
            r["bh_threshold"], "YES" if r["survives_bh"] else "no",
        )
        print("  ".join(f"{str(c):>10s}" for c in cells))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
