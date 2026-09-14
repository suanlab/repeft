#!/usr/bin/env python3
"""Build sweep-style results.csv for the Cell 5 UNCAPPED re-run arms.

Mirrors the CSV schema already used by sweep_runs/cell5_schedule_faithful/results.csv
so that project_a_pairwise_summary.py can consume it unchanged.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
FIELDS = ["job_id", "host_id", "ip", "status", "return_code", "note",
          "seed", "model", "eval_f1", "eval_exact_match"]

ARMS = {
    "sweep_runs/cell5_uncapped_adalora": "uncapped_tinit250_tfinal750",
    "sweep_runs/cell5_uncapped_lora": "uncapped_lora_baseline",
}


def build(arm_dir: str, note: str) -> int:
    root = REPO / arm_dir
    rows = []
    for mf in sorted(root.glob("seed_*/metrics.json"),
                     key=lambda p: int(p.parent.name.split("_")[1])):
        d = json.loads(mf.read_text())
        seed = int(mf.parent.name.split("_")[1])
        # Guard: only accept runs that actually produced an F1
        f1 = d.get("eval_f1")
        if f1 is None:
            continue
        rows.append({
            "job_id": f"seed={seed}", "host_id": "localhost", "ip": "A6000",
            "status": "OK", "return_code": 0, "note": note,
            "seed": seed, "model": d.get("model", ""),
            "eval_f1": f1, "eval_exact_match": d.get("eval_exact_match", ""),
        })
    out = root / "results.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} rows")
    return len(rows)


def main() -> None:
    counts = [build(d, n) for d, n in ARMS.items()]
    print(f"\nshared-seed pairing requires both arms complete: {counts}")


if __name__ == "__main__":
    main()
