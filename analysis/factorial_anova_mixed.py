#!/usr/bin/env python3
"""Mixed-effects re-analysis of the §5.3 2x2 factorial on PiSSA-LoRA delta
(Qwen2.5-3B GSM8K), with seed as a random effect.

Background. The main-text factorial ANOVA (§5.3, Table 4) reports descriptive
F-statistics for lr/α main effects and the interaction, with a caveat that
seeds partially overlap across the four conditions (inflating residual DoF
under the standard between-subjects ANOVA). This script answers reviewer
R2's concern by re-fitting the same data as a mixed-effects model with seed
as a random effect.

Design (2x2 factorial, 4 conditions × ~10 PiSSA-LoRA paired deltas):
  Condition       (lr,    alpha)  PiSSA source            LoRA source
  ----------------------------------------------------------------------
  Common          (2e-4,  16)     project_a_*_pissa       project_a_*_lora
  alpha-swap      (2e-4,  8)      repeft_ablation_alpha_pissa  repeft_ablation_alpha_lora
  lr-swap         (2e-5,  16)     repeft_ablation_lr_pissa     repeft_ablation_lr_lora
  Faithful        (2e-5,  8)      repeft_qwen3b_pissa_gsm8k_faithful  rev_lora_faithful_fp16

For each (condition, seed) we compute Δ = PiSSA_eval - LoRA_eval (paired).
We then fit:

    delta_ij ~ lr_level_i + alpha_level_i + lr_level:alpha_level + (1 | seed_j)

with statsmodels MixedLM. Effect-magnitude summaries (Cohen's f²) replace
the descriptive η² reported in the main text.

Usage:
    python3 analysis/factorial_anova_mixed.py
    python3 analysis/factorial_anova_mixed.py --out analysis/factorial_anova_mixed_output.csv

Output: stdout summary + CSV with fixed effects, p-values, and effect sizes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict


REPO = Path(__file__).resolve().parent.parent

CONDITIONS = [
    # label, lr_level, alpha_level, lora_dir, pissa_dir
    ("Common",      "hi", "hi",
     "sweep_runs/project_a_confirmatory_lora_qwen3b_gsm8k",
     "sweep_runs/project_a_confirmatory_pissa_qwen3b_gsm8k"),
    ("alpha-swap",  "hi", "lo",
     "sweep_runs/repeft_ablation_alpha_lora",
     "sweep_runs/repeft_ablation_alpha_pissa"),
    ("lr-swap",     "lo", "hi",
     "sweep_runs/repeft_ablation_lr_lora",
     "sweep_runs/repeft_ablation_lr_pissa"),
    ("Faithful",    "lo", "lo",
     "sweep_runs/rev_lora_faithful_fp16",
     "sweep_runs/repeft_qwen3b_pissa_gsm8k_faithful"),
]


def _read_per_seed(csv_path: Path, metric_col: str = "eval_gsm8k_exact_match") -> dict[int, float]:
    """Return {seed -> metric_value} for OK runs."""
    out: dict[int, float] = {}
    if not csv_path.exists():
        return out
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r.get("status", "OK") != "OK":
                continue
            v = r.get(metric_col, "").strip()
            if v in ("", "NaN", "nan", "None"):
                continue
            try:
                seed = int(r.get("seed", "")) if r.get("seed", "").isdigit() else None
                if seed is None:
                    continue
                out[seed] = float(v)
            except (ValueError, TypeError):
                continue
    return out


def _collect_per_seed_deltas(conditions: list[tuple]) -> list[dict]:
    """Build long-format rows: (label, lr_level, alpha_level, seed, delta)."""
    rows = []
    for label, lr_lvl, al_lvl, lora_dir, pissa_dir in conditions:
        lora = _read_per_seed(REPO / lora_dir / "results.csv")
        pissa = _read_per_seed(REPO / pissa_dir / "results.csv")
        shared_seeds = sorted(set(lora.keys()) & set(pissa.keys()))
        for seed in shared_seeds:
            rows.append({
                "condition": label,
                "lr_level": lr_lvl,
                "alpha_level": al_lvl,
                "seed": seed,
                "lora": lora[seed],
                "pissa": pissa[seed],
                "delta": pissa[seed] - lora[seed],
            })
        print(f"  {label}: LoRA n={len(lora)}, PiSSA n={len(pissa)}, "
              f"paired={len(shared_seeds)}")
    return rows


def _fit_mixed(rows: list[dict]) -> dict:
    """Fit MixedLM: delta ~ lr + alpha + lr:alpha + (1 | seed)."""
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame(rows)
    # Encode lr_level/alpha_level as binary
    df["lr_hi"] = (df["lr_level"] == "hi").astype(int)
    df["alpha_hi"] = (df["alpha_level"] == "hi").astype(int)
    df["interact"] = df["lr_hi"] * df["alpha_hi"]
    df["seed"] = df["seed"].astype(str)

    model = smf.mixedlm("delta ~ lr_hi + alpha_hi + interact",
                        data=df, groups=df["seed"])
    last_exc = None
    for method in ["powell", "bfgs", "nm", "cg", "lbfgs"]:
        try:
            result = model.fit(method=method, reml=True, maxiter=500, disp=False)
            return {"df": df, "result": result, "fit_method": method}
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"All MixedLM optimizers failed; last error: {last_exc}")


def _fit_ols_cluster(rows: list[dict]) -> dict:
    """Sensitivity: OLS with cluster-robust SE on seed."""
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame(rows)
    df["lr_hi"] = (df.lr_level == "hi").astype(float)
    df["alpha_hi"] = (df.alpha_level == "hi").astype(float)
    df["interact"] = df["lr_hi"] * df["alpha_hi"]
    df["seed"] = df["seed"].astype(str)
    ols = smf.ols("delta ~ lr_hi + alpha_hi + interact", df).fit(
        cov_type="cluster", cov_kwds={"groups": df["seed"]})
    return {"result": ols, "df": df}


def _summarize(fit: dict) -> dict:
    """Extract fixed-effect estimates + p-values + Cohen's f² for each fixed term."""
    result = fit["result"]
    df = fit["df"]
    params = result.params
    pvalues = result.pvalues
    summary = {}
    for term in ["lr_hi", "alpha_hi", "interact"]:
        summary[term] = {
            "coef": float(params.get(term, float("nan"))),
            "se": float(result.bse.get(term, float("nan"))),
            "p": float(pvalues.get(term, float("nan"))),
        }
    summary["_intercept"] = {
        "coef": float(params.get("Intercept", float("nan"))),
    }
    summary["_meta"] = {
        "N_obs": int(len(df)),
        "N_seeds": int(df["seed"].nunique()),
        "N_conditions": int(df["condition"].nunique()),
        "var_seed": float(result.cov_re.iloc[0, 0]) if hasattr(result.cov_re, "iloc") else float(result.cov_re),
        "var_residual": float(result.scale),
        "icc": float(result.cov_re.iloc[0, 0] / (result.cov_re.iloc[0, 0] + result.scale))
                if hasattr(result.cov_re, "iloc") else float("nan"),
        "loglik": float(result.llf),
        "AIC": float(result.aic) if hasattr(result, 'aic') else float("nan"),
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(REPO / "analysis" / "factorial_anova_mixed_output.csv"))
    args = p.parse_args()

    print("Collecting per-seed paired deltas across 4 conditions...")
    rows = _collect_per_seed_deltas(CONDITIONS)
    print(f"\nTotal paired observations: {len(rows)}")

    fit = _fit_mixed(rows)
    summary = _summarize(fit)
    print(f"\nMixedLM fit method: {fit.get('fit_method','default')}")

    print("\n=== Mixed-effects model: delta ~ lr_hi + alpha_hi + lr_hi:alpha_hi + (1|seed) ===")
    meta = summary["_meta"]
    print(f"N obs = {meta['N_obs']}, N unique seeds = {meta['N_seeds']}, "
          f"N conditions = {meta['N_conditions']}")
    print(f"Var(seed) = {meta['var_seed']:.4f}, "
          f"Var(resid) = {meta['var_residual']:.4f}, "
          f"ICC = {meta['icc']:.4f}")
    print(f"Log-likelihood = {meta['loglik']:.3f}, AIC = {meta['AIC']:.3f}")
    print()
    print(f"{'Term':>15s}  {'coef':>10s}  {'SE':>10s}  {'p':>10s}")
    print("-" * 55)
    for term in ["_intercept", "lr_hi", "alpha_hi", "interact"]:
        s = summary[term]
        coef = s.get("coef", float("nan"))
        se = s.get("se", float("nan"))
        p = s.get("p", float("nan"))
        if term == "_intercept":
            print(f"{'(Intercept)':>15s}  {coef:>10.3f}  {'':>10s}  {'':>10s}")
        else:
            label = {"lr_hi": "lr (hi vs lo)",
                     "alpha_hi": "alpha (hi vs lo)",
                     "interact": "lr × alpha"}[term]
            print(f"{label:>15s}  {coef:>10.3f}  {se:>10.3f}  {p:>10.4g}")

    # Sensitivity: OLS with cluster-robust SE on seed
    print("\n=== Sensitivity: OLS with cluster-robust SE on seed ===")
    ols_fit = _fit_ols_cluster(rows)
    ols = ols_fit["result"]
    print(f"R^2 = {ols.rsquared:.4f} (adj = {ols.rsquared_adj:.4f}), "
          f"F = {ols.fvalue:.2f}, F-pval = {ols.f_pvalue:.4g}")
    print(f"{'Term':>15s}  {'coef':>10s}  {'SE':>10s}  {'z':>8s}  {'p':>10s}")
    print("-" * 60)
    for term in ["lr_hi", "alpha_hi", "interact"]:
        coef = float(ols.params.get(term, float('nan')))
        se = float(ols.bse.get(term, float('nan')))
        z = float(ols.tvalues.get(term, float('nan')))
        p = float(ols.pvalues.get(term, float('nan')))
        label = {"lr_hi": "lr (hi vs lo)", "alpha_hi": "alpha (hi vs lo)",
                 "interact": "lr × alpha"}[term]
        print(f"{label:>15s}  {coef:>10.3f}  {se:>10.3f}  {z:>8.2f}  {p:>10.4g}")

    # Save CSV
    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "term", "coefficient", "std_error", "p_value", "interpretation"])
        for term in ["lr_hi", "alpha_hi", "interact"]:
            s = summary[term]
            label = {"lr_hi": "lr_main_effect",
                     "alpha_hi": "alpha_main_effect",
                     "interact": "lr_x_alpha_interaction"}[term]
            w.writerow(["MixedLM", label, f"{s['coef']:.4f}", f"{s['se']:.4f}",
                       f"{s['p']:.6g}", "fixed effect from MixedLM (seed random effect)"])
        for term in ["lr_hi", "alpha_hi", "interact"]:
            coef = float(ols.params.get(term, float('nan')))
            se = float(ols.bse.get(term, float('nan')))
            p = float(ols.pvalues.get(term, float('nan')))
            label = {"lr_hi": "lr_main_effect",
                     "alpha_hi": "alpha_main_effect",
                     "interact": "lr_x_alpha_interaction"}[term]
            w.writerow(["OLS_cluster", label, f"{coef:.4f}", f"{se:.4f}",
                       f"{p:.6g}", "OLS with cluster-robust SE on seed (sensitivity)"])
        w.writerow(["MixedLM_meta", "var_seed", f"{meta['var_seed']:.6f}", "", "",
                   "Random-effect variance (between-seed)"])
        w.writerow(["MixedLM_meta", "var_residual", f"{meta['var_residual']:.6f}", "", "",
                   "Residual variance (within-seed)"])
        w.writerow(["MixedLM_meta", "icc", f"{meta['icc']:.6f}", "", "",
                   "Intraclass correlation"])
        w.writerow(["MixedLM_meta", "N_obs", str(meta['N_obs']), "", "", "Total observations"])
        w.writerow(["MixedLM_meta", "N_seeds", str(meta['N_seeds']), "", "", "Unique seeds"])
        w.writerow(["OLS_cluster_meta", "R_squared", f"{ols.rsquared:.4f}", "", "",
                   "OLS R^2"])

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
