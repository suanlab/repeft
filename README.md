# Auditing PEFT Claims: A Stability Probe Unifying Three LoRA-Variant Fragility Patterns

Code, configs, and per-seed data for the AACL-IJCNLP 2026 (Main Conference) paper.

**Suan Lee** (`suanlee@semyung.ac.kr`) and **Jae Seong Kim** (`kjsqp1010@semyung.ac.kr`)  
School of Computer Science, Semyung University, Jecheon, Republic of Korea

---

## What this is

A paired multi-seed audit of four LoRA variants (DoRA, PiSSA, AdaLoRA, rsLoRA) against vanilla LoRA
across 16 claim cells spanning 125M–7B parameters, plus a 303-run stability landscape characterising
when these methods become fragile.

**Headline results**

- Of 15 confirmatory cells, only **2 remain supported** after Holm–Bonferroni correction; **4 reverse**.
- **DoRA: 0/7 supported**, including a clean 7B cell on DoRA's own task family (commonsense_170k).
- **PiSSA** wins under a common budget at small scale, but reverses under paper-faithful hyperparameters
  and in the claim-matched 7B setting (Gemma-7B, MetaMathQA→GSM8K: Δ = −1.58pp, p ≈ 0.004, 1W/9L).
- **AdaLoRA**'s dramatic SQuAD reversal is **budget-driven, not schedule-driven** — a 2×2 control shows
  the rank-allocation schedule moves F1 by <0.05 while the training budget moves it by ≈40.
- **rsLoRA** collapses once `lr_eff = lr × α/√r` exceeds ≈5e-4 on BERT-base/MNLI/AdamW
  (40/40 stable below the boundary; 11/32 collapse above it).

**Scope of the probe.** `lr_eff` is a calibrated pre-flight check, not a law. It is necessary but not
sufficient, its value is architecture- and optimizer-specific, it does not describe AdaLoRA, and one
prospective 7B test failed — LLaMA-2-7B trains stably in bf16 at 4.5× the boundary. That failed test is
reported in the paper and the cell is withdrawn from the mechanism's supporting evidence.

## Layout

```
analysis/                        paired summaries (BC bootstrap CI, sign-flip p, W/L/T)
  cell5_uncapped.{json,md}       AdaLoRA @12K, uncapped schedule
  cell5_budgetcontrol.json       AdaLoRA @12K, original schedule   } budget-vs-schedule 2x2
  cell5_versioncheck.json        original Cell 5 re-run on newer peft (library control)
  cell17_18_bf16_raw.json        LLaMA-2-7B bf16 paired re-run, 10/10 clean
  union_holm.py                  union-family Holm correction
  factorial_anova_mixed.py       mixed-effects re-analysis of the lr ablation
  holm.py, paired_t_sensitivity.csv, ...
sweep_configs/                   71 sweep configurations covering the 600+ training runs
sweep_runs/                      per-seed metric CSVs for the re-run cells
scripts/                         figure generation (reads directly from analysis/)
tests/                           integration test re-deriving headline statistics
logs/                            preserved fp16 Gemma runs (dtype-sensitivity appendix)
figures/                         paper figures
PREREGISTRATION.md               frozen study design and deviations
requirements.txt                 pinned versions used for the main audit
```

## Reproducing a paired statistic

Every paired Δ / CI / p-value in the paper comes from one driver:

```bash
python3 project_a_pairwise_summary.py \
  --baseline-csv  sweep_runs/cell5_uncapped_lora/results.csv \
  --candidate-csv sweep_runs/cell5_uncapped_adalora/results.csv \
  --metric eval_f1 --baseline-name LoRA --candidate-name AdaLoRA \
  --one-sided none \
  --out-json out.json --out-md out.md
```

Re-derive the headline numbers from the released data:

```bash
python3 tests/test_paired_summary.py      # 6 checks, non-zero exit on mismatch
```

Reproduce the two statistical controls added for the camera-ready:

```bash
python3 analysis/union_holm.py             # union-family Holm over 33 cells
python3 analysis/factorial_anova_mixed.py  # seed as a random effect
```

## Environment

The main audit used the pins in `requirements.txt` (PyTorch 2.3, transformers 4.42, peft 0.11).
The camera-ready re-runs used a newer stack (PyTorch 2.7.1, transformers 5.3.0, peft 0.18.1).
`analysis/cell5_versioncheck.json` is the control showing the two stacks agree to within 0.1 F1 on the
original configuration, so the library upgrade is not a confound for any reported comparison.

## Hardware

50× RTX 3060 (12 GB) for the ≤3B cells and the stability landscape; 2× RTX A6000 (48 GB) for the 7B
chain. Total ≈2,400 GPU-hours.

## Citation

```bibtex
@inproceedings{lee2026auditing,
  title     = {Auditing {PEFT} Claims: A Stability Probe Unifying Three {LoRA}-Variant Fragility Patterns},
  author    = {Lee, Suan and Kim, Jae Seong},
  booktitle = {Proceedings of AACL-IJCNLP 2026},
  year      = {2026}
}
```

## License

MIT for code, configs, and metric CSVs. Pre-trained models follow their upstream licenses
(LLaMA-2 Community License, Gemma Terms of Use, Apache-2.0 for Mistral and Qwen2.5).
