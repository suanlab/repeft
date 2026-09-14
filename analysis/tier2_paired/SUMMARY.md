# Tier 2 Paired Summary (BC Bootstrap CI + Sign-Flip Permutation)

Generated `analysis/tier2_paired/`. All comparisons are paired by seed.
Confirmatory direction (`one-sided greater`) used for ID metrics where the original paper claims method > LoRA; OOD metrics use two-sided (no pre-specified direction).

## Adjudication Outcomes

### ✓ Supported (CI > 0)

| Cell | Metric | Δ (cand − LoRA) | 95% BC CI | p (1-sided) | W/L/T |
|---|---|---:|---|---:|---|
| **PiSSA vs LoRA** | MNLI matched (ID) | **+0.0103** | [+0.0075, +0.0144] | **1.9e-6** | 19/0/0 |
| **PiSSA vs LoRA** | Amazon CF (OOD) | **+0.0247** | [+0.0185, +0.0316] | **3.8e-6** | **19/0/0** |
| DoRA vs LoRA | Amazon CF (OOD) | +0.0035 | [+0.0008, +0.0061] | 0.032 | 12/5/2 |

### ✗ Reversed (CI < 0)

| Cell | Metric | Δ | 95% BC CI | p | W/L/T |
|---|---|---:|---|---:|---|
| **PiSSA vs LoRA** | ANLI R2 (OOD) | **−0.0099** | [−0.0142, −0.0059] | **2.0e-4** | 4/15/0 |

### ○ Unsupported (CI includes 0)

DoRA vs LoRA — mostly null on MNLI/SNLI/ANLI (5 cells, p > 0.07), confirming earlier
finding that DoRA's claimed gain doesn't survive paired multi-seed evaluation.

PiSSA vs LoRA — null on SNLI, ANLI R3 (no clear direction).

PiSSA vs LoRA — modest gain on MNLI mismatched (d=+0.0048, p=0.036, but CI just touches 0).

## Headline Story

**PiSSA's gain is shift-type-dependent**:

| Shift type | Source | Result |
|---|---|---|
| In-distribution (MNLI matched) | nyu-mll/glue | **PiSSA improves** (+1.0%, p<1e-5) |
| Mild OOD (SNLI) | NLI | null |
| Adversarial OOD (ANLI R2) | NLI | **PiSSA reverses** (−1.0%, p=2e-4) |
| Sentiment shift (Amazon CF) | binary class | **PiSSA improves substantially** (+2.5%, p<1e-5) |

This generalizes the earlier 20-seed finding: PiSSA's ID advantage carries over to *sentiment* OOD but reverses on *adversarial NLI* OOD. The shift-type asymmetry is robust across the new 20-seed cells.

**DoRA's gain is mostly null** across NLI ID/OOD; only Amazon CF shows a small positive
effect (d=+0.0035, p=0.032), much smaller than PiSSA's effect on the same metric.

## rsLoRA Rank Sensitivity (descriptive)

| Rank | Mean BoolQ acc | Std | Range |
|---|---:|---:|---|
| r=16 | 0.6412 | 0.0333 | [0.6307, 0.7360] |
| **r=32** | **0.6307** | **0.0000** | **[0.6307, 0.6307]** |

**rsLoRA collapses at rank=32**: all 10 seeds produce identical accuracy = 0.6307,
which equals the BoolQ majority-class baseline. The model fails to learn anything beyond
the prior. This is consistent with — and substantially extends — the earlier rank
sensitivity finding (LoRA degrades at r=32, but rsLoRA fully collapses).

## Caveats / Known Issues

- **ANLI R1**: bootstrap CI is NaN due to one or more nan accuracy values from failed
  seed evaluations. Need to filter or rerun those seeds. ANLI R2/R3 are clean.
- One seed failure each in `e3_mnli_snli_lora` (19/20 OK) and `repeft_sst2_ood_lora_20seed` (19/20 OK)
  → all paired comparisons use n=18-19 not n=20. Underpowered relative to plan but still
  significant for the supported cells.
- e3_mnli_snli used 20 seeds [11, 21, ..., 201] consistent across all 3 methods,
  so paired SHARED SEEDS = 18-19 (intersection of OK seeds).

## Files

```
analysis/tier2_paired/
├── e3_mnli_snli__dora_vs_lora__{mnli_matched,mnli_mismatched,snli,anli_r1,anli_r2,anli_r3}_accuracy.{json,md}
├── e3_mnli_snli__pissa_vs_lora__{...}_accuracy.{json,md}
├── repeft_sst2_ood__dora_vs_lora__{sst2_id,yelp_ood,amazon_counterfactual}_accuracy.{json,md}
├── repeft_sst2_ood__pissa_vs_lora__{...}.{json,md}
└── SUMMARY.md  ← this file
```
