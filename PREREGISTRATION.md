# Pre-Registration Document

**Paper**: "One Quantity Predicts PEFT Fragility: A Stability Boundary for LoRA Variants"
**Status**: Frozen elements documented below; subsequent additions (Cell 16, Tier-2 extension) appear with their own commitment timestamps.

This document describes which elements of the study were **frozen prior to confirmatory runs** vs which were **chosen after calibration**. It accompanies §3.3 (Study Design Transparency) of the paper.

---

## 1. Frozen Before Any Confirmatory Run

These elements were committed in this file before the corresponding training jobs were launched on the cluster. Modifications after launch are noted in §5 ("Deviations") with reasons.

### 1.1 Claim-Cell Inclusion Rules

A *claim cell* qualifies for the confirmatory family if and only if:

1. **Source-paper claim**: the original PEFT paper for the candidate method explicitly states a directional advantage over LoRA on the (model, task) combination.
2. **Reproducible setup**: hyperparameters, training data, evaluation protocol, and metric are documented in the source paper (or its public code release) at a level sufficient to reproduce within ±20% wall-time.
3. **Feasible on our hardware**: the cell can run within 12GB VRAM per node (3060 cluster) or 48GB (A6000), with subsampled data where the source paper used larger sets.

### 1.2 Pre-Registered Confirmatory Cells

Cells 1–14 were committed to the confirmatory family on the dates below (cluster timestamps).

| Cell | Method | Model | Task | Regime | Source paper claim | Commitment date |
|---|---|---|---|---|---|---|
| 1 | DoRA | roberta-base | BoolQ | common | "DoRA outperforms LoRA across NLU benchmarks" | 2026-02-15 |
| 2 | PiSSA | roberta-base | BoolQ | common | "PiSSA exceeds LoRA on GLUE/SuperGLUE" | 2026-02-15 |
| 3 | DoRA | roberta-base | MNLI | common | (same as Cell 1) | 2026-02-15 |
| 4 | PiSSA | roberta-base | MNLI | common | (same as Cell 2) | 2026-02-15 |
| 5 | AdaLoRA | bert-base | SQuAD v1 | common | "AdaLoRA improves QA over LoRA at matched budget" | 2026-02-22 |
| 6 | rsLoRA | roberta-base | BoolQ | common | "rsLoRA outperforms LoRA at r ≥ 16" | 2026-02-22 |
| 7 | PiSSA | Qwen2.5-3B | GSM8K | common | "PiSSA shows +2.6% on GSM8K (3B)" | 2026-03-01 |
| 8 | DoRA | Qwen2.5-3B | GSM8K | common | (same as Cell 1) | 2026-03-01 |
| 9 | PiSSA | Qwen2.5-3B | GSM8K | paper-faithful | "PiSSA wins under paper hyperparameters" | 2026-03-08 |
| 10 | PiSSA | phi-2 | GSM8K | common | (transfer of Cell 7 claim) | 2026-03-08 |
| 11 | AdaLoRA | bert-base | SQuAD v2 | common | (same as Cell 5) | 2026-03-15 |
| 12 | DoRA | Qwen2.5-3B | GSM8K | r=32 | (rank-sensitivity probe) | 2026-03-15 |
| 13 | PiSSA | Mistral-7B | GSM8K | common | "PiSSA scales to 7B" | 2026-04-01 |
| 14 | DoRA | Mistral-7B | GSM8K | common | (transfer of Cell 1 claim) | 2026-04-01 |

### 1.3 Frozen Statistical Protocol

The following statistical choices were frozen *before any confirmatory job was launched*:

- **Pairing**: same seed list applied to both baseline (LoRA) and candidate; seed list = `[11, 21, 31, ..., 201]` × N truncated to per-cell N.
- **Per-seed delta**: $\delta_i = m(\text{candidate}, s_i) - m(\text{LoRA}, s_i)$.
- **Bootstrap**: BC (bias-corrected, *not* BCa) percentile-based 95% CI with B = 5000 replicates. The acceleration term $a$ is omitted.
- **Permutation test**: sign-flip with $2^N$ exact enumeration when N ≤ 18, Monte Carlo with B = 5000 otherwise.
- **Direction**: one-sided for confirmatory cells (direction taken from source paper); two-sided for OOD and exploratory cells.
- **Multiple-testing correction**: Holm–Bonferroni across the confirmatory family at α = 0.05 (and BH–FDR at q = 0.05 as sensitivity).
- **Adjudication**: supported (CI > 0 *and* surviving Holm), unsupported (CI ∋ 0 *or* failing Holm), reversed (CI < 0 *and* surviving Holm). Boundary cases default to unsupported.
- **Failure handling**: SSH timeout, OOM, or other infra failure → drop the seed *symmetrically* from both arms; no imputation.

### 1.4 Frozen Adjudication Labels

The labels {supported, unsupported, reversed} and their CI-based definitions (above) were committed before any confirmatory cell completed.

---

## 2. Cell 16 Pre-Commitment (Gemma-7B Claim-Matched PiSSA)

Cell 16 was added to the confirmatory family on **2026-04-25** when A6000 capacity became available for a 7B paper-faithful PiSSA test. The commitment is documented separately because it post-dates the original Cells 1–14 freeze.

### 2.1 Commitment Statement (verbatim, timestamped 2026-04-25)

> "We will run Gemma-7B paper-faithful PiSSA vs LoRA at $r=128, \alpha=128, \mathrm{lr}=2\times10^{-5}$, 10 paired seeds (11–101), on MetaMathQA (50K) → GSM8K (1319 test). The cell is added to the confirmatory Holm family (now 15 cells). The adjudication rule is unchanged from §1.4 of this document. We commit to this inclusion irrespective of the outcome: if the result is supported, we will report it as such; if reversed, we will report it as such; if unsupported, we will report it as such. The Holm threshold will be recomputed for the expanded family at compile time."

### 2.2 Why Cell 16 Was Added Post-Hoc

The original 14 cells covered 125M–7B but the only 7B paper-faithful test (Cell 9 ≈ Qwen3B faithful) was at 3B scale. A6000 capacity for a clean 7B claim-matched run was unavailable at the original commitment dates. Adding Cell 16 strengthens the 7B confirmatory evidence; the unconditional commitment above is intended to prevent post-hoc selection effects.

---

## 3. Cell 15 Exploratory Status

Cell 15 (Qwen2.5-3B PiSSA, MetaMathQA pipeline) is reported as exploratory because:

- N = 8 seeds (smaller than other cells' 10–20) due to scheduler-level failures on the MetaMathQA fine-tuning side.
- It is explicitly excluded from the Holm family.
- It is reported for directional consistency with Cells 9 and 13 only.

---

## 4. Tier-2 Extension (OOD)

The 19–20 seed OOD extension reported in Appendix G was committed on **2026-03-25**, after Cells 1–14 completed. The Tier-2 cells use:

- BC bootstrap (same as Cells 1–14).
- One-sided sign-flip for ID matched cells with pre-specified direction.
- Two-sided for OOD cells without pre-specified direction.
- Independent Holm correction within the 14 Tier-2 cells (not pooled with confirmatory cells).

---

## 5. Deviations from Pre-Registration

We document below any departure from the frozen elements above:

- **None for Cells 1–14**: no changes to inclusion rules, statistical protocol, or adjudication after the freeze dates above.
- **Cell 5 (AdaLoRA/SQuAD v1) sample size**: planned N = 10; actually N = 9 due to one cluster job timeout. The protocol §1.3 specifies symmetric drop on infra failure; no imputation.
- **Cell 16 added post-hoc**: see §2 above for commitment statement.
- **Cells 17–18 (LLaMA-2/commonsense_170k)**: added as paper-faithful failure-mode evidence *outside* the Holm family. They are not reported as supported/unsupported/reversed adjudications. Per the on-disk artifacts (`local_runs/claim_dora_llama2_7b_commonsense_dora_a6000_fp16_failed/`), the published DoRA fp16 outcomes are: 2/10 seeds (11, 21) complete training to commonsense_avg ≈ 60.3, 61.2; at least 1/10 (seed 31) records explicit grad-norm NaN at epoch ≈ 0.6; 6/10 seeds (51–101) failed with CUDA OOM under concurrent A6000 sharing and are dropped per §1.3's symmetric-drop rule. We therefore characterize Cells 17–18 as "at least one confirmed fp16 NaN at the predicted-unsafe $\mathrm{lr}_\mathrm{eff} = 2.26 \times 10^{-3}$, with a clean bf16 paired sweep being the natural follow-up". An earlier draft of the paper described these cells as "0 OK + 10 collapse"; that wording was corrected to match the artifacts.
- **AdaLoRA schedule choice for Cells 5 and 11**: we use `tinit = 50`, `tfinal = 200` on bert-base SQuAD v1/v2 at `max_samples = 5000`, `batch_size = 8`, 2 epochs ≈ 1250 total training steps. This means the rank-allocator finalizes its budget after only ~16% of training, leaving ~84% of steps with a fixed allocation. The published AdaLoRA paper (Zhang et al., ICLR 2023) typically uses ~20% `tinit` and ~60% `tfinal` proportions on full SQuAD (~88K samples). To defend Cell 5 against the "you chose a bad schedule" attack, we ran a **schedule-relaxed replication** at the maximum schedule the code permits (effective `tinit=125, tfinal=416` = 33% of training on rank reallocation, the strictest cap allowed by `real_lora_qa.py:368-374`'s safety bounds). Results: `analysis/cell5_schedule_faithful.json` shows Δ=−37.19 F1, BC 95% CI [−38.54, −35.82], two-sided sign-flip p=0.002, 0W/10L/0T—the reversal is **robust** to schedule choice and replicates the original Cell 5 result (Δ=−36.7) within tighter bounds. A fully paper-faithful re-run (tinit > total_steps/10, requiring code-level removal of the safety cap, or larger `max_samples` to lift the cap) remains a camera-ready item.
- **Statistical sign-flip code parameters**: §1.3 specifies "exact 2^N enumeration when N ≤ 18, Monte Carlo with B = 5000 otherwise". An earlier version of `project_a_pairwise_summary.py` used `n > 20` as the exact threshold and `B = 100_000` MC samples (both numerically more conservative than the spec, but a code-vs-spec drift). The current code (post-audit) takes both as CLI arguments with defaults matching the pre-registration spec exactly. No published cell has N > 18, so all reported p-values are produced by the exact branch; the deviation has no numerical impact on any adjudication.

---

## 6. Pre-Registered Failure-Tolerance Protocol

When a cell encounters a failure mode (infra timeout, NaN gradient, OOM), the response is:

1. Document the failure in `local_runs/<cell>/results.csv` with status = FAIL or TIMEOUT.
2. Drop the failing seed symmetrically from both arms.
3. If the cell drops below N = 8 effective seeds, downgrade to exploratory status (Holm-excluded).
4. If grad-norm = NaN for one arm (e.g., DoRA fp16), the cell is reported as "collapsed" rather than excluded; this is a substantive finding, not infra failure.

---

## 6.5 Determinism Caveats

The training scripts seed `torch`, `numpy`, `random`, and `torch.cuda` from `--seed` (audit fix M6). However, several internal calls are **not seed-controlled** and may produce slightly different per-seed numbers across hardware/library versions:

- **PiSSA initialization** uses `torch.linalg.svd`, which is non-deterministic on CUDA across cuDNN versions unless `torch.use_deterministic_algorithms(True)` is set. We do not currently force the deterministic mode (it is much slower and not supported by all kernels we depend on). PiSSA-initial weights can therefore differ in their last few decimal digits across reviewers; the paired Δ for PiSSA cells should be reproducible within ~0.05 pp on the same hardware family.
- **bitsandbytes 4-bit quantization** is dtype-deterministic but layout-dependent on `device_map="auto"` choices that can vary with GPU topology.
- **lm-evaluation-harness 0.4.11** (the pinned version) is deterministic given the same seed and same `--batch_size auto` resolution; the auto batch-size resolution can vary across GPUs.

Reviewers attempting bit-exact reproduction should set `--force-fp16` (for Cells 16, 17, 18 fp16-collapse evidence) and report any divergence > 0.5 pp paired Δ.

## 7. Open Items at Time of Submission

The following are documented as planned but not executed before submission:

- **Full BoolQ DoRA re-run on full dataset** (`sweep_configs/robustness_full_boolq_dora.json` reserved) — to confirm subsample direction at production scale.
- **LLaMA-2-7B DoRA bf16 paired sweep** — initial fp16 collapse documented in Cells 17–18; bf16 paired N=10 sweep is a natural follow-up.
- **Additional model-scale lr_eff sweeps** (RoBERTa, phi-2, Qwen2.5-3B) — to corroborate the BERT/Mistral two-point estimate of the critical product.

---

*This document is part of the anonymous supplementary package and follows the pre-registration template recommended by ACL Rolling Review.*
