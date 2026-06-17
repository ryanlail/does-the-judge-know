# Final numbers — reference for the writeup

Every number for the post, with 95% CIs and its source file. Nothing here is prose for the post; it is the lookup table to write from.

**Regenerate:** `uv run python -m src.day1 && uv run python -m src.day3_robustness && uv run python -m src.day3c_escalation && uv run python -m src.day3d_cross_family`; figures at 2× via `uv run python -m src.export_figures`.

**Setup:** data = `composo-ai/llm-judge-criteria-ensembling` @ `e4049a5` (vendored read-only); RewardBench-2, base prompt, k=8. *Bias error* = non-tie error reproduced under resampling (winner stability `f_same ≥ 0.9`, B=1000 per example). *Recall ceiling* = 1 − bias share. CIs are 95% bootstrap over examples (B=2000) unless noted.

---

## 0. Dataset (Step 0) — `step0_summary.md`, `step0_inventory.csv`
| Quantity | Value | Source |
|---|---|---|
| Common usable examples, GPT-5.4 ∩ Sonnet 4.6 | **1,711** (presence 1,730; id-only 1,692) | step0_summary.md |
| — of which **Safety** (headline slice) | **410** | step0_summary.md |
| 3-tier difficulty-strata universe | 1,681 | step0_summary.md |
| GPT-5.4 base n / Safety | 1,730 / 419 | step0_inventory.csv |
| Sonnet 4.6 base n / Safety | 1,763 / 450 | step0_inventory.csv |
| Acceptance gate | PASS (≥ 1,600) | step0_summary.md |

---

## 1. HEADLINE A — within-judge bias share — fig1; `day1_bias_share.csv`
| Judge | Slice | Bias share [95% CI] | n_err | Recall ceiling |
|---|---|---|---|---|
| **GPT-5.4** | overall | **70.0%** [64.2, 75.4] | 243 | **30.0%** |
| **GPT-5.4** | **Safety** | **86.2%** [72.2, 97.0] | 29 | **13.8%** |
| Sonnet 4.6 | overall | 57.6% [51.0, 63.9] | 231 | 42.4% |
| Sonnet 4.6 | Safety | 63.4% [47.6, 78.4] | 41 | 36.6% |

Per category — bias share @0.9 (the fig1 bars), GPT-5.4 / Sonnet 4.6:
Factuality 68.8% / 47.3% · Focus 74.3% / 61.9% · Math 55.9% / 75.0% · Precise IF 66.1% / 53.6% · Safety 86.2% / 63.4%.

Threshold sensitivity (overall, f_same ≥ 0.75 / 0.90 / 0.95): GPT-5.4 **82.7 / 70.0 / 60.1%**; Sonnet 4.6 **72.7 / 57.6 / 51.1%** — `day1_bias_share.csv`.
Error / tie rates: GPT-5.4 14.0% / 4.5% (acc 81.5%); Sonnet 4.6 13.2% / 4.0% (acc 82.7%) — `day1_summary.md`.

---

## 2. HEADLINE B — cross-judge error overlap — fig2; `day1_overlap.csv`
| Quantity | Overall [95% CI] | Safety [95% CI] |
|---|---|---|
| P(GPT err) / P(Sonnet err) / P(both) | 18.2% / 17.3% / 9.7% | 7.6% / 11.5% / 2.8% |
| Unadjusted excess / raw OR | +6.5pp / 11.1 | +1.9pp / 5.6 |
| **Difficulty-adjusted excess** | **+1.9pp** [+1.2, +2.5] | +0.2pp [−0.7, +1.3] |
| **Mantel–Haenszel OR** | **3.07** [2.13, 4.41] | 1.31 [0.46, 3.71] *(n.s.)* |

Negative control: within-stratum permutation excess ≈ **0.0pp** [−0.5, +0.5] (estimator unbiased); within-category +0.4pp (structural, explained); observed +1.9pp, p < 0.001 — `day1_summary.md`.
Per-category MH OR: Factuality 4.71 · Focus 2.36 · Math 4.77 · Precise IF 3.06 · **Safety 1.31 (n.s.)** — `day1_overlap.csv`.

---

## 3. Mechanism / what this caps — `day1_auc.csv`
Variance (and related) signals predicting incorrectness (ROC-AUC):
| Signal | GPT-5.4 | Sonnet 4.6 | Example set |
|---|---|---|---|
| mean response std — **paper's ≈0.60** | **0.601** | 0.587 | all (ties = incorrect) |
| mean response std | 0.643 | 0.631 | non-tie |
| winner std | 0.609 | 0.649 | non-tie |
| instability (1 − f_same) | 0.784 | 0.810 | non-tie |
Anchor: the published "variance predicts incorrectness" AUC ≈ 0.60 reproduced exactly (GPT base 0.601). The bias share (§1) is *why* this ceiling exists. Even the best resample signal (instability, 0.78–0.81) is by construction blind to bias errors.

---

## 4. Robustness (Day 3) — both conclusions stable — `day3_robustness.csv`
- **Criteria injection:** bias share GPT-5.4 70→58% (Safety 86→71%), Sonnet 4.6 58→57% (Safety 63→44%) — lowers it, stays ≥25% everywhere; cross-judge MH OR **3.07 → 3.52** [2.49, 4.99].
- **Tier judges (overall bias share):** GPT-5.4-mini 49.8% · Haiku 4.5 45.2% · GPT-5.4-nano 27.5% — **bias share rises with capability** (frontier judges most affected).
- **f_same threshold:** stable (see §1). **Ties in vs out:** bias share GPT 70→53%, Sonnet 58→44%; MH OR 3.07 vs 2.98 [2.01, 4.40]. **Difficulty proxy:** MH OR 3.07 (tier-count) vs 3.17 [2.25, 4.47] (score-margin quartiles).

---

## 5. Follow-ups — escalation (Analyses C, D) — fig_escalation_recall.png, fig_escalation_family.png

**C — cheap-monitor + escalation** (`day3c_escalation.csv`):
| Tier pair | P(mini wrong \| full bias) | diff-adj baseline | cross-tier MH OR | escalation recall @20% (var / instab) |
|---|---|---|---|---|
| GPT-5.4 → mini | 71% | 41% | 8.2 [6.0, 11.3] | 36% / 46% (AUC 0.61 / 0.70) |
| Sonnet 4.6 → Haiku | 58% | 32% | 5.9 [4.2, 8.3] | 36% / 53% (AUC 0.66 / 0.74) |
Same-family shares bias more than cross-provider (MH OR 6–8 vs 3.1). Escalation trigger is a weak partial check, not blind.

**D — cross-family vs same-family escalation** (`day3d_escalation_family.csv`):
| Monitor | Bias-error repair (same vs cross) | System acc @20% (same vs cross) | Monitor Safety acc |
|---|---|---|---|
| GPT-5.4-mini | 32% vs **53%** (cross wins) | 79.4% vs 80.9% | **92.7%** |
| Haiku 4.5 (mirror) | 36% vs 30% (**reverses**) | 84.1% vs 83.4% | **95.9%** |
Cross-family is **not a robust win** (backstop-specific; the mirror reverses). On Safety the cheap monitors already beat the frontier judges (93–96%), so escalation is flat-to-harmful.

---

## 6. Honest limits (for the limits section)
- One benchmark (RewardBench 2), two providers + their tiers, prompted judges, MCQ-style, k=8.
- **No adversarial optimisation** — the adversarial slice (Step 4) was skipped → named follow-up.
- Safety overlap (§2) and Safety escalation (§5 D) are **underpowered** (~29–41 errors).
- Cross-judge overlap is a **conservative lower bound**: the tier judges used to control for difficulty share provider families with the main judges.
- `all_metrics.json` reflects an earlier, ~40-example-smaller snapshot than the vendored HEAD data; logic validated independently (Factuality accuracy bit-exact; AUC 0.601 ≈ paper 0.60).

---

## Figures (re-exported at 2×, savefig dpi 400)
| File | Role | Backing numbers |
|---|---|---|
| **fig1_bias_share_by_category.png** | HEADLINE | fig1_bias_share_by_category.csv |
| fig2_overlap_by_category.png | headline 2 | fig2_overlap_by_category.csv |
| fig_escalation_recall.png | follow-up (C) | day3c_escalation.csv |
| fig_escalation_family.png | follow-up (D) | day3d_escalation_family.csv |

## Source files (all under `results/`)
step0_inventory.csv · step0_summary.md · day1_bias_share.csv · day1_auc.csv · day1_overlap.csv · day1_summary.md · day3_robustness.csv · day3c_escalation.csv · day3d_escalation_family.csv · day3_summary.md
