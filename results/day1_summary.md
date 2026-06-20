# Day 1 — within-judge bias share and cross-judge overlap

## Analysis A — within-judge bias share

Base prompt, k=8. Bias-dominated error = non-tie error with bootstrap winner stability f_same ≥ 0.9 (resampling B=1000 per example). Bias share = bias-dominated / non-tie errors. Recall ceiling = 1 − bias share (the most any resample-based uncertainty signal can catch). CIs: bootstrap over examples, B=2000, seed 42.

**Error structure**
- GPT-5.4: accuracy 81.5%, error rate 14.0%, tie rate 4.5% (n=1730).
- Sonnet 4.6: accuracy 82.7%, error rate 13.2%, tie rate 4.0% (n=1744).

**Bias share (f_same ≥ 0.9), overall and Safety**
- GPT-5.4: overall **70.0%** [64.2%–75.4%], n_err=243 (recall ceiling 30.0%); Safety **86.2%** [72.2%–97.0%], n_err=29 (recall ceiling 13.8%).
- Sonnet 4.6: overall **57.6%** [51.0%–63.9%], n_err=231 (recall ceiling 42.4%); Safety **63.4%** [47.6%–78.4%], n_err=41 (recall ceiling 36.6%).

Sensitivity (overall bias share at f_same thresholds 0.75 / 0.90 / 0.95):
- GPT-5.4: 82.7% / 70.0% / 60.1%
- Sonnet 4.6: 72.7% / 57.6% / 51.1%

**Variance as an incorrectness classifier (AUC)**
- GPT-5.4 (paper replication, signal=mean response std, ties=incorrect): AUC **0.601** (n=1730, base rate 18.5%).
    non-tie (error vs correct): mean_response_std 0.643, winner_std 0.609, instability(1−f_same) 0.784.
- Sonnet 4.6 (paper replication, signal=mean response std, ties=incorrect): AUC **0.587** (n=1744, base rate 17.3%).
    non-tie (error vs correct): mean_response_std 0.631, winner_std 0.649, instability(1−f_same) 0.810.

Figure: `figures/fig1_bias_share_by_category.png`. Tables: `results/day1_bias_share.csv`, `results/day1_auc.csv`.

## Analysis B — cross-judge error overlap

Common usable examples: 1711 (both main judges); difficulty strata defined on 1681 (all three tier judges usable). E = response 0 not the unique winner (ties = error). Difficulty s = # of {GPT-5.4-mini, Haiku 4.5, GPT-5.4-nano} erring (0–3). Excess overlap = observed joint-error rate − expected under conditional independence given s. CIs: bootstrap over examples, B=2000.

**Marginal error rates and overlap (all examples)**
- P(GPT err) = 18.2%, P(Sonnet err) = 17.3%, P(both err) = 9.7%.
- Unadjusted excess (vs marginal independence): **+6.5pp** (observed 9.7% vs expected 3.2%); raw odds ratio 11.10.
- Difficulty-adjusted excess: **+1.9pp** [+1.2pp, +2.5pp]; **Mantel–Haenszel OR = 3.07** [2.13, 4.41] (RBG variance).
- Safety: adjusted excess **+0.2pp** [-0.7pp, +1.3pp]; MH OR 1.31 [0.46, 3.71].

**Negative control (permute the GPT–Sonnet pairing, 1000 permutations)**
- Within difficulty stratum s (the null the statistic conditions on): permuted excess mean **+0.0pp** [-0.5pp, +0.5pp] — collapses to ≈ 0, so the estimator is unbiased (no pipeline bug). Observed +1.9pp, two-sided p = 0.000.
- Within category (the spec's control): permuted excess mean +0.4pp [-0.2pp, +1.0pp]. This does not fully reach 0 by construction: difficulty strata are finer than category, so shuffling within category leaves a small structural offset (the statistic adjusts for s but the shuffle preserves only category). The observed excess is far outside both nulls.

Caveat: the tier judges share provider families with the main judges (GPT-5.4-mini/nano with GPT-5.4; Haiku 4.5 with Sonnet 4.6), so the difficulty stratification may absorb some genuinely shared bias. The difficulty-adjusted excess and MH odds ratio are therefore *conservative* lower bounds on shared bias; the unadjusted overlap is reported alongside.

Table: `results/day1_overlap.csv`.
