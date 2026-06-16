# Day 3 — robustness

Two Day-1 conclusions are stress-tested: (A) bias errors dominate (high within-judge bias share), and (B) the judges share errors beyond difficulty (MH OR > 1). Difficulty strata always come from the base tier judges.

### Check 1 — criteria injection (Analysis A)
Bias share base→criteria — overall: GPT-5.4 70%→58%, Sonnet 4.6 58%→57%; Safety: GPT-5.4 86%→71%, Sonnet 4.6 63%→44%. Criteria injection *lowers* the bias share — it converts some confidently-wrong errors into resolvable, variance-dominated ones (most visibly Sonnet 4.6 Safety). **Stable** — every cell stays ≥ 25% (the recall-ceiling bar), so most errors remain invisible to resampling; criteria helps only at the margin.

### Check 2 — criteria injection (Analysis B)
MH OR base 3.07 [2.13, 4.41] → criteria 3.52 [2.49, 4.99]; adjusted excess base +1.9pp → criteria +2.3pp. **Stable** — significant residual cross-judge overlap persists under criteria.

### Check 3 — tier judges (Analysis A)
Overall bias share by capability: GPT-5.4 70% / Sonnet 4.6 58% (frontier) > GPT-5.4-mini 50% ≈ Haiku 4.5 45% > GPT-5.4-nano 28%. **Stable (and sharpened)** — bias share *rises with judge capability*: every tier stays ≥ 25% (nano borderline), and the frontier judges — the ones actually deployed as monitors — are the most bias-dominated, so the failure is most acute exactly where it matters.

### Check 4 — f_same threshold sensitivity
- GPT-5.4 overall at f_same ≥ 0.75 / 0.90 / 0.95: 83% / 70% / 60%.
- Sonnet 4.6 overall at f_same ≥ 0.75 / 0.90 / 0.95: 73% / 58% / 51%.
**Stable** — bias errors dominate (≥25%) at every threshold; the headline is not an artifact of the 0.90 cut.

### Check 5 — tie errors included vs excluded
- Analysis A overall bias share, ties excluded→included in denominator: GPT-5.4 70%→53%, Sonnet 4.6 58%→44%.
- Analysis B MH OR, ties-as-error 3.07 [2.13, 4.41] vs ties-excluded 2.98 [2.01, 4.40].
**Stable** — both conclusions survive either tie convention (folding ties in lowers the bias share but it stays dominant; the overlap stays significant).

### Check 6 — difficulty proxy (tier-count vs margin quartiles)
MH OR with tier-error-count strata 3.07 [2.13, 4.41] vs score-margin-quartile strata 3.17 [2.25, 4.47]; adjusted excess +1.9pp vs +2.2pp. **Stable** — the residual overlap does not depend on the choice of difficulty proxy.

Table: `results/day3_robustness.csv`.

## Analysis C — cheap-monitor + escalation (optional follow-up)

Deployment pattern: monitor with a cheap (mini) judge, escalate the top-X% most uncertain cases to the frontier (full) judge. Tests whether the monitor's uncertainty flags the frontier judge's blind spots. Difficulty is stratified by the OTHER provider's full-judge score margin (distinct from both tier judges, to avoid conditioning on the object of study). This is about the escalation architecture, not small-model error prediction (on a bias error both tiers are confidently wrong, so neither variance rises).

- **GPT-5.4 → GPT-5.4-mini** (difficulty proxy: Sonnet 4.6 score-margin). *Shared bias:* of the full judge's bias errors (n=165), the mini judge is also wrong **71%** of the time vs a difficulty-adjusted baseline of 40% (excess +30pp; cross-tier **MH OR 8.2** [6.0, 11.3] — far above the cross-provider 3.1). *Escalation:* the realistic cheap trigger (mini per-response variance) recalls only 36% of those bias errors at a 20% budget (AUC 0.61) — about as poorly as it flags the full judge's *variance* errors (35%, AUC 0.59), i.e. it surfaces hard examples indiscriminately, not bias errors specifically. The costlier instability trigger (mini 1−f_same) does better — 45% recall, AUC 0.70 — but still misses about half.
- **Sonnet 4.6 → Haiku 4.5** (difficulty proxy: GPT-5.4 score-margin). *Shared bias:* of the full judge's bias errors (n=130), the mini judge is also wrong **58%** of the time vs a difficulty-adjusted baseline of 32% (excess +26pp; cross-tier **MH OR 5.9** [4.2, 8.3] — far above the cross-provider 3.1). *Escalation:* the realistic cheap trigger (mini per-response variance) recalls only 36% of those bias errors at a 20% budget (AUC 0.66) — about as poorly as it flags the full judge's *variance* errors (29%, AUC 0.60), i.e. it surfaces hard examples indiscriminately, not bias errors specifically. The costlier instability trigger (mini 1−f_same) does better — 53% recall, AUC 0.74 — but still misses about half.

**Reading:** the escalation architecture *substantially* inherits the blind spot. Same-provider tiers share bias errors even more strongly than cross-provider judges (MH OR ~6–8 vs 3.1), and the cheap monitor's realistic uncertainty signal recalls only ~a third of the frontier judge's bias errors at a 20% budget. A costlier bootstrap-instability trigger roughly halves the miss rate but still leaves the shared confidently-wrong cases — the most dangerous ones — below threshold. So the inheritance is substantial, not total: escalation is a weak partial check on bias errors, not the safety net the architecture assumes. Figure: `figures/fig_escalation_recall.png`; table: `results/day3c_escalation.csv`.
