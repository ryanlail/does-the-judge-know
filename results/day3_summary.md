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
