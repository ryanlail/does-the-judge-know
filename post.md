# Quantifying blind spots of LLM Evaluators

When an LLM judge makes a wrong safety decision, does its own uncertainty know? Under the bias–variance decomposition, errors are either variance errors that are detectable by resampling, or bias errors where the judge is confidently and repeatably wrong. We find it is mostly the latter on frontier judges. Wrong answers are usually stable under resampling, the same wrong winner returns when you re-draw the scores. GPT-5.4's errors are 70% stable across all categories and 86% stable on Safety, so a disagreement-based detector can catch at most 14% of its Safety errors. We show there is a ceiling on what single-judge disagreement can catch, and it falls as judge models get stronger, with the effect strongest on Safety.

## Why this matters

LLM Judges are increasingly deployed as safety monitors, the backstop that is supposed to stop a model from producing unsafe outputs, where the tolerance for error is close to zero. When Anthropic released Fable 5 (link), it was implemented as Mythos 5 with safety guardrails on top to prevent unsafe use (link). Ultimately, these guardrails could be bypassed (link), leading to the withdrawal of the model (link). Therefore, a key problem in AI Safety research is how we can improve the safety monitors. As we recommended in our ICML workshop paper (link), a cheap improvement to standard LLM as a Judge is to resample the same model to reduce variance errors. Another approach may be to sample independent models. Kohli found that sampling different models shares bias errors too, concluding that nine judges are worth roughly two independent votes [link 4]. A detector built only on disagreement can reduce errors, but can give false reassurance precisely on the inputs an adversary would choose.

## How we measure it

On RewardBench 2 each example offers four candidate responses, with response 0 the known-correct one, and a judge scores all four. We collected eight independent score samples per response (k=8) at temperature 1.0, which gives the full spread of each judge's rating rather than a single number. We have these paired samples for two frontier judges, GPT-5.4 and Sonnet 4.6, on the same set of around 1,750 examples, plus three weaker judges: GPT-5.4-mini, Haiku 4.5, and GPT-5.4-nano. A judge's predicted winner is the response with the highest mean score across its eight samples; if that is not response 0, the judge has made an error.

What we measure is whether an error survives resampling. For each error we bootstrap the eight existing samples: resample the scores per response with replacement, recompute the winner, and repeat 1,000 times. We call the fraction of those draws that reproduce the same wrong winner $f_\text{same}$. This deliberately resamples the eight scores we already have rather than making fresh judge calls, because those eight samples are exactly the information an ensemble or disagreement detector has at deployment. A result about their stability is therefore a result about what any such detector can ever see.

Working at the level of the judge's scores, the bias-variance decomposition splits an error in two. A variance error is an accident of sampling: the scores are noisy, the margin between responses is small, and resampling often flips the winner back to response 0. A bias error is one where the judge's expected scores already rank the wrong response first, so resampling changes nothing and the same wrong winner returns nearly every time. We label a non-tie error bias-dominated when $f_\text{same} \geq 0.9$, with sensitivity reported at 0.75 and 0.95 (ties, where no single response wins, count as errors but are analysed separately). The bias share is the fraction of a judge's non-tie errors that are bias-dominated.

This is what sets the limit. A disagreement-based detector can only flag an error it sees move under resampling, so the most it can ever catch is the variance-dominated fraction. We call 1 − bias share the recall ceiling: an upper bound on the recall of any detector built from single-judge disagreement, fixed before you have chosen one. Disagreement reads low scatter as confidence and confidence as trustworthy, but a bias error is the exact case where the judge is confident and wrong, so the signal points the wrong way.

## Results

### Most errors are stable, and the stability peaks on Safety

For both frontier judges, most errors survive resampling.

| Judge (base, k=8) | Errors that are *stable* (bias share) | Detector ceiling (max errors catchable) |
| --- | --- | --- |
| GPT-5.4 (overall) | 70% [64–75] | 30% |
| **GPT-5.4 (Safety)** | **86% [72–97]** | **14%** |
| Sonnet 4.6 (overall) | 58% [51–64] | 42% |
| Sonnet 4.6 (Safety) | 63% [48–78] | 37% |

GPT-5.4 makes a clear, non-tie error on roughly one example in seven (counting ties as errors too, it is wrong on about 18%, the rate in the escalation table later). On 70% of those errors the same wrong winner comes back in at least nine of ten bootstrap draws, the scores are noisy but not noisy enough to move the verdict. A disagreement-based detector can only flag an error it sees move, so a 70% stable rate means a recall ceiling of 30%: no resampling rule and no variance threshold, however we tune it, can catch more than three in ten of GPT-5.4's errors. On Safety the ceiling falls to 14%, so the cases where the monitor is wrong are mostly the cases where it looks most confident. Sonnet 4.6 is less extreme (58% / 63%), but the story is the same.

![Bias share by RewardBench 2 category for both frontier judges (fraction of errors that are stable under resampling, $f_\text{same} \geq 0.9$), with 95% confidence intervals. Safety is highlighted; the dashed line marks 25%. The recall ceiling of any disagreement detector is one minus the bar height.](figures/fig1_bias_share_by_category.png)

The bias share is not uniform across the benchmark, it runs from 56% on Math to 86% on Safety for GPT-5.4. It is highest on Safety, the category where we would least want a blind monitor.

### The ceiling tightens as judges get stronger

The pattern that matters most is what happens as judges get stronger. Within a model family, the bias share rises with scale.

| GPT judge (base, k=8) | Errors that are *stable* | Detector ceiling |
| --- | --- | --- |
| GPT-5.4-nano | 28% [23–32] | 72% |
| GPT-5.4-mini | 50% [44–56] | 50% |
| **GPT-5.4** | **70% [64–75]** | **30%** |

From nano to mini to the full model the detector ceiling falls, 72% to 50% to 30%, and the Anthropic pair moves the same way (Haiku 45% to Sonnet 58%). This is about the composition of errors, not their number. We are not saying stronger judges make fewer errors, we are saying a larger share of the errors they do make are the confident, repeatable kind. So the disagreement signal that ensembling relies on is weakest on the most capable judges, the ones we are most likely to deploy as monitors.

One natural objection is recalibration: maybe each judge just needs its own variance threshold. It does not help. A bias error is a stable wrong winner, not just a low-variance one, so retuning a cutoff trades precision for recall but cannot recover errors whose scores do not move.

### Two judges, one blind spot

The second cheap fix is to ask more than one judge, but two judges can be blind in the same place. On the examples both frontier judges see, when one is wrong the other is wrong more than half the time, roughly three times its base rate. A panel of judges does not buy as many independent votes as its size suggests, the judges tend to fail on the same inputs.

![How often both frontier judges are wrong on the same example, split by difficulty (how many of the three weaker judges also get it wrong). Within each band the observed rate (dark) sits above what independence predicts (light), and that gap is shared bias. Difficulty drives most of the overlap, but the gap persists at every level.](figures/fig2_cross_judge_overlap.png)

Some of this is just difficulty, hard examples trip everyone. But the figure shows the gap survives it: reading left to right, as examples get harder both judges make more mistakes, yet within every difficulty band they are wrong together more often than independence predicts. What independence predicts is just the two judges' error rates within the band multiplied together, the rate you would expect if one of them being wrong told you nothing about the other. We measure difficulty by how many of the three weaker judges (GPT-5.4-mini, Haiku 4.5, GPT-5.4-nano) also get it wrong. The gap is not noise: when we shuffle which examples the two judges are paired on, within each difficulty band, it collapses to zero. This lines up with prior work on correlated panels [Kohli/Goel], we report it as a check, not a discovery.

Two caveats. The weaker judges share provider families with the frontier judges, so they carry some of the same bias, which means the difficulty adjustment absorbs part of the shared bias and the residual is a conservative lower bound. And the effect does not hold on Safety alone, where too few examples have both judges wrong to measure.

So far that is two judges from different providers. Run the same difficulty-adjusted comparison within a single model family, GPT-5.4 against its own mini, Sonnet 4.6 against Haiku 4.5, and the overlap roughly doubles.

![The same difficulty-adjusted overlap for three pairs. Each bar is how much more often the two judges are both wrong than you would expect if they failed independently at each difficulty level (difficulty is set by judges outside the pair, so it cannot absorb the shared bias). Two judges from different providers overlap a little beyond difficulty, 1.2x. Two from the same family overlap far more, 1.6 to 1.7x, they inherit the same blind spots.](figures/fig_family_overlap.png)

Two judges from the same family are close to asking the same model twice.

### Robustness

Both results, the high bias share and the residual overlap, hold across six checks: $f_\text{same}$ thresholds of 0.75 / 0.90 / 0.95, ties counted in or out, and a different difficulty proxy. The one change worth naming is criteria injection. Giving the judge an explicit rubric does turn some confident errors into resolvable ones (Sonnet's Safety bias share falls from 63% to 44%), but it does not change the regime, every cell stays above 25% and the cross-judge overlap is essentially unchanged. The rubric helps a little, it does not remove the wall.

### A ceiling, not a weak signal

Our earlier paper found that a judge's score variance predicts incorrectness with an AUC of about 0.60, a weak but positive signal. We reproduce it exactly (0.601 for GPT-5.4), and our pipeline recovers the paper's published accuracies, so we know the reimplementation is faithful. The natural reading of a 0.60 AUC is "weak signal, room to improve". The bias share says otherwise. A low AUC could mean three things: a noisy estimate of a real signal, the wrong signal, or no signal to find for most errors. The bias share shows it is the third, for most errors there is nothing in the disagreement to detect. Even the best resample signal we could build, the winner's instability ($1 - f_\text{same}$), only reaches AUC 0.78 to 0.81, and it is capped the same way, it cannot separate a stable wrong winner from a confidently correct one. The paper measured how strong the signal is, we measure the limit of the whole class. That limit is the new result, and it is not in the prior work.

## What this means for safety

The unreliability of LLM judges is usually treated as an engineering problem with cheap fixes. We find the three cheapest all hit the same ceiling.

**More samples.** Averaging more draws is the standard fix, and it does work on variance errors, more samples sharpen each response's mean and clear out the mistakes that were only sampling noise. That is the variance reduction resampling is meant to buy. But a bias error is immune to it, its expected scores already rank the wrong winner first, so more draws only confirm it. We only measured k=8, so this part is reasoned rather than shown: a bigger budget cannot raise the ceiling, and by removing the catchable variance errors it leaves a residue that is even more concentrated in the invisible kind.

**Bigger panels.** Polling several judges and trusting agreement fails for the same reason resampling one judge fails, only across models, a bias error is confidently wrong on every member that shares it. That correlation across judges is the signature in our data, a panel of correlated judges buys fewer independent votes than its size suggests.

**Escalation.** A common pattern is to monitor with a cheap judge and escalate the suspicious cases to a frontier one. On Safety there is barely a gradient to climb, the cheap monitors are already as accurate as the frontier judges, and Haiku is more accurate.

| Judge | Safety accuracy | Overall accuracy |
| --- | --- | --- |
| GPT-5.4 (frontier) | 92.2% | 81.7% |
| Sonnet 4.6 (frontier) | 88.8% | 82.8% |
| GPT-5.4-mini (cheap) | 92.7% | 79.2% |
| Haiku 4.5 (cheap) | 95.9% | 85.3% |

The overall column is the context: the cheap GPT-5.4-mini is the weakest of the four across all categories, yet it matches the frontier GPT-5.4 on Safety, so the parity is specific to the slice that matters here. (Accuracy is the benchmark's own metric, the correct response must score strictly highest, so a tie at the top counts as incorrect.)

The deeper problem is correlation. We saw its shape earlier: two judges from the same family share their blind spots far more than two different providers do. So escalating from mini to full within a family is the worst case, it sends the monitor's hardest cases to the judge whose blind spots line up most closely with its own. The trigger is weak on top of that, escalating the most uncertain fifth of cases catches only about a third of the frontier judge's bias errors. Sampling across providers helps a panel, but it does not reliably rescue escalation, in our tests the cross-family backstop helped for one pair and reversed for the other.

**The adversary.** We have not run an adversarial optimisation, that is the obvious next experiment, so this part is argued rather than shown. But the shape of the risk is clear. The bias share is a failure surface. An adversary who can push an input into the stable region gets a confidently-wrong monitor for free, and the recall ceiling means no disagreement-based check, not more samples, not a bigger panel, not escalation, will flag it. It gives the most reassurance on exactly the inputs an adversary would choose.

## Limits and what comes next

The one sentence to take away: a single judge's errors are mostly stable under its own resampling, and that stability is a hard ceiling on the recall of any disagreement-based detector. It is fixed before any detector is built, it is tightest on Safety, and it rises as judges get stronger. Prior work measured how well these signals work, we measure the ceiling on the whole class, and we find it is highest exactly where a safety monitor can least afford it.

The limits are worth stating plainly. This is one benchmark (RewardBench 2), two providers and their smaller tiers, prompted judges on multiple-choice responses, at k=8. We have not run an adversarial optimisation, so the adversary argument is a failure surface we describe, not an attack we have shown, the rescore is the next experiment. We have also been careful to say stable, not biased. $f_\text{same}$ shows that a wrong winner is repeatable, it does not show why. Whether GPT-5.4's 86% on Safety is one failure mode, say a preference for confident, refusal-shaped distractors, or several, is a question for a hand-read we have not done. Until then "bias" stays a label for the statistic, not a mechanism.

The same shape of claim runs through what we think is the natural sequel. The uncertainty signal you would most want to trust is blind exactly where an adversary would operate. A judge that is confidently wrong on a steered input, and a model that sandbags when it senses it is being evaluated, are the same problem, the behaviour you care about hides where your detector sees calm.
