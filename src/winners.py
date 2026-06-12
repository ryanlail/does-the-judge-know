"""Winner / error / tie classification and resample-stability signals (Day 1).

Reproduces the paper protocol from the upstream ``analysis/compute_metrics.py``
(at the pinned commit) and adds the project's new per-example quantities:

- **Outcome.** Predicted winner = response with the strictly-highest mean over
  the k samples (mean over non-null samples only). ``correct`` = unique argmax at
  response 0; ``error`` = unique argmax at a response != 0; ``tie`` = >1 response
  shares the max (counted as not-correct, analysed separately); ``skip`` = some
  response has no valid samples (the example is unusable, matching upstream's
  "skip if any response mean is None").
- **Variance signals.** ``mean_response_std`` (signal a, = the published
  variance-AUC signal): mean over the 4 responses of each response's per-sample
  std (std = 0 if <= 1 valid sample). ``winner_std`` (signal b): std of the
  predicted-winner response (NaN for ties).
- **Winner stability** (new): for one example, resample each response's valid
  samples with replacement and recompute the winner, B times. ``f_same`` =
  fraction of draws reproducing the observed unique winner (NaN for ties);
  ``f_err`` = fraction of draws where response 0 is not the unique winner.
  ``instability`` = 1 - f_same (signal c). For a non-tie *error*, f_same is the
  bias/variance discriminant: high f_same => the wrong winner is stable under
  resampling => a bias error.

The ROC-AUC helper ports the upstream rank-based trapezoid verbatim so that
``mean_response_std`` reproduces the paper's "variance predicts incorrectness"
AUC (~0.60).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.data_loader import K, N_RESPONSES, S_COLS, SEED, SUBSETS


# --------------------------------------------------------------------------- #
# extract a (N, 4, K) sample cube for one (judge_model, condition)            #
# --------------------------------------------------------------------------- #
def example_cube(long: pd.DataFrame, judge_model: str, condition: str):
    """Return (S, keys): S is float (N, N_RESPONSES, K) with NaN for errored
    samples; keys is a DataFrame of (category, example_id), example-aligned."""
    sub = long[(long["judge_model"] == judge_model) & (long["condition"] == condition)]
    sub = sub.sort_values(["category", "example_id", "response_index"])
    n_rows = len(sub)
    if n_rows % N_RESPONSES != 0:
        raise ValueError(f"{judge_model}/{condition}: {n_rows} rows not divisible by {N_RESPONSES}")
    # response_index must be 0..3 within every block
    ri = sub["response_index"].to_numpy().reshape(-1, N_RESPONSES)
    if not np.array_equal(ri, np.tile(np.arange(N_RESPONSES), (ri.shape[0], 1))):
        raise ValueError(f"{judge_model}/{condition}: response_index not 0..{N_RESPONSES-1} per example")
    S = sub[S_COLS].astype("float").to_numpy().reshape(-1, N_RESPONSES, K)
    keys = sub[["category", "example_id"]].iloc[::N_RESPONSES].reset_index(drop=True)
    return S, keys


# --------------------------------------------------------------------------- #
# per-example outcome + variance signals (vectorised)                         #
# --------------------------------------------------------------------------- #
def _means_stds(S: np.ndarray, k: int | None):
    Sk = S[:, :, :k] if k else S
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
        means = np.nanmean(Sk, axis=2)            # (N, 4)
        stds = np.nanstd(Sk, axis=2, ddof=0)      # (N, 4); 0 for single sample
    stds = np.where(np.isnan(stds), 0.0, stds)    # all-null response -> std 0 (masked out by skip)
    return means, stds


def classify(means: np.ndarray):
    """means: (N, 4). Returns (outcome str array, winner_index int array, usable bool array).

    Uses exact float equality for ties, matching upstream ``m == max_score``.
    """
    usable = ~np.isnan(means).any(axis=1)
    safe = np.where(np.isnan(means), -np.inf, means)
    mx = safe.max(axis=1, keepdims=True)
    is_max = safe == mx
    n_winners = is_max.sum(axis=1)
    winner_idx = np.argmax(safe, axis=1)
    outcome = np.full(means.shape[0], "skip", dtype=object)
    tie = usable & (n_winners > 1)
    uniq = usable & (n_winners == 1)
    outcome[tie] = "tie"
    outcome[uniq & (winner_idx == 0)] = "correct"
    outcome[uniq & (winner_idx != 0)] = "error"
    return outcome, winner_idx, usable


# --------------------------------------------------------------------------- #
# winner-stability bootstrap (new quantity)                                   #
# --------------------------------------------------------------------------- #
def winner_stability(S: np.ndarray, k: int | None = K, B: int = 1000, seed: int = SEED):
    """Per-example f_same / f_err over B resamples (NaN where the example is skipped).

    Each response is bootstrapped from its own valid samples (draw n_valid with
    replacement). f_same uses the observed unique winner; for ties f_same is NaN.
    """
    Sk = S[:, :, :k] if k else S
    means, _ = _means_stds(S, k)
    outcome, winner_idx, usable = classify(means)
    n = Sk.shape[0]
    f_same = np.full(n, np.nan)
    f_err = np.full(n, np.nan)
    rng = np.random.default_rng(seed)
    for i in range(n):
        if not usable[i]:
            continue
        boot_means = np.empty((B, N_RESPONSES))
        for j in range(N_RESPONSES):
            vals = Sk[i, j][~np.isnan(Sk[i, j])]
            idx = rng.integers(0, len(vals), size=(B, len(vals)))
            boot_means[:, j] = vals[idx].mean(axis=1)
        mx = boot_means.max(axis=1, keepdims=True)
        uniq = (boot_means == mx).sum(axis=1) == 1
        widx = np.argmax(boot_means, axis=1)
        f_err[i] = np.mean(~(uniq & (widx == 0)))
        if outcome[i] != "tie":
            f_same[i] = np.mean(uniq & (widx == winner_idx[i]))
    return f_same, f_err


# --------------------------------------------------------------------------- #
# assemble a per-example frame                                                #
# --------------------------------------------------------------------------- #
def winners_frame(
    long: pd.DataFrame,
    judge_model: str,
    condition: str,
    k: int | None = K,
    stability: bool = False,
    B: int = 1000,
    seed: int = SEED,
) -> pd.DataFrame:
    """One row per example for a (judge_model, condition). ``skip`` rows dropped."""
    S, keys = example_cube(long, judge_model, condition)
    means, stds = _means_stds(S, k)
    outcome, winner_idx, usable = classify(means)

    mean_response_std = stds.mean(axis=1)
    winner_std = np.where(outcome != "tie", stds[np.arange(len(stds)), winner_idx], np.nan)
    winner_std = np.where(usable, winner_std, np.nan)

    df = keys.copy()
    df["outcome"] = outcome
    df["winner_index"] = np.where(outcome == "tie", -1, winner_idx)
    df["is_correct"] = outcome == "correct"
    df["is_error"] = outcome == "error"
    df["is_tie"] = outcome == "tie"
    df["mean_response_std"] = mean_response_std
    df["winner_std"] = winner_std

    if stability:
        f_same, f_err = winner_stability(S, k=k, B=B, seed=seed)
        df["f_same"] = f_same
        df["f_err"] = f_err
        df["instability"] = 1.0 - f_same  # signal c (NaN for ties)

    return df[usable].reset_index(drop=True)


def accuracy_by_subset(wf: pd.DataFrame) -> pd.DataFrame:
    """n, n_correct, n_tied, n_error, accuracy per subset + overall. Ties count as
    not-correct in the denominator, matching the paper."""
    rows = []
    for sub in (*SUBSETS, "ALL"):
        d = wf if sub == "ALL" else wf[wf["category"] == sub]
        n = len(d)
        nc = int(d["is_correct"].sum())
        rows.append({
            "subset": sub, "n": n, "n_correct": nc,
            "n_tied": int(d["is_tie"].sum()), "n_error": int(d["is_error"].sum()),
            "accuracy": nc / n if n else float("nan"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# ROC-AUC of a variance-style signal predicting "incorrect" (upstream port)   #
# --------------------------------------------------------------------------- #
def variance_auc(scores, labels) -> dict:
    """Rank-based ROC-AUC, ported verbatim from upstream ``variance_as_classifier_auc``.

    ``scores`` higher => more "incorrect"; ``labels`` 1 = incorrect. NaN scores are
    dropped (a signal undefined for ties).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    mask = ~np.isnan(scores)
    scores, labels = scores[mask], labels[mask].astype(int)
    if len(scores) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return {"auc": None, "n": int(len(scores))}

    order = np.argsort(-scores)
    sl = labels[order]
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    tp = fp = 0
    tpr, fpr, precisions, recalls = [0.0], [0.0], [], []
    for y in sl:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)
    auc = float(np.trapezoid(tpr, fpr))

    def _p_at_r(target):
        for p, r in zip(precisions, recalls):
            if r >= target:
                return float(p)
        return None

    return {
        "auc": auc,
        "n": int(len(scores)),
        "n_incorrect": n_pos,
        "n_correct": n_neg,
        "base_rate_incorrect": n_pos / len(labels),
        "precision_at_recall_20": _p_at_r(0.20),
        "precision_at_recall_50": _p_at_r(0.50),
    }
