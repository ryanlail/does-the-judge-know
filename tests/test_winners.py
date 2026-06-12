"""Parity + unit tests for src/winners.py (Day 1).

Parity is checked against the published Claude k=8 accuracies in
data/tables/all_metrics.json. Those metrics were computed on an earlier, slightly
smaller snapshot than the vendored HEAD data (~10 more usable examples per subset
now), so only Factuality matches bit-for-bit; the rest are checked within a small
tolerance. This still pins the winner/error/tie logic to the paper.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import data_loader as dl
from src import winners as W


# --------------------------------------------------------------------------- #
# unit tests on numpy cubes (no vendored data needed)                         #
# --------------------------------------------------------------------------- #
def test_classify_outcomes():
    means = np.array([
        [8.0, 3.0, 2.0, 1.0],   # unique max at 0 -> correct
        [2.0, 9.0, 2.0, 1.0],   # unique max at 1 -> error
        [5.0, 5.0, 2.0, 1.0],   # tie at 0 and 1 -> tie
        [np.nan, 3.0, 2.0, 1.0],  # a null response mean -> skip
    ])
    outcome, winner, usable = W.classify(means)
    assert list(outcome) == ["correct", "error", "tie", "skip"]
    assert winner[1] == 1
    assert list(usable) == [True, True, True, False]


def _const_cube(rows):
    """rows: list of 4 per-response constant values -> (len, 4, K) cube."""
    return np.array([[[v] * W.K for v in r] for r in rows], dtype=float)


def test_winner_stability_bias_vs_variance():
    # example 0: a *bias* error — wrong winner (resp 1) is stable under resampling
    bias = [5.0, 9.0, 4.0, 3.0]
    # example 1: a *variance* error — resp 0 is high-variance, wrong winner (resp 1) is fragile
    var = np.array([[9, 9, 9, 9, 1, 1, 1, 1], [6] * 8, [4] * 8, [3] * 8], dtype=float)
    S = np.stack([_const_cube([bias])[0], var])
    f_same, f_err = W.winner_stability(S, k=8, B=2000, seed=42)
    assert f_same[0] >= 0.99            # bias error: winner reproduced almost always
    assert f_same[1] <= 0.80            # variance error: winner often flips
    assert f_same[0] > f_same[1]
    assert f_err[0] >= 0.99             # response 0 essentially never wins in the bias case


def test_variance_auc_basics():
    # perfectly separating signal -> AUC 1.0
    scores = [0.1, 0.2, 0.3, 0.9, 1.0, 1.1]
    labels = [0, 0, 0, 1, 1, 1]
    assert W.variance_auc(scores, labels)["auc"] == pytest.approx(1.0)
    # NaN scores (ties) are dropped
    r = W.variance_auc([0.1, np.nan, 1.0], [0, 1, 1])
    assert r["n"] == 2


# --------------------------------------------------------------------------- #
# parity against published numbers (skips if data not fetched)                #
# --------------------------------------------------------------------------- #
_HAVE_DATA = any(dl.DATA_RAW.glob("*.jsonl"))


@pytest.mark.skipif(not _HAVE_DATA, reason="run scripts/fetch_data.sh")
def test_accuracy_parity_sonnet_k8():
    long = dl.load_long()
    wf = W.winners_frame(long, "claude-sonnet-4-6", "base", k=8)
    acc = W.accuracy_by_subset(wf).set_index("subset")

    ref = json.load(open(dl.REPO_ROOT / "data" / "tables" / "all_metrics.json"))
    ref = ref["claude_full_ensemble_k8"]["accuracy"]["by_subset"]

    # Factuality is bit-exact across the snapshot drift.
    assert int(acc.loc["Factuality", "n"]) == 472
    assert int(acc.loc["Factuality", "n_correct"]) == 400

    # Every subset's accuracy is within 1pp of the published value.
    for sub in dl.SUBSETS:
        assert acc.loc[sub, "accuracy"] == pytest.approx(ref[sub]["accuracy"], abs=0.01)


@pytest.mark.skipif(not _HAVE_DATA, reason="run scripts/fetch_data.sh")
def test_variance_auc_replicates_paper():
    # The paper's "variance predicts incorrectness" AUC (~0.60): signal a
    # (mean response std), label = incorrect (ties included), over all examples.
    long = dl.load_long()
    wf = W.winners_frame(long, "gpt-5.4", "base", k=8)
    label = (~wf["is_correct"]).astype(int)
    res = W.variance_auc(wf["mean_response_std"], label)
    assert 0.55 <= res["auc"] <= 0.65, res["auc"]
