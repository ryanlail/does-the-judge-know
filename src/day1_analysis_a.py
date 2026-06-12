"""Day 1, Analysis A — within-judge bias share (zero new compute).

For GPT-5.4 base and Sonnet 4.6 base (k=8): error / tie-error rates, the bias
share (bias-dominated errors / non-tie errors) with bootstrap CIs over examples,
the recall ceiling (1 - bias share), and three variance-AUCs for predicting
incorrectness (replicating + extending the paper's ~0.60).

Outputs:
    results/day1_bias_share.csv     judge x category x threshold (point, CI, n_err)
    results/day1_auc.csv            judge x signal x example-set
    figures/fig1_bias_share_by_category.png  (+ .csv with the plotted numbers)
    results/day1_summary.md         Analysis A section (overwrites)

Run:  uv run python -m src.day1_analysis_a
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import winners as W
from src.data_loader import REPO_ROOT, SEED, SUBSETS, load_long

RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"

JUDGES = [("gpt-5.4", "GPT-5.4"), ("claude-sonnet-4-6", "Sonnet 4.6")]
THRESHOLDS = [0.75, 0.90, 0.95]
PRIMARY_THR = 0.90
B_CI = 2000
CATS = (*SUBSETS, "ALL")


def _cell(wf: pd.DataFrame, category: str) -> pd.DataFrame:
    return wf if category == "ALL" else wf[wf["category"] == category]


def bias_share_ci(cell: pd.DataFrame, thr: float, rng) -> tuple[float, float, float, int]:
    """Bias share = (non-tie errors with f_same>=thr) / (non-tie errors), with a
    bootstrap-over-examples 95% CI. Both numerator and denominator are resampled."""
    is_err = cell["is_error"].to_numpy()
    fs = cell["f_same"].to_numpy()
    bias = is_err & (fs >= thr)  # ties have NaN f_same -> False; non-errors -> False
    n = len(cell)
    n_err = int(is_err.sum())
    if n_err == 0:
        return float("nan"), float("nan"), float("nan"), 0
    point = float(bias.sum() / n_err)
    idx = rng.integers(0, n, size=(B_CI, n))
    be = bias[idx].sum(axis=1)
    er = is_err[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = np.where(er > 0, be / er, np.nan)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), n_err


def build_bias_share_table(wfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for jm, disp in JUDGES:
        wf = wfs[jm]
        for cat in CATS:
            cell = _cell(wf, cat)
            for thr in THRESHOLDS:
                point, lo, hi, n_err = bias_share_ci(cell, thr, rng)
                rows.append({
                    "judge_model": jm, "judge_display": disp, "category": cat,
                    "threshold": thr, "n_examples": len(cell), "n_error": n_err,
                    "bias_share": point, "ci_lo": lo, "ci_hi": hi,
                    "recall_ceiling": 1 - point,
                })
    return pd.DataFrame(rows)


def build_auc_table(wfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for jm, disp in JUDGES:
        wf = wfs[jm]
        # (1) paper replication: signal a, label = incorrect (ties incl.), all usable
        r = W.variance_auc(wf["mean_response_std"], (~wf["is_correct"]).astype(int))
        rows.append({"judge_model": jm, "judge_display": disp, "signal": "mean_response_std",
                     "example_set": "all (ties=incorrect)", **r})
        # (2) extension: three signals on non-tie examples, label = error
        nt = wf[~wf["is_tie"]]
        lbl = nt["is_error"].astype(int)
        for sig in ("mean_response_std", "winner_std", "instability"):
            r = W.variance_auc(nt[sig], lbl)
            rows.append({"judge_model": jm, "judge_display": disp, "signal": sig,
                         "example_set": "non-tie (error vs correct)", **r})
    return pd.DataFrame(rows)


def rates(wfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for jm, disp in JUDGES:
        wf = wfs[jm]
        rows.append({
            "judge_model": jm, "judge_display": disp, "n": len(wf),
            "error_rate": float(wf["is_error"].mean()),
            "tie_rate": float(wf["is_tie"].mean()),
            "accuracy": float(wf["is_correct"].mean()),
        })
    return pd.DataFrame(rows)


def make_fig1(bias_df: pd.DataFrame) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot = bias_df[bias_df["threshold"] == PRIMARY_THR].copy()
    cats = list(CATS)
    x = np.arange(len(cats))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    for offset, (jm, disp), color in zip(
        (-width / 2, width / 2), JUDGES, ("#3b6fb6", "#c1543f")
    ):
        d = plot[plot["judge_model"] == jm].set_index("category").reindex(cats)
        vals = d["bias_share"].to_numpy()
        lo = vals - d["ci_lo"].to_numpy()
        hi = d["ci_hi"].to_numpy() - vals
        bars = ax.bar(x + offset, vals, width, label=disp, color=color,
                      yerr=[lo, hi], capsize=3, ecolor="#444", error_kw={"lw": 1})
        # highlight Safety (headline slice)
        si = cats.index("Safety")
        bars[si].set_edgecolor("black")
        bars[si].set_linewidth(2.2)

    ax.axhline(0.25, ls="--", lw=1, color="#888")
    ax.text(len(cats) - 0.5, 0.255, "alive threshold (25%)", ha="right", va="bottom",
            fontsize=8, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel(f"bias share  (f_same ≥ {PRIMARY_THR})")
    ax.set_ylim(0, 1)
    ax.set_title("Fraction of judge errors that are bias errors, by RewardBench-2 category")
    ax.legend(frameon=False)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig1_bias_share_by_category.png")
    plt.close(fig)

    behind = plot[["judge_display", "category", "n_error", "bias_share", "ci_lo", "ci_hi"]]
    behind.to_csv(FIGURES / "fig1_bias_share_by_category.csv", index=False)
    return behind


def _fmt_pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def md_section(bias_df, auc_df, rates_df) -> str:
    L = ["# Day 1 — within-judge bias share and cross-judge overlap", "",
         "## Analysis A — within-judge bias share", "",
         "Base prompt, k=8. Bias-dominated error = non-tie error with bootstrap winner "
         f"stability f_same ≥ {PRIMARY_THR} (resampling B=1000 per example). Bias share = "
         "bias-dominated / non-tie errors. Recall ceiling = 1 − bias share (the most any "
         "resample-based uncertainty signal can catch). CIs: bootstrap over examples, "
         f"B={B_CI}, seed {SEED}.", ""]

    L.append("**Error structure**")
    for _, r in rates_df.iterrows():
        L.append(f"- {r.judge_display}: accuracy {_fmt_pct(r.accuracy)}, error rate "
                 f"{_fmt_pct(r.error_rate)}, tie rate {_fmt_pct(r.tie_rate)} (n={r.n}).")
    L.append("")

    L.append(f"**Bias share (f_same ≥ {PRIMARY_THR}), overall and Safety**")
    for jm, disp in JUDGES:
        ov = bias_df[(bias_df.judge_model == jm) & (bias_df.category == "ALL") & (bias_df.threshold == PRIMARY_THR)].iloc[0]
        sf = bias_df[(bias_df.judge_model == jm) & (bias_df.category == "Safety") & (bias_df.threshold == PRIMARY_THR)].iloc[0]
        L.append(f"- {disp}: overall **{_fmt_pct(ov.bias_share)}** "
                 f"[{_fmt_pct(ov.ci_lo)}–{_fmt_pct(ov.ci_hi)}], n_err={int(ov.n_error)} "
                 f"(recall ceiling {_fmt_pct(ov.recall_ceiling)}); "
                 f"Safety **{_fmt_pct(sf.bias_share)}** [{_fmt_pct(sf.ci_lo)}–{_fmt_pct(sf.ci_hi)}], "
                 f"n_err={int(sf.n_error)} (recall ceiling {_fmt_pct(sf.recall_ceiling)}).")
    L.append("")
    L.append("Sensitivity (overall bias share at f_same thresholds 0.75 / 0.90 / 0.95):")
    for jm, disp in JUDGES:
        vals = [bias_df[(bias_df.judge_model == jm) & (bias_df.category == "ALL") & (bias_df.threshold == t)].iloc[0].bias_share for t in THRESHOLDS]
        L.append(f"- {disp}: " + " / ".join(_fmt_pct(v) for v in vals))
    L.append("")

    L.append("**Variance as an incorrectness classifier (AUC)**")
    for jm, disp in JUDGES:
        rep = auc_df[(auc_df.judge_model == jm) & (auc_df.example_set.str.startswith("all"))].iloc[0]
        L.append(f"- {disp} (paper replication, signal=mean response std, ties=incorrect): "
                 f"AUC **{rep.auc:.3f}** (n={int(rep.n)}, base rate {_fmt_pct(rep.base_rate_incorrect)}).")
        nt = auc_df[(auc_df.judge_model == jm) & (auc_df.example_set.str.startswith("non-tie"))]
        trio = {row.signal: row.auc for _, row in nt.iterrows()}
        L.append(f"    non-tie (error vs correct): mean_response_std {trio['mean_response_std']:.3f}, "
                 f"winner_std {trio['winner_std']:.3f}, instability(1−f_same) {trio['instability']:.3f}.")
    L.append("")
    L.append("Figure: `figures/fig1_bias_share_by_category.png`. "
             "Tables: `results/day1_bias_share.csv`, `results/day1_auc.csv`.")
    L.append("")
    return "\n".join(L)


def run(long):
    wfs = {jm: W.winners_frame(long, jm, "base", k=8, stability=True, B=1000, seed=SEED)
           for jm, _ in JUDGES}
    bias_df = build_bias_share_table(wfs)
    auc_df = build_auc_table(wfs)
    rates_df = rates(wfs)
    return wfs, bias_df, auc_df, rates_df


def main():
    long = load_long()
    _, bias_df, auc_df, rates_df = run(long)
    RESULTS.mkdir(exist_ok=True)
    bias_df.to_csv(RESULTS / "day1_bias_share.csv", index=False)
    auc_df.to_csv(RESULTS / "day1_auc.csv", index=False)
    make_fig1(bias_df)
    (RESULTS / "day1_summary.md").write_text(md_section(bias_df, auc_df, rates_df))
    print(rates_df.to_string(index=False))
    print()
    print(bias_df[bias_df.threshold == PRIMARY_THR][
        ["judge_display", "category", "n_error", "bias_share", "ci_lo", "ci_hi"]
    ].to_string(index=False))
    print("\nWrote day1_bias_share.csv, day1_auc.csv, fig1, day1_summary.md")


if __name__ == "__main__":
    main()
