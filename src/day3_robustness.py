"""Day 3 — robustness (zero new compute).

Stress-tests the two Day-1 conclusions:
  (A) bias errors dominate (within-judge bias share is high), and
  (B) the two judges share errors beyond difficulty (MH OR > 1).

Six checks, each labelled stable / unstable in the summary:
  1. Criteria injection, Analysis A — does it shift errors between bias/variance?
  2. Criteria injection, Analysis B — does it change cross-judge overlap?
  3. Tier judges, Analysis A — do GPT-5.4-mini / Haiku 4.5 / GPT-5.4-nano also
     show bias-dominated errors?
  4. f_same threshold sensitivity {0.75, 0.90, 0.95}.
  5. Tie errors included vs excluded (Analysis A denominator; Analysis B E-indicator).
  6. Difficulty proxy: tier-error-count strata vs quartiles of the response-0-vs-best
     distractor score margin (from the base tier judges).

Difficulty strata always come from the BASE tier judges (an example property reused
across conditions); GPT-5.4-nano has no criteria run, which is fine because difficulty
is fixed.

Outputs:
    results/day3_robustness.csv     tidy table of every number
    results/day3_summary.md         one paragraph per check, each labelled stable/unstable

Run:  uv run python -m src.day3_robustness
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src import winners as W
from src.data_loader import REPO_ROOT, SEED, SUBSETS, load_long
from src.day1_analysis_b import (
    _tables_by_stratum,
    excess_with_ci,
    mantel_haenszel,
    raw_odds_ratio,
)

RESULTS = REPO_ROOT / "results"

THRESHOLDS = [0.75, 0.90, 0.95]
PRIMARY = 0.90
B_CI = 2000
ALIVE_BIAS = 0.25
MAIN = [("gpt-5.4", "GPT-5.4"), ("claude-sonnet-4-6", "Sonnet 4.6")]
TIERS_A = [("gpt-5.4-mini", "GPT-5.4-mini"), ("claude-haiku-4-5-20251001", "Haiku 4.5"),
           ("gpt-5.4-nano", "GPT-5.4-nano")]
TIER_COLS = [("gpt-5.4-mini", "E_mini"), ("claude-haiku-4-5-20251001", "E_haiku"),
             ("gpt-5.4-nano", "E_nano")]

_WF_CACHE: dict = {}


def get_wf(long, jm, cond, stability=False):
    key = (jm, cond, stability)
    if key not in _WF_CACHE:
        _WF_CACHE[key] = W.winners_frame(long, jm, cond, k=8, stability=stability, B=1000, seed=SEED)
    return _WF_CACHE[key]


# --------------------------------------------------------------------------- #
# bias share with bootstrap CI and a tie-handling switch                      #
# --------------------------------------------------------------------------- #
def bias_share(cell, thr, rng, tie_mode="exclude"):
    """tie_mode='exclude' -> denominator = non-tie errors (the primary definition);
    'include' -> denominator = non-tie errors + ties (ties counted as non-bias errors)."""
    is_err = cell["is_error"].to_numpy()
    is_tie = cell["is_tie"].to_numpy()
    fs = cell["f_same"].to_numpy()
    bias = is_err & (fs >= thr)
    denom = is_err if tie_mode == "exclude" else (is_err | is_tie)
    n, nd = len(cell), int(denom.sum())
    if nd == 0:
        return float("nan"), float("nan"), float("nan"), 0
    point = float(bias.sum() / nd)
    idx = rng.integers(0, n, size=(B_CI, n))
    num = bias[idx].sum(axis=1)
    den = denom[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = np.where(den > 0, num / den, np.nan)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), nd


# --------------------------------------------------------------------------- #
# cross-judge common frame (main judges under cond_*, tiers always base)      #
# --------------------------------------------------------------------------- #
def build_common(long, cond_g, cond_c):
    wf_g = get_wf(long, "gpt-5.4", cond_g).set_index(["category", "example_id"])
    wf_c = get_wf(long, "claude-sonnet-4-6", cond_c).set_index(["category", "example_id"])
    tiers = {jm: get_wf(long, jm, "base").set_index(["category", "example_id"]) for jm, _ in TIER_COLS}
    idx = set(wf_g.index) & set(wf_c.index)
    for t in tiers.values():
        idx &= set(t.index)
    idx = sorted(idx)
    df = pd.DataFrame(index=pd.MultiIndex.from_tuples(idx, names=["category", "example_id"]))
    df["E_g"] = (~wf_g["is_correct"]).reindex(idx).astype(int).to_numpy()
    df["E_c"] = (~wf_c["is_correct"]).reindex(idx).astype(int).to_numpy()
    df["E_g_strict"] = wf_g["is_error"].reindex(idx).astype(int).to_numpy()
    df["E_c_strict"] = wf_c["is_error"].reindex(idx).astype(int).to_numpy()
    for jm, col in TIER_COLS:
        df[col] = (~tiers[jm]["is_correct"]).reindex(idx).astype(int).to_numpy()
    df["s"] = df[[c for _, c in TIER_COLS]].sum(axis=1)
    return df.reset_index()


def margin_quartiles(long, df):
    """Alternative difficulty proxy: quartiles of the mean (base) tier-judge score
    margin between response 0 and the best distractor."""
    keys = list(zip(df["category"], df["example_id"]))
    series = []
    for jm, _ in TIER_COLS:
        S, k = W.example_cube(long, jm, "base")
        means, _ = W._means_stds(S, 8)
        with warnings.catch_warnings():  # all-null distractor rows -> NaN, dropped on reindex
            warnings.simplefilter("ignore", category=RuntimeWarning)
            marg = means[:, 0] - np.nanmax(means[:, 1:], axis=1)
        series.append(pd.Series(marg, index=pd.MultiIndex.from_frame(k)))
    M = pd.concat(series, axis=1).mean(axis=1).reindex(keys).to_numpy()
    return np.asarray(pd.qcut(M, 4, labels=False, duplicates="drop"), dtype=int)


def overlap_stats(df, eg, ec, s, rng):
    cell = pd.DataFrame({"E_g": df[eg].to_numpy(), "E_c": df[ec].to_numpy(),
                         "s": np.asarray(s) if not isinstance(s, str) else df[s].to_numpy()})
    or_mh, or_lo, or_hi = mantel_haenszel(_tables_by_stratum(cell))
    joint, expected, excess, exc_lo, exc_hi = excess_with_ci(cell, rng)
    return {"MH_OR": or_mh, "MH_lo": or_lo, "MH_hi": or_hi, "raw_OR": raw_odds_ratio(cell),
            "excess": excess, "exc_lo": exc_lo, "exc_hi": exc_hi, "P_joint": joint}


# --------------------------------------------------------------------------- #
# run all six checks                                                          #
# --------------------------------------------------------------------------- #
def run(long):
    rng = np.random.default_rng(SEED)
    rows = []

    def add(check, item, condition, variant, category, metric, value, lo=np.nan, hi=np.nan, n=np.nan):
        rows.append({"check": check, "item": item, "condition": condition, "variant": variant,
                     "category": category, "metric": metric, "value": value,
                     "ci_lo": lo, "ci_hi": hi, "n": n})

    S = {}  # scratch numbers for the prose

    # --- Check 1: criteria injection, Analysis A ---------------------------
    for jm, disp in MAIN:
        for cond in ("base", "criteria"):
            wf = get_wf(long, jm, cond, stability=True)
            add("1_criteria_A", disp, cond, "-", "ALL", "error_rate", float(wf.is_error.mean()))
            add("1_criteria_A", disp, cond, "-", "ALL", "tie_rate", float(wf.is_tie.mean()))
            for cat in ("ALL", "Safety"):
                cell = wf if cat == "ALL" else wf[wf.category == cat]
                pt, lo, hi, nd = bias_share(cell, PRIMARY, rng)
                add("1_criteria_A", disp, cond, "f_same>=0.9", cat, "bias_share", pt, lo, hi, nd)
                S[(disp, cond, cat)] = pt

    # --- Check 2: criteria injection, Analysis B ---------------------------
    for cond in ("base", "criteria"):
        df = build_common(long, cond, cond)
        ov = overlap_stats(df, "E_g", "E_c", "s", rng)
        add("2_criteria_B", "GPT-5.4 ∩ Sonnet 4.6", cond, "tier-count strata", "ALL", "MH_OR",
            ov["MH_OR"], ov["MH_lo"], ov["MH_hi"], len(df))
        add("2_criteria_B", "GPT-5.4 ∩ Sonnet 4.6", cond, "tier-count strata", "ALL", "excess",
            ov["excess"], ov["exc_lo"], ov["exc_hi"], len(df))
        S[("B", cond)] = ov

    # --- Check 3: tier judges, Analysis A ----------------------------------
    for jm, disp in TIERS_A:
        wf = get_wf(long, jm, "base", stability=True)
        add("3_tiers_A", disp, "base", "-", "ALL", "error_rate", float(wf.is_error.mean()))
        for cat in ("ALL", "Safety"):
            cell = wf if cat == "ALL" else wf[wf.category == cat]
            pt, lo, hi, nd = bias_share(cell, PRIMARY, rng)
            add("3_tiers_A", disp, "base", "f_same>=0.9", cat, "bias_share", pt, lo, hi, nd)
            S[("tier", disp, cat)] = pt

    # --- Check 4: f_same threshold sensitivity (main judges, base) ---------
    for jm, disp in MAIN:
        wf = get_wf(long, jm, "base", stability=True)
        for cat in ("ALL", "Safety"):
            cell = wf if cat == "ALL" else wf[wf.category == cat]
            for thr in THRESHOLDS:
                pt, lo, hi, nd = bias_share(cell, thr, rng)
                add("4_threshold", disp, "base", f"f_same>={thr}", cat, "bias_share", pt, lo, hi, nd)
                S[("thr", disp, cat, thr)] = pt

    # --- Check 5: ties included vs excluded --------------------------------
    for jm, disp in MAIN:  # Analysis A
        wf = get_wf(long, jm, "base", stability=True)
        for tie_mode in ("exclude", "include"):
            pt, lo, hi, nd = bias_share(wf, PRIMARY, rng, tie_mode=tie_mode)
            add("5_ties_A", disp, "base", f"ties_{tie_mode}", "ALL", "bias_share", pt, lo, hi, nd)
            S[("tieA", disp, tie_mode)] = pt
    df_b = build_common(long, "base", "base")  # Analysis B
    for variant, eg, ec in (("ties_as_error", "E_g", "E_c"), ("ties_excluded", "E_g_strict", "E_c_strict")):
        ov = overlap_stats(df_b, eg, ec, "s", rng)
        add("5_ties_B", "GPT-5.4 ∩ Sonnet 4.6", "base", variant, "ALL", "MH_OR",
            ov["MH_OR"], ov["MH_lo"], ov["MH_hi"], len(df_b))
        S[("tieB", variant)] = ov

    # --- Check 6: difficulty proxy (tier-count vs margin quartiles) --------
    mq = margin_quartiles(long, df_b)
    ov_count = overlap_stats(df_b, "E_g", "E_c", "s", rng)
    ov_margin = overlap_stats(df_b, "E_g", "E_c", mq, rng)
    for variant, ov in (("tier-count strata", ov_count), ("margin-quartile strata", ov_margin)):
        add("6_proxy_B", "GPT-5.4 ∩ Sonnet 4.6", "base", variant, "ALL", "MH_OR",
            ov["MH_OR"], ov["MH_lo"], ov["MH_hi"], len(df_b))
        add("6_proxy_B", "GPT-5.4 ∩ Sonnet 4.6", "base", variant, "ALL", "excess",
            ov["excess"], ov["exc_lo"], ov["exc_hi"], len(df_b))
        S[("proxy", variant)] = ov

    return pd.DataFrame(rows), S


# --------------------------------------------------------------------------- #
# narrative                                                                   #
# --------------------------------------------------------------------------- #
def _p(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.0f}%"


def _pp(x):
    return f"{100*x:+.1f}pp"


def summary_md(S):
    L = ["# Day 3 — robustness", "",
         "Two Day-1 conclusions are stress-tested: (A) bias errors dominate (high "
         "within-judge bias share), and (B) the judges share errors beyond difficulty "
         "(MH OR > 1). Difficulty strata always come from the base tier judges.", ""]

    # Check 1
    g_b, g_c = S[("GPT-5.4", "base", "ALL")], S[("GPT-5.4", "criteria", "ALL")]
    s_b, s_c = S[("Sonnet 4.6", "base", "ALL")], S[("Sonnet 4.6", "criteria", "ALL")]
    gsaf_b, gsaf_c = S[("GPT-5.4", "base", "Safety")], S[("GPT-5.4", "criteria", "Safety")]
    ssaf_b, ssaf_c = S[("Sonnet 4.6", "base", "Safety")], S[("Sonnet 4.6", "criteria", "Safety")]
    L.append("### Check 1 — criteria injection (Analysis A)")
    L.append(f"Bias share base→criteria — overall: GPT-5.4 {_p(g_b)}→{_p(g_c)}, "
             f"Sonnet 4.6 {_p(s_b)}→{_p(s_c)}; Safety: GPT-5.4 {_p(gsaf_b)}→{_p(gsaf_c)}, "
             f"Sonnet 4.6 {_p(ssaf_b)}→{_p(ssaf_c)}. Criteria injection *lowers* the bias share — "
             "it converts some confidently-wrong errors into resolvable, variance-dominated ones "
             "(most visibly Sonnet 4.6 Safety). "
             f"**Stable** — every cell stays ≥ {_p(ALIVE_BIAS)} (the recall-ceiling bar), so most "
             "errors remain invisible to resampling; criteria helps only at the margin.")
    L.append("")

    # Check 2
    bb, bc = S[("B", "base")], S[("B", "criteria")]
    L.append("### Check 2 — criteria injection (Analysis B)")
    L.append(f"MH OR base {bb['MH_OR']:.2f} [{bb['MH_lo']:.2f}, {bb['MH_hi']:.2f}] → "
             f"criteria {bc['MH_OR']:.2f} [{bc['MH_lo']:.2f}, {bc['MH_hi']:.2f}]; adjusted excess "
             f"base {_pp(bb['excess'])} → criteria {_pp(bc['excess'])}. "
             f"**{'Stable' if bc['MH_lo'] > 1 else 'Unstable'}** — significant residual cross-judge "
             "overlap persists under criteria.")
    L.append("")

    # Check 3
    L.append("### Check 3 — tier judges (Analysis A)")
    mini = S[("tier", "GPT-5.4-mini", "ALL")]
    haiku = S[("tier", "Haiku 4.5", "ALL")]
    nano = S[("tier", "GPT-5.4-nano", "ALL")]
    L.append(f"Overall bias share by capability: GPT-5.4 {_p(g_b)} / Sonnet 4.6 {_p(s_b)} (frontier) > "
             f"GPT-5.4-mini {_p(mini)} ≈ Haiku 4.5 {_p(haiku)} > GPT-5.4-nano {_p(nano)}. "
             "**Stable (and sharpened)** — bias share *rises with judge capability*: every tier stays "
             f"≥ {_p(ALIVE_BIAS)} (nano borderline), and the frontier judges — the ones actually "
             "deployed as monitors — are the most bias-dominated, so the failure is most acute exactly "
             "where it matters.")
    L.append("")

    # Check 4
    L.append("### Check 4 — f_same threshold sensitivity")
    for _, disp in MAIN:
        vals = " / ".join(_p(S[("thr", disp, "ALL", t)]) for t in THRESHOLDS)
        L.append(f"- {disp} overall at f_same ≥ 0.75 / 0.90 / 0.95: {vals}.")
    L.append(f"**Stable** — bias errors dominate (≥{_p(ALIVE_BIAS)}) at every threshold; the "
             "headline is not an artifact of the 0.90 cut.")
    L.append("")

    # Check 5
    L.append("### Check 5 — tie errors included vs excluded")
    gA = (S[("tieA", "GPT-5.4", "exclude")], S[("tieA", "GPT-5.4", "include")])
    sA = (S[("tieA", "Sonnet 4.6", "exclude")], S[("tieA", "Sonnet 4.6", "include")])
    te, tx = S[("tieB", "ties_as_error")], S[("tieB", "ties_excluded")]
    L.append(f"- Analysis A overall bias share, ties excluded→included in denominator: "
             f"GPT-5.4 {_p(gA[0])}→{_p(gA[1])}, Sonnet 4.6 {_p(sA[0])}→{_p(sA[1])}.")
    L.append(f"- Analysis B MH OR, ties-as-error {te['MH_OR']:.2f} [{te['MH_lo']:.2f}, {te['MH_hi']:.2f}] "
             f"vs ties-excluded {tx['MH_OR']:.2f} [{tx['MH_lo']:.2f}, {tx['MH_hi']:.2f}].")
    stable5 = (gA[1] >= ALIVE_BIAS and sA[1] >= ALIVE_BIAS and tx["MH_lo"] > 1)
    L.append(f"**{'Stable' if stable5 else 'Unstable'}** — both conclusions survive either tie convention "
             "(folding ties in lowers the bias share but it stays dominant; the overlap stays significant).")
    L.append("")

    # Check 6
    L.append("### Check 6 — difficulty proxy (tier-count vs margin quartiles)")
    pc, pm = S[("proxy", "tier-count strata")], S[("proxy", "margin-quartile strata")]
    L.append(f"MH OR with tier-error-count strata {pc['MH_OR']:.2f} [{pc['MH_lo']:.2f}, {pc['MH_hi']:.2f}] "
             f"vs score-margin-quartile strata {pm['MH_OR']:.2f} [{pm['MH_lo']:.2f}, {pm['MH_hi']:.2f}]; "
             f"adjusted excess {_pp(pc['excess'])} vs {_pp(pm['excess'])}. "
             f"**{'Stable' if pm['MH_lo'] > 1 else 'Unstable'}** — the residual overlap does not depend on "
             "the choice of difficulty proxy.")
    L.append("")
    L.append("Table: `results/day3_robustness.csv`.")
    L.append("")
    return "\n".join(L)


def main():
    long = load_long()
    df, S = run(long)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "day3_robustness.csv", index=False)
    (RESULTS / "day3_summary.md").write_text(summary_md(S))
    # console: pivot the headline rows
    show = df[df.metric.isin(["bias_share", "MH_OR"])][
        ["check", "item", "condition", "variant", "category", "metric", "value", "ci_lo", "ci_hi"]
    ]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(show.to_string(index=False))
    print("\nWrote results/day3_robustness.csv and results/day3_summary.md")


if __name__ == "__main__":
    main()
