"""Day 3, Analysis C (optional) — does cheap-monitor + escalation inherit the
frontier judge's blind spots? (zero new compute)

Deployment pattern under test: monitor every case with a cheap (mini) judge and
escalate the suspicious top-X% (highest mini uncertainty) to the frontier (full)
judge. The safety question is whether the cheap monitor's uncertainty flags the
cases where the frontier judge is wrong — or whether shared bias makes the
escalation trigger silent exactly there.

This is NOT "small models predict large-model errors". On a bias error BOTH tiers
are confidently wrong, so neither one's variance rises; the claim is about the
escalation architecture, not small-model predictiveness.

Two parts, per provider tier pair (GPT-5.4↔GPT-5.4-mini, Sonnet 4.6↔Haiku 4.5):
  1. Cross-tier bias overlap: P(mini wrong | full BIAS error) vs a
     difficulty-adjusted baseline (+ the cross-tier MH OR).
  2. Escalation recall: escalate top-X% by mini variance (and, for contrast, by
     mini instability 1−f_same); what fraction of the full judge's BIAS errors
     land in the escalated set? Contrast with the full judge's VARIANCE errors.

Confound control (the tier judge is the OBJECT here, so it cannot also be the
difficulty stratifier): difficulty is the score margin (response 0 minus best
distractor) of the OTHER provider's full judge — distinct from both objects.

Outputs:
    results/day3c_escalation.csv
    figures/fig_escalation_recall.png
    one paragraph appended to results/day3_summary.md

Run:  uv run python -m src.day3c_escalation   (after src.day3_robustness)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src import winners as W
from src.data_loader import REPO_ROOT, SEED, load_long
from src.day1_analysis_b import _tables_by_stratum, mantel_haenszel

RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"

BIAS_THR = 0.90
BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
JUDGES = {"gpt-5.4": "gf", "gpt-5.4-mini": "gm",
          "claude-sonnet-4-6": "sf", "claude-haiku-4-5-20251001": "hk"}
PAIRS = [
    {"name": "GPT-5.4-mini → GPT-5.4", "full": "gf", "mini": "gm",
     "q": "q_gpt_pair", "proxy": "Sonnet 4.6 score-margin"},
    {"name": "Haiku 4.5 → Sonnet 4.6", "full": "sf", "mini": "hk",
     "q": "q_sonnet_pair", "proxy": "GPT-5.4 score-margin"},
]


def judge_cols(long, jm):
    wf = W.winners_frame(long, jm, "base", k=8, stability=True, B=1000, seed=SEED)
    wf = wf.set_index(["category", "example_id"])
    return pd.DataFrame({
        "err": (~wf["is_correct"]).astype(int),
        "bias": (wf["is_error"] & (wf["f_same"] >= BIAS_THR)).astype(int),
        "varerr": (wf["is_error"] & (wf["f_same"] < BIAS_THR)).astype(int),
        "var": wf["mean_response_std"],
        "instab": (1.0 - wf["f_same"]).fillna(1.0),  # ties -> max uncertainty
    })


def margin_series(long, jm):
    S, k = W.example_cube(long, jm, "base")
    means, _ = W._means_stds(S, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = means[:, 0] - np.nanmax(means[:, 1:], axis=1)
    return pd.Series(m, index=pd.MultiIndex.from_frame(k))


def build(long):
    J = {jm: judge_cols(long, jm) for jm in JUDGES}
    idx = None
    for v in J.values():
        idx = set(v.index) if idx is None else (idx & set(v.index))
    idx = sorted(idx)
    df = pd.DataFrame(index=pd.MultiIndex.from_tuples(idx, names=["category", "example_id"]))
    for jm, p in JUDGES.items():
        for c in ("err", "bias", "varerr", "var", "instab"):
            df[f"{p}_{c}"] = J[jm][c].reindex(idx).to_numpy()
    gf_m = margin_series(long, "gpt-5.4").reindex(idx).to_numpy()
    sf_m = margin_series(long, "claude-sonnet-4-6").reindex(idx).to_numpy()
    df["q_gpt_pair"] = pd.qcut(sf_m, 4, labels=False, duplicates="drop")     # Sonnet margin
    df["q_sonnet_pair"] = pd.qcut(gf_m, 4, labels=False, duplicates="drop")  # GPT margin
    return df.reset_index()


# --------------------------------------------------------------------------- #
# Part 1 — cross-tier bias overlap                                            #
# --------------------------------------------------------------------------- #
def part1(df, pair):
    full, mini, q = pair["full"], pair["mini"], pair["q"]
    bias = df[f"{full}_bias"].to_numpy().astype(bool)
    me = df[f"{mini}_err"].to_numpy().astype(float)
    qq = df[q].to_numpy()
    n_bias = int(bias.sum())
    observed = float(me[bias].mean())
    baseline = 0.0
    for u in np.unique(qq):
        inq = qq == u
        nbq = int((bias & inq).sum())
        if nbq:
            baseline += (nbq / n_bias) * float(me[inq].mean())
    cell = pd.DataFrame({"E_g": df[f"{full}_err"], "E_c": df[f"{mini}_err"], "s": qq})
    or_mh, lo, hi = mantel_haenszel(_tables_by_stratum(cell))
    return {"n_bias": n_bias, "p_mini_given_full_bias": observed,
            "mini_err_overall": float(me.mean()), "diff_adj_baseline": baseline,
            "excess": observed - baseline, "mh_or": or_mh, "mh_lo": lo, "mh_hi": hi}


# --------------------------------------------------------------------------- #
# Part 2 — escalation recall                                                  #
# --------------------------------------------------------------------------- #
def recall_at(scores, label, budgets):
    order = np.argsort(-np.asarray(scores, float))
    lab = np.asarray(label, float)[order]
    n = len(lab)
    npos = lab.sum()
    cum = np.cumsum(lab)
    out = []
    for X in budgets:
        k = max(1, int(np.ceil(X * n)))
        out.append(cum[k - 1] / npos if npos > 0 else np.nan)
    return np.array(out)


def part2(df, pair):
    full, mini = pair["full"], pair["mini"]
    bias = df[f"{full}_bias"].to_numpy()
    varerr = df[f"{full}_varerr"].to_numpy()
    tvar = df[f"{mini}_var"].to_numpy()
    tinst = df[f"{mini}_instab"].to_numpy()
    fine = np.linspace(0, 1, 101)
    curves = {
        "bias_by_var": recall_at(tvar, bias, fine),
        "bias_by_instab": recall_at(tinst, bias, fine),
        "varerr_by_var": recall_at(tvar, varerr, fine),
    }
    aucs = {
        "bias_by_var": W.variance_auc(tvar, bias)["auc"],
        "bias_by_instab": W.variance_auc(tinst, bias)["auc"],
        "varerr_by_var": W.variance_auc(tvar, varerr)["auc"],
    }
    return fine, curves, aucs, int(bias.sum()), int(varerr.sum())


# --------------------------------------------------------------------------- #
# figure                                                                      #
# --------------------------------------------------------------------------- #
def make_fig(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200, sharey=True)
    for ax, (pair, fine, curves, aucs, *_rest) in zip(axes, results):
        x = fine * 100
        ax.plot(x, curves["bias_by_var"] * 100, color="#3b6fb6", lw=2,
                label=f"Cheap trigger: monitor variance (AUC {aucs['bias_by_var']:.2f})")
        ax.plot(x, curves["bias_by_instab"] * 100, color="#3b6fb6", lw=1.6, ls="--",
                label=f"Costlier trigger: monitor instability (AUC {aucs['bias_by_instab']:.2f})")
        ax.plot(x, x, color="#888", lw=1, label="Random (no skill)")
        ax.set_title(pair["name"])
        ax.set_xlabel("Escalation budget (% escalated)")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Recall of the frontier judge's bias errors (%)")
    fig.suptitle("Cheap-monitor escalation recalls only a minority of the frontier judge's bias errors")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig_escalation_recall.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# run + outputs                                                               #
# --------------------------------------------------------------------------- #
def run(long):
    df = build(long)
    out = []
    for pair in PAIRS:
        p1 = part1(df, pair)
        fine, curves, aucs, n_bias, n_var = part2(df, pair)
        out.append((pair, fine, curves, aucs, p1, n_bias, n_var))
    return df, out


def to_rows(out):
    rows = []
    for pair, fine, curves, aucs, p1, n_bias, n_var in out:
        for k, v in p1.items():
            rows.append({"pair": pair["name"], "section": "overlap", "key": k,
                         "budget": np.nan, "value": v})
        for c, a in aucs.items():
            rows.append({"pair": pair["name"], "section": "recall_auc", "key": c,
                         "budget": np.nan, "value": a})
        for c, curve in curves.items():
            for X in BUDGETS:
                rec = float(curve[int(round(X * 100))])
                rows.append({"pair": pair["name"], "section": "recall", "key": c,
                             "budget": X, "value": rec})
    return pd.DataFrame(rows)


def summary_section(out):
    L = ["", "## Analysis C — cheap-monitor + escalation (optional follow-up)", "",
         "Deployment pattern: monitor with a cheap (mini) judge, escalate the top-X% most "
         "uncertain cases to the frontier (full) judge. Tests whether the monitor's "
         "uncertainty flags the frontier judge's blind spots. Difficulty is stratified by "
         "the OTHER provider's full-judge score margin (distinct from both tier judges, to "
         "avoid conditioning on the object of study). This is about the escalation "
         "architecture, not small-model error prediction (on a bias error both tiers are "
         "confidently wrong, so neither variance rises).", ""]
    for pair, fine, curves, aucs, p1, n_bias, n_var in out:
        r20_var = float(curves["bias_by_var"][20])
        r20_inst = float(curves["bias_by_instab"][20])
        r20_varerr = float(curves["varerr_by_var"][20])
        L.append(f"- **{pair['name']}** (difficulty proxy: {pair['proxy']}). "
                 f"*Shared bias:* of the full judge's bias errors (n={n_bias}), the mini judge is "
                 f"also wrong **{100*p1['p_mini_given_full_bias']:.0f}%** of the time vs a "
                 f"difficulty-adjusted baseline of {100*p1['diff_adj_baseline']:.0f}% "
                 f"(excess {100*p1['excess']:+.0f}pp; cross-tier **MH OR {p1['mh_or']:.1f}** "
                 f"[{p1['mh_lo']:.1f}, {p1['mh_hi']:.1f}] — far above the cross-provider 3.1). "
                 f"*Escalation:* the realistic cheap trigger (mini per-response variance) recalls "
                 f"only {100*r20_var:.0f}% of those bias errors at a 20% budget "
                 f"(AUC {aucs['bias_by_var']:.2f}) — about as poorly as it flags the full judge's "
                 f"*variance* errors ({100*r20_varerr:.0f}%, AUC {aucs['varerr_by_var']:.2f}), i.e. "
                 f"it surfaces hard examples indiscriminately, not bias errors specifically. The "
                 f"costlier instability trigger (mini 1−f_same) does better — {100*r20_inst:.0f}% "
                 f"recall, AUC {aucs['bias_by_instab']:.2f} — but still misses about half.")
    L.append("")
    L.append("**Reading:** the escalation architecture *substantially* inherits the blind spot. "
             "Same-provider tiers share bias errors even more strongly than cross-provider judges "
             "(MH OR ~6–8 vs 3.1), and the cheap monitor's realistic uncertainty signal recalls only "
             "~a third of the frontier judge's bias errors at a 20% budget. A costlier "
             "bootstrap-instability trigger roughly halves the miss rate but still leaves the "
             "shared confidently-wrong cases — the most dangerous ones — below threshold. So the "
             "inheritance is substantial, not total: escalation is a weak partial check on bias "
             "errors, not the safety net the architecture assumes. Figure: "
             "`figures/fig_escalation_recall.png`; table: `results/day3c_escalation.csv`.")
    L.append("")
    return "\n".join(L)


def main():
    long = load_long()
    df, out = run(long)
    RESULTS.mkdir(exist_ok=True)
    to_rows(out).to_csv(RESULTS / "day3c_escalation.csv", index=False)
    make_fig(out)

    path = RESULTS / "day3_summary.md"
    existing = path.read_text() if path.exists() else "# Day 3 — robustness\n"
    marker = "## Analysis C"
    if marker in existing:
        existing = existing[: existing.index(marker)].rstrip() + "\n"
    path.write_text(existing + summary_section(out))

    for pair, fine, curves, aucs, p1, n_bias, n_var in out:
        print(f"\n=== {pair['name']} === (proxy: {pair['proxy']})")
        print(f"  P(mini wrong | full bias) = {p1['p_mini_given_full_bias']:.3f}  "
              f"diff-adj baseline = {p1['diff_adj_baseline']:.3f}  excess = {p1['excess']:+.3f}  "
              f"MH OR = {p1['mh_or']:.2f} [{p1['mh_lo']:.2f},{p1['mh_hi']:.2f}]")
        print(f"  recall@20% bias-by-var={curves['bias_by_var'][20]:.3f} "
              f"bias-by-instab={curves['bias_by_instab'][20]:.3f} "
              f"varerr-by-var={curves['varerr_by_var'][20]:.3f}")
        print(f"  AUC bias-by-var={aucs['bias_by_var']:.3f} bias-by-instab={aucs['bias_by_instab']:.3f} "
              f"varerr-by-var={aucs['varerr_by_var']:.3f}  (n_bias={n_bias}, n_varerr={n_var})")
    print("\nWrote day3c_escalation.csv, fig_escalation_recall.png, appended Analysis C to day3_summary.md")


if __name__ == "__main__":
    main()
