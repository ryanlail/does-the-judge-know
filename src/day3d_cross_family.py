"""Day 3, Analysis D (optional follow-up) — does CROSS-family escalation perform
better than same-family? (zero new compute)

Analysis C showed same-family tiers share bias errors strongly and the cheap
monitor only weakly flags the frontier judge's bias errors. The natural question:
if you escalate to a DIFFERENT provider's judge instead of the same family, does
the system do better? Analysis B/C measured error *overlap*; this measures the
*system*.

Simulation: a cheap monitor judges everything; the top-X% most uncertain cases
(by the monitor's per-response variance) are escalated and re-judged by a
backstop; the system's verdict is the monitor's (kept) or the backstop's
(escalated). Two monitors, each with a same-family and a cross-family backstop:

    GPT-5.4-mini  → GPT-5.4 (same)   vs  Sonnet 4.6 (cross)
    Haiku 4.5     → Sonnet 4.6 (same) vs  GPT-5.4 (cross)

The Haiku mirror controls the backstop-quality confound: there the same-family
backstop (Sonnet) is the *stronger* judge, so if cross-family (GPT-5.4) still
helps, it is independence, not raw quality.

Headline metrics: system accuracy and Safety accuracy vs escalation budget; and
the repair rate on the monitor's BIAS vs VARIANCE errors (the mechanism).

Outputs:
    results/day3d_escalation_family.csv
    figures/fig_escalation_family.png
    one paragraph appended to results/day3_summary.md

Run:  uv run python -m src.day3d_cross_family   (after src.day3c_escalation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import winners as W
from src.data_loader import REPO_ROOT, SEED, load_long

RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"

BIAS_THR = 0.90
JUDGES = ["gpt-5.4", "gpt-5.4-mini", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
DISP = {"gpt-5.4": "GPT-5.4", "gpt-5.4-mini": "GPT-5.4-mini",
        "claude-sonnet-4-6": "Sonnet 4.6", "claude-haiku-4-5-20251001": "Haiku 4.5"}
SCEN = [
    {"monitor": "gpt-5.4-mini", "same": "gpt-5.4", "cross": "claude-sonnet-4-6"},
    {"monitor": "claude-haiku-4-5-20251001", "same": "claude-sonnet-4-6", "cross": "gpt-5.4"},
]
REPORT_BUDGETS = [0.10, 0.20, 0.30, 0.50, 1.00]


def build(long):
    wf = {j: W.winners_frame(long, j, "base", k=8, stability=True, B=1000, seed=SEED)
          .set_index(["category", "example_id"]) for j in JUDGES}
    idx = sorted(set.intersection(*[set(wf[j].index) for j in JUDGES]))
    df = pd.DataFrame(index=pd.MultiIndex.from_tuples(idx, names=["category", "example_id"]))
    for j in JUDGES:
        w = wf[j].reindex(idx)
        df[f"{j}__correct"] = w["is_correct"].astype(int).to_numpy()
        df[f"{j}__bias"] = (w["is_error"] & (w["f_same"] >= BIAS_THR)).astype(int).to_numpy()
        df[f"{j}__varerr"] = (w["is_error"] & (w["f_same"] < BIAS_THR)).astype(int).to_numpy()
        df[f"{j}__var"] = w["mean_response_std"].to_numpy()
    df = df.reset_index()
    df["is_safety"] = (df["category"] == "Safety").astype(int)
    return df


def repair(df, monitor, backstop, etype, rng):
    mask = df[f"{monitor}__{etype}"].to_numpy().astype(bool)
    cb = df[f"{backstop}__correct"].to_numpy()
    n = int(mask.sum())
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    vals = cb[mask]
    draws = vals[rng.integers(0, n, size=(2000, n))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi), n


def system_curve(df, monitor, backstop):
    T = df[f"{monitor}__var"].to_numpy()
    cm = df[f"{monitor}__correct"].to_numpy()
    cb = df[f"{backstop}__correct"].to_numpy()
    saf = df["is_safety"].to_numpy()
    order = np.argsort(-T)  # most uncertain first
    cm, cb, saf = cm[order], cb[order], saf[order]
    N, n_saf = len(order), int(saf.sum())
    cum_cb = np.concatenate([[0], np.cumsum(cb)])
    cum_cm = np.concatenate([[0], np.cumsum(cm)])
    cum_scb = np.concatenate([[0], np.cumsum(cb * saf)])
    cum_scm = np.concatenate([[0], np.cumsum(cm * saf)])
    tot_cm, tot_scm = cum_cm[-1], cum_scm[-1]
    budgets = np.linspace(0, 1, 101)
    acc, sacc = [], []
    for X in budgets:
        k = int(round(X * N))
        acc.append((cum_cb[k] + (tot_cm - cum_cm[k])) / N)
        sacc.append((cum_scb[k] + (tot_scm - cum_scm[k])) / n_saf)
    return budgets, np.array(acc), np.array(sacc)


def run(long):
    df = build(long)
    rng = np.random.default_rng(SEED)
    res = []
    for sc in SCEN:
        mon = sc["monitor"]
        entry = {"monitor": mon, "same": sc["same"], "cross": sc["cross"], "curves": {}, "repair": {}}
        for fam in ("same", "cross"):
            bs = sc[fam]
            b, acc, sacc = system_curve(df, mon, bs)
            entry["curves"][fam] = {"budgets": b, "acc": acc, "sacc": sacc}
            entry[f"{fam}_backstop_acc"] = float(df[f"{bs}__correct"].mean())
            for et in ("bias", "varerr"):
                entry["repair"][(fam, et)] = repair(df, mon, bs, et, rng)
        entry["monitor_acc"] = float(df[f"{mon}__correct"].mean())
        entry["monitor_safety_acc"] = float(df[df.is_safety == 1][f"{mon}__correct"].mean())
        res.append(entry)
    return df, res


def make_fig(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=200, sharex=True)
    for col, e in enumerate(res):
        for row, (key, lab) in enumerate((("acc", "overall"), ("sacc", "Safety"))):
            ax = axes[row][col]
            for fam, color in (("same", "#c1543f"), ("cross", "#3b6fb6")):
                c = e["curves"][fam]
                bs_disp = DISP[e[fam]]
                ax.plot(c["budgets"] * 100, c[key] * 100, color=color, lw=2,
                        label=f"{'same' if fam=='same' else 'cross'}-family → {bs_disp}")
            ax.set_title(f"{DISP[e['monitor']]} monitor — {lab} accuracy")
            ax.grid(alpha=0.25)
            if row == 1:
                ax.set_xlabel("escalation budget (% escalated)")
            if col == 0:
                ax.set_ylabel(f"{lab} system accuracy (%)")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Cross-family escalation is not a robust win: backstop-dependent, and it hurts Safety")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig_escalation_family.png")
    plt.close(fig)


def to_rows(res):
    rows = []
    for e in res:
        mon = DISP[e["monitor"]]
        for fam in ("same", "cross"):
            bsd = DISP[e[fam]]
            for et in ("bias", "varerr"):
                r, lo, hi, n = e["repair"][(fam, et)]
                rows.append({"monitor": mon, "family": fam, "backstop": bsd,
                             "metric": f"repair_{et}", "budget": np.nan,
                             "value": r, "ci_lo": lo, "ci_hi": hi, "n": n})
            c = e["curves"][fam]
            for which, key in (("overall_acc", "acc"), ("safety_acc", "sacc")):
                for X in REPORT_BUDGETS:
                    rows.append({"monitor": mon, "family": fam, "backstop": bsd,
                                 "metric": which, "budget": X,
                                 "value": float(c[key][int(round(X * 100))]),
                                 "ci_lo": np.nan, "ci_hi": np.nan, "n": np.nan})
    return pd.DataFrame(rows)


def _pp(x):
    return f"{100*x:.1f}%"


def summary_section(res):
    L = ["", "## Analysis D — cross-family vs same-family escalation (optional follow-up)", "",
         "Simulated monitor→escalate system: a cheap monitor judges everything; the top-X% by "
         "the monitor's per-response variance are escalated to a backstop; final verdict = "
         "monitor (kept) or backstop (escalated). Same-family vs cross-family backstop, two "
         "monitors. The Haiku monitor is the mirror — there the same-family backstop (Sonnet 4.6) "
         "is the *stronger* judge, so a genuine cross-family/independence effect should survive it.", ""]
    for e in res:
        mon = DISP[e["monitor"]]
        rb_s, rb_c = e["repair"][("same", "bias")], e["repair"][("cross", "bias")]
        a20_s, a20_c = float(e["curves"]["same"]["acc"][20]), float(e["curves"]["cross"]["acc"][20])
        L.append(f"- **{mon} monitor** (same→{DISP[e['same']]}, cross→{DISP[e['cross']]}; backstop "
                 f"standalone acc {_pp(e['same_backstop_acc'])} vs {_pp(e['cross_backstop_acc'])}; "
                 f"monitor {_pp(e['monitor_acc'])} overall / {_pp(e['monitor_safety_acc'])} Safety). "
                 f"Repair of the monitor's bias errors: same {_pp(rb_s[0])} "
                 f"[{_pp(rb_s[1])},{_pp(rb_s[2])}] vs cross {_pp(rb_c[0])} "
                 f"[{_pp(rb_c[1])},{_pp(rb_c[2])}] (n={rb_s[3]}). Overall system accuracy @20% "
                 f"budget: same {_pp(a20_s)} vs cross {_pp(a20_c)}.")
    L.append("")
    L.append("**Reading — cross-family escalation is NOT a robust win.** The overlap numbers "
             "(cross-provider OR 3.1 < same-family 8.2) predicted it should help; simulating the "
             "actual system says otherwise, in three ways:")
    L.append("- *It depends entirely on the backstop.* For the GPT-5.4-mini monitor, cross-family "
             "(Sonnet 4.6) repairs its bias errors far better than same-family (53% vs 32%, CIs "
             "disjoint) and lifts overall accuracy. But in the Haiku mirror the effect *reverses* — "
             "cross-family (GPT-5.4) repairs slightly worse (30% vs 36%) and lowers accuracy. So the "
             "GPT-mini gain is that Sonnet specifically is a strong, independent repairer — not a "
             "general 'cross-family beats same-family' law.")
    L.append("- *Escalation only helps when the backstop out-scores the monitor.* Haiku (85% "
             "overall, 96% Safety) already beats both frontier backstops, so escalating its cases "
             "*hurts* — the architecture's premise fails before family even matters.")
    L.append("- *On Safety (the headline slice) escalation is flat-to-harmful for both families:* "
             "the cheap monitors are already very accurate on Safety (GPT-5.4-mini 93%, Haiku 96%), "
             "more than the frontier backstops, so routing Safety cases to a 'better' judge does not "
             "help (small-sample; ~30–40 Safety errors).")
    L.append("")
    L.append("**Bottom line:** we cannot conclude cross-family escalation performs better — it is "
             "backstop-specific, the realized gains are small (≤~1.5pp, trigger-limited), and on "
             "Safety the premise inverts. The deeper reason is unchanged from Analyses A–C: the "
             "monitor's confident-bias errors rarely raise its variance, so they are rarely "
             "escalated — no choice of backstop sees them. Figure: "
             "`figures/fig_escalation_family.png`; table: `results/day3d_escalation_family.csv`.")
    L.append("")
    return "\n".join(L)


def main():
    long = load_long()
    df, res = run(long)
    RESULTS.mkdir(exist_ok=True)
    to_rows(res).to_csv(RESULTS / "day3d_escalation_family.csv", index=False)
    make_fig(res)

    path = RESULTS / "day3_summary.md"
    existing = path.read_text() if path.exists() else "# Day 3 — robustness\n"
    marker = "## Analysis D"
    if marker in existing:
        existing = existing[: existing.index(marker)].rstrip() + "\n"
    path.write_text(existing + summary_section(res))

    for e in res:
        print(f"\n=== {DISP[e['monitor']]} monitor ===  same→{DISP[e['same']]} cross→{DISP[e['cross']]}")
        print(f"  monitor acc {e['monitor_acc']:.3f} (Safety {e['monitor_safety_acc']:.3f}); "
              f"backstop standalone same {e['same_backstop_acc']:.3f} cross {e['cross_backstop_acc']:.3f}")
        for et in ("bias", "varerr"):
            rs, rc = e["repair"][("same", et)], e["repair"][("cross", et)]
            print(f"  repair {et}: same {rs[0]:.3f}[{rs[1]:.3f},{rs[2]:.3f}] vs "
                  f"cross {rc[0]:.3f}[{rc[1]:.3f},{rc[2]:.3f}]  (n={rs[3]})")
        for X in (0.10, 0.20, 0.50):
            k = int(round(X * 100))
            print(f"  @{int(X*100):>2}% budget: overall same {e['curves']['same']['acc'][k]:.3f} "
                  f"cross {e['curves']['cross']['acc'][k]:.3f} | "
                  f"Safety same {e['curves']['same']['sacc'][k]:.3f} cross {e['curves']['cross']['sacc'][k]:.3f}")
    print("\nWrote day3d_escalation_family.csv, fig_escalation_family.png, appended Analysis D to day3_summary.md")


if __name__ == "__main__":
    main()
