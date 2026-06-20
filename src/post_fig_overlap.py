"""Post figure: cross-judge overlap broken down by difficulty (main text).

For each difficulty band (how many of the three weaker judges also err on an
example), show how often BOTH frontier judges are wrong: observed vs. what you
would expect if their errors were independent. The gap is shared bias, shown at
every difficulty level. Builds the common frame the same way as Analysis B.

Run:  uv run python -m src.post_fig_overlap
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import data_loader as dl
from src.data_loader import REPO_ROOT
from src.day1_analysis_b import build_common

FIGURES = REPO_ROOT / "figures"


def main():
    long = dl.load_long()
    df, _, _ = build_common(long)
    strata = [0, 1, 2, 3]
    obs, exp, ns = [], [], []
    for s in strata:
        sub = df[df["s"] == s]
        pg, pc = float(sub["E_g"].mean()), float(sub["E_c"].mean())
        obs.append(100 * float((sub["E_g"] & sub["E_c"]).mean()))
        exp.append(100 * pg * pc)
        ns.append(len(sub))

    x = np.arange(len(strata))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=200)
    ax.bar(x - w / 2, exp, w, label="Expected if their errors were independent", color="#bcc8da")
    ax.bar(x + w / 2, obs, w, label="Observed (both judges wrong)", color="#3b6fb6")
    for xi in x:
        ax.text(xi - w / 2, exp[xi] + 1.0, f"{exp[xi]:.0f}", ha="center", fontsize=8.5, color="#5a6b86")
        ax.text(xi + w / 2, obs[xi] + 1.0, f"{obs[xi]:.0f}", ha="center", fontsize=8.5, color="#1f3a5f")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(strata, ns)])
    ax.set_xlabel("Difficulty: how many of the 3 weaker judges also get it wrong  (0 = easiest, 3 = hardest)")
    ax.set_ylabel("Both judges wrong (%)")
    ax.set_ylim(0, 78)
    ax.set_title("Difficulty drives the overlap, but the judges still fail together beyond it")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig2_cross_judge_overlap.png", dpi=400)
    plt.close(fig)
    for s, e, o, n in zip(strata, exp, obs, ns):
        print(f"s={s} n={n}: expected {e:.1f}%  observed {o:.1f}%  excess {o - e:+.1f}pp")
    print("wrote figures/fig2_cross_judge_overlap.png")


if __name__ == "__main__":
    main()
