"""Post figure: difficulty-adjusted error overlap for three judge pairs.

Same machinery as fig2 (observed joint-error rate minus the rate expected if the
two judges erred independently within each difficulty band), computed for one
cross-provider pair and the two same-family pairs. Each pair's difficulty is
defined by judges OUTSIDE that pair's family, so it cannot absorb the shared bias
being measured. Same-family pairs overlap far more than the cross-provider pair.

Run:  uv run python -m src.post_fig_family
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import data_loader as dl
from src.data_loader import REPO_ROOT
from src.day1_analysis_b import _excess, build_common

FIGURES = REPO_ROOT / "figures"


def main():
    long = dl.load_long()
    df, _, _ = build_common(long)
    Eg, Ec, Em, Eh, En = (df[c].to_numpy() for c in ["E_g", "E_c", "E_mini", "E_haiku", "E_nano"])

    # (label, judge A errors, judge B errors, difficulty = #external judges erring, n_strata, same-family)
    pairs = [
        ("GPT-5.4 ∩ Sonnet 4.6\n(two providers)", Eg, Ec, Em + Eh + En, 4, False),
        ("GPT-5.4 ∩ GPT-5.4-mini\n(one family)", Eg, Em, Ec + Eh, 3, True),
        ("Sonnet 4.6 ∩ Haiku 4.5\n(one family)", Ec, Eh, Eg + Em + En, 4, True),
    ]
    labels, exc, ratio, same = [], [], [], []
    for name, a, b, s, nb, fam in pairs:
        joint, expected, e = _excess(a, b, s, nb)
        labels.append(name)
        exc.append(100 * e)
        ratio.append(joint / expected)
        same.append(fam)

    x = np.arange(len(pairs))
    colors = ["#3b6fb6" if f else "#9aa7bd" for f in same]
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=200)
    ax.bar(x, exc, 0.6, color=colors)
    for xi, e, r in zip(x, exc, ratio):
        ax.text(xi, e + 0.12, f"+{e:.1f} pp\n({r:.1f}x more than\ndifficulty predicts)",
                ha="center", va="bottom", fontsize=9.5, color="#1f3a5f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Shared errors beyond what difficulty predicts (percentage points)")
    ax.set_ylim(0, max(exc) + 2.2)
    ax.set_title("Two judges from one family fail together far more than two providers do")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig_family_overlap.png", dpi=400)
    plt.close(fig)
    for l, e, r in zip(labels, exc, ratio):
        print(l.replace("\n", " "), f"  +{e:.1f}pp  {r:.2f}x")
    print("wrote figures/fig_family_overlap.png")


if __name__ == "__main__":
    main()
