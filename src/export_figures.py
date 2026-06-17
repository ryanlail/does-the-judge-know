"""Re-export every post figure at 2x resolution for the writeup.

The analysis modules render at savefig dpi 200 (figure dpi); this bumps the
export to 400 (2x) via the global rcParam, then re-renders all four figures from
the existing result CSVs (fig1, fig2) or by recomputing the escalation curves
(C, D, which need the full 101-point recall curves not stored in the CSVs).

Run:  uv run python -m src.export_figures
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["savefig.dpi"] = 400  # 2x the dev default of 200

import pandas as pd

from src import day1_analysis_a as A
from src import day1_analysis_b as B
from src import day3c_escalation as C
from src import day3d_cross_family as D
from src.data_loader import REPO_ROOT, load_long

RESULTS = REPO_ROOT / "results"


def main():
    # fig1 / fig2 re-render directly from their result tables (deterministic)
    A.make_fig1(pd.read_csv(RESULTS / "day1_bias_share.csv"))
    B.make_fig2(pd.read_csv(RESULTS / "day1_overlap.csv"))
    # C / D need the full curves -> recompute (seed 42, identical numbers)
    long = load_long()
    _, out_c = C.run(long)
    C.make_fig(out_c)
    _, res_d = D.run(long)
    D.make_fig(res_d)
    print("Re-exported 4 figures at 2x (savefig.dpi=400):")
    for f in ("fig1_bias_share_by_category", "fig2_overlap_by_category",
              "fig_escalation_recall", "fig_escalation_family"):
        print(f"  figures/{f}.png")


if __name__ == "__main__":
    main()
