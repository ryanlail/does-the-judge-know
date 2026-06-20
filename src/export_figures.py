"""Re-export every post figure at 2x resolution for the writeup.

The analysis modules render at figure dpi 200; this bumps the export to 400 (2x)
via the global rcParam and renders exactly the three figures the post uses:

  fig1_bias_share_by_category  (Analysis A, from results/day1_bias_share.csv)
  fig2_cross_judge_overlap     (per-difficulty overlap, src.post_fig_overlap)
  fig_family_overlap           (same-family vs cross-provider, src.post_fig_family)

Run:  uv run python -m src.export_figures
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["savefig.dpi"] = 400  # 2x the dev default of 200

import pandas as pd

from src import day1_analysis_a as A
from src import post_fig_family as FAM
from src import post_fig_overlap as OV
from src.data_loader import REPO_ROOT

RESULTS = REPO_ROOT / "results"


def main():
    # fig1 re-renders from its result table (deterministic); rcParam makes it 2x
    A.make_fig1(pd.read_csv(RESULTS / "day1_bias_share.csv"))
    # the two overlap figures render from the common frame (self-contained, dpi=400)
    OV.main()
    FAM.main()
    print("Re-exported 3 post figures at 2x (savefig.dpi=400):")
    for f in ("fig1_bias_share_by_category", "fig2_cross_judge_overlap", "fig_family_overlap"):
        print(f"  figures/{f}.png")


if __name__ == "__main__":
    main()
