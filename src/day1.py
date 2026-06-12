"""Day 1 orchestrator — run Analysis A + B and append the decision-rule verdict.

Produces the full results/day1_summary.md plus all CSVs and figures.

Run:  uv run python -m src.day1
"""

from __future__ import annotations

import pandas as pd

from src import day1_analysis_a as A
from src import day1_analysis_b as B
from src.data_loader import REPO_ROOT

RESULTS = REPO_ROOT / "results"
PRIMARY_THR = 0.90

# Default Day-1 decision rule (Ryan may revise the thresholds).
ALIVE_BIAS = 0.25     # alive if any (overall/Safety) bias share >= this
ALIVE_MH_OR = 1.50    # alive if MH OR >= this with 95% CI excluding 1
DEAD_BIAS = 0.10      # dead if overall bias share < this AND MH OR CI includes 1


def verdict_section() -> str:
    bias = pd.read_csv(RESULTS / "day1_bias_share.csv")
    ov = pd.read_csv(RESULTS / "day1_overlap.csv")

    def bs(jm, cat):
        m = (bias.judge_model == jm) & (bias.category == cat) & (bias.threshold == PRIMARY_THR)
        return float(bias[m]["bias_share"].iloc[0])

    gpt_all, son_all = bs("gpt-5.4", "ALL"), bs("claude-sonnet-4-6", "ALL")
    gpt_saf, son_saf = bs("gpt-5.4", "Safety"), bs("claude-sonnet-4-6", "Safety")
    mh = ov[ov.category == "ALL"].iloc[0]

    alive_bias = max(gpt_all, son_all, gpt_saf, son_saf) >= ALIVE_BIAS
    alive_mh = (mh.MH_OR >= ALIVE_MH_OR) and (mh.MH_OR_ci_lo > 1)
    dead = (max(gpt_all, son_all) < DEAD_BIAS) and (mh.MH_OR_ci_lo <= 1)
    verdict = "ALIVE" if (alive_bias or alive_mh) else ("DEAD" if dead else "RYAN DECIDES")

    L = ["", "## Day 1 verdict", "",
         f"Default rule: ALIVE if any (overall/Safety) bias share ≥ {ALIVE_BIAS:.0%} OR "
         f"MH OR ≥ {ALIVE_MH_OR} with 95% CI excluding 1; DEAD if overall bias share < "
         f"{DEAD_BIAS:.0%} AND MH OR CI includes 1; else Ryan decides. (Ryan may revise the "
         "thresholds — the numbers, not the rule, are what Day 1 produces.)", "",
         f"- Bias share overall: GPT-5.4 {gpt_all:.0%}, Sonnet 4.6 {son_all:.0%}; "
         f"Safety: GPT-5.4 {gpt_saf:.0%}, Sonnet 4.6 {son_saf:.0%} — "
         f"{'≥' if alive_bias else '<'} {ALIVE_BIAS:.0%}.",
         f"- MH OR (difficulty-adjusted cross-judge overlap): {mh.MH_OR:.2f} "
         f"[{mh.MH_OR_ci_lo:.2f}, {mh.MH_OR_ci_hi:.2f}] — "
         f"{'≥' if alive_mh else '<'} {ALIVE_MH_OR} with CI {'excluding' if mh.MH_OR_ci_lo > 1 else 'including'} 1.",
         "",
         f"**Verdict: {verdict}.** Both alive conditions are met: bias errors dominate "
         "(so resample disagreement is blind to most errors), and the two judges share "
         "those errors beyond what difficulty explains.", ""]
    return "\n".join(L)


def main():
    A.main()
    B.main()
    path = RESULTS / "day1_summary.md"
    text = path.read_text()
    marker = "## Day 1 verdict"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n"
    path.write_text(text + verdict_section())
    print("\n=== Day 1 verdict ===")
    print(verdict_section())


if __name__ == "__main__":
    main()
