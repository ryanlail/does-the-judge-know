# does the judge know?

Can LLM-judge resample disagreement detect judge errors on RewardBench 2 (RB2)?

This repo reuses the paired k=8 judge-score collections from
[`composo-ai/llm-judge-criteria-ensembling`](https://github.com/composo-ai/llm-judge-criteria-ensembling)
(vendored read-only under `data/raw/`, pinned to commit `e4049a5`) and asks a
sharper question: when a judge is wrong, does resample disagreement know?

## Layout

- `data/raw/` — vendored raw score collections (read-only). See `data/PROVENANCE.txt`.
- `src/` — one module per analysis step.
- `results/` — CSVs and markdown summaries.
- `figures/` — PNGs (plus a CSV behind every figure).

## Setup

```sh
uv sync                       # create the env
bash scripts/fetch_data.sh    # (re)vendor the data snapshot into data/raw/
```

## Step 0 — data audit

```sh
uv run python -m src.step0_audit
```

Writes `results/step0_inventory.csv` and `results/step0_summary.md`, and prints
the acceptance-gate verdict (per-sample k=8 scores present for both main judges
on >= 1,600 common examples).
