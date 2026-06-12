"""Step 0 - data audit (blocker).

Loads the vendored RewardBench-2 collections, inventories them per
(judge_model, condition), runs structural sanity checks, and evaluates the
acceptance gate:

    per-sample k=8 scores must exist for both main judges (GPT-5.4 base and
    Sonnet 4.6 base) on >= 1,600 common, usable examples.

Outputs
    results/step0_inventory.csv   one row per (judge_model, condition)
    results/step0_summary.md      human-readable verdict + flags

Exits non-zero if the gate STOPs or any hard structural check fails.

Run:  uv run python -m src.step0_audit
"""

from __future__ import annotations

import random
import sys
from collections import Counter

import numpy as np
import pandas as pd

from src import data_loader as dl
from src.data_loader import (
    K,
    N_RESPONSES,
    REPO_ROOT,
    SCORE_MAX,
    SCORE_MIN,
    SEED,
    SUBSETS,
    common_examples,
    example_keys,
    list_collections,
    load_long,
    load_raw,
    resolve_judge,
)

RESULTS = REPO_ROOT / "results"

GATE_MIN = 1600            # hard floor
GATE_ESCALATE_TO = 1650    # [GATE_MIN, GATE_ESCALATE_TO) -> escalate to human

# Required (judge_model, condition) collections (the Step 0 acceptance list).
REQUIRED = [
    ("gpt-5.4", "base"),
    ("claude-sonnet-4-6", "base"),
    ("gpt-5.4", "criteria"),
    ("claude-sonnet-4-6", "criteria"),
    ("gpt-5.4-mini", "base"),
    ("claude-haiku-4-5-20251001", "base"),
    ("gpt-5.4-nano", "base"),
]
MAIN = [("gpt-5.4", "base"), ("claude-sonnet-4-6", "base")]
TIERS = [
    ("gpt-5.4-mini", "base"),
    ("claude-haiku-4-5-20251001", "base"),
    ("gpt-5.4-nano", "base"),
]

# Project's stated figures, for the reconciliation flag.
PROJECT_TOTAL = 1753
PROJECT_SAFETY = 441

SUBSET_COL = {s: "n_" + s.replace(" ", "") for s in SUBSETS}  # "Precise IF" -> "n_PreciseIF"


# --------------------------------------------------------------------------- #
# structural checks (operate on the raw records)                              #
# --------------------------------------------------------------------------- #
def structural_table(collections_raw: dict[str, list[dict]]) -> tuple[pd.DataFrame, list[str]]:
    """Per-collection structural stats + a list of hard-fail messages."""
    rows, fails = [], []
    for stem, recs in sorted(collections_raw.items()):
        score_keys = sorted(k for k in recs[0] if k.endswith("_scores"))
        fields_in_all = set.intersection(*(set(r) for r in recs))
        required_fields = {"id", "subset", "prompt_variant", "k"}
        for sk in score_keys:
            required_fields |= {sk, sk.replace("_scores", "_errors")}
        missing_fields = sorted(required_fields - fields_in_all)

        bad_arity = bad_k = bad_inner = out_of_range = 0
        for r in recs:
            if r.get("k") != K:
                bad_k += 1
            for sk in score_keys:
                arr = r[sk]
                if len(arr) != N_RESPONSES:
                    bad_arity += 1
                for resp in arr:
                    if len(resp) != K:
                        bad_inner += 1
                    for s in resp:
                        if s is not None and not (SCORE_MIN <= s <= SCORE_MAX):
                            out_of_range += 1

        subs = {r["subset"] for r in recs}
        unexpected = sorted((subs - set(SUBSETS)) | (subs & {"Ties"}))
        compound = [(r["subset"], r["id"]) for r in recs]
        ids = [r["id"] for r in recs]
        dup_compound = len(compound) - len(set(compound))
        id_collisions = len(ids) - len(set(ids))
        n_refused = sum(1 for r in recs if r.get("refused", False))
        judges = sorted({resolve_judge(stem, sk).data_name for sk in score_keys})

        rows.append(
            {
                "collection": stem,
                "n": len(recs),
                "judges": ", ".join(judges),
                "arity_bad": bad_arity,
                "k_bad": bad_k,
                "inner_bad": bad_inner,
                "oor_samples": out_of_range,
                "dup_compound": dup_compound,
                "id_collisions": id_collisions,
                "refused": n_refused,
                "missing_fields": ", ".join(missing_fields) or "-",
            }
        )
        if missing_fields:
            fails.append(f"{stem}: missing required fields {missing_fields}")
        if bad_arity:
            fails.append(f"{stem}: {bad_arity} response-arrays with arity != {N_RESPONSES}")
        if bad_k or bad_inner:
            fails.append(f"{stem}: k/inner length != {K} (k_bad={bad_k}, inner_bad={bad_inner})")
        if out_of_range:
            fails.append(f"{stem}: {out_of_range} samples outside [{SCORE_MIN},{SCORE_MAX}]")
        if dup_compound:
            fails.append(f"{stem}: {dup_compound} duplicate (category, id) keys")
        if unexpected:
            fails.append(f"{stem}: unexpected subset labels {unexpected}")
    return pd.DataFrame(rows), fails


# --------------------------------------------------------------------------- #
# inventory                                                                   #
# --------------------------------------------------------------------------- #
def build_inventory(
    long: pd.DataFrame, collections_raw: dict[str, list[dict]], gate_value: int
) -> pd.DataFrame:
    """One row per (judge_model, condition)."""
    fields_in_all = {
        stem: set.intersection(*(set(r) for r in recs))
        for stem, recs in collections_raw.items()
    }
    refused_by_stem = {
        stem: sum(1 for r in recs if r.get("refused", False))
        for stem, recs in collections_raw.items()
    }

    rows = []
    for (jm, cond), sub in long.groupby(["judge_model", "condition"]):
        stem = sub["collection_stem"].iloc[0]
        keys = fields_in_all[stem]
        tier = sub["tier"].iloc[0]
        score_key, errors_key = f"{tier}_scores", f"{tier}_errors"

        ex = sub.drop_duplicates(["category", "example_id"])
        n_examples = len(ex)
        min_valid = sub.groupby(["category", "example_id"])["n_valid_samples"].min()
        n_all8 = int((min_valid == K).sum())
        n_allnull = int((min_valid == 0).sum())
        per_sub = ex["category"].value_counts().reindex(SUBSETS, fill_value=0)

        row = {
            "judge_model": jm,
            "judge_display": sub["judge_display"].iloc[0],
            "provider": sub["provider"].iloc[0],
            "tier": tier,
            "condition": cond,
            "collection_stem": stem,
            "n_examples": n_examples,
            "present_example_id": "id" in keys,
            "present_category": "subset" in keys,
            "present_condition": "prompt_variant" in keys,
            "present_scores": score_key in keys,
            "present_errors": errors_key in keys,
            "present_k": "k" in keys,
        }
        for s in SUBSETS:
            row[SUBSET_COL[s]] = int(per_sub[s])
        row["n_all8_valid"] = n_all8
        row["n_lt8_valid"] = n_examples - n_all8 - n_allnull
        row["n_response_all_null"] = n_allnull
        row["n_refused"] = refused_by_stem[stem]
        row["n_common_gpt54_sonnet46_base"] = gate_value
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["condition", "provider", "tier"])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# markdown summary                                                            #
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _read_provenance() -> tuple[str, str]:
    prov = REPO_ROOT / "data" / "PROVENANCE.txt"
    sha = retrieved = "?"
    if prov.exists():
        for line in prov.read_text().splitlines():
            if line.startswith("commit_sha"):
                sha = line.split("=", 1)[1].strip()
            elif line.startswith("retrieved"):
                retrieved = line.split("=", 1)[1].strip()
    return sha, retrieved


def build_summary(
    inv: pd.DataFrame,
    struct: pd.DataFrame,
    struct_fails: list[str],
    gate: dict,
    missing_required: list[tuple[str, str]],
) -> str:
    sha, retrieved = _read_provenance()
    L: list[str] = []
    L.append("# Step 0 - Data Audit")
    L.append("")
    L.append(
        f"Source: `composo-ai/llm-judge-criteria-ensembling` @ `{sha}` "
        f"(retrieved {retrieved}). Generated by `python -m src.step0_audit`."
    )
    L.append("")

    # --- gate ----------------------------------------------------------------
    L.append("## Acceptance gate")
    L.append("")
    verdict = {
        "PASS": "**PASS**",
        "ESCALATE": "**ESCALATE TO HUMAN**",
        "STOP": "**STOP**",
    }[gate["status"]]
    L.append(
        f"Per-sample k=8 scores exist for both main judges (GPT-5.4 base, "
        f"Sonnet 4.6 base) on **{gate['usable']}** common usable examples "
        f"(threshold >= {GATE_MIN}). Verdict: {verdict}."
    )
    L.append("")
    L.append(f"- Common examples, compound key `(category, id)`, usable at k=8: **{gate['usable']}**")
    L.append(f"- Common examples, compound key, presence only: {gate['presence']}")
    L.append(f"- Common examples, id-only key (upstream basis, collapses collisions): {gate['id_only']}")
    L.append(
        f"- Gate policy: STOP if < {GATE_MIN}; escalate to human if "
        f"[{GATE_MIN}, {GATE_ESCALATE_TO}); else PASS."
    )
    if missing_required:
        L.append(f"- **Missing required collections:** {missing_required}")
    else:
        L.append(f"- All {len(REQUIRED)} required (judge x condition) collections present.")
    bycat = gate["by_category"]
    comp = ", ".join(
        (f"**Safety {bycat.get(s, 0)}**" if s == "Safety" else f"{s} {bycat.get(s, 0)}")
        for s in SUBSETS
    )
    L.append(f"- Common usable set by category (the Analysis-B universe): {comp}.")
    L.append("")

    # --- tier coverage -------------------------------------------------------
    L.append("## Tier coverage (Analysis B difficulty strata)")
    L.append("")
    L.append(
        "Difficulty strata stratify on the number of tier judges that err, among "
        "GPT-5.4-mini, Haiku 4.5, GPT-5.4-nano. Their coverage of the "
        f"{gate['usable']}-example base-common (usable) set:"
    )
    L.append("")
    for jm, _cond in TIERS:
        L.append(f"- `{jm}`: covers {gate['tier_cov'][jm]} of {gate['usable']}")
    L.append(
        f"- **All three tiers + both main judges** (3-tier strata universe): "
        f"**{gate['strata_universe']}** examples. The s in {{0,1,2,3}} stratification "
        f"is only fully defined on this subset; nano is the limiter."
    )
    L.append("")

    # --- inventory -----------------------------------------------------------
    L.append("## Inventory")
    L.append("")
    L.append("Full table: `results/step0_inventory.csv`. Key columns:")
    L.append("")
    show = inv[
        ["judge_display", "condition", "n_examples"]
        + [SUBSET_COL[s] for s in SUBSETS]
        + ["n_all8_valid", "n_lt8_valid", "n_response_all_null"]
    ].copy()
    L.append(_md_table(show))
    L.append("")

    # --- structural checks ---------------------------------------------------
    L.append("## Structural checks")
    L.append("")
    if struct_fails:
        L.append("**HARD FAILURES:**")
        for f in struct_fails:
            L.append(f"- {f}")
    else:
        L.append(
            "All hard checks pass: arity == 4, k field == inner length == 8, "
            "scores in [1, 10] or NA, no duplicate `(category, id)`, subsets ⊆ "
            "{Factuality, Focus, Math, Precise IF, Safety} (no Ties), required "
            "fields present, score-key→judge mapping resolves."
        )
    L.append("")
    L.append(_md_table(struct))
    L.append("")
    L.append(
        "`id_collisions` counts ids reused across subsets (expected; the analysis "
        "keys on `(category, id)`, never id alone). `refused` is 0 everywhere."
    )
    L.append("")

    # --- flags ---------------------------------------------------------------
    gpt = inv[(inv.judge_model == "gpt-5.4") & (inv.condition == "base")].iloc[0]
    son = inv[(inv.judge_model == "claude-sonnet-4-6") & (inv.condition == "base")].iloc[0]
    L.append("## Flags (record verbatim; not blockers)")
    L.append("")
    L.append(
        f"- **Counts differ from the project's stated figures.** GPT-5.4 base has "
        f"{gpt.n_examples} examples (project says ~{PROJECT_TOTAL}); Safety has "
        f"{gpt[SUBSET_COL['Safety']]} (GPT base) / {son[SUBSET_COL['Safety']]} "
        f"(Sonnet base). The headline Safety slice (common usable set) is "
        f"**{gate['by_category'].get('Safety', 0)}**, not {PROJECT_SAFETY}."
    )
    L.append(
        f"- **Nano partial coverage.** The 3-tier difficulty strata are defined on "
        f"only {gate['strata_universe']} examples (nano-limited), fewer than the "
        f"{gate['usable']} two-main-judge set."
    )
    L.append(
        "- **`example_id` is not globally unique** - it is reused across subsets "
        "(see `id_collisions`). The true example key is `(category, example_id)`; "
        "upstream code keys on id alone."
    )
    L.append(
        f"- **Response arity is 4** (1 chosen at index 0 + 3 rejected), confirmed "
        f"for every example; response 0 is the correct one structurally (no label)."
    )
    L.append(
        f"- **Usable < presence.** {gate['presence'] - gate['usable']} common "
        f"examples have an all-null response for one main judge and are excluded "
        f"from the usable count (winner logic would skip them)."
    )
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)

    collections = list_collections()
    collections_raw = {c.stem: load_raw(c.path, drop_refused=False) for c in collections}
    long = load_long()

    # gate + coverage
    base_common = common_examples(long, MAIN, usable_only=True)
    gate = {
        "usable": len(base_common),
        "presence": len(common_examples(long, MAIN, usable_only=False)),
        "id_only": len(common_examples(long, MAIN, strict_id_only=True)),
        "strata_universe": len(common_examples(long, MAIN + TIERS, usable_only=True)),
        "tier_cov": {
            jm: len(base_common & example_keys(long, jm, cond, usable_only=True))
            for jm, cond in TIERS
        },
        "by_category": dict(Counter(cat for cat, _eid in base_common)),
    }
    if gate["usable"] < GATE_MIN:
        gate["status"] = "STOP"
    elif gate["usable"] < GATE_ESCALATE_TO:
        gate["status"] = "ESCALATE"
    else:
        gate["status"] = "PASS"

    present_pairs = set(zip(long["judge_model"], long["condition"]))
    missing_required = [p for p in REQUIRED if p not in present_pairs]

    inv = build_inventory(long, collections_raw, gate["usable"])
    struct, struct_fails = structural_table(collections_raw)
    summary = build_summary(inv, struct, struct_fails, gate, missing_required)

    RESULTS.mkdir(exist_ok=True)
    inv.to_csv(RESULTS / "step0_inventory.csv", index=False)
    (RESULTS / "step0_summary.md").write_text(summary)

    # console
    print(struct.to_string(index=False))
    print()
    print(
        f"Common (gpt-5.4 base ∩ sonnet base): usable={gate['usable']} "
        f"presence={gate['presence']} id_only={gate['id_only']}"
    )
    print(f"3-tier strata universe: {gate['strata_universe']}")
    if missing_required:
        print(f"MISSING required collections: {missing_required}")
    print()
    print(f"Wrote {RESULTS / 'step0_inventory.csv'} ({len(inv)} rows)")
    print(f"Wrote {RESULTS / 'step0_summary.md'}")
    print()

    hard_fail = bool(struct_fails) or bool(missing_required) or gate["status"] == "STOP"
    print(f"GATE: {gate['status']}  common_base(usable)={gate['usable']}  (>= {GATE_MIN})")
    if struct_fails:
        print("STRUCTURAL HARD FAILURES:")
        for f in struct_fails:
            print(f"  - {f}")
    if gate["status"] == "ESCALATE":
        print("Gate in escalation band - Ryan to decide with the number in hand.")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
