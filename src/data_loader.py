"""Canonical loader / normaliser for the vendored RewardBench-2 judge-score data.

The raw JSONL under ``data/raw/`` stores one record per example, each holding
nested per-response arrays of k=8 sample scores. Two facts make the raw schema
awkward and are handled here once, for every later step:

1. The *judge model* is not a field. It is implied by (filename, score-key):
   ``full_scores``/``mini_scores``/``nano_scores``. The Claude files put Sonnet
   under ``full`` and Haiku under ``mini``; the OpenAI files put GPT-5.4 / -mini
   / -nano. See :data:`_JUDGE_TABLE` and :func:`resolve_judge`.
2. The example ``id`` is only unique *within a subset* (e.g. id ``"47"`` appears
   in both Factuality and Focus). The true example identity is therefore the
   compound key ``(category, example_id)``. All joins/intersections here use it.

:func:`load_long` flattens everything into one tidy long-format DataFrame.

Semantics mirror the upstream ``analysis/compute_metrics.py`` at the pinned
commit (see ``data/PROVENANCE.txt``): errored samples are kept as ``<NA>`` (never
zero-filled), a response "mean" is over the non-null samples only, and nothing is
dropped at load time except whole-record refusals (which never occur in the
shipped data). Winner / error classification is intentionally *not* here — it
lands in a separate module on Day 1.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

# --- repo-relative paths ------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"

# --- fixed facts about the data (the audit asserts these; code never assumes) -
SUBSETS = ("Factuality", "Focus", "Math", "Precise IF", "Safety")
N_RESPONSES = 4          # 1 chosen (response_index 0) + 3 rejected
K = 8                    # samples per response
SCORE_MIN, SCORE_MAX = 1, 10
SEED = 42

S_COLS = [f"s_{i}" for i in range(1, K + 1)]


@dataclass(frozen=True)
class JudgeId:
    """Canonical identity of a judge slot."""

    data_name: str       # string used in the data, e.g. "gpt-5.4"
    display_name: str    # name used in the post/tables, e.g. "GPT-5.4"
    tier: str            # "full" | "mini" | "nano"
    provider: str        # "openai" | "anthropic"


# (provider, tier) -> (data_name, display_name). Display names follow the project
# prose ("GPT-5.4", "Sonnet 4.6", "Haiku 4.5", ...).
_JUDGE_TABLE = {
    ("openai", "full"): ("gpt-5.4", "GPT-5.4"),
    ("openai", "mini"): ("gpt-5.4-mini", "GPT-5.4-mini"),
    ("openai", "nano"): ("gpt-5.4-nano", "GPT-5.4-nano"),
    ("anthropic", "full"): ("claude-sonnet-4-6", "Sonnet 4.6"),
    ("anthropic", "mini"): ("claude-haiku-4-5-20251001", "Haiku 4.5"),
}


def resolve_judge(collection_stem: str, score_key: str) -> JudgeId:
    """Map (file stem, score-key) -> canonical judge identity.

    Provider is anthropic iff the stem mentions ``claude``; tier is the score-key
    prefix (``full`` / ``mini`` / ``nano``).
    """
    tier = score_key.split("_")[0]
    provider = "anthropic" if "claude" in collection_stem else "openai"
    try:
        data_name, display = _JUDGE_TABLE[(provider, tier)]
    except KeyError as exc:
        raise KeyError(
            f"no judge mapping for provider={provider!r} tier={tier!r} "
            f"(stem={collection_stem!r}, score_key={score_key!r})"
        ) from exc
    return JudgeId(data_name, display, tier, provider)


@dataclass(frozen=True)
class Collection:
    """A discovered raw collection file."""

    stem: str
    condition: str                 # prompt_variant, e.g. "base" | "criteria"
    provider: str                  # "openai" | "anthropic"
    path: Path
    score_keys: tuple[str, ...]    # e.g. ("full_scores", "mini_scores")


def load_raw(path: str | os.PathLike, drop_refused: bool = True) -> list[dict]:
    """Read a JSONL collection into a list of records.

    Mirrors the upstream ``load_collection``: drops ``refused`` records (a no-op
    on the shipped data, where ``refused`` is always False).
    """
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if drop_refused and r.get("refused", False):
                continue
            out.append(r)
    return out


def list_collections(raw_dir: str | os.PathLike = DATA_RAW) -> list[Collection]:
    """Discover ``*.jsonl`` collections; infer condition + score-keys from row 0."""
    cols: list[Collection] = []
    for fp in sorted(glob.glob(os.path.join(str(raw_dir), "*.jsonl"))):
        path = Path(fp)
        with open(path) as f:
            r0 = json.loads(f.readline())
        score_keys = tuple(sorted(k for k in r0 if k.endswith("_scores")))
        provider = "anthropic" if "claude" in path.stem else "openai"
        cols.append(Collection(path.stem, r0.get("prompt_variant"), provider, path, score_keys))
    return cols


def load_long(
    paths: Iterable[str | os.PathLike] | None = None,
    raw_dir: str | os.PathLike = DATA_RAW,
) -> pd.DataFrame:
    """Flatten the raw collections into one tidy long-format DataFrame.

    One row per (collection, judge_model, example, response_index). Columns:

    ``collection_stem, condition, provider, judge_model, judge_display, tier,
    category, example_id, response_index, s_1..s_8, n_valid_samples, k_field,
    refused``.

    ``s_1..s_8`` are nullable integers (``<NA>`` for an errored sample);
    ``n_valid_samples`` counts the non-null samples (0..8).
    """
    if paths is None:
        files = [c.path for c in list_collections(raw_dir)]
    else:
        files = [Path(p) for p in paths]

    rows: list[dict] = []
    for path in files:
        stem = path.stem
        for r in load_raw(path):
            condition = r.get("prompt_variant")
            refused = bool(r.get("refused", False))
            k_field = r.get("k")
            category = r["subset"]
            example_id = r["id"]
            for sk in (k for k in r if k.endswith("_scores")):
                judge = resolve_judge(stem, sk)
                for resp_idx, samples in enumerate(r[sk]):
                    n_valid = sum(1 for s in samples if s is not None)
                    row = {
                        "collection_stem": stem,
                        "condition": condition,
                        "provider": judge.provider,
                        "judge_model": judge.data_name,
                        "judge_display": judge.display_name,
                        "tier": judge.tier,
                        "category": category,
                        "example_id": example_id,
                        "response_index": resp_idx,
                        "n_valid_samples": n_valid,
                        "k_field": k_field,
                        "refused": refused,
                    }
                    for i, col in enumerate(S_COLS):
                        row[col] = samples[i] if i < len(samples) else None
                    rows.append(row)

    df = pd.DataFrame(rows)
    df[S_COLS] = df[S_COLS].astype("Int64")  # nullable ints; <NA> for errored samples
    return df


def example_keys(
    long: pd.DataFrame,
    judge_model: str,
    condition: str,
    usable_only: bool = False,
    strict_id_only: bool = False,
) -> set:
    """Set of example identities covered by one (judge_model, condition).

    identity = ``example_id`` if ``strict_id_only`` else ``(category, example_id)``.
    ``usable_only`` keeps only examples where every response has >= 1 valid sample
    (no all-null response) — i.e. examples the winner logic would not skip.
    """
    sub = long[(long["judge_model"] == judge_model) & (long["condition"] == condition)]
    if usable_only:
        min_valid = sub.groupby(["category", "example_id"])["n_valid_samples"].min()
        pairs = [key for key, ok in (min_valid >= 1).items() if ok]
    else:
        pairs = list(set(zip(sub["category"], sub["example_id"])))
    if strict_id_only:
        return {eid for (_cat, eid) in pairs}
    return set(pairs)


def common_examples(
    long: pd.DataFrame,
    members: list[tuple[str, str]],
    usable_only: bool = False,
    strict_id_only: bool = False,
) -> set:
    """Intersection of example identities across a list of (judge_model, condition).

    Defaults to the compound ``(category, example_id)`` key. Pass
    ``strict_id_only=True`` to reproduce the upstream id-only intersection (which
    silently collapses cross-subset id collisions) for comparison.
    """
    sets = [
        example_keys(long, jm, cond, usable_only=usable_only, strict_id_only=strict_id_only)
        for jm, cond in members
    ]
    if not sets:
        return set()
    out = set(sets[0])
    for s in sets[1:]:
        out &= s
    return out
