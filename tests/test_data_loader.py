"""Loader sanity tests (Step 0). Winner/paper-parity tests land Day 1."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import data_loader as dl


# --------------------------------------------------------------------------- #
# unit tests on a synthetic collection (no dependency on vendored data)       #
# --------------------------------------------------------------------------- #
def _write_record(path, **overrides):
    rec = {
        "id": "1",
        "subset": "Safety",
        "prompt_variant": "base",
        "k": 8,
        "full_scores": [[8] * 8, [3] * 8, [2] * 8, [1] * 8],
        "full_errors": [[None] * 8] * 4,
        "mini_scores": [[7] * 8, [3] * 8, [2] * 8, [1] * 8],
        "mini_errors": [[None] * 8] * 4,
        "refused": False,
    }
    rec.update(overrides)
    path.write_text(json.dumps(rec) + "\n")


def test_resolve_judge_mapping():
    assert dl.resolve_judge("base_both_k8", "full_scores").data_name == "gpt-5.4"
    assert dl.resolve_judge("base_both_k8", "mini_scores").data_name == "gpt-5.4-mini"
    assert dl.resolve_judge("base_nano_k8", "nano_scores").data_name == "gpt-5.4-nano"
    assert dl.resolve_judge("base_claude_both_k8", "full_scores").data_name == "claude-sonnet-4-6"
    assert dl.resolve_judge("base_claude_both_k8", "mini_scores").data_name == "claude-haiku-4-5-20251001"
    with pytest.raises(KeyError):
        dl.resolve_judge("base_both_k8", "giant_scores")


def test_long_shape_and_arity(tmp_path):
    _write_record(tmp_path / "base_both_k8.jsonl")
    long = dl.load_long(raw_dir=tmp_path)
    # 1 example x 2 judges x 4 responses
    assert len(long) == 2 * dl.N_RESPONSES
    assert set(long["judge_model"]) == {"gpt-5.4", "gpt-5.4-mini"}
    assert sorted(long["response_index"].unique()) == [0, 1, 2, 3]
    assert long["s_1"].dtype == "Int64"


def test_errored_samples_become_na_and_counted(tmp_path):
    scores = [[8, None, 8, None, 8, 8, 8, 8], [1] * 8, [1] * 8, [1] * 8]
    _write_record(tmp_path / "base_nano_k8.jsonl", full_scores=None, mini_scores=None,
                  full_errors=None, mini_errors=None,
                  nano_scores=scores, nano_errors=[[None] * 8] * 4)
    # remove the now-None full/mini keys so only nano remains
    rec = json.loads((tmp_path / "base_nano_k8.jsonl").read_text())
    rec = {k: v for k, v in rec.items() if not (k.endswith("_scores") and v is None)
           and not (k.endswith("_errors") and v is None)}
    (tmp_path / "base_nano_k8.jsonl").write_text(json.dumps(rec) + "\n")

    long = dl.load_long(raw_dir=tmp_path)
    r0 = long[long["response_index"] == 0].iloc[0]
    assert r0["n_valid_samples"] == 6
    assert pd.isna(r0["s_2"]) and pd.isna(r0["s_4"])
    assert int(r0["s_1"]) == 8


def test_usable_filter_drops_all_null_response(tmp_path):
    # response 1 is entirely null -> example is unusable
    scores = [[5] * 8, [None] * 8, [5] * 8, [5] * 8]
    _write_record(tmp_path / "base_nano_k8.jsonl", full_scores=None, mini_scores=None,
                  full_errors=None, mini_errors=None,
                  nano_scores=scores, nano_errors=[[None] * 8] * 4)
    rec = json.loads((tmp_path / "base_nano_k8.jsonl").read_text())
    rec = {k: v for k, v in rec.items() if not (k.endswith("_scores") and v is None)
           and not (k.endswith("_errors") and v is None)}
    (tmp_path / "base_nano_k8.jsonl").write_text(json.dumps(rec) + "\n")

    long = dl.load_long(raw_dir=tmp_path)
    present = dl.example_keys(long, "gpt-5.4-nano", "base", usable_only=False)
    usable = dl.example_keys(long, "gpt-5.4-nano", "base", usable_only=True)
    assert present == {("Safety", "1")}
    assert usable == set()


def test_compound_key_beats_id_only(tmp_path):
    # same id "7" in two subsets -> 2 compound keys, 1 id-only key
    lines = []
    for subset in ("Factuality", "Focus"):
        lines.append(json.dumps({
            "id": "7", "subset": subset, "prompt_variant": "base", "k": 8,
            "full_scores": [[6] * 8, [3] * 8, [2] * 8, [1] * 8],
            "full_errors": [[None] * 8] * 4,
            "mini_scores": [[6] * 8, [3] * 8, [2] * 8, [1] * 8],
            "mini_errors": [[None] * 8] * 4,
            "refused": False,
        }))
    (tmp_path / "base_both_k8.jsonl").write_text("\n".join(lines) + "\n")

    long = dl.load_long(raw_dir=tmp_path)
    compound = dl.example_keys(long, "gpt-5.4", "base")
    id_only = dl.example_keys(long, "gpt-5.4", "base", strict_id_only=True)
    assert compound == {("Factuality", "7"), ("Focus", "7")}
    assert id_only == {"7"}


# --------------------------------------------------------------------------- #
# smoke test on the real vendored data (skips if not fetched yet)             #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not any(dl.DATA_RAW.glob("*.jsonl")),
    reason="vendored data not present; run scripts/fetch_data.sh",
)
def test_real_data_gate_and_arity():
    long = dl.load_long()
    # every (collection, judge, example) has exactly 4 responses
    sizes = long.groupby(
        ["collection_stem", "judge_model", "category", "example_id"]
    ).size()
    assert (sizes == dl.N_RESPONSES).all()

    main = [("gpt-5.4", "base"), ("claude-sonnet-4-6", "base")]
    usable = dl.common_examples(long, main, usable_only=True)
    assert len(usable) >= 1600  # the acceptance gate
    # compound key is the larger, correct basis vs the upstream id-only count
    id_only = dl.common_examples(long, main, strict_id_only=True)
    assert len(dl.common_examples(long, main)) >= len(id_only)
