"""Sub-category model-rankings: normalized-score aggregation across a
sub-category's boards, model identity from HF id, coverage tracking.
Only sub-categories (a category containing "/") are ranked — top-level
categories get no meta-ranking.
"""
from benchhub.hf_results_export import model_identity, compute_aggregates


def _row(model, score):
    k, d = model_identity(model, f"https://huggingface.co/{model}")
    return {"key": k, "model": d, "link": f"https://huggingface.co/{model}", "score": score}


def test_model_identity_prefers_hf_id():
    assert model_identity("X", "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct") \
        == ("qwen/qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct")
    # non-HF link → falls back to the submission name
    assert model_identity("RAFT-Stereo", "https://github.com/x/y")[1] == "RAFT-Stereo"


def test_normalized_mean_and_coverage():
    boards = [
        {"id": 1, "name": "GSM8K", "category": "NLP/Math", "higher_is_better": True,
         "rows": [_row("a/best", 0.9), _row("a/mid", 0.7), _row("a/worst", 0.5)]},
        {"id": 2, "name": "MATH", "category": "NLP/Math", "higher_is_better": True,
         "rows": [_row("a/best", 0.8), _row("a/mid", 0.6), _row("a/worst", 0.4)]},
    ]
    aggs = compute_aggregates(boards)
    # one sub-category (NLP/Math) with 2 boards
    assert len(aggs) == 1
    sub = aggs[0]
    assert sub["scope"] == "NLP/Math" and sub["level"] == "subcategory" and sub["n_boards"] == 2
    by = {m["model"]: m for m in sub["models"]}
    assert by["a/best"]["score"] == 1.0          # best on both → 1.0
    assert by["a/worst"]["score"] == 0.0         # worst on both → 0.0
    assert 0.0 < by["a/mid"]["score"] < 1.0
    assert all(m["coverage"] == 2 for m in sub["models"])  # all on both boards


def test_lower_is_better_is_flipped():
    boards = [
        {"id": 1, "name": "WER-A", "category": "Audio/ASR", "higher_is_better": False,
         "rows": [_row("a/x", 0.1), _row("a/y", 0.3)]},   # lower WER is better
        {"id": 2, "name": "WER-B", "category": "Audio/ASR", "higher_is_better": False,
         "rows": [_row("a/x", 0.2), _row("a/y", 0.5)]},
    ]
    aggs = compute_aggregates(boards)
    by = {m["model"]: m for m in aggs[0]["models"]}
    assert by["a/x"]["score"] == 1.0   # lowest WER on both → best
    assert by["a/y"]["score"] == 0.0


def test_partial_coverage_ranked_with_count():
    boards = [
        {"id": 1, "name": "B1", "category": "Vision/Segmentation", "higher_is_better": True,
         "rows": [_row("a/p", 0.9), _row("a/q", 0.5)]},
        {"id": 2, "name": "B2", "category": "Vision/Segmentation", "higher_is_better": True,
         "rows": [_row("a/q", 0.8)]},   # only q is on B2
    ]
    aggs = compute_aggregates(boards)
    by = {m["model"]: m for m in aggs[0]["models"]}
    assert by["a/p"]["coverage"] == 1   # only B1
    assert by["a/q"]["coverage"] == 2   # both
    # p aced its single board (norm 1.0) — ranked, but coverage exposes it's 1/2
    assert by["a/p"]["score"] == 1.0


def test_subcategory_only_no_toplevel_scope():
    boards = [
        {"id": 1, "name": "M1", "category": "NLP/Math", "higher_is_better": True,
         "rows": [_row("a/x", 0.9), _row("a/y", 0.5)]},
        {"id": 2, "name": "M2", "category": "NLP/Math", "higher_is_better": True,
         "rows": [_row("a/x", 0.8), _row("a/y", 0.6)]},
    ]
    aggs = compute_aggregates(boards)
    scopes = {a["scope"]: a for a in aggs}
    # the sub-category scope exists; the top-level "NLP" scope does NOT
    assert "NLP/Math" in scopes and scopes["NLP/Math"]["level"] == "subcategory"
    assert "NLP" not in scopes


def test_toplevel_only_categories_are_not_ranked():
    # Two boards under bare top-level categories (no "/") → no meta-ranking.
    boards = [
        {"id": 1, "name": "A", "category": "NLP", "higher_is_better": True,
         "rows": [_row("a/x", 0.9), _row("a/y", 0.5)]},
        {"id": 2, "name": "B", "category": "NLP", "higher_is_better": True,
         "rows": [_row("a/x", 0.8), _row("a/y", 0.6)]},
    ]
    assert compute_aggregates(boards) == []
