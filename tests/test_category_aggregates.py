"""Sub-category model-rankings: normalized-score aggregation across a
sub-category's boards, model identity from HF id, coverage tracking.

Only sub-categories (a category containing "/") are ranked. A model must
appear on >=50% of the sub-category's boards to be ranked, and a scope is
only emitted if >=3 such models remain.
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
    # one sub-category (NLP/Math) with 2 boards and 3 valid models
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
         "rows": [_row("a/x", 0.1), _row("a/y", 0.3), _row("a/z", 0.5)]},   # lower WER is better
        {"id": 2, "name": "WER-B", "category": "Audio/ASR", "higher_is_better": False,
         "rows": [_row("a/x", 0.2), _row("a/y", 0.5), _row("a/z", 0.9)]},
    ]
    aggs = compute_aggregates(boards)
    by = {m["model"]: m for m in aggs[0]["models"]}
    assert by["a/x"]["score"] == 1.0   # lowest WER on both → best
    assert by["a/z"]["score"] == 0.0   # highest WER on both → worst


def test_coverage_below_50pct_excluded():
    # 3 boards: p/q/r are on all three (100%); s is on only one (33%, < 50%) → dropped.
    boards = [
        {"id": 1, "name": "B1", "category": "Vision/Segmentation", "higher_is_better": True,
         "rows": [_row("a/p", 0.9), _row("a/q", 0.8), _row("a/r", 0.7), _row("a/s", 0.6)]},
        {"id": 2, "name": "B2", "category": "Vision/Segmentation", "higher_is_better": True,
         "rows": [_row("a/p", 0.9), _row("a/q", 0.8), _row("a/r", 0.7)]},
        {"id": 3, "name": "B3", "category": "Vision/Segmentation", "higher_is_better": True,
         "rows": [_row("a/p", 0.9), _row("a/q", 0.8), _row("a/r", 0.7)]},
    ]
    aggs = compute_aggregates(boards)
    by = {m["model"]: m for m in aggs[0]["models"]}
    assert set(by) == {"a/p", "a/q", "a/r"}   # s (1/3 coverage) excluded
    assert all(m["coverage"] == 3 for m in by.values())


def test_fewer_than_3_valid_models_not_shown():
    # 2 boards but only 2 models → below the 3-model threshold → no scope emitted.
    boards = [
        {"id": 1, "name": "M1", "category": "NLP/Math", "higher_is_better": True,
         "rows": [_row("a/x", 0.9), _row("a/y", 0.5)]},
        {"id": 2, "name": "M2", "category": "NLP/Math", "higher_is_better": True,
         "rows": [_row("a/x", 0.8), _row("a/y", 0.6)]},
    ]
    assert compute_aggregates(boards) == []


def test_subcategory_only_no_toplevel_scope():
    boards = [
        {"id": 1, "name": "M1", "category": "NLP/Math", "higher_is_better": True,
         "rows": [_row("a/x", 0.9), _row("a/y", 0.5), _row("a/z", 0.3)]},
        {"id": 2, "name": "M2", "category": "NLP/Math", "higher_is_better": True,
         "rows": [_row("a/x", 0.8), _row("a/y", 0.6), _row("a/z", 0.2)]},
    ]
    aggs = compute_aggregates(boards)
    scopes = {a["scope"]: a for a in aggs}
    # the sub-category scope exists; the top-level "NLP" scope does NOT
    assert "NLP/Math" in scopes and scopes["NLP/Math"]["level"] == "subcategory"
    assert "NLP" not in scopes


def test_toplevel_only_categories_are_not_ranked():
    # Boards under bare top-level categories (no "/") → no meta-ranking.
    boards = [
        {"id": 1, "name": "A", "category": "NLP", "higher_is_better": True,
         "rows": [_row("a/x", 0.9), _row("a/y", 0.5), _row("a/z", 0.3)]},
        {"id": 2, "name": "B", "category": "NLP", "higher_is_better": True,
         "rows": [_row("a/x", 0.8), _row("a/y", 0.6), _row("a/z", 0.2)]},
    ]
    assert compute_aggregates(boards) == []


def test_per_board_carries_lb_id_for_linking():
    boards = [
        {"id": 11, "name": "B1", "category": "Vision/Detection", "higher_is_better": True,
         "rows": [_row("a/x", 0.9), _row("a/y", 0.5), _row("a/z", 0.3)]},
        {"id": 22, "name": "B2", "category": "Vision/Detection", "higher_is_better": True,
         "rows": [_row("a/x", 0.8), _row("a/y", 0.6), _row("a/z", 0.2)]},
    ]
    aggs = compute_aggregates(boards)
    x = {m["model"]: m for m in aggs[0]["models"]}["a/x"]
    # per_board maps board name -> {lb_id, norm} so the UI can deep-link
    assert x["per_board"]["B1"]["lb_id"] == 11
    assert x["per_board"]["B2"]["lb_id"] == 22
    assert x["per_board"]["B1"]["norm"] == 1.0
