"""Sub-category model-rankings: normalized-score aggregation across a
sub-category's boards, model identity from HF id, coverage tracking.

Only sub-categories (a category containing "/") are ranked. A model must
appear on >=50% of the sub-category's boards to be ranked, and a scope is
only emitted if >=3 such models remain.
"""
from benchhub.hf_results_export import (model_identity, compute_aggregates,
                                        _strip_board_dataset, _dataset_candidates)


def _rowm(model, link, score):
    k, d = model_identity(model, link)
    return {"key": k, "model": d, "link": link, "score": score}


def test_strip_board_dataset_general():
    forklift = _dataset_candidates("Forklift-detection_benchmark")
    plate = _dataset_candidates("LicensePlate-detection_benchmark")
    coco = _dataset_candidates("COCO-val2017-detection_benchmark")
    plane = _dataset_candidates("Plane-detection_benchmark")
    # base architectures of any family collapse off their fine-tune dataset
    assert _strip_board_dataset("detr-resnet50-forklift-detection", forklift) == "detr-resnet50"
    assert _strip_board_dataset("yolos-small-finetuned-license-plate-detection", plate) == "yolos-small"
    # size tokens survive; a model that isn't fine-tuned on this board is untouched
    assert _strip_board_dataset("yolos-small", coco) == "yolos-small"
    # segment-boundary guard: "Plane" must NOT bite "airplane"
    assert _strip_board_dataset("airplane-net", plane) == "airplane-net"
    # LLM model identity is left alone on a knowledge board
    assert _strip_board_dataset("meta-llama/Llama-3.2-1B-Instruct",
                                _dataset_candidates("MMLU-test_benchmark")) == "meta-llama/Llama-3.2-1B-Instruct"
    # Generic board word must NOT strip an architecture name ending in it:
    # "stereo-dataset" board must leave RAFT-Stereo / GMStereo intact.
    stereo = _dataset_candidates("stereo-dataset__stereo-dataset_benchmark")
    assert _strip_board_dataset("RAFT-Stereo", stereo) == "RAFT-Stereo"
    assert _strip_board_dataset("GMStereo", _dataset_candidates("KITTI-2015-stereo_benchmark")) == "GMStereo"


def test_non_yolo_finetunes_group_across_dataset_boards():
    # A non-arch-listed model (DETR) fine-tuned per dataset, one per board.
    HF = "https://huggingface.co/acme/{}"
    boards = [
        {"id": 1, "name": "Forklift-detection_benchmark", "category": "Vision/Object Detection",
         "higher_is_better": True, "rows": [
            _rowm("detr-r50-forklift-detection", HF.format("detr-r50-forklift-detection"), 0.7),
            _rowm("yolov8s-forklift-detection", HF.format("yolov8s-forklift-detection"), 0.8),
            _rowm("rtdetr-forklift-detection", HF.format("rtdetr-forklift-detection"), 0.6)]},
        {"id": 2, "name": "Plane-detection_benchmark", "category": "Vision/Object Detection",
         "higher_is_better": True, "rows": [
            _rowm("detr-r50-plane-detection", HF.format("detr-r50-plane-detection"), 0.5),
            _rowm("yolov8s-plane-detection", HF.format("yolov8s-plane-detection"), 0.9),
            _rowm("rtdetr-plane-detection", HF.format("rtdetr-plane-detection"), 0.4)]},
    ]
    aggs = compute_aggregates(boards)
    by = {m["model"]: m for m in aggs[0]["models"]}
    # acme/detr-r50 (owner kept for non-arch), yolov8s (arch token), acme/rtdetr
    assert "acme/detr-r50" in by and by["acme/detr-r50"]["coverage"] == 2
    assert "yolov8s" in by and by["yolov8s"]["coverage"] == 2
    assert "acme/rtdetr" in by


def _row(model, score):
    k, d = model_identity(model, f"https://huggingface.co/{model}")
    return {"key": k, "model": d, "link": f"https://huggingface.co/{model}", "score": score}


def test_model_identity_prefers_hf_id():
    assert model_identity("X", "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct") \
        == ("qwen/qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct")
    # non-HF link → falls back to the submission name
    assert model_identity("RAFT-Stereo", "https://github.com/x/y")[1] == "RAFT-Stereo"


def test_model_identity_strips_variant_suffix():
    # Fine-tune / config variants of the same base model collapse to one identity.
    assert model_identity("RAFT-Stereo (ETH3D)", "https://github.com/x") == ("raft-stereo", "RAFT-Stereo")
    assert model_identity("RAFT-Stereo (fast)", "https://github.com/x") == ("raft-stereo", "RAFT-Stereo")
    assert model_identity("CREStereo (combined, iter10)", "")[0] == "crestereo"
    assert model_identity("StereoSGBM (block5, 128disp)", "")[1] == "StereoSGBM"
    # no parenthetical → unchanged; all-parenthetical → not collapsed to empty
    assert model_identity("HITNet", "")[1] == "HITNet"
    assert model_identity("(only)", "")[1] == "(only)"


def test_arch_family_groups_finetunes_across_datasets():
    # keremberke's per-dataset YOLO fine-tunes collapse to the architecture token.
    assert model_identity("yolov8s-forklift-detection",
                          "https://huggingface.co/keremberke/yolov8s-forklift-detection") == ("yolov8s", "yolov8s")
    assert model_identity("yolov8s-blood-cell-detection",
                          "https://huggingface.co/keremberke/yolov8s-blood-cell-detection")[0] == "yolov8s"
    assert model_identity("yolov8n-plane-detection",
                          "https://huggingface.co/keremberke/yolov8n-plane-detection")[0] == "yolov8n"
    assert model_identity("yolov8m-hard-hat-detection",
                          "https://huggingface.co/keremberke/yolov8m-hard-hat-detection")[0] == "yolov8m"
    # Different sizes stay distinct; non-YOLO names are untouched.
    assert model_identity("yolov8n-x", "")[0] != model_identity("yolov8s-x", "")[0]
    assert model_identity("yolos-small", "https://huggingface.co/hustvl/yolos-small") == ("hustvl/yolos-small", "hustvl/yolos-small")
    assert model_identity("RAFT-Stereo", "")[1] == "RAFT-Stereo"          # not "RAFT"
    assert model_identity("X", "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")[1] == "meta-llama/Llama-3.2-1B-Instruct"


def test_variants_group_as_one_model_best_per_board():
    # Two RAFT-Stereo variants + two HITNet variants across 2 boards. Each base
    # model should appear once, scored by its BEST variant per board.
    boards = [
        {"id": 1, "name": "B1", "category": "Vision/Stereo", "higher_is_better": True,
         "rows": [_row("RAFT-Stereo (ETH3D)", 0.6), _row("RAFT-Stereo (fast)", 0.9),
                  _row("HITNet (SceneFlow)", 0.4), _row("CREStereo (iter10)", 0.2)]},
        {"id": 2, "name": "B2", "category": "Vision/Stereo", "higher_is_better": True,
         "rows": [_row("RAFT-Stereo (ETH3D)", 0.5), _row("HITNet (Middlebury)", 0.8),
                  _row("CREStereo (iter20)", 0.3)]},
    ]
    aggs = compute_aggregates(boards)
    by = {m["model"]: m for m in aggs[0]["models"]}
    assert set(by) == {"RAFT-Stereo", "HITNet", "CREStereo"}   # variants collapsed
    # B1 RAFT norms: fast 0.9 is best (→1.0), ETH3D 0.6, HITNet 0.4, CRE 0.2.
    raft = by["RAFT-Stereo"]
    assert raft["coverage"] == 2                                # both boards, not 3 rows
    assert raft["per_board"]["B1"]["norm"] == 1.0              # best variant (fast) on B1


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
