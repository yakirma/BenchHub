"""Tests for `_pwc_task_to_category` and `_pwc_task_area`-style
classification. Covers:

- Domain-modality prefixes (Medical, Aerial, Few-Shot, …) are stripped
  before classification, so "Medical Image Segmentation" lands under
  Vision/Image Segmentation rather than Medical/Medical Image
  Segmentation.
- Title-casing after strip collapses "image segmentation" and
  "Image Segmentation" into one node.
- `_PWC_AREA_RULES` priority: Speech / Code / Medical match before
  Vision when their tokens are present, so "Speech Recognition" isn't
  swallowed by the `recognition` Vision rule.
- Unknown / empty input falls back to None or 'Other'."""
import pytest

from app import _pwc_task_to_category


@pytest.mark.parametrize("task, expected", [
    # Domain-prefix stripping → Vision
    ("Medical Image Segmentation", "Vision/Image Segmentation"),
    ("Aerial Image Classification", "Vision/Image Classification"),
    ("Satellite Image Classification", "Vision/Image Classification"),
    ("Biomedical Image Segmentation", "Vision/Image Segmentation"),

    # No-prefix vision tasks stay in Vision
    ("Image Segmentation", "Vision/Image Segmentation"),
    ("Semantic Segmentation", "Vision/Semantic Segmentation"),
    ("Instance Segmentation", "Vision/Instance Segmentation"),
    ("Object Detection", "Vision/Object Detection"),
    ("Depth Estimation", "Vision/Depth Estimation"),
    ("Image Generation", "Vision/Image Generation"),

    # 3D qualifier is kept (not in _DOMAIN_PREFIXES) — 3D Pose is its
    # own thing.
    ("3D Human Pose Estimation", "Vision/3D Human Pose Estimation"),
    # Monocular IS in _DOMAIN_PREFIXES (input-modality metadata),
    # so it strips to the abstract task name.
    ("Monocular Depth Estimation", "Vision/Depth Estimation"),
    ("Stereo Depth Estimation", "Vision/Depth Estimation"),

    # Priority-order: Speech / Audio beats the broad Vision `recognition`
    ("Speech Recognition", "Speech & Audio/Speech Recognition"),
    ("Keyword Spotting", "Speech & Audio/Keyword Spotting"),

    # Priority-order: Code beats Vision `generation`
    ("Code Generation", "Code/Code Generation"),

    # NLP tasks
    ("Question Answering", "NLP/Question Answering"),
    ("Relation Extraction", "NLP/Relation Extraction"),
    ("Common Sense Reasoning", "NLP/Common Sense Reasoning"),
    ("Visual Reasoning", "NLP/Visual Reasoning"),  # `reasoning` matches NLP

    # Graph
    ("Link Prediction", "Graph/Link Prediction"),

    # Few-Shot prefix strips → falls back to Other when stripped task
    # has no Vision/NLP token
    ("Few-Shot Learning", "Other/Learning"),
])
def test_pwc_task_to_category(task, expected):
    assert _pwc_task_to_category(task) == expected


def test_empty_or_none_task_returns_none():
    assert _pwc_task_to_category(None) is None
    assert _pwc_task_to_category("") is None


def test_title_casing_collapses_source_casing():
    """PWC source casing varies — make sure the category task part
    normalises so two LBs of the same task land under one node in the
    /explore tree."""
    a = _pwc_task_to_category("image segmentation")
    b = _pwc_task_to_category("Image Segmentation")
    c = _pwc_task_to_category("IMAGE SEGMENTATION")
    # "IMAGE SEGMENTATION" — we only title-case lowercase words, not
    # already-uppercase ones, so it stays uppercase. Document the
    # current behavior and at least make sure the lowercase / title
    # cases produce the same node.
    assert a == b


def test_priority_order_medical_before_vision_for_modality_tokens():
    """When the prefix-strip leaves a non-vision task, the Medical
    bucket still wins on modality tokens like 'mri'/'ct.scan'. Make
    sure that path still works after the prefix changes."""
    # "MRI Reconstruction" doesn't match any _DOMAIN_PREFIXES so the
    # raw form is classified. `\bmri\b` matches Medical first.
    assert _pwc_task_to_category("MRI Reconstruction").startswith("Medical/")


def test_unknown_task_falls_back_to_other():
    """A task name with no recognised tokens lands under Other."""
    cat = _pwc_task_to_category("Whimsical Frobnication")
    assert cat.startswith("Other/")
    assert "Whimsical Frobnication" in cat
