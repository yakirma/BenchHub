"""Baseline-runner stub for semantic-segmentation leaderboards.

Loads each `MODELS` entry, runs it on every RGB sample, and returns
per-sample summary stats keyed by the LB's pred-fields. The simplest
case is "predicted dominant class index" → `label_pred` (matches the
auto-proposer's top-1 metric on a ClassLabel-shaped GT). For real
segmentation comparison you'll want per-pixel mask IoU, which is a
heavier upload — replace `predictor_fn` accordingly.
"""
from pathlib import Path

from PIL import Image
import numpy as np


def _hf_seg_loader(repo_id):
    def _load():
        from transformers import (
            AutoImageProcessor, AutoModelForSemanticSegmentation,
        )
        return (
            AutoModelForSemanticSegmentation.from_pretrained(repo_id),
            AutoImageProcessor.from_pretrained(repo_id),
        )
    return _load


MODELS = [
    {'name': 'segformer-b0-ade',    'repo_id': 'nvidia/segformer-b0-finetuned-ade-512-512',
     'load': _hf_seg_loader('nvidia/segformer-b0-finetuned-ade-512-512')},
    {'name': 'segformer-b1-ade',    'repo_id': 'nvidia/segformer-b1-finetuned-ade-512-512',
     'load': _hf_seg_loader('nvidia/segformer-b1-finetuned-ade-512-512')},
    {'name': 'segformer-b2-ade',    'repo_id': 'nvidia/segformer-b2-finetuned-ade-512-512',
     'load': _hf_seg_loader('nvidia/segformer-b2-finetuned-ade-512-512')},
    {'name': 'segformer-b3-ade',    'repo_id': 'nvidia/segformer-b3-finetuned-ade-512-512',
     'load': _hf_seg_loader('nvidia/segformer-b3-finetuned-ade-512-512')},
    {'name': 'segformer-b4-ade',    'repo_id': 'nvidia/segformer-b4-finetuned-ade-512-512',
     'load': _hf_seg_loader('nvidia/segformer-b4-finetuned-ade-512-512')},
    {'name': 'segformer-b5-ade',    'repo_id': 'nvidia/segformer-b5-finetuned-ade-640-640',
     'load': _hf_seg_loader('nvidia/segformer-b5-finetuned-ade-640-640')},
    {'name': 'mask2former-tiny-ade','repo_id': 'facebook/mask2former-swin-tiny-ade-semantic',
     'load': _hf_seg_loader('facebook/mask2former-swin-tiny-ade-semantic')},
    {'name': 'mask2former-large',   'repo_id': 'facebook/mask2former-swin-large-ade-semantic',
     'load': _hf_seg_loader('facebook/mask2former-swin-large-ade-semantic')},
    {'name': 'oneformer-coco',      'repo_id': 'shi-labs/oneformer_coco_swin_large',
     'load': _hf_seg_loader('shi-labs/oneformer_coco_swin_large')},
    {'name': 'oneformer-ade20k',    'repo_id': 'shi-labs/oneformer_ade20k_swin_large',
     'load': _hf_seg_loader('shi-labs/oneformer_ade20k_swin_large')},
]


def load_inputs(gt_root: Path, sample_name: str) -> dict:
    for sub in gt_root.iterdir():
        if not (sub.is_dir() and sub.name.startswith('image_')):
            continue
        for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'):
            candidate = sub / f'{sample_name}{ext}'
            if candidate.exists():
                return {'rgb': Image.open(candidate).convert('RGB')}
    raise FileNotFoundError(
        f"No RGB image found for sample {sample_name!r} under {gt_root}"
    )


def predictor_fn(spec, model, processor, inputs) -> dict:
    """Predict the most-frequent pixel class index. For an LB whose
    GT scalar is `label` (ClassLabel), the auto-proposer expects
    `label_pred`. REPLACE with full mask + IoU when running for real."""
    import torch
    rgb = inputs['rgb']
    enc = processor(images=rgb, return_tensors='pt')
    with torch.no_grad():
        out = model(**enc)
    # logits: [1, n_classes, H, W] → argmax per pixel → mode = dominant class.
    pred_pixels = out.logits.argmax(dim=1).squeeze().cpu().numpy()
    classes, counts = np.unique(pred_pixels, return_counts=True)
    dominant = int(classes[np.argmax(counts)])
    return {'label_pred': dominant}
