"""Baseline-runner stub for monocular depth-estimation leaderboards.

Loads each `MODELS` entry, runs it on every sample's RGB input, and
returns a per-sample mean depth (the simplest scalar) under whatever
pred-field key the LB expects (typically `depth_pred`).

You almost certainly want to replace `predictor_fn` with something
domain-correct (per-pixel error vs GT depth map, etc.). This stub just
gets the upload mechanics working end-to-end.

Run via:
    from seed_baselines import seed_baselines
    from baselines_depth import MODELS, predictor_fn, load_inputs
    seed_baselines(leaderboard_id=..., api_token=..., gt_zip_url=...,
                   models=MODELS, predictor_fn=predictor_fn,
                   load_inputs=load_inputs)
"""
from pathlib import Path

from PIL import Image
import numpy as np


# Each entry needs `name` (used as the submission name on BenchHub),
# `repo_id` (HF hub), and a `load` callable returning (model, processor).
def _hf_loader(repo_id):
    """Lazy import so this file is cheap when only the model list is read."""
    def _load():
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        return (
            AutoModelForDepthEstimation.from_pretrained(repo_id),
            AutoImageProcessor.from_pretrained(repo_id),
        )
    return _load


MODELS = [
    {'name': 'dpt-large',           'repo_id': 'Intel/dpt-large',
     'load': _hf_loader('Intel/dpt-large')},
    {'name': 'dpt-hybrid-midas',    'repo_id': 'Intel/dpt-hybrid-midas',
     'load': _hf_loader('Intel/dpt-hybrid-midas')},
    {'name': 'glpn-nyu',            'repo_id': 'vinvino02/glpn-nyu',
     'load': _hf_loader('vinvino02/glpn-nyu')},
    {'name': 'glpn-kitti',          'repo_id': 'vinvino02/glpn-kitti',
     'load': _hf_loader('vinvino02/glpn-kitti')},
    {'name': 'depth-anything-small','repo_id': 'LiheYoung/depth-anything-small-hf',
     'load': _hf_loader('LiheYoung/depth-anything-small-hf')},
    {'name': 'depth-anything-base', 'repo_id': 'LiheYoung/depth-anything-base-hf',
     'load': _hf_loader('LiheYoung/depth-anything-base-hf')},
    {'name': 'depth-anything-large','repo_id': 'LiheYoung/depth-anything-large-hf',
     'load': _hf_loader('LiheYoung/depth-anything-large-hf')},
    {'name': 'zoedepth-nyu',        'repo_id': 'Intel/zoedepth-nyu',
     'load': _hf_loader('Intel/zoedepth-nyu')},
    {'name': 'zoedepth-kitti',      'repo_id': 'Intel/zoedepth-kitti',
     'load': _hf_loader('Intel/zoedepth-kitti')},
    {'name': 'midas-v2-1-small',    'repo_id': 'Intel/midas-2.1-small',
     'load': _hf_loader('Intel/midas-2.1-small')},
]


def load_inputs(gt_root: Path, sample_name: str) -> dict:
    """Locate the sample's RGB image under any image_*/ folder and
    return it as a PIL.Image. Imports are lazy so this module loads
    fast even on a CPU-only host."""
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
    """Run the loaded model on the sample's RGB image and return a
    scalar prediction. The LB's auto-proposer creates `<col>_pred`
    pred-field names from each GT scalar — for a depth dataset the
    GT scalar is typically the per-sample mean / median depth, so we
    return mean predicted depth here.

    REPLACE this with the metric your LB actually expects (per-pixel
    map, dense depth bytes, etc.). This stub just gets the plumbing
    working."""
    import torch
    rgb = inputs['rgb']
    enc = processor(images=rgb, return_tensors='pt')
    with torch.no_grad():
        pred = model(**enc).predicted_depth
    pred_np = pred.squeeze().cpu().numpy()
    return {'depth_pred': float(np.mean(pred_np))}
