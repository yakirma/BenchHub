"""Baseline-runner stub for image denoising / restoration leaderboards.

Loads a list of restoration models and returns a per-sample summary
scalar (mean intensity of the denoised output). REPLACE with PSNR /
SSIM against the GT clean image once you've wired the LB metric to
take both images. The stub just gets the upload mechanics flowing.
"""
from pathlib import Path

from PIL import Image
import numpy as np


def _hf_pipeline_loader(repo_id, task=None):
    """Many denoising models on HF are wrapped as a `pipeline` rather
    than a head class. This loader returns a callable as the 'model'
    so the predictor can apply it uniformly."""
    def _load():
        from transformers import pipeline
        pipe = pipeline(task or 'image-to-image', model=repo_id)
        return pipe, None
    return _load


def _generic_torch_loader(repo_id, processor_cls=None, model_cls=None):
    def _load():
        from transformers import AutoImageProcessor, AutoModelForImageToImage
        return (
            (model_cls or AutoModelForImageToImage).from_pretrained(repo_id),
            (processor_cls or AutoImageProcessor).from_pretrained(repo_id),
        )
    return _load


MODELS = [
    # Pre-trained super-resolution / restoration models — Hugging Face
    # has fewer canonical denoising checkpoints than for depth or seg,
    # so this list mixes Swin2SR + a few image-to-image pipelines.
    # Verify each repo ID is still live before running.
    {'name': 'swin2sr-classical-sr-x2', 'repo_id': 'caidas/swin2SR-classical-sr-x2-64',
     'load': _generic_torch_loader('caidas/swin2SR-classical-sr-x2-64')},
    {'name': 'swin2sr-classical-sr-x4', 'repo_id': 'caidas/swin2SR-classical-sr-x4-64',
     'load': _generic_torch_loader('caidas/swin2SR-classical-sr-x4-64')},
    {'name': 'swin2sr-real-sr-x4',      'repo_id': 'caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr',
     'load': _generic_torch_loader('caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr')},
    {'name': 'swin2sr-compressed',      'repo_id': 'caidas/swin2SR-compressed-sr-x4-48',
     'load': _generic_torch_loader('caidas/swin2SR-compressed-sr-x4-48')},
    {'name': 'swin2sr-lightweight',     'repo_id': 'caidas/swin2SR-lightweight-x2-64',
     'load': _generic_torch_loader('caidas/swin2SR-lightweight-x2-64')},
    # Generic image-to-image pipelines as fallback baselines:
    {'name': 'instruct-pix2pix-denoise','repo_id': 'timbrooks/instruct-pix2pix',
     'load': _hf_pipeline_loader('timbrooks/instruct-pix2pix')},
    {'name': 'sd-x4-upscaler',          'repo_id': 'stabilityai/stable-diffusion-x4-upscaler',
     'load': _hf_pipeline_loader('stabilityai/stable-diffusion-x4-upscaler')},
    {'name': 'ldm-super-resolution',    'repo_id': 'CompVis/ldm-super-resolution-4x-openimages',
     'load': _hf_pipeline_loader('CompVis/ldm-super-resolution-4x-openimages')},
    {'name': 'real-esrgan',             'repo_id': 'ai-forever/Real-ESRGAN',
     'load': _hf_pipeline_loader('ai-forever/Real-ESRGAN')},
    {'name': 'restormer',               'repo_id': 'caidas/swin2SR-color-jpeg-cer-x1-48',
     'load': _generic_torch_loader('caidas/swin2SR-color-jpeg-cer-x1-48')},
]


def load_inputs(gt_root: Path, sample_name: str) -> dict:
    """Find the noisy input under any image_*/ folder."""
    for sub in gt_root.iterdir():
        if not (sub.is_dir() and sub.name.startswith('image_')):
            continue
        for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'):
            candidate = sub / f'{sample_name}{ext}'
            if candidate.exists():
                return {'noisy': Image.open(candidate).convert('RGB')}
    raise FileNotFoundError(f"No input image for sample {sample_name!r}")


def predictor_fn(spec, model, processor, inputs) -> dict:
    """Run the model on the noisy input and return the mean-intensity
    scalar of the denoised output as a stand-in. REPLACE with full
    image upload + PSNR/SSIM metric on the LB side."""
    noisy = inputs['noisy']
    if processor is None:  # pipeline path
        out = model(noisy)
        if isinstance(out, list) and out:
            out_img = out[0].get('image', out[0])
        else:
            out_img = out
        arr = np.asarray(out_img.convert('RGB') if hasattr(out_img, 'convert') else out_img)
    else:
        import torch
        enc = processor(images=noisy, return_tensors='pt')
        with torch.no_grad():
            out = model(**enc)
        arr = out.reconstruction.squeeze().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
    return {'image_pred': float(np.asarray(arr).mean())}
