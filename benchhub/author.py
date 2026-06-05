"""Local authoring kit — iterate on a metric or visualization against your
own sample data BEFORE uploading it with ``client.create_metric`` /
``client.create_visualization``. Pure in-process execution (no server, no
sandbox) so you get instant values + real tracebacks while developing.

    import benchhub as bh

    def my_iou(gt: bh.Mask, pred: bh.Mask):
        g, p = gt.array, pred.array
        inter = ((g == 1) & (p == 1)).sum()
        union = ((g == 1) | (p == 1)).sum()
        return float(inter / union) if union else 1.0

    # try it on one sample
    print(bh.author.test_metric(my_iou, gt=gt_mask, pred=pred_mask))
    # happy? ship it (input_kinds/roles auto-derive from the annotations)
    client.create_metric("my_iou", my_iou)

Visualizations are the same, but return a PIL.Image:

    def confusion(gt, pred):
        ...  # returns Image.Image
    img = bh.author.test_visualization(confusion, gt=gts, pred=preds)
    img.show()
    client.create_visualization("confusion", confusion)
"""
from __future__ import annotations


def test_metric(fn, **kwargs):
    """Run a metric function on one sample's kwargs and return its value as
    a float — exactly what the server stores. Re-raises whatever the
    function raises so you see the traceback before uploading."""
    return float(fn(**kwargs))


def test_metric_batch(fn, samples):
    """Run `fn` over an iterable of kwargs-dicts and return the list of
    per-sample values (mirrors how the server pools a per-sample metric)."""
    return [test_metric(fn, **s) for s in samples]


def test_visualization(fn, *, save_to=None, **kwargs):
    """Run a visualization function and return the ``PIL.Image`` it
    produces (optionally saving to `save_to`). Raises if it returns
    anything other than a PIL.Image — the same contract the server
    enforces."""
    from PIL import Image
    img = fn(**kwargs)
    if not isinstance(img, Image.Image):
        raise TypeError(
            f"visualization must return a PIL.Image, got {type(img).__name__}")
    if save_to:
        img.save(save_to)
    return img
