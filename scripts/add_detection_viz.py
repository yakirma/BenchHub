#!/usr/bin/env python
"""Add a COCO-style detection-overlay visualization (GT + predictions) to the
COCO detection board (and any LB sharing the detection JSON format).

Creates one `detection_overlay` GlobalVisualization that draws boxes + labels
(+ scores) on the image, robust to BOTH:
  - {"boxes": [[x1,y1,x2,y2],...], "labels": [name,...], "scores": [...]}  (our LBs)
  - [{"bbox": [x,y,w,h], "category_name"/"category_id": ...}, ...]          (COCO)
then binds it twice per LB: once on the GT boxes, once on the prediction.

Usage:
    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/add_detection_viz.py <lb_id> [<lb_id> ...] \
            --gt <gt_field> --pred <pred_field>
"""
import os
import sys
import json

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

VIZ_CODE = r'''def detection_overlay(image, detections):
    """Draw detection boxes (+labels, +scores) on the image. Accepts our
    {boxes:[[x1,y1,x2,y2]], labels:[name], scores?:[...]} dict OR a COCO
    list[dict] ({bbox:[x,y,w,h], category_name/category_id, score?})."""
    import os, json, colorsys
    import numpy as np
    from PIL import Image as PIL_Image, ImageDraw, ImageFont

    # --- image -> RGBA ---
    im = None
    arr = getattr(image, 'array', None)
    if arr is not None:
        im = PIL_Image.fromarray(np.asarray(arr).astype('uint8')).convert('RGBA')
    elif isinstance(image, PIL_Image.Image):
        im = image.convert('RGBA')
    elif isinstance(image, str):
        p = image
        if not os.path.isabs(p):
            dd = os.environ.get('BENCHHUB_DATA_DIR') or os.path.expanduser('~/.dtofbenchmarking')
            p = os.path.join(dd, 'uploads', p)
        try:
            im = PIL_Image.open(p).convert('RGBA')
        except Exception:
            im = None
    elif image is not None:
        try:
            im = PIL_Image.fromarray(np.asarray(image).astype('uint8')).convert('RGBA')
        except Exception:
            im = None
    if im is None:
        return PIL_Image.new('RGB', (320, 240), (40, 40, 40))
    W, H = im.size

    # --- detections -> list of (x1,y1,x2,y2,label,score) ---
    d = detections
    if hasattr(d, 'data'):
        d = d.data
    if hasattr(d, 'value'):
        d = d.value
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except Exception:
            d = []
    dets = []
    if isinstance(d, dict) and 'boxes' in d:
        boxes = d.get('boxes') or []
        labels = d.get('labels') or []
        scores = d.get('scores') or []
        for i, b in enumerate(boxes):
            if not b or len(b) != 4:
                continue
            lab = labels[i] if i < len(labels) else '?'
            sc = scores[i] if i < len(scores) else None
            dets.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(lab), sc))
    elif isinstance(d, list):
        for det in d:
            if not isinstance(det, dict):
                continue
            bb = det.get('bbox') or det.get('box')
            if not bb or len(bb) != 4:
                continue
            x, y, w, h = [float(t) for t in bb]
            lab = det.get('category_name') or det.get('label') or det.get('category_id') or '?'
            dets.append((x, y, x + w, y + h, str(lab), det.get('score')))

    overlay = PIL_Image.new('RGBA', im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', max(12, min(22, H // 28)))
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    def hue(s):
        v = (sum(ord(c) for c in s) * 137) % 360 / 360.0   # deterministic per label
        r, g, b = colorsys.hsv_to_rgb(v, 0.85, 1.0)
        return (int(r * 255), int(g * 255), int(b * 255))

    def tsize(t):
        if font is None:
            return (len(t) * 6, 11)
        try:
            x0, y0, x1, y1 = draw.textbbox((0, 0), t, font=font)
            return (x1 - x0, y1 - y0)
        except Exception:
            return (len(t) * 6, 12)

    for (x1, y1, x2, y2, lab, sc) in dets:
        rgb = hue(lab)
        stroke = (rgb[0], rgb[1], rgb[2], 235)
        draw.rectangle([x1, y1, x2, y2], outline=stroke, width=2)
        txt = lab if sc is None else '%s %.2f' % (lab, float(sc))
        tw, th = tsize(txt)
        pad = 3
        ly = y1 - th - 2 * pad
        if ly < 0:
            ly = y1 + 1
        draw.rectangle([x1, ly, x1 + tw + 2 * pad, ly + th + 2 * pad], fill=stroke)
        if font is not None:
            try:
                draw.text((x1 + pad, ly + pad), txt, fill=(255, 255, 255, 255), font=font)
            except Exception:
                pass
    return PIL_Image.alpha_composite(im, overlay).convert('RGB')
'''


def main():
    args = sys.argv[1:]
    lb_ids = [int(a) for a in args if a.isdigit()]
    gt_field = 'objects'
    pred_field = 'detections_pred'
    if '--gt' in args:
        gt_field = args[args.index('--gt') + 1]
    if '--pred' in args:
        pred_field = args[args.index('--pred') + 1]
    if not lb_ids:
        print('usage: add_detection_viz.py <lb_id> [...] [--gt objects] [--pred detections_pred]')
        return 2

    import app as A
    from app import db, GlobalVisualization, LeaderboardVisualization
    with A.app.app_context():
        gv = GlobalVisualization.query.filter_by(name='detection_overlay').first()
        if gv is None:
            gv = GlobalVisualization(
                name='detection_overlay',
                description='Draw detection boxes + labels (+scores) on the image.',
                python_code=VIZ_CODE, is_aggregated=0, accepts_aggregated_inputs=0,
                input_kinds=json.dumps(['image', 'json']),
                owner_user_id=None, visibility='public')
            db.session.add(gv); db.session.commit()
            print(f'created detection_overlay viz id={gv.id}')
        else:
            gv.python_code = VIZ_CODE; db.session.commit()
            print(f'updated detection_overlay viz id={gv.id}')

        for lb_id in lb_ids:
            for target, mapping in [
                ('GT detections', {'image': 'gt_image', 'detections': f'gt_{gt_field}'}),
                ('Predicted detections', {'image': 'gt_image', 'detections': f'sub_{pred_field}'}),
            ]:
                exists = LeaderboardVisualization.query.filter_by(
                    leaderboard_id=lb_id, global_visualization_id=gv.id,
                    target_name=target).first()
                if exists:
                    exists.arg_mappings = json.dumps(mapping); db.session.commit()
                    print(f'  lb{lb_id}: updated "{target}"')
                    continue
                lv = LeaderboardVisualization(
                    leaderboard_id=lb_id, global_visualization_id=gv.id,
                    arg_mappings=json.dumps(mapping), target_name=target)
                db.session.add(lv); db.session.commit()
                print(f'  lb{lb_id}: bound "{target}" (lv={lv.id})')
        print('DETECTION_VIZ_DONE')


if __name__ == '__main__':
    raise SystemExit(main())
