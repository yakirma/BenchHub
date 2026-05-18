#!/usr/bin/env python3
"""
Create one Leaderboard for Semantic Segmentation on NYUv2, attached
directly to the HuggingFace repo `tanganke/nyuv2` (Parquet, val split,
654 examples = canonical Silberman test split).

No data download — purely DB seeding. The existing
_iter_hf_attachment_samples flow streams the parquet at eval time
and resolves the `val` split via _HF_SPLIT_PREFERENCE / persistence.

Schema on tanganke/nyuv2:
    image         array3_d float32 [3, 288, 384]
    segmentation  array2_d int64   [288, 384]      <- our GT
    depth         array3_d float32 [1, 288, 384]
    normal        array3_d float32 [3, 288, 384]
    noise         array3_d float32 [1, 288, 384]

We only consume `image` (input) and `segmentation` (gt). Depth +
normals stay unused on this LB — they're scoring targets for the
existing NYUv2 depth LB (id=3, different HF mirror) and a potential
future normals LB.

Metric: mIoU (existing GlobalMetric, kinds=["mask","mask"]).

Internal — run once, idempotent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))


LB_NAME = 'Semantic Segmentation on NYUv2'
CATEGORY = 'Vision/Image Segmentation'
HF_REPO = 'tanganke/nyuv2'
HF_SPLIT = 'val'  # 654-image canonical test split.

MAPPING = [
    {'column': 'image',
     'target_kind': 'image',
     'target_field': 'image_image',
     'reason': 'RGB input given to the segmentation model',
     'role': 'input'},
    {'column': 'segmentation',
     'target_kind': 'mask',
     'target_field': 'segmentation',
     'reason': 'Per-pixel class label (40-class NYUv2 taxonomy)',
     'role': 'gt'},
    # depth + normal + noise are present but we skip them on this LB
    # — depth has its own LB on a different mirror, normals/noise
    # aren't a segmentation target.
    {'column': 'depth', 'target_kind': 'skip',
     'target_field': '', 'role': 'gt'},
    {'column': 'normal', 'target_kind': 'skip',
     'target_field': '', 'role': 'gt'},
    {'column': 'noise', 'target_kind': 'skip',
     'target_field': '', 'role': 'gt'},
]


def main() -> int:
    import app as app_mod
    with app_mod.app.app_context():
        existing = app_mod.Leaderboard.query.filter_by(name=LB_NAME).first()
        if existing:
            print(f'LB "{LB_NAME}" already exists (id={existing.id}) — '
                  f'no-op.')
            return 0

        miou = app_mod.GlobalMetric.query.filter_by(name='miou').first()
        if miou is None:
            print('GlobalMetric "miou" not found — abort.')
            return 1

        lb = app_mod.Leaderboard(
            name=LB_NAME,
            category=CATEGORY,
            summary_metrics='',
            visibility='public',
            owner_user_id=None,
            canonical_for_repo=HF_REPO,
        )
        lb.required_pred_fields_json = json.dumps([
            {
                'name': 'segmentation_pred',
                'kind': 'mask',
                'description': 'Per-pixel class prediction over the 40-class '
                               'NYUv2 taxonomy. Same H×W as the input image.',
            },
        ])
        app_mod.db.session.add(lb)
        app_mod.db.session.flush()

        att = app_mod.Attachment(
            leaderboard_id=lb.id,
            hf_repo_id=HF_REPO,
            hf_split=HF_SPLIT,
            hf_mapping_json=json.dumps(MAPPING),
            role='primary',
        )
        app_mod.db.session.add(att)

        lm = app_mod.LeaderboardMetric(
            leaderboard_id=lb.id,
            global_metric_id=miou.id,
            arg_mappings=json.dumps({
                'gt': 'gt_segmentation',
                'pred': 'sub_segmentation_pred',
            }),
            target_name='mIoU',
            pooling_type='mean',
            sort_direction='higher_is_better',
        )
        app_mod.db.session.add(lm)
        app_mod.db.session.flush()
        lb.summary_metrics = f'lm_{lm.id}'
        app_mod.db.session.commit()
        print(f'-> created LB id={lb.id} ({LB_NAME})')
        print(f'   attached to HF repo {HF_REPO}, split={HF_SPLIT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
