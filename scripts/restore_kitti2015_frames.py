#!/usr/bin/env python
"""Restore the missing KITTI-2015-flow input frames (frame1/frame2).

The dataset's GT (flow_u/flow_v depth) is intact, but the two RGB input frames
were lost from disk, so the catalog card 404'd. The CustomField rows still point
at datasets/317/frame1|frame2/<name>.png — we just re-extract those PNGs from
the cached KITTI zip (training/image_2/<sid>_10.png -> frame1, _11.png -> frame2;
sample name is k15_<sid>) and write them back to the exact paths.

Read-only on the DB (sqlite3), file-writes only — does NOT import app.
"""
import os
import re
import sqlite3
import zipfile

DB = os.path.expanduser('~/.dtofbenchmarking/database.db')
UP = os.path.expanduser('~/.dtofbenchmarking/uploads')
ZIP = os.path.expanduser('~/.dtofbenchmarking/_cache_data_scene_flow.zip')
DS_ID = 317
IMG_DIR = 'image_2'   # KITTI-2015 frames

con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
rows = con.execute(
    "SELECT cf.name, cf.value_text FROM custom_field cf "
    "JOIN sample s ON cf.sample_id = s.id "
    "WHERE s.dataset_id=? AND cf.submission_id IS NULL AND cf.data_type='image'",
    (DS_ID,)).fetchall()
con.close()

z = zipfile.ZipFile(ZIP)
zip_names = set(z.namelist())

restored = missing_zip = bad = already = 0
for field_name, vt in rows:
    if not vt:
        bad += 1
        continue
    base = os.path.basename(vt)                       # k15_000033.png
    m = re.match(r'k15_(\d+)\.png$', base)
    if not m:
        bad += 1
        continue
    sid = m.group(1)
    suffix = '_10' if field_name == 'frame1' else '_11' if field_name == 'frame2' else None
    if suffix is None:
        bad += 1
        continue
    src = f'training/{IMG_DIR}/{sid}{suffix}.png'
    if src not in zip_names:
        print(f'  MISSING in zip: {src}')
        missing_zip += 1
        continue
    dst = os.path.join(UP, vt)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        already += 1
        continue
    with open(dst, 'wb') as fh:
        fh.write(z.read(src))
    restored += 1

print(f'KITTI-2015 frames: restored={restored} already_present={already} '
      f'missing_in_zip={missing_zip} bad_rows={bad} (of {len(rows)} image CFs)')
