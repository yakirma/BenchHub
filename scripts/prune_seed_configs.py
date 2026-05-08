"""Prune seed_data/*.json down to entries whose HF schema BenchHub
can actually read. Run on a host with HF network access (e.g. the
fly machine):

    python scripts/prune_seed_configs.py

Hits _hf_fetch_features for each repo in each config; rewrites the
JSON in place, keeping only entries that returned a non-empty
feature dict. Original is backed up next to it as `<name>.json.bak`.
"""
import json
import os
import shutil
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, _hf_fetch_features


CONFIGS = ['depth.json', 'segmentation.json', 'llm.json', 'denoising.json']


def main():
    seed_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'seed_data')
    )
    with app.app_context():
        for fname in CONFIGS:
            path = os.path.join(seed_dir, fname)
            if not os.path.exists(path):
                print(f"  [skip] {fname} not found")
                continue
            with open(path) as f:
                cfg = json.load(f)
            kept, dropped = [], []
            for entry in cfg.get('datasets', []):
                repo = entry['hf_repo_id']
                try:
                    feats = _hf_fetch_features(repo)
                except Exception as e:
                    feats = {}
                    note = f"raised {type(e).__name__}"
                else:
                    note = ''
                if feats:
                    kept.append(entry)
                    print(f"  KEEP  {repo}")
                else:
                    dropped.append((repo, note or 'no features'))
                    print(f"  DROP  {repo}  -- {note or 'no features'}")
            shutil.copy2(path, path + '.bak')
            cfg['datasets'] = kept
            cfg.setdefault('_pruned_dropped', [
                {'hf_repo_id': r, 'reason': n} for r, n in dropped
            ])
            with open(path, 'w') as f:
                json.dump(cfg, f, indent=2)
                f.write('\n')
            print(f"=> {fname}: kept {len(kept)} / {len(kept) + len(dropped)} "
                  f"(backup at {fname}.bak)\n")


if __name__ == '__main__':
    main()
