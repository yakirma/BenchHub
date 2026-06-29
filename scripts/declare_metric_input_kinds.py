#!/usr/bin/env python
"""Declare input_kinds on the remaining typeless metrics so the whole metric
catalog participates in the typed contract (the /metrics 'Accepts:' row + future
kind validation), and make the 3 that used bare str() typed-safe first.

Audit (scratchpad/audit_all.py) found 14 bound metrics + 1 orphan with NULL
input_kinds. Kinds are the per-arg field kinds the metrics are ACTUALLY bound
with across all their boards (consistent for every one). 11 already contained
an unwrap()/arr() helper that handles bh.<Kind>, so declaring is a safe no-op
(verified identical on real samples). The 3 letter/number metrics (mmlu, mcq,
gsm8k) read gt/pred via bare str() — str(bh.Text(..)) == 'Text({})', which would
silently break them (mcq=11 boards, mmlu=5) — so they get a _raw() unwrap here
BEFORE declaring. Verified: patched+typed == current primitive on every sampled
board (scratchpad/verify14.py); the unpatched versions provably broke
(scratchpad/neg2.py). Existing MetricResults stay valid (identical scoring).

    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking ~/benchhub/.venv/bin/python scripts/declare_metric_input_kinds.py
"""
import os, sys, json
sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

# metric name -> input_kinds (signature arg order)
KINDS = {
    'map50':         ['json', 'json'],
    'squad_f1':      ['json', 'text'],
    'squad_em':      ['json', 'text'],
    'ner_f1':        ['json', 'json'],
    'rouge1_f':      ['text', 'text'],
    'rouge2_f':      ['text', 'text'],
    'rougeL_f':      ['text', 'text'],
    'bleu':          ['text', 'text'],
    'tapvid_aj':     ['json', 'json', 'json', 'json'],
    'tapvid_delta':  ['json', 'json', 'json', 'json'],
    'tapvid_oa':     ['json', 'json', 'json', 'json'],
    'gsm8k_accuracy': ['text', 'text'],
    'mmlu_accuracy': ['text', 'text'],
    'mcq_accuracy':  ['text', 'text'],
    'lstq':          ['point_panoptic', 'point_panoptic'],   # orphaned 4D agg metric
}

# Typed-safe rewrites for the 3 metrics that used bare str() (added a _raw()
# unwrap so a bh.Text/bh.Json arg is normalised to its raw value).
NEW_CODE = {}
NEW_CODE['gsm8k_accuracy'] = '''def gsm8k_accuracy(gt, pred):
    """GSM8K final-answer exact match (higher is better). Pulls the last number
    out of the model's generation and compares it to the gold answer; 1.0 on a
    numeric match, else 0.0. Tolerant of $, thousands-commas, and trailing
    periods so formatting doesn't cost a correct answer."""
    import re

    def _raw(x):
        if hasattr(x, 'text'): return x.text
        if hasattr(x, 'data'): return x.data
        if hasattr(x, 'value'): return x.value
        return x

    def last_number(s):
        s = str(_raw(s)).replace('$', '').replace(',', '')
        marked = re.findall(r'####\\s*(-?\\d+\\.?\\d*)', s)
        cands = marked if marked else re.findall(r'-?\\d+\\.?\\d*', s)
        if not cands:
            return None
        x = cands[-1].rstrip('.')
        try:
            return float(x)
        except ValueError:
            return None

    g = last_number(gt)
    p = last_number(pred)
    if g is None or p is None:
        return 0.0
    return 1.0 if abs(g - p) < 1e-6 else 0.0
'''
NEW_CODE['mmlu_accuracy'] = '''def mmlu_accuracy(gt, pred):
    """MMLU letter exact-match (higher is better). Extracts the chosen option
    letter from the model's generation and compares it to the gold letter;
    1.0 on match, else 0.0."""
    import re

    def _raw(x):
        if hasattr(x, 'text'): return x.text
        if hasattr(x, 'data'): return x.data
        if hasattr(x, 'value'): return x.value
        return x

    gold = str(_raw(gt)).strip().upper()[:1]
    m = re.search(r'\\b([A-D])\\b', str(_raw(pred)).upper())
    return 1.0 if (m and m.group(1) == gold) else 0.0
'''
NEW_CODE['mcq_accuracy'] = '''def mcq_accuracy(gt, pred):
    """N-way MCQ letter exact-match (higher is better). Extracts the first
    standalone A-Z letter from the model's generation and compares it to the
    gold option letter; 1.0 on match else 0.0. Works for any option count."""
    import re

    def _raw(x):
        if hasattr(x, 'text'): return x.text
        if hasattr(x, 'data'): return x.data
        if hasattr(x, 'value'): return x.value
        return x

    gold = str(_raw(gt)).strip().upper()[:1]
    m = re.search(r'\\b([A-Z])\\b', str(_raw(pred)).upper())
    return 1.0 if (m and m.group(1) == gold) else 0.0
'''


def main():
    import app as A
    from app import db, GlobalMetric
    with A.app.app_context():
        for name, kinds in KINDS.items():
            gm = GlobalMetric.query.filter_by(name=name).first()
            if gm is None:
                print(f'  ! {name}: not found'); continue
            changed = []
            if name in NEW_CODE and gm.python_code != NEW_CODE[name]:
                gm.python_code = NEW_CODE[name]; changed.append('code')
            newik = json.dumps(kinds)
            if gm.input_kinds != newik:
                gm.input_kinds = newik; changed.append('input_kinds')
            print(f'  {name}: {("updated " + "+".join(changed)) if changed else "already current"} -> {kinds}')
        db.session.commit()
        print('DONE')


if __name__ == '__main__':
    main()
