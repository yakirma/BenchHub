#!/usr/bin/env python
"""Generic N-way multiple-choice LLM board builder — the curated benchmark
addition hook for the weekly growth job. Generalizes build_mcq.py to ANY option
count (2..26) via a per-dataset parser, scored by `mcq_accuracy` (first
standalone A-Z letter vs the gold letter). Same pinned-protocol recipe: the
exact prompt is baked into the input, zero-shot.

Every spec here is hand-vetted (schema + correct split with VISIBLE gold), so
the boards stay trustworthy — never arbitrary auto-discovery.

    BENCHHUB_DATA_DIR=$HOME/.dtofbenchmarking \
        ~/benchhub/.venv/bin/python scripts/build_benchmark.py <key> [n]

Keys: winogrande commonsenseqa mmlu_pro sciq medmcqa
"""
import os
import sys
import json
import string
import tempfile
import shutil
import hashlib
from pathlib import Path

sys.path.insert(0, '/home/ymatri/Git/BenchHub')
os.environ.setdefault('BENCHHUB_DATA_DIR', os.path.expanduser('~/.dtofbenchmarking'))
os.environ['BENCHHUB_AUTO_MIGRATE'] = '0'

LETTERS = string.ascii_uppercase
KEY = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9

MCQ_ACCURACY = '''
def mcq_accuracy(gt, pred):
    """N-way MCQ letter exact-match (higher is better). Extracts the first
    standalone A-Z letter from the model's generation and compares it to the
    gold option letter; 1.0 on match else 0.0. Works for any option count."""
    import re
    gold = str(gt).strip().upper()[:1]
    m = re.search(r'\\b([A-Z])\\b', str(pred).upper())
    return 1.0 if (m and m.group(1) == gold) else 0.0
'''


# --- per-dataset parsers: yield (stem_text, [choices], gold_idx) ---
def p_winogrande(df):
    for d in df.to_dict('records'):
        a = str(d['answer']).strip()
        if a not in ('1', '2'):
            continue
        yield str(d['sentence']), [str(d['option1']), str(d['option2'])], int(a) - 1


def p_commonsenseqa(df):
    for d in df.to_dict('records'):
        ch = d['choices']
        texts = list(ch['text'])
        labels = [str(x) for x in ch['label']]
        ak = str(d['answerKey']).strip()
        if not ak or ak not in labels or len(texts) != 5:
            continue
        yield str(d['question']), [str(t) for t in texts], labels.index(ak)


def p_mmlu_pro(df):
    for d in df.to_dict('records'):
        opts = [str(o) for o in list(d['options'])]
        gi = d.get('answer_index')
        if gi is None or int(gi) < 0 or int(gi) >= len(opts) or len(opts) < 2:
            continue
        yield str(d['question']), opts, int(gi)


def p_sciq(df):
    for d in df.to_dict('records'):
        correct = str(d['correct_answer'])
        opts = [correct, str(d['distractor1']), str(d['distractor2']), str(d['distractor3'])]
        # deterministic per-question shuffle so the gold isn't always option A
        h = int(hashlib.md5(str(d['question']).encode()).hexdigest(), 16)
        order = sorted(range(4), key=lambda i: (h >> (i * 4)) & 0xf)
        shuffled = [opts[i] for i in order]
        yield str(d['question']), shuffled, shuffled.index(correct)


def p_medmcqa(df):
    for d in df.to_dict('records'):
        cop = d.get('cop')
        if cop is None or int(cop) < 0 or int(cop) > 3:
            continue
        opts = [str(d['opa']), str(d['opb']), str(d['opc']), str(d['opd'])]
        yield str(d['question']), opts, int(cop)


def p_race(df):
    for d in df.to_dict('records'):
        opts = [str(o) for o in list(d['options'])]
        ak = str(d['answer']).strip().upper()
        if len(opts) != 4 or ak not in 'ABCD':
            continue
        stem = f"{d['article']}\n\nQuestion: {d['question']}"
        yield stem, opts, 'ABCD'.index(ak)


def p_boolq(df):
    for d in df.to_dict('records'):
        a = d.get('answer')
        if a is None:
            continue
        gold = 0 if bool(a) else 1                       # A=Yes(True), B=No(False)
        stem = f"{d['passage']}\n\nQuestion: {d['question']}"
        yield stem, ['Yes', 'No'], gold


def p_qasc(df):
    for d in df.to_dict('records'):
        ch = d['choices']
        texts = list(ch['text'])
        labels = [str(x) for x in ch['label']]
        ak = str(d['answerKey']).strip()
        if not ak or ak not in labels or len(texts) < 2:
            continue
        yield str(d['question']), [str(t) for t in texts], labels.index(ak)


def p_aqua(df):
    import re
    for d in df.to_dict('records'):
        opts = []
        for o in list(d['options']):
            m = re.match(r'^\s*([A-E])\)\s*(.*)$', str(o), re.S)   # strip the "A)" prefix
            opts.append(m.group(2) if m else str(o))
        c = str(d['correct']).strip().upper()
        if len(opts) != 5 or c not in 'ABCDE':
            continue
        yield str(d['question']), opts, 'ABCDE'.index(c)


def p_truthfulqa(df):
    for d in df.to_dict('records'):
        mc = d['mc1_targets']
        choices = list(mc['choices'])
        labels = [int(x) for x in mc['labels']]
        if 1 not in labels or len(choices) < 2 or len(choices) != len(labels):
            continue
        yield str(d['question']), [str(c) for c in choices], labels.index(1)


def p_copa(df):
    for d in df.to_dict('records'):
        lab = d.get('label')
        if lab is None or int(lab) not in (0, 1):     # validation has visible 0/1 gold (test is -1)
            continue
        q = str(d['question']).strip().lower()
        framing = 'What was the cause?' if q == 'cause' else 'What was the effect?'
        stem = f"{d['premise']}\n{framing}"
        yield stem, [str(d['choice1']), str(d['choice2'])], int(lab)


SPECS = {
    'winogrande': {
        'repo': 'allenai/winogrande',
        'parquet': 'winogrande_debiased/validation-00000-of-00001.parquet',
        'ds_name': 'Winogrande-validation', 'stem': 'Sentence',
        'source': 'https://huggingface.co/datasets/allenai/winogrande',
        'desc': 'Winogrande (Sakaguchi et al., 2019) — pronoun-resolution '
                'commonsense, 2-option (validation, debiased). Pinned zero-shot '
                'prompt; scored by letter exact match.',
        'instr': 'Fill the blank (_) in the sentence. Respond with only the '
                 'letter (A or B) of the option that best completes it.',
        'parse': p_winogrande},
    'commonsenseqa': {
        'repo': 'tau/commonsense_qa', 'parquet': 'data/validation-00000-of-00001.parquet',
        'ds_name': 'CommonsenseQA-validation', 'stem': 'Question',
        'source': 'https://huggingface.co/datasets/tau/commonsense_qa',
        'desc': 'CommonsenseQA (Talmor et al., 2019) — 5-option commonsense '
                'question answering (validation). Pinned zero-shot prompt; '
                'scored by letter exact match.',
        'instr': 'Answer the commonsense question. Respond with only the letter '
                 '(A, B, C, D, or E).',
        'parse': p_commonsenseqa},
    'mmlu_pro': {
        'repo': 'TIGER-Lab/MMLU-Pro', 'parquet': 'data/test-00000-of-00001.parquet',
        'ds_name': 'MMLU-Pro-test', 'stem': 'Question',
        'source': 'https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro',
        'desc': 'MMLU-Pro (Wang et al., 2024) — harder, 10-option multitask '
                'knowledge & reasoning (test). Pinned zero-shot prompt; scored '
                'by letter exact match.',
        'instr': 'Answer the multiple-choice question. Respond with only the '
                 'letter of the correct option.',
        'parse': p_mmlu_pro},
    'sciq': {
        'repo': 'allenai/sciq', 'parquet': 'data/test-00000-of-00001.parquet',
        'ds_name': 'SciQ-test', 'stem': 'Question',
        'source': 'https://huggingface.co/datasets/allenai/sciq',
        'desc': 'SciQ (Welbl et al., 2017) — crowdsourced science-exam '
                'questions, 4-option (test). Pinned zero-shot prompt; scored by '
                'letter exact match.',
        'instr': 'Answer the multiple-choice science question. Respond with '
                 'only the letter (A, B, C, or D).',
        'parse': p_sciq},
    'medmcqa': {
        'repo': 'openlifescienceai/medmcqa', 'parquet': 'data/validation-00000-of-00001.parquet',
        'ds_name': 'MedMCQA-validation', 'stem': 'Question',
        'source': 'https://huggingface.co/datasets/openlifescienceai/medmcqa',
        'desc': 'MedMCQA (Pal et al., 2022) — medical entrance-exam MCQs, '
                '4-option (validation). Pinned zero-shot prompt; scored by '
                'letter exact match.',
        'instr': 'Answer the multiple-choice medical question. Respond with '
                 'only the letter (A, B, C, or D).',
        'parse': p_medmcqa},
    # --- NLP/Reading Comprehension (new sub-category, widens coverage) ---
    'race': {
        'repo': 'ehovy/race', 'parquet': 'all/test-00000-of-00001.parquet',
        'ds_name': 'RACE-test', 'stem': 'Passage',
        'category': 'NLP/Reading Comprehension',
        'source': 'https://huggingface.co/datasets/ehovy/race',
        'desc': 'RACE (Lai et al., 2017) — English-exam reading comprehension, '
                '4-option, over a passage (test). Pinned zero-shot prompt; '
                'scored by letter exact match.',
        'instr': 'Read the passage and answer the question. Respond with only '
                 'the letter (A, B, C, or D).',
        'parse': p_race},
    'boolq': {
        'repo': 'google/boolq', 'parquet': 'data/validation-00000-of-00001.parquet',
        'ds_name': 'BoolQ-validation', 'stem': 'Passage',
        'category': 'NLP/Reading Comprehension',
        'source': 'https://huggingface.co/datasets/google/boolq',
        'desc': 'BoolQ (Clark et al., 2019) — yes/no reading-comprehension '
                'questions over a passage (validation). Pinned zero-shot '
                'prompt; scored by letter exact match.',
        'instr': 'Read the passage and answer the yes/no question. Respond with '
                 'only the letter (A for Yes, B for No).',
        'parse': p_boolq},
    # --- more new sub-categories (widen coverage) ---
    'qasc': {
        'repo': 'allenai/qasc', 'parquet': 'data/validation-00000-of-00001.parquet',
        'ds_name': 'QASC-validation', 'stem': 'Question',
        'category': 'NLP/Science QA',
        'source': 'https://huggingface.co/datasets/allenai/qasc',
        'desc': 'QASC (Khot et al., 2020) — multi-hop science QA, 8-option '
                '(validation). Pinned zero-shot prompt; scored by letter exact '
                'match.',
        'instr': 'Answer the multiple-choice science question. Respond with '
                 'only the letter of the correct option.',
        'parse': p_qasc},
    'aqua_rat': {
        'repo': 'deepmind/aqua_rat', 'parquet': 'raw/test-00000-of-00001.parquet',
        'ds_name': 'AQuA-RAT-test', 'stem': 'Problem',
        'category': 'NLP/Mathematical Reasoning',
        'source': 'https://huggingface.co/datasets/deepmind/aqua_rat',
        'desc': 'AQuA-RAT (Ling et al., 2017) — algebraic math word problems, '
                '5-option multiple choice (test). Pinned zero-shot prompt; '
                'scored by letter exact match.',
        'instr': 'Solve the math word problem. Respond with only the letter '
                 '(A, B, C, D, or E) of the correct option.',
        'parse': p_aqua},
    'truthfulqa': {
        'repo': 'truthfulqa/truthful_qa', 'parquet': 'multiple_choice/validation-00000-of-00001.parquet',
        'ds_name': 'TruthfulQA-MC1', 'stem': 'Question',
        'category': 'NLP/Truthfulness',
        'source': 'https://huggingface.co/datasets/truthfulqa/truthful_qa',
        'desc': 'TruthfulQA MC1 (Lin et al., 2022) — single-true-answer multiple '
                'choice testing truthfulness (validation). Pinned zero-shot '
                'prompt; scored by letter exact match.',
        'instr': 'Answer the question truthfully. Respond with only the letter '
                 'of the single correct option.',
        'parse': p_truthfulqa},
    # --- NLP/Commonsense Reasoning (pairs with HellaSwag → forms a ranking) ---
    'copa': {
        'repo': 'aps/super_glue', 'parquet': 'copa/validation-00000-of-00001.parquet',
        'ds_name': 'COPA-validation', 'stem': 'Situation',
        'category': 'NLP/Commonsense Reasoning',
        'source': 'https://huggingface.co/datasets/aps/super_glue',
        'desc': 'COPA (Roemmele et al., 2011; SuperGLUE) — Choice of Plausible '
                'Alternatives, 2-option causal commonsense reasoning '
                '(validation). Pinned zero-shot prompt; scored by letter exact '
                'match.',
        'instr': 'Choose the more plausible option. Respond with only the '
                 'letter (A or B).',
        'parse': p_copa},
}
DEFAULT_CATEGORY = 'NLP/Reasoning & Knowledge'   # most boards join the combined LLM ranking


def build_staging(staging, spec):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    for f in ('prompt', 'answer'):
        (staging / f).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(hf_hub_download(spec['repo'], spec['parquet'], repo_type='dataset'))
    out = []
    for i, (stem, choices, gold) in enumerate(spec['parse'](df)):
        if i >= N:
            break
        lettered = '\n'.join(f'{LETTERS[j]}. {c}' for j, c in enumerate(choices))
        name = f'q_{i:05d}'
        (staging / 'prompt' / f'{name}.txt').write_text(
            f"{spec['instr']}\n\n{spec['stem']}: {stem}\n{lettered}\nAnswer:")
        (staging / 'answer' / f'{name}.txt').write_text(LETTERS[gold])
        out.append(name)
    manifest = {
        'name': spec['ds_name'], 'version': '1.0',
        'fields': [
            {'name': 'prompt', 'kind': 'text', 'role': 'input', 'params': {}},
            {'name': 'answer', 'kind': 'text', 'role': 'gt', 'params': {}},
        ],
        'samples': out,
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return out


def main():
    spec = SPECS[KEY]
    category = spec.get('category', DEFAULT_CATEGORY)
    import app as A
    from app import (db, Dataset, Sample, CustomField, DatasetField,
                     Leaderboard, LeaderboardMetric, GlobalMetric)
    from benchhub.manifest import import_typed_dataset
    with A.app.app_context():
        ds = Dataset.query.filter_by(name=spec['ds_name']).first()
        if ds is None:
            staging = Path(tempfile.mkdtemp(prefix=f'{KEY}_'))
            try:
                n = len(build_staging(staging, spec))
                ds_id, summary = import_typed_dataset(
                    staging, db_session=db.session, Dataset=Dataset, Sample=Sample,
                    CustomField=CustomField, DatasetField=DatasetField,
                    upload_folder=A.app.config['UPLOAD_FOLDER'],
                    owner_user_id=2, visibility='public', preview_only=False)
                db.session.commit()
                ds = Dataset.query.get(ds_id)
                ds.category = category
                ds.source_url = spec['source']
                ds.source_kind = 'local-llm'
                ds.card_description = spec['desc']
                db.session.commit()
                print(f'imported dataset id={ds_id}: {summary["samples"]} samples ({n} built)')
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        gm = GlobalMetric.query.filter_by(name='mcq_accuracy').first()
        if gm is None:
            gm = GlobalMetric(name='mcq_accuracy', python_code=MCQ_ACCURACY.strip(),
                              owner_user_id=2, visibility='public', is_aggregated=False)
            db.session.add(gm)
            db.session.commit()
        lb_name = f"{spec['ds_name']}_benchmark"
        lb = Leaderboard.query.filter_by(name=lb_name).first()
        if lb is None:
            lb = Leaderboard(
                name=lb_name, owner_user_id=2, visibility='public', category=category,
                required_pred_fields_json=json.dumps(
                    [{"name": "answer_pred", "kind": "text", "params": {}, "role": "pred"}]),
                field_roles_json=json.dumps({'prompt': 'input', 'answer': 'gt'}),
                summary_metrics='')
            lb.datasets.append(ds)
            db.session.add(lb)
            db.session.commit()
        lm = LeaderboardMetric.query.filter_by(leaderboard_id=lb.id, global_metric_id=gm.id).first()
        if lm is None:
            lm = LeaderboardMetric(
                leaderboard_id=lb.id, global_metric_id=gm.id, target_name='Accuracy',
                arg_mappings=json.dumps({"gt": "gt_answer", "pred": "sub_answer_pred"}),
                pooling_type='mean', sort_direction='higher_is_better')
            db.session.add(lm)
            db.session.commit()
        lb.summary_metrics = f'lm_{lm.id}'
        db.session.commit()
        print(f'BENCHMARK_BUILT {KEY} lb_id={lb.id} ds={ds.id} name={lb_name}')


if __name__ == '__main__':
    main()
