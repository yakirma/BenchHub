"""Baseline-runner stub for text-classification leaderboards (sentiment,
topic, NLI). Returns the predicted class index → `label_pred`.

For QA / summarization / translation you'll need a different head class
(`AutoModelForQuestionAnswering` etc.) and a different output schema —
this stub covers the simplest LB shape (text in, label index out).
"""
from pathlib import Path


def _hf_classifier_loader(repo_id):
    def _load():
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
        )
        return (
            AutoModelForSequenceClassification.from_pretrained(repo_id),
            AutoTokenizer.from_pretrained(repo_id),
        )
    return _load


MODELS = [
    # Sentiment
    {'name': 'distilbert-sst2',           'repo_id': 'distilbert-base-uncased-finetuned-sst-2-english',
     'load': _hf_classifier_loader('distilbert-base-uncased-finetuned-sst-2-english')},
    {'name': 'roberta-sst2',              'repo_id': 'textattack/roberta-base-SST-2',
     'load': _hf_classifier_loader('textattack/roberta-base-SST-2')},
    {'name': 'twitter-roberta-sentiment', 'repo_id': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
     'load': _hf_classifier_loader('cardiffnlp/twitter-roberta-base-sentiment-latest')},
    # Topic / NLI
    {'name': 'bart-mnli',                 'repo_id': 'facebook/bart-large-mnli',
     'load': _hf_classifier_loader('facebook/bart-large-mnli')},
    {'name': 'deberta-mnli',              'repo_id': 'microsoft/deberta-large-mnli',
     'load': _hf_classifier_loader('microsoft/deberta-large-mnli')},
    {'name': 'distilbert-emotion',        'repo_id': 'bhadresh-savani/distilbert-base-uncased-emotion',
     'load': _hf_classifier_loader('bhadresh-savani/distilbert-base-uncased-emotion')},
    {'name': 'finbert',                   'repo_id': 'ProsusAI/finbert',
     'load': _hf_classifier_loader('ProsusAI/finbert')},
    {'name': 'distilbert-toxic',          'repo_id': 'unitary/toxic-bert',
     'load': _hf_classifier_loader('unitary/toxic-bert')},
    {'name': 'roberta-go-emotions',       'repo_id': 'SamLowe/roberta-base-go_emotions',
     'load': _hf_classifier_loader('SamLowe/roberta-base-go_emotions')},
    {'name': 'distilbert-ag-news',        'repo_id': 'textattack/distilbert-base-uncased-ag-news',
     'load': _hf_classifier_loader('textattack/distilbert-base-uncased-ag-news')},
]


def load_inputs(gt_root: Path, sample_name: str) -> dict:
    """Look for the sample's text under any text-typed folder. The HF
    auto-importer puts free-form text in a bare-name folder named after
    the source column (e.g. `text/`, `sentence/`, `premise/`)."""
    candidates = []
    for sub in gt_root.iterdir():
        if not sub.is_dir() or sub.name.startswith(('image_', 'raw_', 'hist_')):
            continue
        candidate = sub / f'{sample_name}.txt'
        if candidate.exists():
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"No text input for sample {sample_name!r}")
    # If multiple text columns exist (NLI: premise + hypothesis), join.
    return {'text': '\n'.join(c.read_text() for c in candidates)}


def predictor_fn(spec, model, processor, inputs) -> dict:
    import torch
    enc = processor(inputs['text'], return_tensors='pt',
                    truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**enc).logits
    return {'label_pred': int(logits.argmax(dim=-1).item())}
