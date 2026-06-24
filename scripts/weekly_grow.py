#!/usr/bin/env python
"""Weekly growth job — keeps the BenchHub LLM boards (and the HF Space mirror)
current so the leaderboard is the place that *already benchmarked* the model
people are searching for.

Each run:
  1. Discovers the LLM boards (input field `prompt`, pred field `answer_pred`).
  2. Picks a few NEW notable instruct models (HF trending, reputable orgs,
     size-capped, accessible, not already submitted) and benchmarks them on
     every LLM board — per (model,board) error-isolated so one bad model can't
     wedge the run.
  3. Builds the next benchmark from a CURATED queue (vetted MCQ datasets via
     scripts/build_mcq.py) and benchmarks a couple of baselines on it.
  4. Triggers the standings sync so the Space refreshes with the real new
     content (freshness bump = genuine new results, not churn).

Safety: only reputable orgs + a hard size cap are auto-benchmarked; new boards
come from a hand-vetted queue, never arbitrary auto-discovery — a broken public
board would undercut the whole "reliable scores" point.

Run:  ~/benchhub/.venv/bin/python scripts/weekly_grow.py [--dry-run] [--max-models N]
"""
import os
import sys
import time
import sqlite3
import subprocess

REPO = '/home/ymatri/Git/BenchHub'
BENCHHUB_PY = '/home/ymatri/benchhub/.venv/bin/python'
CLIENT_PY = os.path.expanduser('~/miniconda3/envs/BenchClient/bin/python')
SUBMIT = os.path.expanduser('~/Git/BenchClient/submit_llm.py')
BUILD_MCQ = os.path.join(REPO, 'scripts', 'build_mcq.py')
BASE_URL = os.environ.get('BENCHHUB_BASE_URL', 'http://127.0.0.1:6060')
DB = os.path.expanduser('~/.dtofbenchmarking/database.db')

# Reputable orgs we auto-benchmark from (keeps junk / mis-tagged / unsafe repos
# out of an unattended pipeline). Extend deliberately.
ALLOWED_ORGS = {
    'qwen', 'meta-llama', 'google', 'mistralai', 'microsoft', 'huggingfacetb',
    'deepseek-ai', 'allenai', 'ibm-granite', 'tiiuae', 'nvidia', 'stabilityai',
}
MAX_PARAMS_B = 9.0      # bf16 size cap the local GPU can run
DEFAULT_MAX_MODELS = 3  # new models per run

# Curated queue of NEXT benchmarks to add (keys must exist in build_mcq.py CONFIG).
# The job builds the first one whose dataset isn't already imported.
BENCHMARK_QUEUE = ['arc_easy', 'openbookqa']
# Baselines to seed onto a freshly-built board.
BASELINE_MODELS = ['Qwen/Qwen2.5-3B-Instruct', 'meta-llama/Llama-3.2-1B-Instruct']

DRY = '--dry-run' in sys.argv
if '--max-models' in sys.argv:
    DEFAULT_MAX_MODELS = int(sys.argv[sys.argv.index('--max-models') + 1])


def log(*a):
    print('[grow]', time.strftime('%H:%M:%S'), *a, flush=True)


def _db(readonly=True):
    uri = f'file:{DB}?mode=ro' if readonly else f'file:{DB}'
    con = sqlite3.connect(uri, uri=True)
    con.execute('PRAGMA busy_timeout=10000')
    con.row_factory = sqlite3.Row
    return con


def llm_boards():
    """(lb_id, name, max_new) for public boards taking prompt->answer_pred."""
    con = _db()
    out = []
    rows = con.execute(
        "SELECT id, name FROM leaderboard "
        "WHERE (visibility='public' OR owner_user_id IS NULL) "
        "AND required_pred_fields_json LIKE '%answer_pred%'").fetchall()
    for r in rows:
        # confirm a 'prompt' input field exists on the board's dataset samples
        # (excludes e.g. SQuAD, whose inputs are context+question, not prompt).
        has_prompt = con.execute(
            "SELECT 1 FROM custom_field cf JOIN sample s ON s.id=cf.sample_id "
            "JOIN leaderboard_datasets ld ON ld.dataset_id=s.dataset_id "
            "WHERE ld.leaderboard_id=? AND cf.name='prompt' LIMIT 1", (r['id'],)).fetchone()
        if not has_prompt:
            continue
        max_new = 512 if 'gsm8k' in r['name'].lower() else 8
        out.append((r['id'], r['name'], max_new))
    con.close()
    return out


def already_submitted(lb_id, model):
    con = _db()
    n = con.execute("SELECT COUNT(*) FROM submission WHERE leaderboard_id=? AND name=?",
                    (lb_id, model)).fetchone()[0]
    con.close()
    return n > 0


def sub_count(lb_id):
    con = _db()
    n = con.execute("SELECT COUNT(*) FROM submission WHERE leaderboard_id=? "
                    "AND is_archived=0", (lb_id,)).fetchone()[0]
    con.close()
    return n


def seed_sparse(boards):
    """Benchmark the baseline models onto under-populated boards (e.g. a
    freshly-built benchmark) so no board ships empty."""
    for b in boards:
        if sub_count(b[0]) < 3:
            for m in BASELINE_MODELS:
                benchmark(m, [b])


def _size_b(info):
    try:
        total = (info.safetensors.total if info.safetensors else None)
        if total:
            return total / 1e9
    except Exception:
        pass
    # fall back to a digit-in-name hint (…-7b…)
    import re
    m = re.search(r'(\d+(?:\.\d+)?)\s*b\b', (info.id or '').lower())
    return float(m.group(1)) if m else None


def pick_models(boards, n):
    from huggingface_hub import list_models, model_info
    board_ids = [b[0] for b in boards]
    picked = []
    try:
        trending = list_models(filter='text-generation', sort='trendingScore', limit=80)
    except Exception as e:
        log('trending query failed:', e)
        return picked
    for m in trending:
        org = m.id.split('/')[0].lower()
        low = m.id.lower()
        if org not in ALLOWED_ORGS:
            continue
        if not any(t in low for t in ('instruct', '-it', 'chat')):
            continue
        if all(already_submitted(lb, m.id) for lb in board_ids):
            continue
        try:
            info = model_info(m.id)
        except Exception:
            continue
        sz = _size_b(info)
        if sz and sz > MAX_PARAMS_B:
            continue
        picked.append(m.id)
        if len(picked) >= n:
            break
    return picked


def benchmark(model, boards):
    """Run the submitter for `model` on each board it's not yet on."""
    for lb, name, max_new in boards:
        if already_submitted(lb, model):
            continue
        log(f'benchmark {model} -> lb{lb} ({name}, max_new={max_new})')
        env = dict(os.environ, BENCHHUB_BASE_URL=BASE_URL)
        try:
            subprocess.run([CLIENT_PY, SUBMIT, str(lb), model, '32', str(max_new)],
                           env=env, cwd=os.path.dirname(SUBMIT),
                           timeout=5400, check=False)
        except subprocess.TimeoutExpired:
            log(f'TIMEOUT {model} on lb{lb}')
        except Exception as e:
            log(f'ERROR {model} on lb{lb}: {e}')


def build_next_benchmark():
    """Build the first queued benchmark whose dataset isn't imported yet."""
    con = _db()
    have = {r['name'].lower() for r in con.execute('SELECT name FROM dataset').fetchall()}
    con.close()
    # build_mcq CONFIG ds_name → import name; quick map (kept in sync manually).
    DS_NAME = {'arc_easy': 'arc-easy-test', 'openbookqa': 'openbookqa-test'}
    for key in BENCHMARK_QUEUE:
        if DS_NAME.get(key, key).lower() in have:
            continue
        log(f'building new benchmark: {key}')
        env = dict(os.environ, BENCHHUB_DATA_DIR=os.path.expanduser('~/.dtofbenchmarking'))
        try:
            r = subprocess.run([BENCHHUB_PY, BUILD_MCQ, key],
                               env=env, cwd=REPO, timeout=1800, check=False)
            if r.returncode == 0:
                return key
            log(f'build_mcq {key} exited {r.returncode}')
        except Exception as e:
            log(f'build {key} failed: {e}')
        return None
    log('benchmark queue empty — nothing to build')
    return None


def publish():
    log('triggering standings sync (publish to HF Space)')
    try:
        subprocess.run([BENCHHUB_PY, '-c', 'import tasks; tasks.push_standings_to_hf.delay()'],
                       cwd=os.path.expanduser('~/benchhub'), timeout=120, check=False)
    except Exception as e:
        log('publish trigger failed:', e)


def main():
    boards = llm_boards()
    log('LLM boards:', [(b[0], b[1]) for b in boards])
    if not boards:
        log('no LLM boards found; abort')
        return

    models = pick_models(boards, DEFAULT_MAX_MODELS)
    log('new models to benchmark:', models or '(none new)')

    if DRY:
        nb = [k for k in BENCHMARK_QUEUE]
        log('DRY-RUN — would benchmark the above, build next of', nb, 'and publish.')
        return

    for m in models:
        benchmark(m, boards)

    built = build_next_benchmark()
    if built:
        boards = llm_boards()      # the new board now exists
    seed_sparse(boards)            # populate any sparse/new board with baselines

    publish()
    log('done')


if __name__ == '__main__':
    main()
