"""User-declared file-tree importer.

Croissant/parquet only covers tabular datasets. Many benchmark repos are
a tree of paired files — RGB `.png` + a packed `.npz` of depth/events,
quality-variant folders, per-sequence archives — where *no* heuristic can
reliably infer the meaning (e.g. an `.npz` keyed `arr_0` of shape (N, 4)
is an event stream, not depth). So the user declares each modality and
points it at a source; this engine resolves + decodes those sources into
the typed-manifest staging dir the rest of the pipeline already ingests.

A spec is a list of field descriptors:

    {"name": "image", "kind": "image", "role": "input",
     "loader": "file",  "pattern": "train/{seq}/normal/{id}.png"}

    {"name": "depth", "kind": "depth", "role": "gt",
     "loader": "npz",   "pattern": "train/{seq}/normal/depth.npz",
     "key": "depth", "shared": true, "axis": 0}

    {"name": "events", "kind": "json", "role": "input",
     "loader": "npz",   "pattern": "train/{seq}/low/{id}.npz", "key": "arr_0"}

Loaders:
  - `file`  : one file per sample; `pattern` carries `{tokens}`, at least
              one of which (the per-sample one) varies across the split.
  - `npz`   : a NumPy archive. `shared=False` (default) → one archive per
              sample (`pattern` includes the per-sample token), value =
              `arr[key]`. `shared=True` → ONE archive holding all samples
              stacked along `axis`; each sample takes its frame by its
              ordinal within the archive's group, in sorted-filename order.

The first `file` field is the INDEX modality: matching its pattern against
the repo file list enumerates the samples (and their token tuples). Every
other field joins to those samples by the tokens its own pattern shares
with the index.
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
from collections import Counter, defaultdict

import numpy as np

# Per-kind canonical staging extension (mirrors the typed-manifest spec
# import_typed_dataset reads). Inline kinds (scalar/label/text/json) still
# land as small files here; the importer decodes them into value_*.
_STAGE_EXT = {
    'image': '.png', 'mask': '.png', 'depth': '.npz', 'audio': '.wav',
    'text': '.txt', 'json': '.json', 'scalar': '.txt', 'label': '.txt',
    'label_list': '.json',
}

_TOKEN_RE = re.compile(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}')


def _pattern_to_regex(pattern):
    """Compile a `{token}` pattern into a regex with named groups. Each
    token matches a single path segment (no `/`)."""
    out, tokens, last = '', [], 0
    for m in _TOKEN_RE.finditer(pattern):
        out += re.escape(pattern[last:m.start()])
        name = m.group(1)
        tokens.append(name)
        out += rf'(?P<{name}>[^/]+)'
        last = m.end()
    out += re.escape(pattern[last:])
    return re.compile('^' + out + '$'), tokens


def _substitute(pattern, tokens):
    return _TOKEN_RE.sub(lambda m: str(tokens.get(m.group(1), m.group(0))), pattern)


def match_files(pattern, files):
    """Files matching `pattern`, each as a dict of captured tokens plus
    `_path`. Returns (matches, token_names)."""
    rx, names = _pattern_to_regex(pattern)
    matches = []
    for f in files:
        m = rx.match(f)
        if m:
            d = dict(m.groupdict())
            d['_path'] = f
            matches.append(d)
    return matches, names


def _sample_name(tokens, token_names):
    """Stable, filesystem-safe sample name from the captured tokens."""
    raw = '_'.join(str(tokens[t]) for t in token_names)
    return re.sub(r'[^A-Za-z0-9._-]', '_', raw) or 'sample'


def inspect_repo(files):
    """Cheap structural summary for the mapping UI: extension histogram,
    plus auto-suggested `<dir>/{id}.<ext>` patterns grouped by directory +
    extension, and the shared-archive candidates (a lone non-image file in
    a directory of many same-stem files)."""
    exts = Counter((f.rsplit('.', 1)[-1].lower() if '.' in f else '(noext)')
                   for f in files if not f.endswith('/'))
    # Group files by (dir, ext); a dir+ext with many files → a per-sample
    # file pattern `dir/{id}.ext`.
    by_dir_ext = defaultdict(list)
    for f in files:
        if '/' not in f or '.' not in f.rsplit('/', 1)[-1]:
            continue
        d, base = f.rsplit('/', 1)
        ext = base.rsplit('.', 1)[-1].lower()
        by_dir_ext[(d, ext)].append(f)
    suggestions = []
    seen_patterns = set()
    for (d, ext), fs in sorted(by_dir_ext.items(), key=lambda kv: -len(kv[1])):
        if len(fs) < 2:
            continue
        # Generalise the directory's variable segments into tokens by
        # comparing two sibling paths — segments that differ become {seg}.
        pat = _generalise_dir(d) + '/{id}.' + ext
        if pat not in seen_patterns:
            seen_patterns.add(pat)
            suggestions.append({'pattern': pat, 'ext': ext, 'count': len(fs)})
    return {
        'file_count': sum(exts.values()),
        'ext_histogram': dict(exts.most_common()),
        'suggested_patterns': suggestions[:20],
    }


def _generalise_dir(d):
    """Turn a concrete directory like `train/i_0/normal` into a tokenised
    `train/{dir1}/normal` by tokenising segments that look like an id
    (contain a digit and aren't a fixed split/quality word)."""
    fixed = {'train', 'test', 'val', 'validation', 'dev', 'normal', 'low',
             'high', 'images', 'depth', 'gt', 'rgb', 'masks'}
    segs = d.split('/')
    out, n = [], 0
    for s in segs:
        if s.lower() in fixed or not any(c.isdigit() for c in s):
            out.append(s)
        else:
            n += 1
            out.append('{seq%d}' % n if n > 1 else '{seq}')
    return '/'.join(out)


def resolve_samples(spec, files):
    """Enumerate samples from the index (first `file`) field. Returns
    (samples, index_field) where each sample is
    {'name', 'tokens': {...}, '_index_path': str}. Raises ValueError with
    an actionable message when the spec can't be resolved."""
    file_fields = [f for f in spec if f.get('loader') == 'file']
    if not file_fields:
        raise ValueError("Need at least one field with loader 'file' to "
                         "enumerate samples (the index modality).")
    index = file_fields[0]
    matches, names = match_files(index['pattern'], files)
    if not matches:
        raise ValueError(
            f"Pattern {index['pattern']!r} matched no files in the repo. "
            f"Check the path + token placeholders.")
    matches.sort(key=lambda m: m['_path'])  # sorted-filename order
    samples = []
    seen = set()
    for m in matches:
        nm = _sample_name(m, names)
        # Disambiguate the rare name collision deterministically.
        base, k = nm, 1
        while nm in seen:
            k += 1
            nm = f"{base}#{k}"
        seen.add(nm)
        samples.append({'name': nm, 'tokens': {t: m[t] for t in names},
                        '_index_path': m['_path']})
    return samples, index


def _stage_value(kind, raw, dest_noext):
    """Write one decoded value to `dest_noext + ext` and return the
    relative-style basename written. `raw` is either bytes (file loaders)
    or an ndarray / python value (archive loaders)."""
    ext = _STAGE_EXT.get(kind, '.bin')
    dest = dest_noext + ext
    if isinstance(raw, (bytes, bytearray)):
        if kind == 'depth':
            # bytes already an .npz/.npy on disk would be copied upstream;
            # here a file-loader depth is unusual — decode via numpy.
            arr = _load_array_bytes(raw)
            np.savez_compressed(dest, depth=_as_2d(arr))
        else:
            with open(dest, 'wb') as fh:
                fh.write(raw)
        return os.path.basename(dest)

    # ndarray / python value from an archive loader.
    if kind == 'depth':
        np.savez_compressed(dest, depth=_as_2d(np.asarray(raw)))
    elif kind in ('image', 'mask'):
        from PIL import Image as _Img
        a = np.asarray(raw)
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
        mode = None
        if kind == 'mask' and a.ndim == 2:
            a = a.astype(np.uint16); mode = 'I;16'
        _Img.fromarray(a.astype(np.uint8) if mode is None else a, mode).save(dest)
    elif kind in ('text', 'scalar', 'label'):
        with open(dest, 'w') as fh:
            fh.write(str(raw if not isinstance(raw, np.generic) else raw.item()))
    else:  # json / label_list / anything else → JSON
        with open(dest, 'w') as fh:
            json.dump(_jsonable(raw), fh)
    return os.path.basename(dest)


def _as_2d(arr):
    arr = np.asarray(arr)
    while arr.ndim > 2 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _load_array_bytes(b):
    bio = io.BytesIO(b)
    obj = np.load(bio, allow_pickle=True)
    if hasattr(obj, 'files'):  # npz
        return obj[obj.files[0]]
    return obj


class _SafeFmt(dict):
    def __missing__(self, k):
        return '{' + k + '}'


def _resolve_json_pointer(obj, pointer, subs):
    """Walk a dotted pointer into a decoded JSON object. Each segment is
    a dict key or (if all-digits) a list index; `{token}`/`{ordinal}`
    placeholders are substituted from `subs` first. Empty pointer → whole
    object."""
    cur = obj
    if not pointer:
        return cur
    for part in pointer.split('.'):
        if part == '':
            continue
        if '{' in part:
            part = part.format_map(_SafeFmt(subs))
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:
            raise KeyError(f"can't descend into {type(cur).__name__} at {part!r}")
    return cur


def distinct_token_values(spec, files, token):
    """Distinct values a token takes across the index modality's matched
    files — used to enumerate variant folders (e.g. quality=low/normal)."""
    samples, _ = resolve_samples(spec, files)
    vals = []
    seen = set()
    for s in samples:
        v = s['tokens'].get(token)
        if v is not None and v not in seen:
            seen.add(v)
            vals.append(v)
    return vals


def materialize_file_tree(spec, files, fetch, staging_dir, *,
                          sample_cap=-1, dataset_name='dataset',
                          token_filter=None, progress_cb=None):
    """Resolve + decode every field into a typed-manifest staging dir.

    `fetch(repo_relpath) -> local filesystem path` downloads (or locates)
    one repo file; the engine stays storage-agnostic so tests pass a
    local-dir fetch and prod passes hf_hub_download.

    Returns a summary dict; the dir then goes straight to
    import_typed_dataset(preview_only=True)."""
    def _progress(phase, cur, total, msg):
        if progress_cb:
            progress_cb({'phase': phase, 'current': cur, 'total': total,
                         'message': msg})

    samples, _index = resolve_samples(spec, files)
    # Variant filter: keep only samples whose tokens match (e.g. import
    # just the `normal` quality when token_filter={'quality': 'normal'}).
    if token_filter:
        samples = [s for s in samples
                   if all(s['tokens'].get(k) == v for k, v in token_filter.items())]
    total_in_split = len(samples)
    if sample_cap and sample_cap > 0:
        samples = samples[:sample_cap]
    n = len(samples)
    _progress('resolving', 0, n, f"{n} samples across {len(spec)} field(s)")

    # Pre-compute ordinals for any field that reads from a shared source
    # (one archive/file holding many samples): group samples by the tokens
    # that field's pattern references, rank within group in sorted order.
    # Covers shared npz, shared json, and csv (always table-shaped).
    shared_ordinals = {}
    for f in spec:
        ldr = f.get('loader')
        if (ldr == 'csv') or (ldr in ('npz', 'json') and f.get('shared')):
            gtoks = _TOKEN_RE.findall(f['pattern'])
            groups = defaultdict(list)
            for i, s in enumerate(samples):
                gkey = tuple(s['tokens'].get(t, '') for t in gtoks)
                groups[gkey].append(i)
            ordinals = {}
            for gkey, idxs in groups.items():
                for rank, i in enumerate(idxs):
                    ordinals[i] = rank
            shared_ordinals[f['name']] = ordinals

    os.makedirs(staging_dir, exist_ok=True)
    for f in spec:
        os.makedirs(os.path.join(staging_dir, f['name']), exist_ok=True)

    # Cache shared sources so a .npz / .json / .csv is loaded once, not
    # per sample.
    archive_cache, json_cache, csv_cache = {}, {}, {}

    def _load_archive(path):
        if path not in archive_cache:
            archive_cache[path] = np.load(fetch(path), allow_pickle=True)
        return archive_cache[path]

    def _load_json(path):
        if path not in json_cache:
            with open(fetch(path)) as fh:
                json_cache[path] = json.load(fh)
        return json_cache[path]

    def _load_csv(path):
        if path not in csv_cache:
            import csv as _csv
            with open(fetch(path), newline='') as fh:
                csv_cache[path] = list(_csv.DictReader(fh))
        return csv_cache[path]

    written = defaultdict(int)
    for i, s in enumerate(samples):
        for f in spec:
            name, kind, loader = f['name'], f['kind'], f.get('loader', 'file')
            dest_noext = os.path.join(staging_dir, name, s['name'])
            try:
                if loader == 'file':
                    src = fetch(_substitute(f['pattern'], s['tokens']))
                    with open(src, 'rb') as fh:
                        raw = fh.read()
                    # Copy file-backed kinds verbatim with their real ext.
                    ext = os.path.splitext(src)[1] or _STAGE_EXT.get(kind, '')
                    if kind in ('image', 'mask', 'audio'):
                        shutil.copyfile(src, dest_noext + ext)
                    else:
                        _stage_value(kind, raw, dest_noext)
                elif loader == 'npz':
                    if f.get('shared'):
                        arc = _load_archive(_substitute(f['pattern'], s['tokens']))
                        full = arc[f.get('key') or arc.files[0]]
                        val = np.take(full, shared_ordinals[name][i],
                                      axis=int(f.get('axis', 0)))
                    else:
                        arc = np.load(fetch(_substitute(f['pattern'], s['tokens'])),
                                      allow_pickle=True)
                        val = arc[f.get('key') or arc.files[0]]
                    _stage_value(kind, val, dest_noext)
                elif loader == 'json':
                    doc = _load_json(_substitute(f['pattern'], s['tokens']))
                    subs = {**s['tokens'],
                            'ordinal': shared_ordinals.get(name, {}).get(i, 0)}
                    val = _resolve_json_pointer(doc, f.get('pointer') or '', subs)
                    _stage_value(kind, val, dest_noext)
                elif loader == 'csv':
                    rows = _load_csv(_substitute(f['pattern'], s['tokens']))
                    idcol = (f.get('id_column') or '').strip()
                    if idcol:
                        sid = s['tokens'].get(f.get('id_token') or 'id')
                        row = next((r for r in rows
                                    if str(r.get(idcol)) == str(sid)), None)
                    else:  # align by sorted order
                        ordn = shared_ordinals.get(name, {}).get(i, 0)
                        row = rows[ordn] if ordn < len(rows) else None
                    if row is None:
                        raise FileNotFoundError('csv row')
                    col = (f.get('column') or '').strip()
                    val = row.get(col) if col else row
                    _stage_value(kind, val, dest_noext)
                else:
                    raise ValueError(f"unknown loader {loader!r}")
                written[name] += 1
            except FileNotFoundError:
                # A sample missing one modality's file is skipped for that
                # field; the importer tolerates ragged fields.
                continue
        if (i + 1) % 100 == 0 or i + 1 == n:
            _progress('decoding', i + 1, n, f"Decoded {i + 1}/{n} samples")

    manifest = {
        'name': dataset_name, 'version': '1.0',
        'fields': [{'name': f['name'], 'kind': f['kind'],
                    'role': f.get('role', 'gt'), 'params': f.get('params') or {}}
                   for f in spec],
        'samples': [s['name'] for s in samples],
    }
    with open(os.path.join(staging_dir, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh)
    return {
        'name': dataset_name, 'samples': n, 'fields': len(spec),
        'total_rows_in_split': total_in_split,
        'rows_written': dict(written),
    }
