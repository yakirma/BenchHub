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
    'label_list': '.json', 'sequence': '.zip',
}

_TOKEN_RE = re.compile(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}')

# Folder names that read as a MODALITY (a separate field) rather than a
# class label. Used to decide whether `<X>/<id>.ext` siblings are paired
# modalities (image/ + mask/) or class folders (Alex_Brush/ + Cookie/).
_MODALITY_WORDS = {
    'image', 'images', 'img', 'imgs', 'rgb', 'rgba', 'color', 'colour',
    'photo', 'photos', 'frame', 'frames', 'input', 'inputs',
    'depth', 'depths', 'disparity', 'mask', 'masks', 'seg', 'segmentation',
    'semantic', 'instance', 'instances', 'label_map', 'labelmap', 'labels',
    'gt', 'groundtruth', 'ground_truth', 'target', 'targets', 'annotation',
    'annotations', 'normal', 'normals', 'flow', 'ir', 'nir', 'thermal',
    'left', 'right', 'audio', 'video', 'text', 'texts', 'caption', 'captions',
    'pose', 'keypoints', 'kpts', 'points', 'pointcloud', 'lidar', 'events',
}


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

    # Label-folder shape: files like `<X>/<file>.<ext>` (one folder level)
    # with several sibling folders. Two readings:
    #   - folders are MODALITIES (image/, depth/, mask/ …) → one field per
    #     folder, each its own `folder/{id}.ext` pattern.
    #   - folders are CLASS LABELS (Alex_Brush/, Cookie/ …) → ONE field
    #     `{label}/{id}.ext` (pair with a `token` label field).
    # The name heuristic (modality folders are semantic words) only picks
    # the DEFAULT; both readings are always emitted, tagged with `group`
    # ('label' | 'folder'), and the UI renders a toggle between them —
    # `folder_toggle` below carries the preferred default.
    two_seg = defaultdict(set)
    two_seg_count = defaultdict(int)
    for f in files:
        parts = f.split('/')
        if len(parts) == 2 and '.' in parts[1]:
            ext = parts[1].rsplit('.', 1)[-1].lower()
            two_seg[ext].add(parts[0])
            two_seg_count[ext] += 1
    ambiguous_exts = {}     # ext -> True when the label reading is preferred
    for ext, folders in two_seg.items():
        if len(folders) < 2:
            continue
        modality_like = sum(1 for d in folders if d.lower() in _MODALITY_WORDS)
        # Mostly arbitrary names → class labels.
        ambiguous_exts[ext] = modality_like / len(folders) < 0.5
    folder_toggle = None
    if ambiguous_exts:
        # Default reading follows the dominant (most files) ambiguous ext.
        top = max(ambiguous_exts, key=lambda e: two_seg_count[e])
        folder_toggle = 'label' if ambiguous_exts[top] else 'modality'

    label_suggestions = []
    for ext in sorted(ambiguous_exts, key=lambda e: -two_seg_count[e]):
        pat = '{label}/{id}.' + ext
        seen_patterns.add(pat)
        label_suggestions.append({'pattern': pat, 'ext': ext,
                                  'count': two_seg_count[ext],
                                  'label_folder': True, 'group': 'label'})
    if folder_toggle == 'label':
        suggestions.extend(label_suggestions)

    for (d, ext), fs in sorted(by_dir_ext.items(), key=lambda kv: -len(kv[1])):
        if len(fs) < 2:
            continue
        # Generalise the directory's variable segments into tokens by
        # comparing two sibling paths — segments that differ become {seg}.
        pat = _generalise_dir(d) + '/{id}.' + ext
        if pat in seen_patterns:
            continue
        seen_patterns.add(pat)
        s = {'pattern': pat, 'ext': ext, 'count': len(fs)}
        # A top-level folder participating in the label/modality ambiguity
        # → tag it so the toggle can show one reading at a time.
        if '/' not in d and ext in ambiguous_exts:
            s['group'] = 'folder'
        suggestions.append(s)

    # The non-preferred {label} alternates ride at the end.
    if folder_toggle == 'modality':
        suggestions.extend(label_suggestions)

    # Cap, but never drop a {label} alternate — the toggle needs it.
    capped = suggestions[:20] + [s for s in suggestions[20:]
                                 if s.get('group') == 'label']
    return {
        'file_count': sum(exts.values()),
        'ext_histogram': dict(exts.most_common()),
        'suggested_patterns': capped,
        'folder_toggle': folder_toggle,
    }


# --- Path-level role analysis: the user labels each directory level, and
#     we generate the field rows from that (modality/property/split/group/
#     id/fixed) instead of guessing from folder names. ---

_META_EXTS = {'md', 'gitattributes', 'gitignore', 'json5'}
_SPLIT_WORDS = {'train', 'test', 'val', 'valid', 'validation', 'dev',
                'eval', 'trainval', 'training', 'testing'}
_MASK_NAME_WORDS = {'mask', 'masks', 'seg', 'segmentation', 'semantic',
                    'instance', 'instances', 'label_map', 'labelmap',
                    'annotation', 'annotations'}
_EXT_KIND_LOADER = {
    'png': ('image', 'file'), 'jpg': ('image', 'file'), 'jpeg': ('image', 'file'),
    'bmp': ('image', 'file'), 'tif': ('image', 'file'), 'tiff': ('image', 'file'),
    'webp': ('image', 'file'), 'gif': ('image', 'file'),
    'jxl': ('image', 'file'),
    'wav': ('audio', 'file'), 'mp3': ('audio', 'file'), 'flac': ('audio', 'file'),
    'txt': ('text', 'file'), 'json': ('json', 'file'),
    'npz': ('depth', 'npz'), 'npy': ('depth', 'file'),
}


def _kind_loader_for_ext(ext, name=None):
    kind, loader = _EXT_KIND_LOADER.get((ext or '').lower(), ('json', 'file'))
    if kind == 'image' and name and str(name).lower() in _MASK_NAME_WORDS:
        kind = 'mask'
    return kind, loader


def _data_files(files):
    """Files that look like sample data (drop dotfiles + repo meta)."""
    out = []
    for f in files:
        base = f.rsplit('/', 1)[-1]
        if base.startswith('.') or '.' not in base:
            continue
        if base.rsplit('.', 1)[-1].lower() in _META_EXTS:
            continue
        out.append(f)
    return out


def _dominant_depth_files(files):
    data = _data_files(files)
    if not data:
        return [], 0
    depth_counts = Counter(f.count('/') + 1 for f in data)
    depth = depth_counts.most_common(1)[0][0]
    return [f for f in data if f.count('/') + 1 == depth], depth


def _default_level_role(is_file, distinct, exts):
    if is_file:
        return 'id'
    n = len(distinct)
    if n == 1:
        return 'fixed'
    low = [d.lower() for d in distinct]
    if all(d in _SPLIT_WORDS for d in low):
        return 'split'
    if sum(1 for d in low if d in _MODALITY_WORDS) / n >= 0.5:
        return 'modality'
    if all(any(c.isdigit() for c in d) for d in distinct):
        return 'group'
    return 'property'


def analyze_levels(files):
    """Describe the repo's directory skeleton at the dominant depth: one
    entry per path level with example values, distinct count, file-level
    extensions, and a smart default role. Drives the 'describe the
    structure' UI."""
    sel, depth = _dominant_depth_files(files)
    levels = []
    for i in range(depth):
        is_file = (i == depth - 1)
        if is_file:
            exts = Counter(f.rsplit('.', 1)[-1].lower() for f in sel)
            distinct = sorted({f.split('/')[i].rsplit('.', 1)[0] for f in sel})
        else:
            exts = None
            distinct = sorted({f.split('/')[i] for f in sel})
        levels.append({
            'index': i, 'is_file': is_file,
            'distinct_count': len(distinct),
            'examples': distinct[:6],
            'exts': dict(exts.most_common()) if exts else None,
            'default_role': _default_level_role(is_file, distinct, exts),
        })
    return {'levels': levels, 'depth': depth, 'file_count': len(sel)}


def _spec_field(name, kind, loader, pattern):
    name = re.sub(r'[^A-Za-z0-9_]', '_', str(name)).strip('_') or 'field'
    f = {'name': name, 'kind': kind,
         'role': 'input' if kind == 'image' else 'gt',
         'loader': loader, 'pattern': pattern}
    if loader == 'npz':
        f.update({'key': None, 'shared': False, 'axis': 0})
    return f


def generate_spec_from_roles(files, roles):
    """Turn a per-level role assignment into the field-row spec. `roles`
    is a list indexed by path level (id / modality / property / split /
    group / fixed). A `modality` level fans out one field per value; a
    file level with several extensions fans out one field per ext; each
    `property` level adds a `token` label field; split/group become tokens
    in the patterns."""
    sel, depth = _dominant_depth_files(files)
    if not depth:
        return []
    roles = list(roles) + ['id' if i == depth - 1 else 'fixed'
                           for i in range(len(roles), depth)]

    # Decide what each non-file level contributes to a pattern.
    counts = defaultdict(int)
    level_tok = {}        # i -> ('literal', v) | ('token', name) | ('modality',) | ('id',)
    prop_tokens = []      # [(i, token_name)] → token label fields
    for i in range(depth):
        r = (roles[i] or '').lower()
        if i == depth - 1:
            level_tok[i] = ('id',)
            continue
        if r == 'modality':
            level_tok[i] = ('modality',)
        elif r == 'split':
            level_tok[i] = ('token', 'split')
        elif r == 'group':
            counts['seq'] += 1
            level_tok[i] = ('token', 'seq' if counts['seq'] == 1 else f"seq{counts['seq']}")
        elif r == 'property':
            counts['label'] += 1
            nm = 'label' if counts['label'] == 1 else f"label{counts['label']}"
            level_tok[i] = ('token', nm)
            prop_tokens.append((i, nm))
        elif r == 'fixed':
            vals = sorted({f.split('/')[i] for f in sel})
            if len(vals) == 1:
                level_tok[i] = ('literal', vals[0])
            else:
                counts['x'] += 1
                level_tok[i] = ('token', f"x{counts['x']}")
        else:  # 'id' on a non-file level
            level_tok[i] = ('token', 'id')

    def build_pattern(mod_value, ext):
        segs = []
        for i in range(depth):
            tok = level_tok[i]
            if i == depth - 1:
                segs.append('{id}.' + ext)
            elif tok[0] == 'modality':
                segs.append(mod_value)
            elif tok[0] == 'literal':
                segs.append(tok[1])
            else:  # token
                segs.append('{%s}' % tok[1])
        return '/'.join(segs)

    spec = []
    mod_levels = [i for i in range(depth) if level_tok[i][0] == 'modality']
    if mod_levels:
        ml = mod_levels[0]
        for V in sorted({f.split('/')[ml] for f in sel}):
            sub = [f for f in sel if f.split('/')[ml] == V]
            ext = Counter(f.rsplit('.', 1)[-1].lower() for f in sub).most_common(1)[0][0]
            kind, loader = _kind_loader_for_ext(ext, name=V)
            spec.append(_spec_field(V, kind, loader, build_pattern(V, ext)))
    else:
        by_ext = defaultdict(list)
        for f in sel:
            by_ext[f.rsplit('.', 1)[-1].lower()].append(f)
        multi = len(by_ext) > 1
        for ext, fs in sorted(by_ext.items(), key=lambda kv: -len(kv[1])):
            kind, loader = _kind_loader_for_ext(ext)
            name = ('image' if (kind == 'image' and not multi) else kind)
            spec.append(_spec_field(name, kind, loader, build_pattern(None, ext)))

    for _i, nm in prop_tokens:
        spec.append({'name': nm, 'kind': 'label', 'role': 'gt',
                     'loader': 'token', 'token': nm})
    return spec


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


def _container_member_names(loader, local_path):
    """List the (file) member names inside a zip/tar/tar.gz container."""
    if loader == 'zip':
        import zipfile
        with zipfile.ZipFile(local_path) as z:
            return [n for n in z.namelist() if not n.endswith('/')]
    import tarfile
    with tarfile.open(local_path, 'r:*') as t:
        return [m.name for m in t.getmembers() if m.isfile()]


def _match_container_members(field, files, fetch):
    """Enumerate samples from a container (zip/tar) index field: for each
    container file matching the field `pattern`, open it and match its
    members against the `member` pattern. Tokens = container-path tokens
    ∪ member tokens. Returns (matches, token_names)."""
    crx, ctoks = _pattern_to_regex(field['pattern'])
    mrx, mtoks = _pattern_to_regex(field.get('member') or '')
    names, seen_names = [], set()
    for t in ctoks + mtoks:
        if t not in seen_names:
            seen_names.add(t); names.append(t)
    matches = []
    for f in files:
        cm = crx.match(f)
        if not cm:
            continue
        ctok = cm.groupdict()
        for mname in _container_member_names(field['loader'], fetch(f)):
            mm = mrx.match(mname)
            if not mm:
                continue
            d = {**ctok, **mm.groupdict()}
            d['_path'] = f + '!' + mname
            matches.append(d)
    return matches, names


def resolve_samples(spec, files, fetch=None):
    """Enumerate samples from the index modality. The index is the first
    `file` field, or — when there's none — the first `zip`/`tar` field
    whose `member` uses `{id}` (container index). Returns (samples,
    index_field); each sample is {'name', 'tokens', '_index_path'}.

    A container index needs `fetch` to list members; when called without
    one (route pre-validation), it returns [] rather than raising, leaving
    the real enumeration to materialize (which has fetch)."""
    file_fields = [f for f in spec if f.get('loader') == 'file']
    container_fields = [f for f in spec if f.get('loader') in ('zip', 'tar')
                        and '{id}' in (f.get('member') or '')]
    seq_fields = [f for f in spec if f.get('loader') == 'sequence'
                  and 'frame' in _TOKEN_RE.findall(f.get('pattern') or '')]
    if file_fields:
        index = file_fields[0]
        matches, names = match_files(index['pattern'], files)
    elif container_fields:
        index = container_fields[0]
        if fetch is None:
            return [], index   # defer enumeration to materialize
        matches, names = _match_container_members(index, files, fetch)
    elif seq_fields:
        # A sequence is the index: samples are the GROUPS of frames (the
        # pattern tokens other than {frame}); each group = one sample.
        index = seq_fields[0]
        raw, names = match_files(index['pattern'], files)
        id_toks = [t for t in names if t != 'frame']
        seen_keys, matches, names = set(), [], id_toks
        for m in sorted(raw, key=lambda m: m['_path']):
            key = tuple(m[t] for t in id_toks)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            d = {t: m[t] for t in id_toks}
            d['_path'] = m['_path']
            matches.append(d)
    else:
        raise ValueError("Need an index modality: a field with loader "
                         "'file', a 'zip'/'tar' field whose member uses {id}, "
                         "or a 'sequence' field whose pattern uses {frame}.")
    if not matches:
        raise ValueError(
            f"Pattern {index.get('pattern')!r} matched no files/members. "
            f"Check the path + token placeholders.")
    matches.sort(key=lambda m: m['_path'])  # sorted order
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


def _transcode_to_canonical(kind, raw, dest_noext, src_label=''):
    """Re-encode a non-canonical source (jpg/webp/jxl image, flac/mp3
    audio, …) into the kind's canonical staging format. Decode failures
    raise ValueError with a readable message — a wrong kind mapping or an
    unsupported codec should fail the import loudly, not stage garbage.
    (The decode preview runs this same path, so users can catch it before
    committing.)"""
    dest = dest_noext + _STAGE_EXT[kind]
    try:
        if kind in ('image', 'mask'):
            try:
                import pillow_jxl  # noqa: F401 — registers .jxl with PIL
            except ImportError:
                pass
            from PIL import Image as _Img
            im = _Img.open(io.BytesIO(raw))
            im.load()
            # Masks keep their integer modes (P / L / I;16 all save as
            # PNG); photographic oddballs (CMYK, YCbCr…) normalise to RGB.
            if kind == 'image' and im.mode not in ('RGB', 'RGBA', 'L'):
                im = im.convert('RGB')
            im.save(dest, format='PNG')
        elif kind == 'audio':
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(raw))
            sf.write(dest, data, sr, format='WAV')
        else:
            with open(dest, 'wb') as fh:
                fh.write(raw)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"can't decode {os.path.basename(str(src_label)) or 'file'} as "
            f"{kind} ({type(e).__name__}: {e}) — wrong kind mapping, or a "
            f"codec the server doesn't have?")
    return os.path.basename(dest)


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
                          sample_cap=-1, sample_offset=0, dataset_name='dataset',
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

    samples, _index = resolve_samples(spec, files, fetch=fetch)
    # Variant filter: keep only samples whose tokens match (e.g. import
    # just the `normal` quality when token_filter={'quality': 'normal'}).
    if token_filter:
        samples = [s for s in samples
                   if all(s['tokens'].get(k) == v for k, v in token_filter.items())]
    total_in_split = len(samples)
    # `sample_offset` skips into the resolved list before the cap — the
    # decode preview uses (offset=i, cap=1) to materialize the i-th sample.
    if sample_offset:
        samples = samples[sample_offset:]
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
        if (ldr in ('csv', 'parquet', 'zip', 'tar')
                or (ldr in ('npz', 'json', 'hdf5') and f.get('shared'))):
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

    # `sequence` fields: group all matching frames by the sample-identity
    # tokens (pattern tokens minus {frame}); per sample we zip the frames
    # (sorted by {frame}) into one Sequence. Also stamp item_kind/fps onto
    # the field params so the stored zip decodes/renders correctly.
    seq_frames = {}
    for f in spec:
        if f.get('loader') == 'sequence':
            rx, names = _pattern_to_regex(f['pattern'])
            id_toks = [t for t in names if t != 'frame']
            by_key = defaultdict(list)
            for path in files:
                m = rx.match(path)
                if m:
                    d = m.groupdict()
                    by_key[tuple(d.get(t) for t in id_toks)].append(
                        (d.get('frame', ''), path))
            for k in by_key:
                by_key[k].sort(key=lambda fp: fp[0])
            seq_frames[f['name']] = (id_toks, by_key)
            # Infer item_kind from the first frame's extension if unset.
            any_frames = next((v for v in by_key.values() if v), None)
            ext = (any_frames[0][1].rsplit('.', 1)[-1].lower() if any_frames else 'png')
            item_kind = f.get('item_kind') or _kind_loader_for_ext(ext)[0]
            f.setdefault('params', {})
            f['params'].setdefault('item_kind', item_kind)
            f['params'].setdefault('fps', int(f.get('fps', 6) or 6))
            f['_frame_ext'] = ext

    # `token` fields take their value from a captured path token (e.g. the
    # class folder in `<class>/<id>.png`). For label kind, build a vocab so
    # the value stores an int index + a `names` list (categorical), like
    # ClassLabel — renders as the class name in the UI.
    token_vocab = {}
    for f in spec:
        if f.get('loader') == 'token' and f.get('kind') == 'label':
            tok = f.get('token') or 'id'
            vals = sorted({s['tokens'].get(tok) for s in samples
                           if s['tokens'].get(tok) is not None})
            token_vocab[f['name']] = vals
            f.setdefault('params', {})
            f['params'].setdefault('names', vals)

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

    parquet_cache, h5_cache = {}, {}

    def _load_parquet(path):
        if path not in parquet_cache:
            import pandas as pd
            parquet_cache[path] = pd.read_parquet(fetch(path)).to_dict('records')
        return parquet_cache[path]

    def _load_h5(path):
        if path not in h5_cache:
            import h5py
            h5_cache[path] = h5py.File(fetch(path), 'r')
        return h5_cache[path]

    def _table_row(rows, f, s, i):
        """Pick a sample's row from a shared table (csv/parquet): by an
        id column matched to the sample's id token, else by sorted order."""
        idcol = (f.get('id_column') or '').strip()
        if idcol:
            sid = s['tokens'].get(f.get('id_token') or 'id')
            return next((r for r in rows if str(r.get(idcol)) == str(sid)), None)
        ordn = shared_ordinals.get(f['name'], {}).get(i, 0)
        return rows[ordn] if ordn < len(rows) else None

    # Container (zip/tar) handles, opened once and closed after the run.
    container_cache = {}

    def _open_container(loader, path):
        key = (loader, path)
        if key not in container_cache:
            local = fetch(path)
            if loader == 'zip':
                import zipfile
                container_cache[key] = ('zip', zipfile.ZipFile(local))
            else:
                import tarfile
                container_cache[key] = ('tar', tarfile.open(local, 'r:*'))
        return container_cache[key]

    def _read_member(loader, path, member):
        kind_, c = _open_container(loader, path)
        if kind_ == 'zip':
            try:
                return c.read(member)
            except KeyError:
                raise FileNotFoundError(member)
        m = c.extractfile(member)
        if m is None:
            raise FileNotFoundError(member)
        return m.read()

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
                    ext = os.path.splitext(src)[1] or _STAGE_EXT.get(kind, '')
                    if kind in ('image', 'mask', 'audio'):
                        # The typed-manifest contract is ONE canonical ext
                        # per kind (image/mask → .png, audio → .wav); the
                        # importer's existence check + every downstream
                        # consumer assume it. Canonical sources copy
                        # verbatim; anything else (jpg, webp, jxl, flac …)
                        # transcodes here, at the boundary — so e.g. JXL
                        # needs its PIL plugin only on the server.
                        if ext.lower() == _STAGE_EXT[kind]:
                            shutil.copyfile(src, dest_noext + ext)
                        else:
                            _transcode_to_canonical(kind, raw, dest_noext,
                                                    src_label=src)
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
                elif loader in ('csv', 'parquet'):
                    rows = (_load_csv if loader == 'csv' else _load_parquet)(
                        _substitute(f['pattern'], s['tokens']))
                    row = _table_row(rows, f, s, i)
                    if row is None:
                        raise FileNotFoundError(f'{loader} row')
                    col = (f.get('column') or '').strip()
                    val = row.get(col) if col else row
                    _stage_value(kind, val, dest_noext)
                elif loader == 'hdf5':
                    h5 = _load_h5(_substitute(f['pattern'], s['tokens']))
                    dset = h5[f.get('key') or list(h5.keys())[0]]
                    if f.get('shared'):
                        val = np.take(dset[...], shared_ordinals[name][i],
                                      axis=int(f.get('axis', 0)))
                    else:
                        val = dset[...]
                    _stage_value(kind, np.asarray(val), dest_noext)
                elif loader == 'token':
                    tok = f.get('token') or 'id'
                    raw = s['tokens'].get(tok)
                    if raw is None:
                        raise FileNotFoundError(f'token {tok}')
                    if kind == 'label' and name in token_vocab:
                        raw = token_vocab[name].index(raw)  # categorical index
                    _stage_value(kind, raw, dest_noext)
                elif loader in ('zip', 'tar'):
                    subs = {**s['tokens'],
                            'ordinal': shared_ordinals.get(name, {}).get(i, 0)}
                    data = _read_member(loader,
                                        _substitute(f['pattern'], s['tokens']),
                                        _substitute(f.get('member') or '', subs))
                    _stage_value(kind, data, dest_noext)
                elif loader == 'gz':
                    import gzip
                    with gzip.open(fetch(_substitute(f['pattern'], s['tokens'])),
                                   'rb') as gf:
                        data = gf.read()
                    _stage_value(kind, data, dest_noext)
                elif loader == 'sequence':
                    import zipfile
                    id_toks, by_key = seq_frames[name]
                    key = tuple(s['tokens'].get(t) for t in id_toks)
                    frame_paths = [p for _fr, p in by_key.get(key, [])]
                    if not frame_paths:
                        raise FileNotFoundError('no frames')
                    fext = f.get('_frame_ext', 'png')
                    zbuf = io.BytesIO()
                    with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_STORED) as zf:
                        for fi, p in enumerate(frame_paths):
                            with open(fetch(p), 'rb') as fh:
                                zf.writestr(f"{fi:06d}.{fext}", fh.read())
                    with open(dest_noext + '.zip', 'wb') as fh:
                        fh.write(zbuf.getvalue())
                else:
                    raise ValueError(f"unknown loader {loader!r}")
                written[name] += 1
            except FileNotFoundError:
                # A sample missing one modality's file is skipped for that
                # field; the importer tolerates ragged fields.
                continue
        if (i + 1) % 100 == 0 or i + 1 == n:
            _progress('decoding', i + 1, n, f"Decoded {i + 1}/{n} samples")

    for _h5 in h5_cache.values():
        try:
            _h5.close()
        except Exception:
            pass
    for _kind, _c in container_cache.values():
        try:
            _c.close()
        except Exception:
            pass

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
