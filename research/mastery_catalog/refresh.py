"""Incremental catalog refresh refresh-v1 (tasks MS-701..MS-704, design.md section 9).

    .venv/bin/python -m research.mastery_catalog.refresh --previous catalog-v1 --version catalog-v2 --live
    .venv/bin/python -m research.mastery_catalog.refresh --previous catalog-v1 --version catalog-vX --replay DIR

Fetches only documented metadata endpoints (Codeforces API, AtCoder Problems resources), serialized
and rate-gated; stops on 403/429. Classifies a delta against the previous version, encodes only new
or changed titles (cache keyed by text hash + model revision), recomputes exact neighbors, and builds
an immutable catalog-vN+1. Never overwrites a run or catalog directory; never reads statements.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from research.mastery_catalog import build as builder
from src.learning import catalog_sources as sources
from research.full_catalog import catalog as fullcat

ROOT = Path(__file__).resolve().parents[2]
MASTERY = ROOT / 'private-learning/mastery'
MODEL_REVISION = 'sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41+c2-translation'
GAP_SECONDS = 2.1
STORED_K = 30
ENDPOINTS = {
    'codeforces_problems': 'https://codeforces.com/api/problemset.problems?lang=en',
    'codeforces_contests': 'https://codeforces.com/api/contest.list?gym=false',
    'atcoder_problems': 'https://kenkoooo.com/atcoder/resources/problems.json',
    'atcoder_contests': 'https://kenkoooo.com/atcoder/resources/contests.json',
    'atcoder_contest_problem': 'https://kenkoooo.com/atcoder/resources/contest-problem.json',
    'atcoder_models': 'https://kenkoooo.com/atcoder/resources/problem-models.json',
}
REQUIRED = ('codeforces_problems', 'atcoder_problems')
V1_ROWS = ROOT / 'private-learning/full-catalog/catalog-v1/catalog.v1.jsonl'
V1_EMBEDDING = ROOT / 'private-learning/unified-map/full-v2'
DELTA_FIELDS = ('title', 'url', 'tags', 'native_difficulty', 'contest', 'eligibility', 'exclusion_reasons')


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    path = Path(path)
    with open(path, 'xb') as handle:
        handle.write(data if isinstance(data, bytes) else data.encode('utf-8'))


# 1. fetch / replay ------------------------------------------------------------------------------
def fetch_all(raw_dir, fetch=sources.bounded_fetch, sleep=time.sleep, now=lambda: datetime.now(timezone.utc)):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for i, (name, url) in enumerate(ENDPOINTS.items()):
        if i:
            sleep(GAP_SECONDS)
        result = fetch(url)
        entry = {'url': url, 'http_status': result.http_status, 'error': result.error, 'retrieved_at': now().isoformat(timespec='seconds')}
        if result.http_status in (403, 429) or (result.error and 'http_error_4' in result.error):
            manifest[name] = entry
            write_new(raw_dir / 'manifest.json', json.dumps(manifest, indent=1, sort_keys=True))
            raise RuntimeError(f'{name}: refused with HTTP {result.http_status}; stopping without retry')
        if result.error or result.http_status != 200:
            entry['status'] = 'failed'
        else:
            write_new(raw_dir / f'{name}.raw', result.body)
            entry.update(status='ok', sha256=sha256_bytes(result.body), bytes=len(result.body))
        manifest[name] = entry
    write_new(raw_dir / 'manifest.json', json.dumps(manifest, indent=1, sort_keys=True))
    missing = [n for n in REQUIRED if manifest[n].get('status') != 'ok']
    if missing:
        raise RuntimeError(f'required sources failed: {missing}; previous version stays active')
    return manifest


def load_raw(raw_dir):
    raw_dir = Path(raw_dir)
    manifest = json.loads((raw_dir / 'manifest.json').read_text())
    bodies = {}
    for name, entry in manifest.items():
        if entry.get('status') != 'ok':
            continue
        body = (raw_dir / f'{name}.raw').read_bytes()
        if sha256_bytes(body) != entry['sha256']:
            raise ValueError(f'raw {name} hash mismatch')
        bodies[name] = body
    return manifest, bodies


# 2. normalize ------------------------------------------------------------------------------------
def canonical_atcoder_contests(bodies):
    """task_id -> canonical contest. AtCoder Problems' problems.json points shared ABC/ARC tasks at
    their latest AtCoder Daily Training (adt_*) reuse, which drifts daily. Prefer the contest named by
    the task prefix (abc235_a -> abc235), else the earliest non-ADT contest that lists the task."""
    if 'atcoder_contest_problem' not in bodies:
        return {}
    starts = {c['id']: c.get('start_epoch_second') or 0 for c in json.loads(bodies.get('atcoder_contests', b'[]'))}
    listed = {}
    for item in json.loads(bodies['atcoder_contest_problem']):
        if isinstance(item.get('problem_id'), str) and isinstance(item.get('contest_id'), str):
            listed.setdefault(item['problem_id'], set()).add(item['contest_id'])
    canonical = {}
    for task, contests in listed.items():
        prefix = task.rsplit('_', 1)[0]
        non_adt = sorted((c for c in contests if not c.startswith('adt_')), key=lambda c: (starts.get(c, 0), c))
        if prefix in contests:
            canonical[task] = prefix
        elif non_adt:
            canonical[task] = non_adt[0]
    return canonical


def normalize_external(bodies):
    cf_rows, cf_rejects = sources.parse_codeforces(bodies['codeforces_problems'])
    at_rows, at_rejects = sources.parse_atcoder(bodies['atcoder_problems'])
    canonical = canonical_atcoder_contests(bodies)
    corrected = 0
    for row in at_rows:
        task = row['problem_key'].split(':', 1)[1]
        contest = canonical.get(task)
        url = f'https://atcoder.jp/contests/{contest}/tasks/{task}' if contest else row['url']
        if url != row['url']:
            row['url'] = url
            corrected += 1
    lines = [json.dumps(r) for r in cf_rows + at_rows]
    return fullcat.load_external(lines), {'codeforces': cf_rejects, 'atcoder': at_rejects, 'atcoder_canonical_url_corrections': corrected}


def contests_from(bodies):
    contests, cf_titles = [], {}
    if 'codeforces_contests' in bodies:
        payload = json.loads(bodies['codeforces_contests'])
        for c in payload.get('result', []):
            if not isinstance(c.get('id'), int):
                continue
            start = datetime.fromtimestamp(c['startTimeSeconds'], timezone.utc).isoformat().replace('+00:00', 'Z') if isinstance(c.get('startTimeSeconds'), int) else None
            cf_titles[str(c['id'])] = c.get('name', str(c['id']))
            contests.append({'contest_id': str(c['id']), 'platform': 'codeforces', 'title': c.get('name', str(c['id'])), 'kind': contest_kind('codeforces', str(c['id']), c.get('name', '')),
                             'start_at': start, 'duration_minutes': c['durationSeconds'] // 60 if isinstance(c.get('durationSeconds'), int) else None,
                             'url': f"https://codeforces.com/contest/{c['id']}", 'problems': []})
    if 'atcoder_contests' in bodies:
        for c in json.loads(bodies['atcoder_contests']):
            if not isinstance(c.get('id'), str):
                continue
            start = datetime.fromtimestamp(c['start_epoch_second'], timezone.utc).isoformat().replace('+00:00', 'Z') if isinstance(c.get('start_epoch_second'), int) else None
            contests.append({'contest_id': c['id'], 'platform': 'atcoder', 'title': c.get('title', c['id']), 'kind': contest_kind('atcoder', c['id'], c.get('title', '')),
                             'start_at': start, 'duration_minutes': c['duration_second'] // 60 if isinstance(c.get('duration_second'), int) else None,
                             'url': f"https://atcoder.jp/contests/{c['id']}", 'problems': []})
    return contests


def contest_kind(platform, contest_id, title):
    import re
    if platform == 'atcoder':
        m = re.match(r'(abc|arc|agc|ahc)\d+', contest_id, re.I)
        return m.group(1).lower() if m else 'other'
    if re.search(r'div\.?\s*1\s*\+\s*(div\.?\s*)?2', title, re.I):
        return 'div1+2'
    m = re.search(r'div\.?\s*([1-4])', title, re.I)
    if m:
        return 'div' + m.group(1)
    if re.search(r'educational', title, re.I):
        return 'educational'
    if re.search(r'global', title, re.I):
        return 'global'
    return 'other'


def attach_contest_problems(contests, rows, bodies):
    by_id = {c['contest_id']: c for c in contests}
    keys = {r['problem_key'] for r in rows}
    for r in rows:
        if r['platform'] == 'codeforces' and r.get('contest') is not None and str(r['contest']) in by_id:
            by_id[str(r['contest'])]['problems'].append((r['problem_key'], r['problem_key'].split(':')[1][len(str(r['contest'])):]))
    if 'atcoder_contest_problem' in bodies:
        for item in json.loads(bodies['atcoder_contest_problem']):
            key = f"atcoder:{item.get('problem_id')}"
            if item.get('contest_id') in by_id and key in keys:
                by_id[item['contest_id']]['problems'].append((key, item.get('problem_index')))
    for c in contests:
        c['problems'] = sorted(set(c['problems']), key=lambda p: (str(p[1]), p[0]))
    return contests


# 3. delta -----------------------------------------------------------------------------------------
def row_signature(row):
    return hashlib.sha256(json.dumps({f: row.get(f) for f in DELTA_FIELDS}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def classify_delta(previous, current, refreshed_platforms=('codeforces', 'atcoder')):
    prev = {r['problem_key']: r for r in previous}
    cur = {r['problem_key']: r for r in current}
    delta = {'added': [], 'changed': [], 'removed': [], 'unchanged': 0}
    for key, row in cur.items():
        if key not in prev:
            delta['added'].append(key)
        elif row_signature(row) != row_signature(prev[key]) or prev[key].get('status') == 'removed_upstream':
            delta['changed'].append(key)
        else:
            delta['unchanged'] += 1
    for key, row in prev.items():
        if key not in cur and row['platform'] in refreshed_platforms and row.get('status', 'active') == 'active':
            delta['removed'].append(key)
    for k in ('added', 'changed', 'removed'):
        delta[k].sort()
    return delta


# 4. embeddings ------------------------------------------------------------------------------------
class EmbeddingCache:
    """Vector reuse keyed first by (problem_key, embedded text), then by text alone, per model revision.

    Identical texts encoded in different batches differ by ~1e-7 (batch padding), so an unchanged node
    keeps its own previous vector bit-for-bit; only new nodes fall back to a shared-text vector, and
    only texts never seen are encoded.
    """

    def __init__(self, keys, texts, vectors, model_revision=MODEL_REVISION):
        self.model_revision = model_revision
        self.by_node, self.by_text = {}, {}
        for key, text, vector in zip(keys, texts, vectors):
            self.by_node[(key, self.key(text))] = vector
            self.by_text.setdefault(self.key(text), vector)

    def key(self, text):
        return sha256_bytes((self.model_revision + '\0' + text).encode('utf-8'))

    def vectors(self, keys, texts, encode):
        missing = sorted({t for k, t in zip(keys, texts) if (k, self.key(t)) not in self.by_node and self.key(t) not in self.by_text})
        if missing:
            encoded = encode(missing)
            if encoded.shape[0] != len(missing):
                raise ValueError('encoder returned wrong row count')
            for text, vector in zip(missing, encoded):
                self.by_text[self.key(text)] = np.asarray(vector, dtype=np.float32)
        rows = [self.by_node.get((k, self.key(t)), self.by_text.get(self.key(t))) for k, t in zip(keys, texts)]
        return np.vstack(rows).astype(np.float32), len(missing)


def exact_neighbors(keys, platforms, representative, groups, vectors, k=STORED_K):
    from research.unified_map import full_quality as fq
    from research.full_catalog.retrieve import dense_cosine
    rows = fq.Rows(keys, platforms, representative, groups)
    out = {}
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for mode, name in (('all', 'any'), ('cross', 'cross')):
            out[name] = fq.neighbor_arrays(rows, dense_cosine(vectors), k, mode)
    return out


# 5. orchestration ---------------------------------------------------------------------------------
def previous_state(previous_version, root=MASTERY):
    prev_run = Path(root) / previous_version / 'run'
    if prev_run.exists():
        rows = [json.loads(l) for l in (prev_run / 'rows.jsonl').read_text().splitlines() if l.strip()]
        emb = prev_run
    else:
        rows = [dict(json.loads(l, object_pairs_hook=builder._drop_statements), status='active', first_seen_version=previous_version)
                for l in V1_ROWS.read_text().splitlines() if l.strip()]
        emb = V1_EMBEDDING
    keys = json.loads((emb / 'keys.json').read_text())
    texts = {}
    for line in (emb / 'embedded_text.jsonl').read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        texts[item['problem_key']] = item
    meta = {json.loads(l)['problem_key']: json.loads(l) for l in (emb / 'vector_meta.jsonl').read_text().splitlines() if l.strip()}
    vectors = np.load(emb / 'vectors.npy', allow_pickle=False)
    return rows, keys, texts, meta, vectors


def refresh(previous_version, version, *, raw_dir, encoder=None, translator=None, tokenizer=None, problem_concepts=None,
            concept_map=None, manual_additions=None, archive=None, output_root=MASTERY, now=None):
    out_dir = Path(output_root) / version
    if out_dir.exists():
        raise FileExistsError(out_dir)
    manifest, bodies = load_raw(raw_dir)
    external, rejects = normalize_external(bodies)
    prev_rows, prev_keys, prev_texts, prev_meta, prev_vectors = previous_state(previous_version, output_root)
    rules = json.loads((ROOT / 'research/full_catalog/exclusions.v1.json').read_text())['rules']
    aliases = json.loads((ROOT / 'assets/problem_identity_aliases.v1.json').read_text())['aliases']
    lc_rows = [{k: r.get(k) for k in ('problem_key', 'platform', 'title', 'tags', 'native_difficulty', 'url', 'contest')} for r in prev_rows if r['platform'] == 'leetcode']
    manual = []
    if manual_additions:
        document = json.loads(Path(manual_additions).read_text())
        for item in document['rows']:
            if item['problem_key'] in {r['problem_key'] for r in lc_rows + external}:
                continue
            manual.append({k: item.get(k) for k in ('problem_key', 'platform', 'title', 'tags', 'native_difficulty', 'url', 'contest')})
    catalog_rows, groups, _ = fullcat.build_catalog(lc_rows + manual, external, rules, aliases)
    prev_by_key = {r['problem_key']: r for r in prev_rows}
    delta = classify_delta([r for r in prev_rows if r['platform'] != 'leetcode'], [r for r in catalog_rows if r['platform'] != 'leetcode'])
    manual_keys = {m['problem_key'] for m in manual}
    for row in catalog_rows:
        prior = prev_by_key.get(row['problem_key'])
        row['status'] = 'active'
        row['first_seen_version'] = prior.get('first_seen_version', previous_version) if prior else version
        row['provenance'] = 'user_entered:manual-additions' if row['problem_key'] in manual_keys else prior.get('provenance', f'catalog:{previous_version}') if prior and row['platform'] == 'leetcode' else f'refresh:{version}'
    for key in delta['removed']:
        kept = dict(prev_by_key[key], status='removed_upstream')
        catalog_rows.append(kept)
    catalog_rows.sort(key=lambda r: r['problem_key'])
    # embedded text: reuse previous text for unchanged titles; translate new/changed titles
    from research.language_bias import languages as L
    from research.language_bias.translate import translate_all
    from research.language_bias.rebuild import reliability
    texts, sources_used, need = [], [], []
    for row in catalog_rows:
        prior = prev_by_key.get(row['problem_key'])
        if prior and prior.get('title') == row['title'] and row['problem_key'] in prev_texts:
            texts.append(prev_texts[row['problem_key']]['text'])
            sources_used.append(prev_texts[row['problem_key']]['source'])
        else:
            texts.append(None)
            sources_used.append(None)
            need.append(len(texts) - 1)
    if need:
        titles = [catalog_rows[i]['title'] for i in need]
        classes = [L.language_class(t) for t in titles]
        translated = translate_all(titles, classes, translator)
        for i, item in zip(need, translated):
            texts[i], sources_used[i] = item['text'], item['source']
    cache = EmbeddingCache(prev_keys, [prev_texts[k]['text'] for k in prev_keys], prev_vectors)
    keys = [r['problem_key'] for r in catalog_rows]
    vectors, encoded = cache.vectors(keys, texts, encoder.encode if encoder else (lambda t: (_ for _ in ()).throw(RuntimeError('encoder required for new titles'))))
    keys = [r['problem_key'] for r in catalog_rows]
    meta = []
    for i, row in enumerate(catalog_rows):
        prior = prev_meta.get(row['problem_key'])
        if prior and i not in set(need):
            meta.append(prior)
            continue
        ids = tokenizer(texts[i]) if tokenizer else None
        unk = ids['unk_share'] if ids else 0.0
        meta.append({'problem_key': row['problem_key'], 'vector_source': 'c2_minilm_on_translated_title', 'unk_share': round(float(unk), 4),
                     'reliability': reliability(sources_used[i], unk), 'token_count': ids['count'] if ids else 0,
                     'embedded_source': sources_used[i], 'language_class': L.language_class(row['title'])})
    neighbors = exact_neighbors(keys, [r['platform'] for r in catalog_rows], [r.get('representative_key') or r['problem_key'] for r in catalog_rows],
                                [r.get('duplicate_candidate_group') for r in catalog_rows], vectors)
    run = out_dir / 'run'
    run.mkdir(parents=True)
    write_new(run / 'rows.jsonl', ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in catalog_rows))
    write_new(run / 'keys.json', json.dumps(keys, ensure_ascii=False) + '\n')
    with open(run / 'vectors.npy', 'xb') as handle:
        np.save(handle, vectors, allow_pickle=False)
    write_new(run / 'embedded_text.jsonl', ''.join(json.dumps({'problem_key': k, 'text': t, 'source': s}, ensure_ascii=False, sort_keys=True) + '\n'
                                                   for k, t, s in zip(keys, texts, sources_used)))
    write_new(run / 'vector_meta.jsonl', ''.join(json.dumps(m, sort_keys=True) + '\n' for m in meta))
    (run / 'neighbors').mkdir()
    nb_receipt = {}
    for name, (idx, sc) in neighbors.items():
        for part, arr in (('idx', idx), ('score', sc)):
            with open(run / 'neighbors' / f'{name}-{part}.npy', 'xb') as handle:
                np.save(handle, arr, allow_pickle=False)
        nb_receipt[name] = {f'{p}_sha256': builder.sha256_file(run / 'neighbors' / f'{name}-{p}.npy') for p in ('idx', 'score')}
    write_new(run / 'neighbors' / 'receipt.json', json.dumps(nb_receipt, indent=1, sort_keys=True))
    contests = attach_contest_problems(contests_from(bodies), [r for r in catalog_rows if r['status'] == 'active'], bodies)
    starts = {c['contest_id']: c['start_at'] for c in contests}
    for r in catalog_rows:
        r['contest_start'] = starts.get(str(r.get('contest'))) if r.get('contest') is not None else None
    catalog_jsonl = run / 'catalog-input.jsonl'
    write_new(catalog_jsonl, ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in catalog_rows))
    inputs = {'catalog': catalog_jsonl, 'keys': run / 'keys.json', 'vector_meta': run / 'vector_meta.jsonl', 'neighbors_dir': run / 'neighbors',
              'aliases': ROOT / 'assets/problem_identity_aliases.v1.json', 'concepts': ROOT / 'assets/mastery/concepts.v1.json'}
    if 'atcoder_models' in bodies:
        inputs['atcoder_models'] = Path(raw_dir) / 'atcoder_models.raw'
    else:
        inputs['atcoder_models'] = builder.ROOT / builder.DEFAULTS['atcoder_models']
    extra = json.loads(Path(problem_concepts).read_text())['rows'] if problem_concepts else None
    db_dir = out_dir / 'db'
    archive_rows = None
    if archive:
        from research.mastery_catalog.archive import compile_rows
        archive_rows = compile_rows(json.loads(Path(archive).read_text(encoding='utf-8')))
    receipt = builder.build(db_dir, version, inputs, parent=previous_version, concept_map=concept_map, contests=contests, problem_concepts=extra,
                            archive=archive_rows, verify_atcoder=False, built_at=now)
    # catalog file lives at <root>/<version>/<version>.sqlite3 for the CLI
    (db_dir / f'{version}.sqlite3').rename(out_dir / f'{version}.sqlite3')
    receipt_path = out_dir / 'receipt.json'
    write_new(receipt_path, (db_dir / 'receipt.json').read_text())
    (db_dir / 'receipt.json').unlink()
    db_dir.rmdir()
    refresh_receipt = {'schema': 'mastery-refresh-receipt.v1', 'version': version, 'previous_version': previous_version, 'raw_manifest': manifest,
                       'rejects': rejects, 'delta': {k: (len(v) if isinstance(v, list) else v) for k, v in delta.items()},
                       'delta_keys': {k: v for k, v in delta.items() if isinstance(v, list)}, 'manual_additions': sorted(manual_keys),
                       'encoded_titles': encoded, 'translated_titles': len(need), 'model_revision': MODEL_REVISION,
                       'neighbors': nb_receipt, 'catalog': {'output_sha256': receipt['output_sha256'], 'counts': receipt['counts']},
                       'contests': len(contests), 'statement_fields_read': False}
    write_new(out_dir / 'refresh-receipt.json', json.dumps(refresh_receipt, indent=1, sort_keys=True))
    return refresh_receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--previous', required=True)
    parser.add_argument('--version', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--live', action='store_true', help='NETWORK: fetch documented metadata endpoints (approved weekly refresh)')
    mode.add_argument('--replay', type=Path, help='replay a stored raw directory with manifest.json')
    parser.add_argument('--concept-map', type=Path, default=ROOT / 'assets/mastery/concept-map.v1.json')
    parser.add_argument('--problem-concepts', type=Path, default=ROOT / 'assets/mastery/atcoder-pilot-concepts.v1.json')
    parser.add_argument('--manual-additions', type=Path, help='learner-entered rows {"rows": [...]}, provenance user_entered')
    parser.add_argument('--archive', type=Path, help='learner archive JSON (mastery-archive.v1)')
    parser.add_argument('--public-receipt', type=Path)
    args = parser.parse_args(argv)
    raw_dir = args.replay
    if args.live:
        raw_dir = MASTERY / 'raw' / f'{args.version}-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
        print(json.dumps(fetch_all(raw_dir), indent=1))
    from research.language_bias.encoders import MiniLMEncoder
    minilm = MiniLMEncoder()
    tok = minilm.encoder.tokenizer

    def token_info(text):
        ids = tok(text, add_special_tokens=False)['input_ids']
        return {'count': len(ids), 'unk_share': (sum(1 for i in ids if i == tok.unk_token_id) / len(ids)) if ids else 1.0}

    class LazyTranslator:
        def __init__(self):
            self.mt = None

        def __call__(self, texts):
            if self.mt is None:
                from research.language_bias.translate import MarianTranslator
                self.mt = MarianTranslator()
            return self.mt(texts)

    receipt = refresh(args.previous, args.version, raw_dir=raw_dir, encoder=minilm, translator=LazyTranslator(), tokenizer=token_info,
                      problem_concepts=args.problem_concepts, concept_map=args.concept_map, manual_additions=args.manual_additions,
                      archive=args.archive)
    if args.public_receipt:
        public = {k: v for k, v in receipt.items() if k != 'delta_keys'}
        public['raw_manifest'] = {k: {f: e.get(f) for f in ('url', 'http_status', 'status', 'sha256', 'bytes', 'retrieved_at')} for k, e in receipt['raw_manifest'].items()}
        write_new(args.public_receipt, json.dumps(public, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: receipt[k] for k in ('version', 'delta', 'encoded_titles', 'translated_titles', 'contests')}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
