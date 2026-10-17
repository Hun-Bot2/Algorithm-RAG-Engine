"""Build an immutable catalog-vN.sqlite3 for the mastery system (tasks MS-301, MS-704).

Reads existing private research artifacts (never modifies them), drops statement fields
while parsing, verifies input hashes against their receipts, and refuses to overwrite.
Outputs stay under private-learning/ (titles and URLs are private).
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
STATEMENT_FIELDS = {'content', 'content_clean', 'full_statement', 'statement', 'description'}
LC_LEVEL = {'Easy': 1, 'Medium': 2, 'Hard': 3}
ATC_SENTINEL_FLOOR = -2000  # research/mastery_catalog/atcoder-model-provenance.v1.json outlier rule

DEFAULTS = {
    'catalog': 'private-learning/full-catalog/catalog-v1/catalog.v1.jsonl',
    'keys': 'private-learning/unified-map/full-v2/keys.json',
    'vector_meta': 'private-learning/unified-map/full-v2/vector_meta.jsonl',
    'neighbors_dir': 'private-learning/unified-map/full-v2/neighbors',
    'atcoder_models': 'private-learning/algorithm-system/atcoder-problem-models-2026-10-16-v1/problem-models.json',
    'aliases': 'assets/problem_identity_aliases.v1.json',
    'concepts': 'assets/mastery/concepts.v1.json',
}
ATCODER_MODELS_SHA256 = '2bdb4d6cdde49d76a40dceb8e02be5ad59f86717bf83b57374de625ffb890a58'

SCHEMA = """
CREATE TABLE catalog_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE problems (
  problem_key TEXT PRIMARY KEY, platform TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL,
  contest_id TEXT, contest_start TEXT, difficulty_kind TEXT NOT NULL CHECK (difficulty_kind IN ('lc_level','cf_rating','atc_irt','none')),
  difficulty_value REAL, atc_discrimination REAL, atc_experimental INTEGER,
  access TEXT NOT NULL CHECK (access IN ('free','free_at_collection','premium','unknown')),
  eligible INTEGER NOT NULL CHECK (eligible IN (0,1)), exclusion_reason TEXT,
  status TEXT NOT NULL CHECK (status IN ('active','removed_upstream')),
  first_seen_version TEXT NOT NULL, row_hash TEXT NOT NULL, provenance TEXT NOT NULL
);
CREATE INDEX problems_platform ON problems(platform, difficulty_value);
CREATE TABLE problem_tags (problem_key TEXT NOT NULL REFERENCES problems(problem_key), native_tag TEXT NOT NULL, PRIMARY KEY (problem_key, native_tag));
CREATE TABLE aliases (alias_key TEXT PRIMARY KEY, canonical_key TEXT NOT NULL, evidence TEXT NOT NULL);
CREATE TABLE concepts (concept_id TEXT PRIMARY KEY, label TEXT NOT NULL);
CREATE TABLE concept_map (platform TEXT NOT NULL, native_tag TEXT NOT NULL, concept_id TEXT NOT NULL REFERENCES concepts(concept_id),
  evidence_status TEXT NOT NULL CHECK (evidence_status IN ('approved','ai_reviewed','abstained','proposed')), judgment_ref TEXT NOT NULL,
  PRIMARY KEY (platform, native_tag, concept_id));
CREATE TABLE problem_concepts (problem_key TEXT NOT NULL REFERENCES problems(problem_key), concept_id TEXT NOT NULL,
  evidence_status TEXT NOT NULL, via TEXT NOT NULL, PRIMARY KEY (problem_key, concept_id, via));
CREATE INDEX problem_concepts_concept ON problem_concepts(concept_id, evidence_status);
CREATE TABLE neighbors (src TEXT NOT NULL, dst TEXT NOT NULL, channel TEXT NOT NULL, rank INTEGER NOT NULL, score REAL NOT NULL,
  PRIMARY KEY (src, channel, rank));
CREATE INDEX neighbors_dst ON neighbors(dst, channel);
CREATE TABLE reliability (problem_key TEXT PRIMARY KEY, tier TEXT NOT NULL, language_class TEXT NOT NULL);
CREATE TABLE contests (contest_id TEXT PRIMARY KEY, platform TEXT NOT NULL, title TEXT NOT NULL, kind TEXT NOT NULL,
  start_at TEXT, duration_minutes INTEGER, url TEXT NOT NULL);
CREATE TABLE contest_problems (contest_id TEXT NOT NULL, problem_key TEXT NOT NULL, label TEXT, PRIMARY KEY (contest_id, problem_key));
CREATE TABLE archive_records (record_id TEXT NOT NULL, version INTEGER NOT NULL, archive_version TEXT NOT NULL, category TEXT NOT NULL,
  organization TEXT, exam TEXT, year INTEGER, rights_status TEXT NOT NULL, confidence TEXT NOT NULL, recorded_by TEXT NOT NULL,
  source_kind TEXT NOT NULL, acquisition TEXT NOT NULL, superseded INTEGER NOT NULL, record TEXT NOT NULL, PRIMARY KEY (record_id, version));
CREATE TABLE archive_links (record_id TEXT NOT NULL, version INTEGER NOT NULL, problem_key TEXT NOT NULL, relation TEXT NOT NULL,
  status TEXT NOT NULL, evidence TEXT NOT NULL, PRIMARY KEY (record_id, version, problem_key, relation));
CREATE TABLE ai_judgments (judgment_id TEXT PRIMARY KEY, object_kind TEXT NOT NULL, object_ref TEXT NOT NULL, record TEXT NOT NULL);
"""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def _drop_statements(pairs):
    # Statement text is never loaded (quarantine rule, research/full_catalog/catalog.py precedent).
    return {k: v for k, v in pairs if k not in STATEMENT_FIELDS}


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line, object_pairs_hook=_drop_statements)


def difficulty(row, models):
    native = row.get('native_difficulty')
    if row['platform'] == 'leetcode':
        return ('lc_level', LC_LEVEL[native], None, None) if native in LC_LEVEL else ('none', None, None, None)
    if row['platform'] == 'codeforces':
        return ('cf_rating', float(native), None, None) if isinstance(native, (int, float)) else ('none', None, None, None)
    model = models.get(row['problem_key'].split(':', 1)[1])
    if model and isinstance(model.get('difficulty'), (int, float)) and model['difficulty'] >= ATC_SENTINEL_FLOOR and isinstance(model.get('discrimination'), (int, float)):
        return ('atc_irt', float(model['difficulty']), float(model['discrimination']), int(bool(model.get('is_experimental'))))
    return ('none', None, None, None)


def problem_record(row, models, version, provenance):
    kind, value, discrimination, experimental = difficulty(row, models)
    record = {
        'problem_key': row['problem_key'], 'platform': row['platform'], 'title': row['title'], 'url': row['url'],
        'contest_id': None if row.get('contest') is None else str(row['contest']), 'contest_start': row.get('contest_start'),
        'difficulty_kind': kind, 'difficulty_value': value, 'atc_discrimination': discrimination, 'atc_experimental': experimental,
        'access': 'free_at_collection' if row['platform'] == 'leetcode' else 'free',
        'eligible': 1 if row.get('eligibility') == 'eligible' else 0,
        'exclusion_reason': ';'.join(row.get('exclusion_reasons') or []) or None,
        'status': row.get('status', 'active'), 'tags': sorted(row.get('tags') or []),
    }
    record['row_hash'] = hashlib.sha256(canonical(record).encode()).hexdigest()
    record['first_seen_version'] = row.get('first_seen_version', version)
    record['provenance'] = provenance
    return record


def load_concept_map(path, vocabulary):
    if not path:
        return []
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if document.get('schema') != 'mastery-concept-map.v1':
        raise ValueError('unsupported concept map schema')
    rows = []
    for entry in document['entries']:
        for concept in entry['concepts']:
            if concept['concept_id'] not in vocabulary:
                raise ValueError(f"unknown concept {concept['concept_id']}")
            rows.append((entry['platform'], entry['native_tag'], concept['concept_id'], concept['evidence_status'], entry['judgment_ref']))
    return rows


def build(output, version, inputs, parent=None, parent_catalog=None, concept_map=None, contests=None, archive=None,
          problem_concepts=None, ai_judgments=None, built_at=None, extra_rows=None, verify_atcoder=True):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    paths = {name: Path(inputs[name]) for name in inputs}
    hashes = {name: sha256_file(path) for name, path in paths.items() if path.is_file()}
    if 'atcoder_models' in hashes and verify_atcoder and hashes['atcoder_models'] != ATCODER_MODELS_SHA256:
        raise ValueError('AtCoder problem-models hash differs from the validated provenance receipt')
    neighbor_receipt = paths.get('neighbors_dir') and json.loads((paths['neighbors_dir'] / 'receipt.json').read_text())
    if neighbor_receipt:
        for channel in ('any', 'cross'):
            for part in ('idx', 'score'):
                actual = sha256_file(paths['neighbors_dir'] / f'{channel}-{part}.npy')
                if actual != neighbor_receipt[channel][f'{part}_sha256']:
                    raise ValueError(f'neighbor {channel}-{part} hash mismatch')
                hashes[f'neighbors_{channel}_{part}'] = actual

    models = json.loads(paths['atcoder_models'].read_text()) if 'atcoder_models' in paths else {}
    vocabulary = {c['id']: c['label'] for c in json.loads(paths['concepts'].read_text())['concepts']}
    provenance = f'catalog:{paths["catalog"].name}'
    rows = list(read_jsonl(paths['catalog']))
    if extra_rows:
        rows.extend(extra_rows)
    keys = [r['problem_key'] for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate problem keys')
    records = [problem_record(r, models, version, r.get('provenance', provenance)) for r in rows]

    output.mkdir(parents=True, mode=0o700)
    db_path = output / f'{version}.sqlite3'
    con = sqlite3.connect(db_path)
    try:
        con.executescript('BEGIN;' + SCHEMA + 'COMMIT;')
        con.execute('BEGIN')
        con.executemany('INSERT INTO concepts VALUES (?,?)', sorted(vocabulary.items()))
        con.executemany('INSERT INTO problems VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', [
            (r['problem_key'], r['platform'], r['title'], r['url'], r['contest_id'], r['contest_start'], r['difficulty_kind'],
             r['difficulty_value'], r['atc_discrimination'], r['atc_experimental'], r['access'], r['eligible'], r['exclusion_reason'],
             r['status'], r['first_seen_version'], r['row_hash'], r['provenance']) for r in records])
        con.executemany('INSERT INTO problem_tags VALUES (?,?)', [(r['problem_key'], t) for r in records for t in r['tags']])
        known = set(keys)
        aliases = json.loads(paths['aliases'].read_text())['aliases'] if 'aliases' in paths else []
        con.executemany('INSERT INTO aliases VALUES (?,?,?)', [(a['legacy_key'], a['native_key'], a['evidence']) for a in aliases])
        if 'vector_meta' in paths:
            con.executemany('INSERT INTO reliability VALUES (?,?,?)', [
                (m['problem_key'], m['reliability'], m['language_class']) for m in read_jsonl(paths['vector_meta']) if m['problem_key'] in known])
        neighbor_count = 0
        if 'neighbors_dir' in paths:
            import numpy as np
            order = json.loads(paths['keys'].read_text())
            for channel, name in (('title_v2', 'any'), ('title_v2_cross', 'cross')):
                idx = np.load(paths['neighbors_dir'] / f'{name}-idx.npy')
                score = np.load(paths['neighbors_dir'] / f'{name}-score.npy')
                batch = []
                for i, src in enumerate(order):
                    if src not in known:
                        continue
                    for rank, (j, s) in enumerate(zip(idx[i].tolist(), score[i].tolist()), start=1):
                        if j < 0 or order[j] not in known:
                            continue
                        batch.append((src, order[j], channel, rank, round(float(s), 6)))
                con.executemany('INSERT INTO neighbors VALUES (?,?,?,?,?)', batch)
                neighbor_count += len(batch)
        # No statement channel: the CodeContests statement run (statement-run-v1) retrieves only across
        # platforms and its AtCoder rows lack native identities, so no edge joins two catalog nodes.
        cmap = load_concept_map(concept_map, vocabulary)
        con.executemany('INSERT INTO concept_map VALUES (?,?,?,?,?)', cmap)
        con.execute("""INSERT INTO problem_concepts SELECT DISTINCT p.problem_key, m.concept_id, m.evidence_status, 'tag:' || m.native_tag
                       FROM problem_tags t JOIN problems p ON p.problem_key=t.problem_key
                       JOIN concept_map m ON m.platform=p.platform AND m.native_tag=t.native_tag""")
        for item in problem_concepts or []:
            if item['problem_key'] in known and item['concept_id'] in vocabulary:
                con.execute('INSERT OR IGNORE INTO problem_concepts VALUES (?,?,?,?)', (item['problem_key'], item['concept_id'], item['evidence_status'], item['via']))
        for contest in contests or []:
            con.execute('INSERT INTO contests VALUES (?,?,?,?,?,?,?)', (contest['contest_id'], contest['platform'], contest['title'], contest['kind'],
                                                                         contest.get('start_at'), contest.get('duration_minutes'), contest['url']))
            for problem_key, label in contest.get('problems', []):
                if problem_key in known:
                    con.execute('INSERT OR IGNORE INTO contest_problems VALUES (?,?,?)', (contest['contest_id'], problem_key, label))
        for record in archive or []:
            con.execute('INSERT INTO archive_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', record['row'])
            for link in record['links']:
                con.execute('INSERT INTO archive_links VALUES (?,?,?,?,?,?)', link)
        for judgment in ai_judgments or []:
            con.execute('INSERT INTO ai_judgments VALUES (?,?,?,?)', (judgment['judgment_id'], judgment['object_kind'], judgment['object_ref'], canonical(judgment)))
        counts = {
            'problems': len(records),
            'by_platform': {p: sum(1 for r in records if r['platform'] == p) for p in sorted({r['platform'] for r in records})},
            'eligible': sum(r['eligible'] for r in records),
            'difficulty_kind': {k: sum(1 for r in records if r['difficulty_kind'] == k) for k in ('lc_level', 'cf_rating', 'atc_irt', 'none')},
            'neighbors': neighbor_count,
            'neighbor_channels': dict(con.execute('SELECT channel, COUNT(*) FROM neighbors GROUP BY channel').fetchall()),
            'concept_map_rows': len(cmap),
            'problem_concepts': dict(con.execute('SELECT evidence_status, COUNT(*) FROM problem_concepts GROUP BY evidence_status').fetchall()),
            'problems_with_concepts': con.execute("SELECT COUNT(DISTINCT problem_key) FROM problem_concepts WHERE evidence_status IN ('approved','ai_reviewed')").fetchone()[0],
            'contests': len(contests or []), 'archive_records': len(archive or []), 'ai_judgments': len(ai_judgments or []),
            'status': {s: sum(1 for r in records if r['status'] == s) for s in ('active', 'removed_upstream')},
        }
        meta = {
            'schema_version': str(SCHEMA_VERSION), 'version': version, 'parent_version': parent or '',
            'built_at': built_at or datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'input_sha256': canonical(hashes), 'counts': canonical(counts),
            'statement_fields_read': 'false',
        }
        con.executemany('INSERT INTO catalog_meta VALUES (?,?)', sorted(meta.items()))
        con.commit()
    finally:
        con.close()
    os.chmod(db_path, 0o400)
    digest = sha256_file(db_path)
    receipt = {'schema': 'mastery-catalog-receipt.v1', 'version': version, 'parent_version': parent, 'built_at': meta['built_at'],
               'output': db_path.name, 'output_sha256': digest, 'input_sha256': hashes, 'counts': counts,
               'statement_fields_read': False, 'concept_map': str(concept_map) if concept_map else None}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--concept-map', type=Path)
    parser.add_argument('--problem-concepts', type=Path, help='JSON list of per-problem concept rows (AtCoder AI judgments)')
    parser.add_argument('--archive', type=Path, help='learner archive JSON (mastery-archive.v1), compiled after validation')
    parser.add_argument('--public-receipt', type=Path, help='committed receipt copy (counts and hashes only)')
    args = parser.parse_args(argv)
    inputs = {name: ROOT / rel for name, rel in DEFAULTS.items()}
    extra = json.loads(args.problem_concepts.read_text())['rows'] if args.problem_concepts else None
    archive = None
    if args.archive:
        from research.mastery_catalog.archive import compile_rows
        archive = compile_rows(json.loads(args.archive.read_text(encoding='utf-8')))
    receipt = build(args.output, args.version, inputs, concept_map=args.concept_map, problem_concepts=extra, archive=archive)
    if args.public_receipt:
        if args.public_receipt.exists():
            raise FileExistsError(args.public_receipt)
        args.public_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt['counts'], indent=2))


if __name__ == '__main__':
    sys.exit(main())
