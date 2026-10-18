"""Offline relatedness evaluation (task MS-901, design.md section 8). Never runs in the runtime core.

    python3 -m research.mastery_catalog.relatedness export --db private-learning/hosted.sqlite3 --catalog <catalog.sqlite3> --output batch.json
    python3 -m research.mastery_catalog.relatedness merge --batch batch.json --draft d.json --review r.json --draft-model M1 --review-model M2 --output judgments.json
    python3 -m research.mastery_catalog.relatedness agreement --db private-learning/hosted.sqlite3 --judgments judgments.json

`export` collects related-practice candidates actually shown in plans plus a deterministic 10% sample of
unshown neighbor candidates (titles/tags only, no statements). `merge` records two independent AI
judgments (agreement -> ai_reviewed verdict, disagreement -> abstained). `agreement` compares AI verdicts
with the learner's own verdicts. Learner verdicts are the primary signal; AI judgments are secondary,
and channels are never evaluated with labels derived from the same channel's features.
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from research.mastery_catalog.judgments import validate_record

VERDICTS = ('related', 'partly', 'unrelated')
SAMPLE_RATE = 0.10


def _ro(path):
    return sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True)


def shown_pairs(db_path, learner_id='local-owner'):
    con = _ro(db_path)
    pairs = []
    for (payload,) in con.execute("SELECT payload FROM recommendation_snapshots WHERE learner_id=? AND kind='plan' ORDER BY local_date", (learner_id,)):
        plan = json.loads(payload)
        for slot in plan.get('slots', []):
            if slot.get('sub_kind') == 'related_channel' and slot.get('components', {}).get('source_key'):
                pairs.append({'source_key': slot['components']['source_key'], 'candidate_key': slot['problem_key'],
                              'channel': slot['components'].get('channel'), 'shown': True, 'plan_date': plan['date']})
    return pairs


def sample_unshown(catalog_path, shown, rate=SAMPLE_RATE, salt='relatedness-v1'):
    """Deterministic hash sample of other neighbor candidates for the same sources."""
    con = _ro(catalog_path)
    have = {(p['source_key'], p['candidate_key']) for p in shown}
    out = []
    for source in sorted({p['source_key'] for p in shown}):
        for dst, channel, rank in con.execute("SELECT dst, channel, rank FROM neighbors WHERE src=? AND channel IN ('title_v2','title_v2_cross') ORDER BY channel, rank", (source,)):
            if (source, dst) in have:
                continue
            bucket = int(hashlib.sha256(f'{salt}\0{source}\0{dst}'.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            if bucket < rate:
                have.add((source, dst))
                out.append({'source_key': source, 'candidate_key': dst, 'channel': channel, 'shown': False, 'rank': rank})
    return out


def export_batch(db_path, catalog_path, learner_id='local-owner'):
    shown = shown_pairs(db_path, learner_id)
    items = shown + sample_unshown(catalog_path, shown)
    con = _ro(catalog_path)
    describe = lambda key: (lambda row: {'key': key, 'title': row[0] if row else None, 'platform': key.split(':')[0],
                                         'tags': [t for (t,) in con.execute('SELECT native_tag FROM problem_tags WHERE problem_key=? ORDER BY native_tag', (key,))]})(
        con.execute('SELECT title FROM problems WHERE problem_key=?', (key,)).fetchone())
    for i, item in enumerate(items):
        item['item_id'] = f"rel-{hashlib.sha256((item['source_key'] + '|' + item['candidate_key']).encode()).hexdigest()[:16]}"
        item['source'] = describe(item['source_key'])
        item['candidate'] = describe(item['candidate_key'])
    return {'schema': 'mastery-relatedness-batch.v1', 'sample_rate': SAMPLE_RATE, 'items': items,
            'instructions': 'Judge whether practicing the candidate is likely to exercise the same technique as the source. '
                            'Verdict related/partly/unrelated/abstain with a non-empty rationale. Titles and tags only; no statements.'}


def merge(batch, draft, review, meta):
    ids = {i['item_id'] for i in batch['items']}
    def index(entries):
        by = {}
        for e in entries:
            if set(e) != {'item_id', 'verdict', 'rationale'} or e['item_id'] not in ids or e['item_id'] in by:
                raise ValueError(f"invalid relatedness entry: {e.get('item_id')}")
            if e['verdict'] not in VERDICTS + ('abstain',) or not str(e['rationale']).strip():
                raise ValueError(f"invalid verdict or empty rationale: {e['item_id']}")
            by[e['item_id']] = e
        if set(by) != ids:
            raise ValueError('relatedness coverage mismatch')
        return by
    d, r = index(draft), index(review)
    records, merged = [], []
    for item in batch['items']:
        a, b = d[item['item_id']], r[item['item_id']]
        agreed = a['verdict'] == b['verdict'] and a['verdict'] != 'abstain'
        merged.append({'item_id': item['item_id'], 'source_key': item['source_key'], 'candidate_key': item['candidate_key'], 'channel': item['channel'],
                       'shown': item['shown'], 'verdict': a['verdict'] if agreed else None, 'status': 'ai_reviewed' if agreed else 'abstained'})
        for role, e in (('draft', a), ('review', b)):
            records.append(validate_record({'judgment_id': f"{item['item_id']}:{role}", 'object_kind': 'relatedness', 'object_ref': f"{item['source_key']}->{item['candidate_key']}",
                'model': meta[role], 'role': role, 'run_date': meta['run_date'], 'prompt_sha256': meta['prompt_sha256'], 'input_sha256': meta['input_sha256'],
                'verdict': e['verdict'], 'rationale': e['rationale']}))
    by_channel = {}
    for m in merged:
        c = by_channel.setdefault(m['channel'] or 'unknown', {'judged': 0, 'related_or_partly': 0, 'abstained': 0})
        if m['status'] == 'abstained':
            c['abstained'] += 1
        else:
            c['judged'] += 1
            c['related_or_partly'] += m['verdict'] in ('related', 'partly')
    return {'schema': 'mastery-relatedness-judgments.v1', 'not_gold': True, 'items': merged, 'records': records, 'precision_by_channel': by_channel}


def agreement(learner_verdicts, ai_items):
    """Agreement between learner verdicts and ai_reviewed verdicts on the same pairs (descriptive)."""
    ai = {(i['source_key'], i['candidate_key']): i['verdict'] for i in ai_items if i['status'] == 'ai_reviewed'}
    pairs = [(v['verdict'], ai[(v['source_key'], v['candidate_key'])]) for v in learner_verdicts
             if v['verdict'] in VERDICTS and (v['source_key'], v['candidate_key']) in ai]
    n = len(pairs)
    if not n:
        return {'n': 0, 'status': 'insufficient_evidence', 'note': 'No pair has both a learner verdict and an ai_reviewed verdict.'}
    observed = sum(a == b for a, b in pairs) / n
    expected = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in VERDICTS)
    kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    return {'n': n, 'status': 'estimated' if n >= 20 else 'insufficient_evidence', 'observed_agreement': round(observed, 3),
            'cohen_kappa': None if kappa is None else round(kappa, 3), 'confusion': {f'{a}|{b}': sum(1 for x in pairs if x == (a, b)) for a in VERDICTS for b in VERDICTS}}


def learner_verdicts(db_path, learner_id='local-owner'):
    con = _ro(db_path)
    rows = [json.loads(c) for (c,) in con.execute('SELECT canonical FROM practice_feedback WHERE learner_id=? ORDER BY sequence', (learner_id,))]
    superseded = {r['supersedes'] for r in rows if r.get('supersedes')}
    return [{'source_key': r['payload']['source_key'], 'candidate_key': r['payload']['candidate_key'], 'verdict': r['payload']['verdict']}
            for r in rows if r['entry_id'] not in superseded and r['payload']['source_key']]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export'); e.add_argument('--db', required=True); e.add_argument('--catalog', required=True); e.add_argument('--output', type=Path, required=True)
    m = sub.add_parser('merge')
    for name in ('batch', 'draft', 'review', 'output'):
        m.add_argument(f'--{name}', type=Path, required=True)
    m.add_argument('--draft-model', required=True); m.add_argument('--review-model', required=True); m.add_argument('--run-date', required=True)
    a = sub.add_parser('agreement'); a.add_argument('--db', required=True); a.add_argument('--judgments', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'export':
        batch = export_batch(args.db, args.catalog)
        with open(args.output, 'x', encoding='utf-8') as handle:
            json.dump(batch, handle, ensure_ascii=False, indent=1)
        print(json.dumps({'items': len(batch['items']), 'shown': sum(i['shown'] for i in batch['items'])}))
    elif args.command == 'merge':
        batch = json.loads(args.batch.read_text())
        meta = {'draft': args.draft_model, 'review': args.review_model, 'run_date': args.run_date,
                'prompt_sha256': hashlib.sha256(batch['instructions'].encode()).hexdigest(), 'input_sha256': hashlib.sha256(args.batch.read_bytes()).hexdigest()}
        result = merge(batch, json.loads(args.draft.read_text()), json.loads(args.review.read_text()), meta)
        with open(args.output, 'x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=1)
        print(json.dumps(result['precision_by_channel']))
    else:
        print(json.dumps(agreement(learner_verdicts(args.db), json.loads(args.judgments.read_text())['items']), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
