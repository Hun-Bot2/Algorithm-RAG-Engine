"""Full-catalog AtCoder concept classification, metadata only (spec: specs/mastery-system/atcoder-concepts.md).

    python3 -m research.mastery_catalog.atcoder_concepts export --catalog <catalog.sqlite3> --out-dir <dir> --batch-size 150
    python3 -m research.mastery_catalog.atcoder_concepts merge --dir <dir> --draft-model M1 --review-model M2 --run-date D --asset assets/...
    python3 -m research.mastery_catalog.atcoder_concepts confirm --file <jsonl> --problem K --concept C [--reject] --note TEXT

Judges see key, title and task letter only (no statements, no difficulty). Blind decoys are mixed into every batch:
fabricated tasks and real keys carrying another task's title. They are stripped before merging and reported.
AI output never creates `approved` or `rejected`; only the `confirm` command (a user action) does.
"""
import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
from datetime import date
from pathlib import Path

from research.mastery_catalog.judgments import validate_record

BASES = {'recognized', 'title_inference', 'unknown'}
FABRICATED_SHARE = 4   # per batch
SWAPPED_SHARE = 4
ITEM_STATUSES = ('labeled', 'proposed_only', 'abstained', 'unknown')

_ADJECTIVES = ['Amber', 'Quiet', 'Folded', 'Hollow', 'Lunar', 'Brittle', 'Tidal', 'Narrow', 'Velvet', 'Crooked', 'Silent', 'Gilded',
               'Drifting', 'Twin', 'Hidden', 'Rusted', 'Painted', 'Scattered', 'Woven', 'Distant']
_NOUNS = ['Lanterns', 'Orchard', 'Bridges', 'Ledger', 'Compass', 'Harbor', 'Mosaic', 'Kettles', 'Staircase', 'Marbles', 'Windmills',
          'Archive', 'Meadow', 'Pendulum', 'Tapestry', 'Foxes', 'Cranes', 'Chimes', 'Quarry', 'Beacons']
_SUFFIXES = ['', '', '', ' II', ' on a Ring', ' and Mirrors', ' Again', ' Festival']


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def task_letter(problem_key):
    return problem_key.rsplit('_', 1)[-1].upper()


def contest_family(problem_key):
    m = re.match(r'atcoder:([a-z]+)', problem_key)
    return m.group(1) if m else 'other'


def load_items(catalog_path):
    """Eligible AtCoder problems from an immutable catalog, sorted by key (deterministic)."""
    con = sqlite3.connect(f'file:{Path(catalog_path).resolve()}?mode=ro', uri=True)
    try:
        rows = con.execute("SELECT problem_key, title FROM problems WHERE platform='atcoder' AND eligible=1 AND status='active' ORDER BY problem_key").fetchall()
    finally:
        con.close()
    return [{'problem_key': k, 'title': t, 'task_letter': task_letter(k)} for k, t in rows]


def fabricate(items, count, rng, taken_titles, first_contest):
    """Invented tasks: made-up title, contest number past the catalog's newest ABC, never a catalog title."""
    out, used = [], set()
    while len(out) < count:
        title = f'{rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)}{rng.choice(_SUFFIXES)}'
        if title.lower() in taken_titles or title in used:
            continue
        used.add(title)
        number = first_contest + rng.randrange(0, 40)
        letter = rng.choice('CDEF')
        out.append({'problem_key': f'atcoder:abc{number}_{letter.lower()}', 'title': title, 'task_letter': letter})
    return out


def newest_abc(items):
    numbers = [int(m.group(1)) for i in items if (m := re.match(r'atcoder:abc(\d+)_', i['problem_key']))]
    return max(numbers) if numbers else 0


def make_batches(items, batch_size, seed=20261017):
    """Deterministic batches. Each real item is judged once. Controls are added per batch and shuffled in.

    Returns (batches, controls): batches[i] is the list shown to the judges (neutral item_id only); controls maps
    item_id -> {'kind': 'fabricated'|'swapped', 'real_key', 'shown_title'...} and real maps item_id -> problem_key.
    """
    rng = random.Random(seed)
    taken = {i['title'].lower() for i in items}
    first_contest = newest_abc(items) + 60
    order = list(items)
    rng.shuffle(order)   # seeded: every batch mixes contest families, so any three batches are a random sample
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    batches, truth = [], {}
    for n, chunk in enumerate(chunks):
        shown = [{'role': 'real', **i} for i in chunk]
        for f in fabricate(items, FABRICATED_SHARE, rng, taken, first_contest):
            shown.append({'role': 'fabricated', **f})
        in_batch = {i['problem_key'] for i in chunk}
        pool = [i for i in items if i['problem_key'] not in in_batch]
        for _ in range(SWAPPED_SHARE):
            host = rng.choice(pool)
            donor = rng.choice(pool)
            while donor['title'] == host['title']:
                donor = rng.choice(pool)
            shown.append({'role': 'swapped', 'problem_key': host['problem_key'], 'title': donor['title'], 'task_letter': host['task_letter'],
                          'title_owner': donor['problem_key']})
        rng.shuffle(shown)
        batch = []
        for j, entry in enumerate(shown):
            item_id = f'b{n:03d}-{j:03d}'
            truth[item_id] = {k: v for k, v in entry.items() if k not in ('title',)} | {'shown_title': entry['title']}
            batch.append({'item_id': item_id, 'problem_key': entry['problem_key'], 'title': entry['title'], 'task_letter': entry['task_letter']})
        batches.append(batch)
    return batches, truth


def export(catalog_path, out_dir, batch_size=150, families=None, exclude_truth=None, exclude_batches=None):
    out_dir = Path(out_dir)
    if (out_dir / 'truth.json').exists():
        raise FileExistsError(out_dir / 'truth.json')
    items = load_items(catalog_path)
    if families:
        items = [i for i in items if contest_family(i['problem_key']) in families]
    if exclude_truth:   # real items already judged in an earlier export
        prefixes = tuple(f'b{n:03d}-' for n in (exclude_batches or []))
        done = {t['problem_key'] for k, t in json.loads(Path(exclude_truth).read_text()).items() if t['role'] == 'real' and k.startswith(prefixes)}
        items = [i for i in items if i['problem_key'] not in done]
    batches, truth = make_batches(items, batch_size)
    (out_dir / 'in').mkdir(parents=True, exist_ok=True)
    (out_dir / 'out').mkdir(parents=True, exist_ok=True)
    for n, batch in enumerate(batches):
        (out_dir / 'in' / f'batch-{n:03d}.json').write_text(json.dumps(batch, ensure_ascii=False, indent=0) + '\n')
    (out_dir / 'truth.json').write_text(json.dumps(truth, ensure_ascii=False, indent=0, sort_keys=True) + '\n')
    return {'items': len(items), 'batches': len(batches), 'batch_size': batch_size,
            'controls_per_batch': FABRICATED_SHARE + SWAPPED_SHARE}


def check_judgment(entries, expected_ids, vocabulary, role):
    by = {}
    for e in entries:
        if not isinstance(e, dict) or set(e) != {'item_id', 'concepts', 'basis', 'rationale'}:
            raise ValueError(f'{role}: invalid entry fields: {str(e)[:80]}')
        if e['item_id'] not in expected_ids or e['item_id'] in by:
            raise ValueError(f'{role}: unexpected or duplicate item {e["item_id"]}')
        if e['basis'] not in BASES or not str(e['rationale']).strip():
            raise ValueError(f'{role}: invalid basis/rationale {e["item_id"]}')
        if e['basis'] == 'unknown' and e['concepts']:
            raise ValueError(f'{role}: unknown basis with concepts {e["item_id"]}')
        if len(e['concepts']) > 3 or len(set(e['concepts'])) != len(e['concepts']) or set(e['concepts']) - vocabulary:
            raise ValueError(f'{role}: invalid concepts {e["item_id"]}')
        by[e['item_id']] = e
    if set(by) != expected_ids:
        raise ValueError(f'{role}: coverage mismatch (missing {len(expected_ids - set(by))})')
    return by


def decide(a, b):
    """Merge rule R4 for one item. Returns (ai_reviewed, proposed, abstained) concept sets."""
    both_known = a['basis'] != 'unknown' and b['basis'] != 'unknown'
    shared = set(a['concepts']) & set(b['concepts']) if both_known else set()
    trusted = shared if 'recognized' in (a['basis'], b['basis']) else set()
    proposed = shared - trusted
    abstained = (set(a['concepts']) | set(b['concepts'])) - shared
    return trusted, proposed, abstained


def item_status(trusted, proposed, abstained):
    if trusted:
        return 'labeled'
    if proposed:
        return 'proposed_only'
    return 'abstained' if abstained else 'unknown'


def merge(truth, draft, review, vocabulary, meta, via='atcoder-full-v1'):
    """Merge two judges' per-item output. Controls are scored and removed; real items become rows and statuses."""
    expected = set(truth)
    d = check_judgment(draft, expected, vocabulary, 'draft')
    r = check_judgment(review, expected, vocabulary, 'review')
    rows, judgments, statuses = [], [], {}
    pairs = {'both_known': 0, 'both_unknown': 0, 'one_unknown': 0, 'agree_any': 0, 'disagree_known': 0}
    controls = {'fabricated': {'items': 0, 'draft_known': 0, 'review_known': 0, 'either_known': 0, 'both_known': 0, 'ai_reviewed_label': 0},
                'swapped': {'items': 0, 'draft_known': 0, 'review_known': 0, 'either_known': 0, 'both_known': 0, 'ai_reviewed_label': 0}}
    swapped_detail = []
    for item_id in sorted(truth):
        t = truth[item_id]
        a, b = d[item_id], r[item_id]
        trusted, proposed, abstained = decide(a, b)
        if t['role'] != 'real':
            c = controls[t['role']]
            c['items'] += 1
            c['draft_known'] += a['basis'] != 'unknown'
            c['review_known'] += b['basis'] != 'unknown'
            c['either_known'] += a['basis'] != 'unknown' or b['basis'] != 'unknown'
            c['both_known'] += a['basis'] != 'unknown' and b['basis'] != 'unknown'
            c['ai_reviewed_label'] += bool(trusted)
            if t['role'] == 'swapped':
                swapped_detail.append({'item_id': item_id, 'problem_key': t['problem_key'], 'title_owner': t['title_owner'],
                                       'draft': a['concepts'], 'review': b['concepts'], 'basis': [a['basis'], b['basis']]})
            continue
        key = t['problem_key']
        if key in statuses:
            raise ValueError(f'duplicate real item {key}')
        known_a, known_b = a['basis'] != 'unknown', b['basis'] != 'unknown'
        pairs['both_known'] += known_a and known_b
        pairs['both_unknown'] += not known_a and not known_b
        pairs['one_unknown'] += known_a != known_b
        if known_a and known_b:
            pairs['agree_any'] += bool(set(a['concepts']) & set(b['concepts']))
            pairs['disagree_known'] += not (set(a['concepts']) & set(b['concepts']))
        statuses[key] = item_status(trusted, proposed, abstained)
        for status, concepts in (('ai_reviewed', trusted), ('proposed', proposed), ('abstained', abstained)):
            for c in sorted(concepts):
                rows.append({'problem_key': key, 'concept_id': c, 'evidence_status': status, 'via': via})
        for role, e in (('draft', a), ('review', b)):
            judgments.append(validate_record({'judgment_id': f'{via}:{key}:{role}', 'object_kind': 'problem_concept', 'object_ref': key,
                'model': meta[role]['model'], 'role': role, 'run_date': meta['run_date'], 'prompt_sha256': meta['prompt_sha256'],
                'input_sha256': meta['input_sha256'], 'verdict': {'concepts': e['concepts'], 'basis': e['basis']}, 'rationale': e['rationale']}))
    counts = {s: sum(1 for v in statuses.values() if v == s) for s in ITEM_STATUSES}
    return {'rows': rows, 'statuses': statuses, 'judgments': judgments, 'controls': controls, 'swapped_detail': swapped_detail,
            'pair_stats': pairs, 'item_counts': counts, 'items': len(statuses)}


def cohen_kappa(pairs):
    """Chance-corrected agreement on a binary label (list of (x, y) booleans)."""
    n = len(pairs)
    if not n:
        return None
    po = sum(x == y for x, y in pairs) / n
    px = sum(x for x, _ in pairs) / n
    py = sum(y for _, y in pairs) / n
    pe = px * py + (1 - px) * (1 - py)
    return None if pe == 1 else (po - pe) / (1 - pe)


def confirmation_rows(path, vocabulary, known_keys):
    """Validated user decisions (append-only JSONL). Later records for a pair supersede earlier ones."""
    latest = {}
    if not Path(path).exists():
        return [], []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        if set(rec) != {'problem_key', 'concept_id', 'decision', 'decided_by', 'decided_at', 'note'}:
            raise ValueError(f'confirmation line {n}: invalid fields')
        if rec['decided_by'] != 'user' or rec['decision'] not in ('confirm', 'reject') or not str(rec['note']).strip():
            raise ValueError(f'confirmation line {n}: invalid decision record')
        if rec['concept_id'] not in vocabulary or rec['problem_key'] not in known_keys:
            raise ValueError(f'confirmation line {n}: unknown problem or concept')
        date.fromisoformat(rec['decided_at'])
        latest[(rec['problem_key'], rec['concept_id'])] = rec
    approved = [{'problem_key': k, 'concept_id': c, 'evidence_status': 'approved', 'via': 'user:confirm'} for (k, c), r in sorted(latest.items()) if r['decision'] == 'confirm']
    rejected = sorted(k for k, r in latest.items() if r['decision'] == 'reject')
    return approved, rejected


def compile_rows(ai_rows, approved, rejected):
    """Final per-problem concept rows. A user rejection removes every AI row for that pair and records it as `rejected`;
    a user confirmation adds `approved` (the AI row stays, so provenance shows both)."""
    drop = set(rejected)
    rows = [r for r in ai_rows if (r['problem_key'], r['concept_id']) not in drop]
    rows += [{'problem_key': k, 'concept_id': c, 'evidence_status': 'rejected', 'via': 'user:reject'} for k, c in rejected]
    rows += approved
    return rows


def append_confirmation(path, problem_key, concept_id, decision, note, vocabulary, known_keys, today=None):
    rec = {'problem_key': problem_key, 'concept_id': concept_id, 'decision': decision, 'decided_by': 'user',
           'decided_at': (today or date.today()).isoformat(), 'note': note}
    probe = Path(path)
    lines = probe.read_text().splitlines() if probe.exists() else []
    tmp = [json.loads(x) for x in lines if x.strip()] + [rec]
    # validate the whole file plus the new record before appending
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False) as handle:
        handle.write('\n'.join(json.dumps(x, sort_keys=True) for x in tmp) + '\n')
    try:
        confirmation_rows(handle.name, vocabulary, known_keys)
    finally:
        Path(handle.name).unlink()
    with open(path, 'a') as out:
        out.write(json.dumps(rec, sort_keys=True) + '\n')
    return rec


def _vocabulary(root):
    return {c['id'] for c in json.loads((Path(root) / 'assets/mastery/concepts.v1.json').read_text())['concepts']}


def load_merged_inputs(directory, batch_ids):
    """Truth subset and both judges' outputs for the listed batch numbers."""
    directory = Path(directory)
    truth_all = json.loads((directory / 'truth.json').read_text())
    prefixes = tuple(f'b{n:03d}-' for n in batch_ids)
    truth = {k: v for k, v in truth_all.items() if k.startswith(prefixes)}
    draft, review = [], []
    for n in batch_ids:
        draft += json.loads((directory / 'out' / f'batch-{n:03d}.draft.json').read_text())
        review += json.loads((directory / 'out' / f'batch-{n:03d}.review.json').read_text())
    inputs = b''.join((directory / 'in' / f'batch-{n:03d}.json').read_bytes() for n in batch_ids)
    return truth, draft, review, sha256_bytes(inputs)


def public_asset(result, meta, via='atcoder-full-v1'):
    """Titles-free asset: keys, concepts, evidence status. Controls and judge text stay private."""
    return {'schema': 'mastery-problem-concepts.v2', 'not_gold': True, 'via': via,
            'status_rule': 'ai_reviewed: both judges include the concept, neither unknown, at least one recognized; proposed: agreement on title inference only; abstained: one judge only or disagreement',
            'models': {'draft': meta['draft']['model'], 'review': meta['review']['model']}, 'run_date': meta['run_date'],
            'prompt_sha256': meta['prompt_sha256'], 'input_sha256': meta['input_sha256'],
            'stats': {'items': result['items'], 'item_status': result['item_counts'], 'pair_stats': result['pair_stats'],
                      'rows_by_status': {s: sum(1 for r in result['rows'] if r['evidence_status'] == s) for s in ('ai_reviewed', 'proposed', 'abstained')}},
            'item_status': dict(sorted(result['statuses'].items())), 'rows': result['rows']}


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export')
    e.add_argument('--catalog', type=Path, required=True)
    e.add_argument('--out-dir', type=Path, required=True)
    e.add_argument('--batch-size', type=int, default=150)
    e.add_argument('--families', help='comma-separated contest families, e.g. abc,arc,agc')
    e.add_argument('--exclude-batches', help='batch range of that truth to skip, e.g. 0-2')
    e.add_argument('--exclude-truth', type=Path, help='truth.json of an earlier export whose real items are skipped')
    m = sub.add_parser('merge')
    m.add_argument('--dir', type=Path, required=True)
    m.add_argument('--batches', required=True, help='e.g. 0-2 or 0-64')
    m.add_argument('--draft-model', required=True)
    m.add_argument('--review-model', required=True)
    m.add_argument('--run-date', required=True)
    m.add_argument('--output', type=Path, required=True, help='private merged result (includes controls and judgments)')
    m.add_argument('--asset', type=Path, help='public titles-free asset')
    c = sub.add_parser('confirm')
    c.add_argument('--catalog', type=Path, required=True)
    c.add_argument('--file', type=Path, required=True)
    c.add_argument('--problem', required=True)
    c.add_argument('--concept', required=True)
    c.add_argument('--reject', action='store_true')
    c.add_argument('--note', required=True)
    args = parser.parse_args(argv)
    if args.command == 'export':
        print(json.dumps(export(args.catalog, args.out_dir, args.batch_size, args.families.split(',') if args.families else None, args.exclude_truth,
                                 range(int(args.exclude_batches.split('-')[0]), int(args.exclude_batches.split('-')[1]) + 1) if args.exclude_batches else None)))
    elif args.command == 'merge':
        lo, hi = (int(x) for x in args.batches.split('-'))
        truth, draft, review, input_sha = load_merged_inputs(args.dir, range(lo, hi + 1))
        meta = {'draft': {'model': args.draft_model}, 'review': {'model': args.review_model}, 'run_date': args.run_date,
                'prompt_sha256': sha256_bytes((root / 'research/mastery_catalog/prompts/atcoder-full.v1.txt').read_bytes()), 'input_sha256': input_sha}
        result = merge(truth, draft, review, _vocabulary(root), meta)
        for path in (args.output, args.asset):
            if path and path.exists():
                raise FileExistsError(path)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
        if args.asset:
            args.asset.write_text(json.dumps(public_asset(result, meta), ensure_ascii=False, indent=1) + '\n')
        print(json.dumps({k: result[k] for k in ('items', 'item_counts', 'pair_stats', 'controls')}))
    elif args.command == 'confirm':
        known = {i['problem_key'] for i in load_items(args.catalog)}
        print(json.dumps(append_confirmation(args.file, args.problem, args.concept, 'reject' if args.reject else 'confirm', args.note,
                                             _vocabulary(root), known)))
    return 0


if __name__ == '__main__':
    sys.exit(main())


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round((centre - half) / d, 3), round((centre + half) / d, 3)]


def evaluate(directory, catalog_path, pilot_asset, batch_ids, cost):
    """Aggregate evaluation of a judged slice (no titles). `cost` = {'draft_tokens': n, 'review_tokens': n}."""
    directory = Path(directory)
    truth, draft, review, _ = load_merged_inputs(directory, batch_ids)
    d = {e['item_id']: e for e in draft}
    r = {e['item_id']: e for e in review}
    merged = json.loads((directory / 'merged-calibration.json').read_text())
    real = [i for i in truth if truth[i]['role'] == 'real']
    key_of = {i: truth[i]['problem_key'] for i in real}
    con = sqlite3.connect(f'file:{Path(catalog_path).resolve()}?mode=ro', uri=True)
    diff = {k: v for k, v in con.execute("SELECT problem_key, difficulty_value FROM problems WHERE platform='atcoder' AND difficulty_kind='atc_irt'")}
    tier = {k: t for k, t in con.execute("SELECT problem_key, tier FROM reliability")}
    population = [p for (p,) in con.execute("SELECT problem_key FROM problems WHERE platform='atcoder' AND eligible=1 AND status='active'")]
    con.close()
    by = lambda keyfn: {g: {s: sum(1 for i in real if keyfn(i) == g and merged['statuses'][key_of[i]] == s) for s in ITEM_STATUSES}
                        for g in sorted({keyfn(i) for i in real})}
    fam = lambda i: (lambda f: f if f in ('abc', 'arc', 'agc', 'awc', 'past', 'joi') else 'other')(contest_family(key_of[i]))
    bins = lambda i: 'no_difficulty' if key_of[i] not in diff else ('<400' if diff[key_of[i]] < 400 else '400-1199' if diff[key_of[i]] < 1200 else '1200+')
    rel = lambda i: tier.get(key_of[i], 'unknown')
    pop_family = {}
    for p in population:
        f = contest_family(p)
        f = f if f in ('abc', 'arc', 'agc', 'awc', 'past', 'joi') else 'other'
        pop_family[f] = pop_family.get(f, 0) + 1
    labeled_rate = {f: (c['labeled'] / sum(c.values())) for f, c in by(fam).items()}
    projection = {'labeled_if_abc_arc_agc': round(sum(labeled_rate.get(f, 0) * pop_family.get(f, 0) for f in ('abc', 'arc', 'agc'))),
                  'labeled_if_full_catalog': round(sum(labeled_rate.get(f, 0) * n for f, n in pop_family.items())),
                  'population_by_family': pop_family}
    known = lambda e: e['basis'] != 'unknown'
    pairs = [(known(d[i]), known(r[i])) for i in real]
    both = [i for i in real if known(d[i]) and known(r[i])]
    jacc = [len(set(d[i]['concepts']) & set(r[i]['concepts'])) / max(1, len(set(d[i]['concepts']) | set(r[i]['concepts']))) for i in both]
    ctrl = merged['controls']
    # title following: a judge produced a trusted-looking answer for a swapped title
    swapped_recognized = sum(1 for s in merged['swapped_detail'] if 'recognized' in s['basis'])
    # test-retest against the pilot (items judged in both)
    pilot = json.loads(Path(pilot_asset).read_text())
    pilot_status = {}
    for row in pilot['rows']:
        pilot_status.setdefault(row['problem_key'], set())
        if row['evidence_status'] == 'ai_reviewed':
            pilot_status[row['problem_key']].add(row['concept_id'])
    pilot_keys = {row['problem_key'] for row in pilot['rows']}
    overlap = [k for k in key_of.values() if k in pilot_keys or k in {j['object_ref'] for j in []}]
    retest = {'items_in_both': len(overlap)}
    new_by_key = {}
    for row in merged['rows']:
        if row['evidence_status'] == 'ai_reviewed':
            new_by_key.setdefault(row['problem_key'], set()).add(row['concept_id'])
    agree = [bool(pilot_status.get(k) and new_by_key.get(k) and pilot_status[k] & new_by_key[k]) for k in overlap if pilot_status.get(k) and new_by_key.get(k)]
    retest.update({'both_labeled': len(agree), 'concept_overlap': sum(agree)})
    shown = len(truth)
    tokens = cost['draft_tokens'] + cost['review_tokens']
    return {
        'schema': 'atcoder-concepts-eval.v1', 'batches': list(batch_ids), 'sample_items': len(real), 'eligible_population': len(population),
        'item_status': merged['item_counts'],
        'coverage_by_family': by(fam), 'coverage_by_difficulty': by(bins), 'coverage_by_title_reliability': by(rel),
        'judges': {'draft_known': sum(a for a, _ in pairs), 'review_known': sum(b for _, b in pairs), 'known_kappa': round(cohen_kappa(pairs), 3),
                   'both_known': len(both), 'mean_concept_jaccard_when_both_known': round(sum(jacc) / len(jacc), 3) if jacc else None,
                   'basis_draft': dict(sorted(__import__('collections').Counter(d[i]['basis'] for i in real).items())),
                   'basis_review': dict(sorted(__import__('collections').Counter(r[i]['basis'] for i in real).items()))},
        'controls': {'fabricated': {**ctrl['fabricated'], 'either_known_wilson95': wilson(ctrl['fabricated']['either_known'], ctrl['fabricated']['items'])},
                     'swapped': {**ctrl['swapped'], 'either_known_wilson95': wilson(ctrl['swapped']['either_known'], ctrl['swapped']['items']),
                                 'recognized_despite_wrong_title': swapped_recognized}},
        'retest_vs_pilot': retest,
        'cost': {'items_shown_incl_controls': shown, 'draft_tokens': cost['draft_tokens'], 'review_tokens': cost['review_tokens'],
                 'tokens_per_real_item_both_judges': round(tokens / len(real)), 'tokens_per_labeled_item': round(tokens / max(1, merged['item_counts']['labeled'])),
                 'projected_tokens_abc_arc_agc_remaining': round(4308 * tokens / len(real)), 'projected_tokens_full_catalog': round(len(population) * tokens / len(real))},
        'projection': projection,
    }
