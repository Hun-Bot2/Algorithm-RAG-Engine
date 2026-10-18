"""AI-assisted judgment records and the tag-to-concept merge (tasks MS-601, MS-602).

Protocol (design.md section 8): two different model roles judge independently; a (tag, concept)
pair both include becomes ai_reviewed, a pair only one includes becomes abstained. Records keep
model, role, date, prompt hash, input hash and a non-empty rationale. Never call these gold.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REQUIRED_RECORD = {'judgment_id', 'object_kind', 'object_ref', 'model', 'role', 'run_date', 'prompt_sha256', 'input_sha256', 'verdict', 'rationale'}
ROLES = {'draft', 'review', 'adjudicator'}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def validate_record(record):
    """Rejects incomplete AI judgment records (R9.4)."""
    if not isinstance(record, dict) or set(record) != REQUIRED_RECORD:
        raise ValueError('invalid judgment record fields')
    for field in REQUIRED_RECORD - {'verdict'}:
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f'empty judgment field: {field}')
    if record['role'] not in ROLES:
        raise ValueError('invalid judgment role')
    for field in ('prompt_sha256', 'input_sha256'):
        if len(record[field]) != 64 or any(c not in '0123456789abcdef' for c in record[field]):
            raise ValueError(f'invalid {field}')
    return record


def validate_crosswalk(entries, expected_tags, vocabulary):
    if not isinstance(entries, list):
        raise ValueError('crosswalk must be a list')
    seen = set()
    for entry in entries:
        if set(entry) != {'platform', 'native_tag', 'concepts', 'rationale'}:
            raise ValueError(f'invalid crosswalk entry fields: {entry}')
        key = (entry['platform'], entry['native_tag'])
        if key in seen:
            raise ValueError(f'duplicate crosswalk entry: {key}')
        seen.add(key)
        if not isinstance(entry['rationale'], str) or not entry['rationale'].strip():
            raise ValueError(f'empty rationale: {key}')
        if not isinstance(entry['concepts'], list) or len(entry['concepts']) > 3 or len(set(entry['concepts'])) != len(entry['concepts']):
            raise ValueError(f'invalid concept list: {key}')
        unknown = set(entry['concepts']) - vocabulary
        if unknown:
            raise ValueError(f'unknown concepts {unknown} for {key}')
    if seen != expected_tags:
        raise ValueError(f'tag coverage mismatch: missing {sorted(expected_tags - seen)[:5]}, extra {sorted(seen - expected_tags)[:5]}')
    return {(e['platform'], e['native_tag']): e for e in entries}


def merge(draft, review, meta):
    entries, judgments = [], []
    stats = {'pairs_agreed': 0, 'pairs_abstained': 0, 'tags_identical': 0, 'tags_both_empty': 0}
    for key in sorted(draft):
        a, b = set(draft[key]['concepts']), set(review[key]['concepts'])
        ref = f'crosswalk-v1:{key[0]}:{key[1]}'
        concepts = [{'concept_id': c, 'evidence_status': 'ai_reviewed'} for c in sorted(a & b)]
        concepts += [{'concept_id': c, 'evidence_status': 'abstained'} for c in sorted(a ^ b)]
        stats['pairs_agreed'] += len(a & b)
        stats['pairs_abstained'] += len(a ^ b)
        stats['tags_identical'] += a == b
        stats['tags_both_empty'] += not a and not b
        entries.append({'platform': key[0], 'native_tag': key[1], 'concepts': concepts, 'judgment_ref': ref})
        for role, source in (('draft', draft), ('review', review)):
            judgments.append(validate_record({
                'judgment_id': f'{ref}:{role}', 'object_kind': 'tag_concept', 'object_ref': f'{key[0]}:{key[1]}',
                'model': meta[role]['model'], 'role': role, 'run_date': meta['run_date'], 'prompt_sha256': meta['prompt_sha256'],
                'input_sha256': meta['input_sha256'], 'verdict': source[key]['concepts'], 'rationale': source[key]['rationale']}))
    return {'schema': 'mastery-concept-map.v1', 'status_rule': 'pair in both independent judgments -> ai_reviewed; in one -> abstained',
            'not_gold': True, 'run_date': meta['run_date'], 'models': {r: meta[r]['model'] for r in ('draft', 'review')},
            'prompt_sha256': meta['prompt_sha256'], 'input_sha256': meta['input_sha256'], 'stats': stats,
            'entries': entries, 'judgments': judgments}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--tags', type=Path, required=True, help='JSON {platform: [tags]} the judges received')
    parser.add_argument('--vocabulary', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--draft-model', required=True)
    parser.add_argument('--review-model', required=True)
    parser.add_argument('--run-date', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    vocabulary = {c['id'] for c in json.loads(args.vocabulary.read_text())['concepts']}
    tag_doc = json.loads(args.tags.read_text())
    expected = {(p, t) for p, tags in tag_doc.items() for t in tags}
    draft = validate_crosswalk(json.loads(args.draft.read_text()), expected, vocabulary)
    review = validate_crosswalk(json.loads(args.review.read_text()), expected, vocabulary)
    meta = {'draft': {'model': args.draft_model}, 'review': {'model': args.review_model}, 'run_date': args.run_date,
            'prompt_sha256': sha256_bytes(args.prompt.read_bytes()),
            'input_sha256': sha256_bytes(json.dumps(tag_doc, sort_keys=True).encode() + args.vocabulary.read_bytes())}
    document = merge(draft, review, meta)
    args.output.write_text(json.dumps(document, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(document['stats']))


if __name__ == '__main__':
    sys.exit(main())


def merge_problem_concepts(draft, review, items, vocabulary, meta):
    """AtCoder per-problem pilot (MS-603): ai_reviewed only when both judges include the concept and
    neither marked the item unknown. Self-reported 'recognized' is not trusted on its own."""
    keys = {i['problem_key'] for i in items}
    def check(entries):
        by = {}
        for e in entries:
            if set(e) != {'problem_key', 'concepts', 'basis', 'rationale'} or e['problem_key'] not in keys or e['problem_key'] in by:
                raise ValueError(f'invalid pilot entry: {e.get("problem_key")}')
            if e['basis'] not in {'recognized', 'title_inference', 'unknown'} or not str(e['rationale']).strip():
                raise ValueError(f'invalid basis/rationale: {e["problem_key"]}')
            if e['basis'] == 'unknown' and e['concepts']:
                raise ValueError(f'unknown basis with concepts: {e["problem_key"]}')
            if len(e['concepts']) > 3 or set(e['concepts']) - vocabulary:
                raise ValueError(f'invalid concepts: {e["problem_key"]}')
            by[e['problem_key']] = e
        if set(by) != keys:
            raise ValueError('pilot coverage mismatch')
        return by
    d, r = check(draft), check(review)
    rows, judgments = [], []
    stats = {'items': len(keys), 'both_known': 0, 'both_recognized': 0, 'both_recognized_overlap': 0, 'items_with_ai_reviewed': 0,
             'pairs_ai_reviewed': 0, 'pairs_abstained': 0, 'basis_draft': {}, 'basis_review': {}}
    for key in sorted(keys):
        a, b = d[key], r[key]
        stats['basis_draft'][a['basis']] = stats['basis_draft'].get(a['basis'], 0) + 1
        stats['basis_review'][b['basis']] = stats['basis_review'].get(b['basis'], 0) + 1
        both_known = a['basis'] != 'unknown' and b['basis'] != 'unknown'
        shared = set(a['concepts']) & set(b['concepts']) if both_known else set()
        stats['both_known'] += both_known
        if a['basis'] == b['basis'] == 'recognized':
            stats['both_recognized'] += 1
            stats['both_recognized_overlap'] += bool(set(a['concepts']) & set(b['concepts']))
        stats['items_with_ai_reviewed'] += bool(shared)
        for c in sorted(shared):
            rows.append({'problem_key': key, 'concept_id': c, 'evidence_status': 'ai_reviewed', 'via': 'atcoder-pilot-v1'})
        for c in sorted((set(a['concepts']) | set(b['concepts'])) - shared):
            rows.append({'problem_key': key, 'concept_id': c, 'evidence_status': 'abstained', 'via': 'atcoder-pilot-v1'})
        stats['pairs_ai_reviewed'] += len(shared)
        stats['pairs_abstained'] += len((set(a['concepts']) | set(b['concepts'])) - shared)
        for role, e in (('draft', a), ('review', b)):
            judgments.append(validate_record({'judgment_id': f'atcoder-pilot-v1:{key}:{role}', 'object_kind': 'problem_concept', 'object_ref': key,
                'model': meta[role]['model'], 'role': role, 'run_date': meta['run_date'], 'prompt_sha256': meta['prompt_sha256'],
                'input_sha256': meta['input_sha256'], 'verdict': {'concepts': e['concepts'], 'basis': e['basis']}, 'rationale': e['rationale']}))
    return rows, judgments, stats
