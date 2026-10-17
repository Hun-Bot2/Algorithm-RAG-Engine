"""Provenance-aware exam/interview archive archive-v1 (tasks MS-801, MS-802; design.md section 10).

Learner-authored JSON (kept in a private repository) is validated and compiled into catalog rows.
Records are append-only: a correction is a new version that supersedes the previous one; earlier
versions stay readable. Third-party statement text is rejected unless the learner owns or licensed it.

    python3 -m research.mastery_catalog.archive validate PATH
"""
import json
import re
import sys
from pathlib import Path

CATEGORIES = {'company_interview', 'grad_entrance', 'certification', 'contest_past', 'other'}
SOURCE_KINDS = {'official_publication', 'personal_recollection', 'third_party_report', 'book'}
ACQUISITIONS = {'link_only', 'manual_transcription', 'recollection', 'purchased_book'}
RIGHTS = {'link_only', 'personal_notes', 'own_work', 'licensed'}
CONFIDENCE = {'high', 'medium', 'low'}
RECORDED_BY = {'learner', 'ai_draft'}
RELATIONS = {'practice_for', 'concept_only', 'same_problem'}
LINK_STATUS = {'approved', 'ai_reviewed', 'proposed'}
KEY = re.compile(r'^(leetcode|atcoder|codeforces|baekjoon):[A-Za-z0-9][A-Za-z0-9_.-]*$')
FIELDS = {'record_id', 'version', 'supersedes', 'category', 'origin', 'source', 'acquisition', 'rights_status', 'private_copy_sha256',
          'confidence', 'recorded_by', 'statement_text', 'notes', 'links'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(document):
    _require(isinstance(document, dict) and document.get('schema') == 'mastery-archive.v1', 'archive schema must be mastery-archive.v1')
    _require(isinstance(document.get('archive_version'), str) and document['archive_version'], 'archive_version required')
    records = document.get('records')
    _require(isinstance(records, list), 'records must be a list')
    seen, successors = {}, set()
    for r in records:
        rid = r.get('record_id') if isinstance(r, dict) else None
        _require(isinstance(r, dict) and set(r) == FIELDS, f'invalid record fields: {rid}')
        _require(isinstance(rid, str) and re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,120}', rid), f'invalid record_id: {rid}')
        _require(isinstance(r['version'], int) and r['version'] >= 1, f'invalid version: {rid}')
        ref = f"{rid}@{r['version']}"
        _require(ref not in seen, f'duplicate record version: {ref}')
        _require(r['category'] in CATEGORIES, f'invalid category: {ref}')
        origin, source = r['origin'], r['source']
        _require(isinstance(origin, dict) and set(origin) == {'organization', 'exam', 'year', 'session', 'label'}, f'invalid origin: {ref}')
        _require(origin['year'] is None or (isinstance(origin['year'], int) and 1950 <= origin['year'] <= 2100), f'invalid year: {ref}')
        _require(isinstance(source, dict) and set(source) == {'kind', 'citation', 'url', 'accessed'} and source['kind'] in SOURCE_KINDS, f'invalid source: {ref}')
        _require(isinstance(source['citation'], str) and source['citation'].strip(), f'source citation required: {ref}')
        _require(source['url'] is None or (isinstance(source['url'], str) and source['url'].startswith('https://')), f'source URL must be HTTPS: {ref}')
        _require(source['accessed'] is None or re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(source['accessed'])), f'invalid access date: {ref}')
        _require(r['acquisition'] in ACQUISITIONS and r['rights_status'] in RIGHTS and r['confidence'] in CONFIDENCE and r['recorded_by'] in RECORDED_BY, f'invalid provenance enum: {ref}')
        _require(r['private_copy_sha256'] is None or re.fullmatch(r'[0-9a-f]{64}', str(r['private_copy_sha256'])), f'invalid private copy hash: {ref}')
        # Rights: no third-party statement text (R12.3).
        _require(r['statement_text'] is None or r['rights_status'] in ('own_work', 'licensed'), f'statement text requires own_work or licensed rights: {ref}')
        _require(not (r['acquisition'] == 'recollection' and r['confidence'] == 'high'), f'recollection cannot have high confidence: {ref}')
        _require(r['notes'] is None or isinstance(r['notes'], str), f'invalid notes: {ref}')
        _require(isinstance(r['links'], list), f'links must be a list: {ref}')
        for link in r['links']:
            _require(isinstance(link, dict) and set(link) == {'problem_key', 'relation', 'evidence', 'status'}, f'invalid link: {ref}')
            _require(KEY.fullmatch(str(link['problem_key'])) and link['relation'] in RELATIONS and link['status'] in LINK_STATUS, f'invalid link values: {ref}')
            _require(isinstance(link['evidence'], list) and all(isinstance(e, str) and e.strip() for e in link['evidence']), f'invalid link evidence: {ref}')
            if link['relation'] == 'same_problem':
                # Identity needs two verifiable citations and is never resolved by AI alone (R12.5).
                _require(len(link['evidence']) >= 2, f'same_problem needs two evidence citations: {ref}')
                _require(r['recorded_by'] != 'ai_draft' and link['status'] == 'approved', f'same_problem cannot come from AI alone: {ref}')
        if r['supersedes'] is not None:
            target = r['supersedes']
            _require(target in seen, f'dangling supersedes: {ref} -> {target}')
            _require(target.split('@')[0] == rid and int(target.split('@')[1]) < r['version'], f'supersedes must point to an earlier version of the same record: {ref}')
            _require(target not in successors, f'branching correction: {target}')
            successors.add(target)
        else:
            _require(r['version'] == 1, f'version > 1 must supersede an earlier version: {ref}')
        seen[ref] = r
    return seen, successors


def compile_rows(document):
    """Rows for research.mastery_catalog.build (archive_records / archive_links)."""
    seen, successors = validate(document)
    out = []
    for ref, r in seen.items():
        o, s = r['origin'], r['source']
        row = (r['record_id'], r['version'], document['archive_version'], r['category'], o['organization'], o['exam'], o['year'],
               r['rights_status'], r['confidence'], r['recorded_by'], s['kind'], r['acquisition'], int(ref in successors),
               json.dumps(r, ensure_ascii=False, sort_keys=True))
        links = [(r['record_id'], r['version'], l['problem_key'], l['relation'], l['status'], json.dumps(l['evidence'], ensure_ascii=False)) for l in r['links']]
        out.append({'row': row, 'links': links})
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 2 or argv[0] != 'validate':
        print('usage: python3 -m research.mastery_catalog.archive validate PATH')
        return 1
    seen, successors = validate(json.loads(Path(argv[1]).read_text(encoding='utf-8')))
    print(json.dumps({'records': len(seen), 'current': len(seen) - len(successors), 'superseded': len(successors)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
