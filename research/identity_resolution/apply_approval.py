"""Apply explicit user-approved identities as metadata, without rewriting sources."""
from copy import deepcopy
from research.identity_resolution.build import validate_evidence

APPROVAL = {
    'basis': 'explicit_user_instruction',
    'date': '2026-10-17',
    'scope': 'Approve 36 ai_proposed mappings; keep p03510 unresolved and abc084_c quality flags.',
    'independent_adjudication': False,
}


def approved_overlay(rows, corpus, catalog, required_keys, evidence_sha256):
    validate_evidence(rows, corpus, catalog, required_keys)
    mappings, unresolved = {}, {}
    for row in rows:
        record = {
            'source_key': row['source_key'],
            'publisher_title': row.get('publisher_title'),
            'source_content_sha256': row['source_content_sha256'],
            'native_key': row['selected_native_key'] if row['status'] == 'ai_proposed' else None,
            'status': 'user_approved' if row['status'] == 'ai_proposed' else 'unresolved',
            'canonical_url': row.get('canonical_url'),
            'quality_flags': deepcopy(row.get('quality_flags', [])),
        }
        target = mappings if row['status'] == 'ai_proposed' else unresolved
        target[row['source_key']] = record
    return {'schema_version': 1, 'evidence_sha256': evidence_sha256,
            'approval': deepcopy(APPROVAL), 'mappings': mappings, 'unresolved': unresolved}


def mapping_for(overlay, source_key, source_content_sha256):
    """Resolve metadata only when the caller's source content matches approval."""
    record = overlay['mappings'].get(source_key)
    if record is None:
        return None
    if record['source_content_sha256'] != source_content_sha256:
        raise ValueError('mapping source hash mismatch')
    return deepcopy(record)
