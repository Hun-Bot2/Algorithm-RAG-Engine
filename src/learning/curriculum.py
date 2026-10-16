"""Offline validation for proposed and reviewed curriculum documents."""
from urllib.parse import urlsplit

from .events import problem_key


_DOCUMENT_FIELDS = {'schema_version', 'curriculum_version', 'problems', 'relationships'}
_PROBLEM_FIELDS = {
    'problem_key', 'canonical_url', 'native_difficulty', 'access_state',
    'concepts', 'review',
}
_RELATION_FIELDS = {'source', 'target', 'type', 'review'}
_REVIEW_FIELDS = {'status', 'reviewer', 'evidence'}
_PLATFORM_HOSTS = {
    'leetcode': 'leetcode.com', 'atcoder': 'atcoder.jp',
    'codeforces': 'codeforces.com', 'baekjoon': 'www.acmicpc.net',
}
_ACCESS = {'free', 'paid', 'unknown', 'unavailable'}
_RELATIONS = {'similarity', 'prerequisite', 'transfer'}


def _exact_fields(value, fields, label):
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f'{label} must contain exactly {sorted(fields)}')


def _nonempty_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be nonempty text')


def _review(value, label):
    _exact_fields(value, _REVIEW_FIELDS, label)
    if not isinstance(value['status'], str) or value['status'] not in {'proposed', 'approved'}:
        raise ValueError(f'{label} has invalid status')
    for field in ('reviewer', 'evidence'):
        item = value[field]
        if item is not None and (not isinstance(item, str) or not item.strip()):
            raise ValueError(f'{label} {field} must be nonempty text or null')
    if value['status'] == 'approved' and (
            not isinstance(value['reviewer'], str) or not value['reviewer'].strip()
            or not isinstance(value['evidence'], str) or not value['evidence'].strip()):
        raise ValueError(f'{label} approval requires reviewer and evidence')


def _canonical_url(value, key):
    _nonempty_text(value, 'canonical_url')
    try:
        parsed = urlsplit(value)
        # Accessing port also rejects malformed port syntax.
        port = parsed.port
    except ValueError as exc:
        raise ValueError('invalid canonical_url') from exc
    platform = key.split(':', 1)[0]
    if (parsed.scheme != 'https' or parsed.hostname != _PLATFORM_HOSTS[platform] or port is not None
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or not parsed.path
            or parsed.netloc != parsed.hostname):
        raise ValueError('canonical_url must be HTTPS on an allowed host with a path')


def _acyclic_prerequisites(edges):
    graph, indegree = {}, {}
    for source, target in edges:
        graph.setdefault(source, []).append(target)
        indegree.setdefault(source, 0)
        indegree[target] = indegree.get(target, 0) + 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for neighbor in graph.get(node, ()):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                ready.append(neighbor)
    if visited != len(indegree):
        raise ValueError('prerequisite relationships must be acyclic')


def validate_curriculum(document, require_release=False):
    """Validate a curriculum object and return the original object unchanged."""
    _exact_fields(document, _DOCUMENT_FIELDS, 'curriculum')
    if type(document['schema_version']) is not int or document['schema_version'] != 1:
        raise ValueError('schema_version must be 1')
    _nonempty_text(document['curriculum_version'], 'curriculum_version')
    if not isinstance(document['problems'], list) or not isinstance(document['relationships'], list):
        raise ValueError('problems and relationships must be lists')

    problems = {}
    for item in document['problems']:
        _exact_fields(item, _PROBLEM_FIELDS, 'problem')
        key = item['problem_key']
        problem_key(key)
        if key in problems:
            raise ValueError('duplicate problem_key')
        _canonical_url(item['canonical_url'], key)
        _nonempty_text(item['native_difficulty'], 'native_difficulty')
        if not isinstance(item['access_state'], str) or item['access_state'] not in _ACCESS:
            raise ValueError('invalid access_state')
        concepts = item['concepts']
        if not isinstance(concepts, list) or any(
                not isinstance(concept, str) or not concept.strip() for concept in concepts):
            raise ValueError('concepts must be a list of nonempty strings')
        if len(concepts) != len(set(concepts)):
            raise ValueError('concepts must be unique')
        _review(item['review'], 'problem review')
        problems[key] = item

    seen_edges, prerequisites = set(), []
    for item in document['relationships']:
        _exact_fields(item, _RELATION_FIELDS, 'relationship')
        source, target, kind = item['source'], item['target'], item['type']
        if (not isinstance(source, str) or not isinstance(target, str)
                or source not in problems or target not in problems):
            raise ValueError('relationship endpoint does not exist')
        if source == target:
            raise ValueError('self relationships are not allowed')
        if not isinstance(kind, str) or kind not in _RELATIONS:
            raise ValueError('invalid relationship type')
        pair = (source, target)
        identity = (source, target, kind)
        if identity in seen_edges:
            raise ValueError('duplicate relationship')
        seen_edges.add(identity)
        _review(item['review'], 'relationship review')
        if item['review']['status'] == 'approved' and (
                problems[source]['review']['status'] != 'approved'
                or problems[target]['review']['status'] != 'approved'):
            raise ValueError('approved relationship requires approved endpoints')
        if kind == 'prerequisite':
            prerequisites.append(pair)
    _acyclic_prerequisites(prerequisites)

    if require_release:
        if len(problems) < 90:
            raise ValueError('release requires at least 90 problems')
        if any(p['review']['status'] != 'approved' for p in problems.values()):
            raise ValueError('release requires all problems approved')
        if any(p['access_state'] != 'free' for p in problems.values()):
            raise ValueError('release requires explicit free access')
        if any(r['review']['status'] != 'approved' for r in document['relationships']):
            raise ValueError('release requires all relationships approved')
    return document
