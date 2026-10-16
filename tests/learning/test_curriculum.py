import copy
import unittest

from src.learning.curriculum import validate_curriculum


def problem(key, *, status='proposed', access='free'):
    approved = status == 'approved'
    host = {'leetcode': 'leetcode.com', 'atcoder': 'atcoder.jp',
            'codeforces': 'codeforces.com', 'baekjoon': 'www.acmicpc.net'}[key.split(':')[0]]
    return {
        'problem_key': key,
        'canonical_url': f'https://{host}/problems/example/',
        'native_difficulty': 'Medium',
        'access_state': access,
        'concepts': ['graphs'],
        'review': {'status': status, 'reviewer': 'reviewer-1' if approved else None,
                   'evidence': 'source-check' if approved else None},
    }


def curriculum():
    return {'schema_version': 1, 'curriculum_version': 'draft-1',
            'problems': [problem('leetcode:example'), problem('atcoder:abc001_a')],
            'relationships': [{'source': 'leetcode:example', 'target': 'atcoder:abc001_a',
                               'type': 'prerequisite',
                               'review': {'status': 'proposed', 'reviewer': None,
                                          'evidence': None}}]}


class CurriculumTests(unittest.TestCase):
    def test_valid_proposed_document_returns_same_object_without_mutation(self):
        doc = curriculum()
        before = copy.deepcopy(doc)
        self.assertIs(validate_curriculum(doc), doc)
        self.assertEqual(doc, before)

    def test_cycle_including_proposed_edges_rejected(self):
        doc = curriculum()
        doc['relationships'].append({'source': 'atcoder:abc001_a',
                                     'target': 'leetcode:example', 'type': 'prerequisite',
                                     'review': {'status': 'proposed', 'reviewer': None,
                                                'evidence': None}})
        with self.assertRaisesRegex(ValueError, 'acyclic'):
            validate_curriculum(doc)

    def test_relationship_direction_is_preserved(self):
        doc = curriculum()
        self.assertEqual(validate_curriculum(doc)['relationships'][0]['source'],
                         'leetcode:example')

    def test_missing_endpoint_rejected(self):
        doc = curriculum()
        doc['relationships'][0]['target'] = 'leetcode:missing'
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            validate_curriculum(doc)

    def test_platform_url_host_must_match_problem_key(self):
        doc = curriculum()
        doc['problems'][1]['canonical_url'] = 'https://leetcode.com/problems/example/'
        with self.assertRaisesRegex(ValueError, 'canonical_url'):
            validate_curriculum(doc)

    def test_url_credentials_query_and_fragment_rejected(self):
        for suffix in ('https://user:pass@leetcode.com/problems/example/',
                       'https://leetcode.com/problems/example/?ref=x',
                       'https://leetcode.com/problems/example/#part'):
            with self.subTest(url=suffix):
                doc = curriculum()
                doc['problems'][0]['canonical_url'] = suffix
                with self.assertRaises(ValueError):
                    validate_curriculum(doc)

    def test_same_direction_different_types_are_distinct_edges(self):
        doc = curriculum()
        doc['relationships'].append({'source': 'leetcode:example',
                                     'target': 'atcoder:abc001_a', 'type': 'similarity',
                                     'review': {'status': 'proposed', 'reviewer': None,
                                                'evidence': None}})
        self.assertIs(validate_curriculum(doc), doc)

    def test_duplicate_typed_edge_and_self_edge_rejected(self):
        doc = curriculum()
        doc['relationships'].append(copy.deepcopy(doc['relationships'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_curriculum(doc)
        doc = curriculum()
        doc['relationships'][0]['target'] = 'leetcode:example'
        with self.assertRaisesRegex(ValueError, 'self'):
            validate_curriculum(doc)

    def test_unhashable_status_access_kind_and_endpoints_raise_value_error(self):
        mutations = (
            lambda d: d['problems'][0]['review'].__setitem__('status', []),
            lambda d: d['problems'][0].__setitem__('access_state', {}),
            lambda d: d['relationships'][0].__setitem__('type', []),
            lambda d: d['relationships'][0].__setitem__('source', []),
            lambda d: d['relationships'][0].__setitem__('target', {}),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                doc = curriculum()
                mutate(doc)
                with self.assertRaises(ValueError):
                    validate_curriculum(doc)

    def test_long_prerequisite_chain_avoids_recursion_limit(self):
        count = 1100
        doc = {'schema_version': 1, 'curriculum_version': 'large-draft',
               'problems': [problem(f'leetcode:item-{i}') for i in range(count)],
               'relationships': [
                   {'source': f'leetcode:item-{i}', 'target': f'leetcode:item-{i + 1}',
                    'type': 'prerequisite',
                    'review': {'status': 'proposed', 'reviewer': None, 'evidence': None}}
                   for i in range(count - 1)]}
        self.assertIs(validate_curriculum(doc), doc)

    def test_malformed_schema_and_exact_fields_rejected(self):
        doc = curriculum()
        doc['schema_version'] = True
        with self.assertRaises(ValueError):
            validate_curriculum(doc)
        doc = curriculum()
        doc['unexpected'] = 1
        with self.assertRaises(ValueError):
            validate_curriculum(doc)

    def test_approval_requires_reviewer_and_evidence(self):
        doc = curriculum()
        doc['problems'][0]['review'] = {'status': 'approved', 'reviewer': None,
                                        'evidence': None}
        with self.assertRaisesRegex(ValueError, 'requires reviewer'):
            validate_curriculum(doc)

    def test_approved_relationship_requires_approved_endpoints(self):
        doc = curriculum()
        doc['relationships'][0]['review'] = {
            'status': 'approved', 'reviewer': 'reviewer', 'evidence': 'judgment'}
        with self.assertRaisesRegex(ValueError, 'approved endpoints'):
            validate_curriculum(doc)

    def test_release_requires_90_approved_free_problems(self):
        doc = curriculum()
        with self.assertRaisesRegex(ValueError, '90 problems'):
            validate_curriculum(doc, require_release=True)
        doc['problems'] = [problem(f'leetcode:item-{i}', status='approved')
                           for i in range(90)]
        doc['relationships'] = []
        self.assertIs(validate_curriculum(doc, require_release=True), doc)
        doc['problems'][0]['access_state'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'explicit free'):
            validate_curriculum(doc, require_release=True)


if __name__ == '__main__':
    unittest.main()
