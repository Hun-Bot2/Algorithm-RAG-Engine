import unittest
import numpy as np
from research.recommendation_inputs.leetcode import rankings
from research.recommendation_inputs.viewer import response_data, safe_source_url


class LeetCodeTests(unittest.TestCase):
    def problems(self):
        return {key: {'problem_key': key, 'title': key, 'platform': 'leetcode', 'native_key': key, 'source_tags': tags,
                      'eligible': eligible, 'status': 'active', 'statement_text': None, 'original_url': 'https://leetcode.com/problems/'+key.split(':')[1]+'/'}
                for key, tags, eligible in [('leetcode:a', ['array'], True), ('leetcode:b', ['array'], True), ('leetcode:c', [], True), ('leetcode:d', ['array'], False)]}

    def test_same_platform_cosine_and_eligible_targets(self):
        problems = self.problems()
        result, count = rankings(problems, np.array([[1., 0.], [.8, .6], [0., 1.], [1., 0.]], dtype=np.float32))
        self.assertEqual(count, 3)
        self.assertEqual(result['lc_title_cosine']['leetcode:a'][0][0], 'leetcode:b')
        self.assertAlmostEqual(result['lc_title_cosine']['leetcode:a'][0][1], .8, places=6)
        for method in result.values():
            for source, targets in method.items():
                self.assertTrue(all(key.startswith('leetcode:') and key != source and key != 'leetcode:d' for key, score in targets))

    def test_missing_or_disjoint_tags_do_not_fabricate_results(self):
        result, _ = rankings(self.problems(), np.eye(4, dtype=np.float32))
        self.assertEqual(result['lc_tag_jaccard']['leetcode:c'], [])
        self.assertEqual(result['lc_tag_jaccard']['leetcode:a'], [('leetcode:b', 1.)])

    def test_invalid_vector_alignment_fails(self):
        for vectors in [np.eye(3), np.zeros((4, 2)), np.full((4, 2), np.nan)]:
            with self.assertRaises(ValueError):
                rankings(self.problems(), vectors)

    def test_scope_keeps_LeetCode_separate(self):
        problems = self.problems()
        result, _ = rankings(problems, np.eye(4))
        scoped = {'scopes': {'leetcode': {'problems': problems, 'rankings': result, 'methods': {'lc_title_cosine': {'label': 'title'}}, 'coverage': 'LC', 'default_key': 'leetcode:a'}}}
        meta = response_data(scoped, '/api/meta', {})
        self.assertEqual(meta['scope'], 'leetcode')
        data = response_data(scoped, '/api/compare', {'scope': ['leetcode'], 'key': ['leetcode:a']})
        self.assertIsNone(data['source']['statement_text'])
        self.assertEqual(data['methods']['lc_title_cosine'][0]['platform'], 'leetcode')
        with self.assertRaises(KeyError):
            response_data(scoped, '/api/meta', {'scope': ['missing']})
        self.assertEqual(safe_source_url('https://leetcode.com/problems/buddy-strings/', 'leetcode'), 'https://leetcode.com/problems/buddy-strings/')


if __name__ == '__main__':
    unittest.main()
