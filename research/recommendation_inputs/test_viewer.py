import unittest
from research.recommendation_inputs.viewer import response_data, safe_source_url


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.dataset = {'problems': {
            'a': {'problem_key': 'a', 'title': 'Alpha', 'platform': 'codeforces', 'native_key': 'codeforces:1A', 'statement_text': '<script>inert text</script>', 'original_url': 'https://codeforces.com/problemset/problem/1/A'},
            'b': {'problem_key': 'b', 'title': 'Beta', 'platform': 'atcoder', 'native_key': None, 'statement_text': 'Fixture problem', 'original_url': None},
        }, 'rankings': {'minilm_title': {'a': [['b', .7]], 'b': [['a', .6]]}}}

    def test_queries_return_same_recorded_candidates_and_scores(self):
        response = response_data(self.dataset, '/api/compare', {'key': ['a']})
        self.assertEqual(response['source']['problem_key'], 'a')
        self.assertEqual(response['methods']['minilm_title'][0]['score'], .7)
        self.assertEqual(response['methods']['minilm_title'][0]['problem_key'], 'b')

    def test_search_and_unresolved_identity(self):
        self.assertEqual(response_data(self.dataset, '/api/search', {'q': ['alpha']})[0]['problem_key'], 'a')
        p = response_data(self.dataset, '/api/problem', {'key': ['b']})
        self.assertIsNone(p['native_key'])
        self.assertIsNone(p['original_url'])
        self.assertNotIn('path', p)

    def test_unknown_routes_and_ids_are_not_file_access(self):
        for path, params in [('/../../AGENTS.md', {}), ('/api/problem', {'key': ['../../AGENTS.md']})]:
            with self.assertRaises(KeyError):
                response_data(self.dataset, path, params)

    def test_source_url_is_not_guessed_or_cross_platform(self):
        self.assertEqual(safe_source_url('https://atcoder.jp/contests/abc001/tasks/abc001_1', 'atcoder'), 'https://atcoder.jp/contests/abc001/tasks/abc001_1')
        for value in [None, 'javascript:alert(1)', 'https://example.test/', 'https://evil@codeforces.com/', 'https://codeforces.com.evil.test/']:
            self.assertIsNone(safe_source_url(value, 'codeforces'))
        self.assertIsNone(safe_source_url('https://codeforces.com/', 'atcoder'))


if __name__ == '__main__':
    unittest.main()
