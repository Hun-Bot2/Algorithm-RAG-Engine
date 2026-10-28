import unittest
import numpy as np

from research.recommendation_inputs import maps
from research.recommendation_inputs.experiments import retrieve
from research.recommendation_inputs.viewer import response_data


def fixture(n=24, seed=3):
    rng = np.random.default_rng(seed)
    keys = [f'leetcode:p{i:02d}' for i in range(n)]
    vectors = rng.normal(size=(n, 16)).astype(np.float32)
    problems = {k: {'problem_key': k, 'title': f'Problem {i}', 'platform': 'leetcode' if i % 2 else 'codeforces'} for i, k in enumerate(keys)}
    with np.errstate(all='ignore'):
        rankings = {'idea': retrieve(problems, vectors, k=5), 'cross_encoder': {k: [] for k in keys}}
    return keys, vectors, {'problems': problems, 'rankings': rankings}


class MapTests(unittest.TestCase):
    def setUp(self):
        maps._cache.clear()

    def test_vectors_attach_by_key_and_must_reproduce_stored_rankings(self):
        keys, vectors, scope = fixture()
        subset = {'problems': {k: scope['problems'][k] for k in keys[::-1]}, 'rankings': {}}
        other = {'problems': {'x': {}}, 'rankings': {'idea': {}}}
        subset['rankings'] = {'idea': {}}
        maps.attach_vectors({'all': scope, 'subset': subset, 'other': other}, keys, {'idea': vectors})
        self.assertTrue(np.array_equal(scope['vectors']['idea'], vectors))
        self.assertTrue(np.array_equal(subset['vectors']['idea'], vectors[::-1]), 'rows follow the scope problem order')
        self.assertNotIn('vectors', other, 'scopes with unknown keys get no vectors')
        with np.errstate(all='ignore'):
            self.assertTrue(maps.verified(scope, 'idea'))
            self.assertFalse(maps.verified(scope, 'cross_encoder'), 'methods without vectors get no map')
            scope['vectors']['idea'] = vectors[::-1].copy()
            self.assertFalse(maps.verified(scope, 'idea'), 'misaligned rows are rejected, not drawn')

    def test_map_is_deterministic_normalized_and_reports_preservation(self):
        keys, vectors, scope = fixture()
        maps.attach_vectors({'s': scope}, keys, {'idea': vectors})
        with np.errstate(all='ignore'):
            first = maps.scope_map('s', scope, 'idea')
            maps._cache.clear()
            second = maps.scope_map('s', scope, 'idea')
        self.assertTrue(first['available'])
        self.assertEqual(first, second)
        self.assertEqual([p['problem_key'] for p in first['points']], keys)
        xs = [p['x'] for p in first['points']] + [p['y'] for p in first['points']]
        self.assertGreaterEqual(min(xs), 0.0)
        self.assertLessEqual(max(xs), 1.0)
        self.assertTrue(0.0 <= first['neighbor_preservation'] <= 1.0)
        self.assertNotIn('vectors', first)
        self.assertFalse(maps.scope_map('s', scope, 'cross_encoder')['available'])

    def test_same_platform_map_recomputes_neighbors_within_the_platform(self):
        keys, vectors, scope = fixture()
        maps.attach_vectors({'s': scope}, keys, {'idea': vectors})
        with np.errstate(all='ignore'):
            whole = maps.scope_map('s', scope, 'idea')
            same = maps.scope_map('s', scope, 'idea', 'leetcode')
        self.assertEqual(whole['neighbors'][keys[1]], [k for k, _ in scope['rankings']['idea'][keys[1]]])
        self.assertEqual({p['platform'] for p in same['points']}, {'leetcode'})
        for key, rows in same['neighbors'].items():
            self.assertNotIn(key, rows)
            self.assertTrue(all(scope['problems'][k]['platform'] == 'leetcode' for k in rows))
            self.assertEqual(len(rows), maps.K, 'top-k is filled from the same platform, not truncated from the mixed list')
        self.assertFalse(maps.scope_map('s', scope, 'idea', 'atcoder')['available'])

    def test_viewer_route_rejects_unknown_methods(self):
        keys, vectors, scope = fixture()
        maps.attach_vectors({'s': scope}, keys, {'idea': vectors})
        dataset = {'scopes': {'s': scope}}
        with np.errstate(all='ignore'):
            self.assertTrue(response_data(dataset, '/api/map', {'scope': ['s'], 'method': ['idea']})['available'])
        with self.assertRaises(KeyError):
            response_data(dataset, '/api/map', {'scope': ['s'], 'method': ['../../AGENTS.md']})
        with self.assertRaises(KeyError):
            response_data(dataset, '/api/map', {'scope': ['s'], 'method': ['idea'], 'platform': ['nowhere']})


if __name__ == '__main__':
    unittest.main()
