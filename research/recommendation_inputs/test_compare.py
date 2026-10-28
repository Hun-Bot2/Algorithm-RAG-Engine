import json
from pathlib import Path
import tempfile
import unittest

from research.recommendation_inputs.compare import compare_rankings, digest, load_rankings, verify


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'rankings.jsonl'

    def write(self, rows):
        self.path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return self.path

    def test_known_overlap_and_concentration(self):
        runs = {'a': {'q1': ['x', 'y'], 'q2': ['x', 'z']},
                'b': {'q1': ['y', 'z'], 'q2': ['x', 'y']}}
        report = compare_rankings(runs, k=2)
        pair = report['pairwise'][0]
        self.assertEqual(pair['mean_shared_top10'], 1)
        self.assertEqual(pair['top10_overlap_fraction'], .5)
        self.assertEqual(pair['top1_agreement_fraction'], .5)
        self.assertEqual(pair['zero_overlap_queries'], 0)
        self.assertEqual(report['concentration']['a']['max_indegree'], 2)
        self.assertEqual(report['concentration']['a']['distinct_targets'], 3)

    def test_mismatched_queries_are_not_silently_intersected(self):
        with self.assertRaisesRegex(ValueError, 'cohort mismatch'):
            compare_rankings({'a': {'q1': ['x']}, 'b': {'q2': ['x']}}, k=1)

    def test_duplicate_results_are_rejected(self):
        with self.assertRaises(ValueError):
            compare_rankings({'a': {'q1': ['x', 'x']}}, k=2)

    def test_loader_checks_identity_and_scores(self):
        valid = [{'query': 'a', 'results': [['b', .8]]}, {'query': 'b', 'results': [['a', .7]]}]
        self.assertEqual(load_rankings(self.write(valid), {'a', 'b'}, k=1), {'a': ['b'], 'b': ['a']})
        for invalid in [valid[:1], valid + [valid[0]],
                        [{'query': 'a', 'results': [['a', .8]]}, valid[1]],
                        [{'query': 'a', 'results': [['unknown', .8]]}, valid[1]],
                        [{'query': 'a', 'results': [['b', float('nan')]]}, valid[1]],
                        [{'query': 'a', 'results': [['b', True]]}, valid[1]]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                load_rankings(self.write(invalid), {'a', 'b'}, k=1)

    def test_hash_change_fails_closed(self):
        self.path.write_text('original')
        expected = digest(self.path)
        self.assertEqual(verify(self.path, expected), expected)
        self.path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            verify(self.path, expected)


if __name__ == '__main__':
    unittest.main()
