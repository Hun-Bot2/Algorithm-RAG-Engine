import json
import unittest
from collections import Counter
from pathlib import Path

from src.learning.curriculum import validate_curriculum


ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


class ReviewedSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = load('curriculum/seed.v1.json')
        cls.annotations = load('curriculum/annotations.v1.json')['annotations']
        cls.judgments = load('research/relationship-judgments.v1.json')

    def test_reviewed_seed_is_valid_dag_with_registered_annotations(self):
        self.assertIs(validate_curriculum(self.seed), self.seed)
        problems = self.seed['problems']
        edges = self.seed['relationships']
        self.assertEqual(len(problems), 90)
        self.assertEqual(Counter(p['problem_key'].split(':')[0] for p in problems),
                         {'leetcode': 60, 'atcoder': 20, 'codeforces': 10})
        self.assertEqual(Counter(r['type'] for r in edges),
                         {'prerequisite': 28, 'similarity': 8, 'transfer': 4})
        self.assertEqual(set(self.annotations), {p['problem_key'] for p in problems})
        for p in problems:
            self.assertEqual(p['review']['status'], 'approved')
            self.assertEqual(p['review']['reviewer'], 'sol-independent-review')
            self.assertTrue(p['review']['evidence'])
            annotation = self.annotations[p['problem_key']]
            self.assertEqual(annotation['source_url'], p['canonical_url'])
            self.assertEqual(annotation['platform'], p['problem_key'].split(':')[0])
            self.assertTrue(annotation['concept_rationale'])
        self.assertTrue(all(r['review']['status'] == 'approved' and
                            r['review']['reviewer'] == 'sol-independent-review' and
                            r['review']['evidence'] for r in edges))

    def test_access_evidence_is_explicit_and_release_gate_passes(self):
        states = Counter(p['access_state'] for p in self.seed['problems'])
        self.assertEqual(states, {'free': 90})
        for p in self.seed['problems']:
            ann = self.annotations[p['problem_key']]
            self.assertEqual(ann['observed_at'], '2026-10-16')
            self.assertIn('official', ann['access_provenance'].lower())
            self.assertIn('statement', ann['access_provenance'].lower())
            self.assertIn('submission was not tested', ann['access_provenance'])
        self.assertFalse({'leetcode:encode-and-decode-strings',
                          'leetcode:graph-valid-tree',
                          'leetcode:meeting-rooms-ii'} & set(self.annotations))
        self.assertIs(validate_curriculum(self.seed, require_release=True),
                      self.seed)

    def test_judgment_registry_is_reviewed_and_evaluation_only(self):
        doc = self.judgments
        self.assertEqual(doc['permitted_uses'], ['evaluation'])
        rows = doc['judgments']
        self.assertEqual(len(rows), 180)
        self.assertEqual(len({row['id'] for row in rows}), 180)
        pairs = {(row['source'], row['target']) for row in rows}
        self.assertEqual(len(pairs), 60)
        self.assertEqual(Counter(row['type'] for row in rows),
                         {'similarity': 60, 'prerequisite': 60, 'transfer': 60})
        for row in rows:
            self.assertIn(row['source'], self.annotations)
            self.assertIn(row['target'], self.annotations)
            self.assertEqual(row['direction'], 'source_to_target')
            self.assertIs(type(row['relevance']), int)
            self.assertIn(row['relevance'], range(4))
            self.assertEqual(row['annotator'], 'luna-draft')
            self.assertEqual(row['adjudication'], 'accepted')
            self.assertEqual(row['adjudicator'], 'sol-independent-review')
            self.assertTrue(row['family'])
            self.assertTrue(row['platform_pair'])
            self.assertTrue(row['evidence'])
        for pair in pairs:
            self.assertEqual({row['type'] for row in rows if
                              (row['source'], row['target']) == pair},
                             {'similarity', 'prerequisite', 'transfer'})


if __name__ == '__main__':
    unittest.main()
