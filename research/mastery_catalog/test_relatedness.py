import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from research.mastery_catalog import relatedness as R


class RelatednessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.db = root / 'learner.sqlite3'
        con = sqlite3.connect(self.db)
        con.execute('CREATE TABLE recommendation_snapshots (learner_id TEXT, kind TEXT, local_date TEXT, payload TEXT)')
        con.execute('CREATE TABLE practice_feedback (sequence INTEGER PRIMARY KEY, learner_id TEXT, canonical TEXT)')
        plan = {'date': '2026-10-20', 'slots': [{'sub_kind': 'related_channel', 'problem_key': 'codeforces:1A', 'components': {'source_key': 'leetcode:s', 'channel': 'concept'}},
                                                {'sub_kind': None, 'problem_key': 'leetcode:x', 'components': {}}]}
        con.execute('INSERT INTO recommendation_snapshots VALUES (?,?,?,?)', ('local-owner', 'plan', '2026-10-20', json.dumps(plan)))
        for i, (verdict, sup) in enumerate((('unrelated', None), ('related', 'f0'))):
            con.execute('INSERT INTO practice_feedback (learner_id, canonical) VALUES (?,?)', ('local-owner', json.dumps({'entry_id': f'f{i}', 'supersedes': sup,
                        'payload': {'source_key': 'leetcode:s', 'candidate_key': 'codeforces:1A', 'verdict': verdict}})))
        con.commit()
        self.catalog = root / 'catalog.sqlite3'
        cat = sqlite3.connect(self.catalog)
        cat.execute('CREATE TABLE problems (problem_key TEXT, title TEXT)')
        cat.execute('CREATE TABLE problem_tags (problem_key TEXT, native_tag TEXT)')
        cat.execute('CREATE TABLE neighbors (src TEXT, dst TEXT, channel TEXT, rank INTEGER)')
        cat.executemany('INSERT INTO problems VALUES (?,?)', [('leetcode:s', 'Source'), ('codeforces:1A', 'Shown')] + [(f'codeforces:{i}B', f'N{i}') for i in range(200)])
        cat.executemany('INSERT INTO neighbors VALUES (?,?,?,?)', [('leetcode:s', f'codeforces:{i}B', 'title_v2', i + 1) for i in range(200)])
        cat.commit()

    def test_export_shown_plus_deterministic_sample(self):
        a = R.export_batch(self.db, self.catalog)
        b = R.export_batch(self.db, self.catalog)
        self.assertEqual(a, b, 'deterministic')
        shown = [i for i in a['items'] if i['shown']]
        self.assertEqual([(i['source_key'], i['candidate_key']) for i in shown], [('leetcode:s', 'codeforces:1A')])
        unshown = len(a['items']) - 1
        self.assertTrue(5 <= unshown <= 40, unshown)
        self.assertNotIn('statement', json.dumps(a).lower().replace('no statements', ''))

    def test_merge_and_agreement(self):
        batch = R.export_batch(self.db, self.catalog)
        ids = [i['item_id'] for i in batch['items']]
        draft = [{'item_id': i, 'verdict': 'related', 'rationale': 'same technique'} for i in ids]
        review = [{'item_id': i, 'verdict': 'related' if n == 0 else 'unrelated', 'rationale': 'judged'} for n, i in enumerate(ids)]
        meta = {'draft': 'm1', 'review': 'm2', 'run_date': '2026-10-17', 'prompt_sha256': 'a' * 64, 'input_sha256': 'b' * 64}
        result = R.merge(batch, draft, review, meta)
        self.assertEqual(result['items'][0]['status'], 'ai_reviewed')
        self.assertTrue(all(i['status'] == 'abstained' for i in result['items'][1:]))
        self.assertTrue(result['not_gold'])
        with self.assertRaisesRegex(ValueError, 'empty rationale'):
            R.merge(batch, [dict(d, rationale=' ') for d in draft], review, meta)
        learner = R.learner_verdicts(self.db)
        self.assertEqual(learner, [{'source_key': 'leetcode:s', 'candidate_key': 'codeforces:1A', 'verdict': 'related'}], 'corrections supersede')
        report = R.agreement(learner, result['items'])
        self.assertEqual((report['n'], report['observed_agreement'], report['status']), (1, 1.0, 'insufficient_evidence'))
        self.assertEqual(R.agreement([], result['items'])['status'], 'insufficient_evidence')


if __name__ == '__main__':
    unittest.main()
