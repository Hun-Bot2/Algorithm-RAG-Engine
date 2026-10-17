import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from research.mastery_catalog import atcoder_concepts as ac

VOCAB = {'greedy', 'dynamic-programming', 'implementation', 'graph-traversal', 'sorting'}
META = {'draft': {'model': 'm-draft'}, 'review': {'model': 'm-review'}, 'run_date': '2026-10-17',
        'prompt_sha256': 'a' * 64, 'input_sha256': 'b' * 64}


def items(n=40):
    return [{'problem_key': f'atcoder:abc{100 + i // 4}_{"abcd"[i % 4]}', 'title': f'Invented Task {i}', 'task_letter': 'ABCD'[i % 4]} for i in range(n)]


def judge(truth, concepts_for, basis_for):
    return [{'item_id': k, 'concepts': concepts_for(k, truth[k]), 'basis': basis_for(k, truth[k]),
             'rationale': 'x'} for k in sorted(truth)]


class BatchTests(unittest.TestCase):
    def test_every_real_item_once_and_deterministic(self):
        batches, truth = ac.make_batches(items(40), 15)
        again, truth2 = ac.make_batches(items(40), 15)
        self.assertEqual(batches, again)
        self.assertEqual(truth, truth2)
        real = [t['problem_key'] for t in truth.values() if t['role'] == 'real']
        self.assertEqual(sorted(real), sorted(i['problem_key'] for i in items(40)))
        self.assertEqual(len(batches), 3)

    def test_controls_are_blind_and_valid(self):
        batches, truth = ac.make_batches(items(40), 20)
        for batch in batches:
            for entry in batch:
                self.assertEqual(set(entry), {'item_id', 'problem_key', 'title', 'task_letter'})
        fab = [t for t in truth.values() if t['role'] == 'fabricated']
        swp = [t for t in truth.values() if t['role'] == 'swapped']
        self.assertEqual(len(fab), 2 * ac.FABRICATED_SHARE)
        self.assertEqual(len(swp), 2 * ac.SWAPPED_SHARE)
        real_keys = {i['problem_key'] for i in items(40)}
        real_titles = {i['title'].lower() for i in items(40)}
        for t in fab:
            self.assertNotIn(t['problem_key'], real_keys)
            self.assertNotIn(t['shown_title'].lower(), real_titles)
        for t in swp:
            self.assertIn(t['problem_key'], real_keys)
            self.assertNotEqual(t['shown_title'], next(i['title'] for i in items(40) if i['problem_key'] == t['problem_key']))


class MergeTests(unittest.TestCase):
    def setUp(self):
        _, self.truth = ac.make_batches(items(40), 20)
        self.real = [k for k, t in self.truth.items() if t['role'] == 'real']

    def run_merge(self, draft_fn, review_fn, basis_d, basis_r):
        draft = judge(self.truth, draft_fn, basis_d)
        review = judge(self.truth, review_fn, basis_r)
        return ac.merge(self.truth, draft, review, VOCAB, META)

    def test_decide_branches(self):
        rec = {'basis': 'recognized', 'concepts': ['greedy', 'sorting']}
        inf = {'basis': 'title_inference', 'concepts': ['greedy']}
        unk = {'basis': 'unknown', 'concepts': []}
        self.assertEqual(ac.decide(rec, {'basis': 'recognized', 'concepts': ['greedy']}), ({'greedy'}, set(), {'sorting'}))
        self.assertEqual(ac.decide(rec, inf), ({'greedy'}, set(), {'sorting'}))
        self.assertEqual(ac.decide(inf, inf), (set(), {'greedy'}, set()))
        self.assertEqual(ac.decide(rec, unk), (set(), set(), {'greedy', 'sorting'}))
        self.assertEqual(ac.decide(unk, unk), (set(), set(), set()))
        self.assertEqual(ac.decide(rec, {'basis': 'recognized', 'concepts': ['dynamic-programming']}), (set(), set(), {'greedy', 'sorting', 'dynamic-programming'}))

    def test_status_names(self):
        self.assertEqual(ac.item_status({'a'}, set(), set()), 'labeled')
        self.assertEqual(ac.item_status(set(), {'a'}, {'b'}), 'proposed_only')
        self.assertEqual(ac.item_status(set(), set(), {'b'}), 'abstained')
        self.assertEqual(ac.item_status(set(), set(), set()), 'unknown')

    def test_controls_removed_and_counted(self):
        def known(k, t):
            return 'recognized'
        result = self.run_merge(lambda k, t: ['greedy'], lambda k, t: ['greedy'], known, known)
        self.assertEqual(result['items'], len(self.real))
        self.assertEqual(set(result['statuses']), {self.truth[k]['problem_key'] for k in self.real})
        self.assertEqual(result['controls']['fabricated']['items'], 2 * ac.FABRICATED_SHARE)
        self.assertEqual(result['controls']['fabricated']['both_known'], 2 * ac.FABRICATED_SHARE)
        self.assertEqual(result['controls']['fabricated']['ai_reviewed_label'], 2 * ac.FABRICATED_SHARE)
        self.assertTrue(all(r['evidence_status'] == 'ai_reviewed' for r in result['rows']))
        self.assertTrue(all(j['object_ref'].startswith('atcoder:abc1') for j in result['judgments']))

    def test_unknown_everywhere_leaves_no_rows(self):
        result = self.run_merge(lambda k, t: [], lambda k, t: [], lambda k, t: 'unknown', lambda k, t: 'unknown')
        self.assertEqual(result['rows'], [])
        self.assertEqual(result['item_counts']['unknown'], len(self.real))
        self.assertEqual(result['controls']['fabricated']['either_known'], 0)

    def test_title_inference_only_agreement_is_proposed(self):
        result = self.run_merge(lambda k, t: ['sorting'], lambda k, t: ['sorting'], lambda k, t: 'title_inference', lambda k, t: 'title_inference')
        self.assertEqual({r['evidence_status'] for r in result['rows']}, {'proposed'})
        self.assertEqual(result['item_counts']['proposed_only'], len(self.real))

    def test_invalid_output_rejected(self):
        draft = judge(self.truth, lambda k, t: [], lambda k, t: 'unknown')
        review = judge(self.truth, lambda k, t: [], lambda k, t: 'unknown')
        bad = [dict(draft[0], concepts=['not-a-concept'], basis='recognized')] + draft[1:]
        with self.assertRaises(ValueError):
            ac.merge(self.truth, bad, review, VOCAB, META)
        with self.assertRaises(ValueError):
            ac.merge(self.truth, draft[:-1], review, VOCAB, META)
        with self.assertRaises(ValueError):
            ac.merge(self.truth, [dict(draft[0], concepts=['greedy'])] + draft[1:], review, VOCAB, META)
        with self.assertRaises(ValueError):
            ac.merge(self.truth, [dict(draft[0], rationale=' ')] + draft[1:], review, VOCAB, META)

    def test_kappa(self):
        self.assertAlmostEqual(ac.cohen_kappa([(True, True), (False, False)] * 5), 1.0)
        self.assertAlmostEqual(ac.cohen_kappa([(True, False), (False, True)] * 5), -1.0)
        self.assertIsNone(ac.cohen_kappa([(True, True)] * 4))
        self.assertIsNone(ac.cohen_kappa([]))


class ConfirmationTests(unittest.TestCase):
    def test_confirm_reject_precedence(self):
        keys = {'atcoder:abc100_a', 'atcoder:abc100_b'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'c.jsonl'
            ac.append_confirmation(path, 'atcoder:abc100_a', 'greedy', 'confirm', 'checked by me', VOCAB, keys)
            ac.append_confirmation(path, 'atcoder:abc100_b', 'sorting', 'reject', 'wrong', VOCAB, keys)
            approved, rejected = ac.confirmation_rows(path, VOCAB, keys)
            ai = [{'problem_key': 'atcoder:abc100_a', 'concept_id': 'greedy', 'evidence_status': 'ai_reviewed', 'via': 'atcoder-full-v1'},
                  {'problem_key': 'atcoder:abc100_b', 'concept_id': 'sorting', 'evidence_status': 'ai_reviewed', 'via': 'atcoder-full-v1'}]
            rows = ac.compile_rows(ai, approved, rejected)
            by = {(r['problem_key'], r['concept_id'], r['evidence_status']) for r in rows}
            self.assertIn(('atcoder:abc100_a', 'greedy', 'approved'), by)
            self.assertIn(('atcoder:abc100_a', 'greedy', 'ai_reviewed'), by)   # provenance of both stays
            self.assertNotIn(('atcoder:abc100_b', 'sorting', 'ai_reviewed'), by)
            self.assertIn(('atcoder:abc100_b', 'sorting', 'rejected'), by)
            # a later decision supersedes an earlier one
            ac.append_confirmation(path, 'atcoder:abc100_b', 'sorting', 'confirm', 'changed my mind', VOCAB, keys)
            approved, rejected = ac.confirmation_rows(path, VOCAB, keys)
            self.assertEqual(rejected, [])
            self.assertEqual(len(approved), 2)

    def test_invalid_records_rejected_and_not_written(self):
        keys = {'atcoder:abc100_a'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'c.jsonl'
            for args in (('atcoder:zzz', 'greedy', 'confirm', 'n'), ('atcoder:abc100_a', 'nope', 'confirm', 'n'),
                         ('atcoder:abc100_a', 'greedy', 'confirm', ' '), ('atcoder:abc100_a', 'greedy', 'maybe', 'n')):
                with self.assertRaises(ValueError):
                    ac.append_confirmation(path, *args, VOCAB, keys)
            self.assertFalse(path.exists())

    def test_ai_authored_record_rejected(self):
        keys = {'atcoder:abc100_a'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'c.jsonl'
            path.write_text(json.dumps({'problem_key': 'atcoder:abc100_a', 'concept_id': 'greedy', 'decision': 'confirm',
                                        'decided_by': 'ai', 'decided_at': '2026-10-17', 'note': 'n'}) + '\n')
            with self.assertRaises(ValueError):
                ac.confirmation_rows(path, VOCAB, keys)


class LoadTests(unittest.TestCase):
    def test_load_items_filters_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'c.sqlite3'
            con = sqlite3.connect(db)
            con.execute('CREATE TABLE problems (problem_key TEXT, platform TEXT, title TEXT, eligible INTEGER, status TEXT)')
            con.executemany('INSERT INTO problems VALUES (?,?,?,?,?)', [
                ('atcoder:abc100_b', 'atcoder', 'Two', 1, 'active'), ('atcoder:abc100_a', 'atcoder', 'One', 1, 'active'),
                ('atcoder:ahc001_a', 'atcoder', 'Heuristic', 0, 'active'), ('codeforces:1A', 'codeforces', 'X', 1, 'active')])
            con.commit()
            con.close()
            got = ac.load_items(db)
            self.assertEqual([i['problem_key'] for i in got], ['atcoder:abc100_a', 'atcoder:abc100_b'])
            self.assertEqual(got[0]['task_letter'], 'A')
            self.assertEqual(ac.contest_family('atcoder:abc100_a'), 'abc')


if __name__ == '__main__':
    unittest.main()
