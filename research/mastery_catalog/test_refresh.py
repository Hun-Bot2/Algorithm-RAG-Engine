import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.mastery_catalog import refresh as R
from src.learning.catalog_sources import FetchResult


class DeltaTests(unittest.TestCase):
    def test_classifies_add_change_remove_unchanged(self):
        base = {'platform': 'codeforces', 'url': 'u', 'tags': ['dp'], 'native_difficulty': 800, 'contest': 1, 'eligibility': 'eligible', 'exclusion_reasons': []}
        prev = [dict(base, problem_key='codeforces:1A', title='a'), dict(base, problem_key='codeforces:1B', title='b'), dict(base, problem_key='codeforces:1C', title='c'),
                dict(base, problem_key='leetcode:x', platform='leetcode', title='x')]
        cur = [dict(base, problem_key='codeforces:1A', title='a'), dict(base, problem_key='codeforces:1B', title='b', native_difficulty=900),
               dict(base, problem_key='codeforces:2A', title='new')]
        delta = R.classify_delta(prev, cur)
        self.assertEqual(delta, {'added': ['codeforces:2A'], 'changed': ['codeforces:1B'], 'removed': ['codeforces:1C'], 'unchanged': 1})


class CacheTests(unittest.TestCase):
    def test_unchanged_nodes_keep_their_own_vectors_bit_for_bit(self):
        v = np.array([[1, 0], [0.99999994, 1e-7], [0, 1]], dtype=np.float32)  # same text, batch-dependent float noise
        cache = R.EmbeddingCache(['a', 'b', 'c'], ['same', 'same', 'other'], v)
        calls = []
        out, encoded = cache.vectors(['a', 'b', 'c', 'd', 'e'], ['same', 'same', 'other', 'same', 'brand new'],
                                     lambda texts: calls.append(texts) or np.ones((len(texts), 2), dtype=np.float32))
        self.assertTrue(np.array_equal(out[:3], v))
        self.assertTrue(np.array_equal(out[3], v[0]), 'new node with known text reuses a cached vector')
        self.assertEqual(calls, [['brand new']])
        self.assertEqual(encoded, 1)
        other = R.EmbeddingCache(['a'], ['same'], v[:1], model_revision='different-model')
        self.assertNotEqual(other.key('same'), cache.key('same'), 'model revision is part of the cache key')


class FetchTests(unittest.TestCase):
    def test_stops_on_refusal_and_records_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, sleeps = [], []
            def fetch(url):
                calls.append(url)
                return FetchResult(b'', 429, 'http_error_429') if len(calls) == 2 else FetchResult(b'{"status":"OK"}', 200)
            with self.assertRaisesRegex(RuntimeError, 'refused'):
                R.fetch_all(Path(tmp) / 'raw', fetch=fetch, sleep=sleeps.append)
            self.assertEqual(len(calls), 2, 'no further requests after a refusal')
            self.assertEqual(sleeps, [R.GAP_SECONDS])
            manifest = json.loads((Path(tmp) / 'raw' / 'manifest.json').read_text())
            self.assertEqual(manifest['codeforces_contests']['http_status'], 429)
            with self.assertRaises(FileExistsError):
                R.fetch_all(Path(tmp) / 'raw', fetch=fetch, sleep=sleeps.append)

    def test_replay_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            (raw / 'codeforces_problems.raw').write_bytes(b'abc')
            (raw / 'manifest.json').write_text(json.dumps({'codeforces_problems': {'status': 'ok', 'sha256': R.sha256_bytes(b'abc')}}))
            self.assertIn('codeforces_problems', R.load_raw(raw)[1])
            (raw / 'codeforces_problems.raw').write_bytes(b'abd')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                R.load_raw(raw)


class CanonicalContestTests(unittest.TestCase):
    def test_adt_reuse_does_not_move_canonical_contest(self):
        bodies = {'atcoder_contests': json.dumps([{'id': 'abc235', 'start_epoch_second': 1}, {'id': 'adt_easy_20261016_2', 'start_epoch_second': 9}, {'id': 'arc130', 'start_epoch_second': 2}]).encode(),
                  'atcoder_contest_problem': json.dumps([{'contest_id': 'adt_easy_20261016_2', 'problem_id': 'abc235_a'}, {'contest_id': 'abc235', 'problem_id': 'abc235_a'},
                                                         {'contest_id': 'adt_all_1', 'problem_id': 'arc130_a'}, {'contest_id': 'abc229', 'problem_id': 'arc130_a'},
                                                         {'contest_id': 'adt_only', 'problem_id': 'adt_x_a'}]).encode()}
        canonical = R.canonical_atcoder_contests(bodies)
        self.assertEqual(canonical['abc235_a'], 'abc235')
        self.assertEqual(canonical['arc130_a'], 'abc229', 'earliest non-ADT listing when the prefix contest does not list it')
        self.assertNotIn('adt_x_a', canonical, 'ADT-only tasks keep their URL')


class NeighborTests(unittest.TestCase):
    def test_exact_neighbors_match_brute_force(self):
        rng = np.random.default_rng(7)
        v = rng.normal(size=(40, 8)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        keys = [f'codeforces:{i:03d}A' if i % 2 else f'atcoder:t{i:03d}' for i in range(40)]
        order = np.argsort(keys)
        keys = [keys[i] for i in order]
        v = v[order]
        platforms = [k.split(':')[0] for k in keys]
        out = R.exact_neighbors(keys, platforms, keys, [None] * 40, v, k=5)
        sims = v @ v.T
        for i in range(40):
            row = sims[i].copy()
            row[i] = -np.inf
            self.assertEqual(set(out['any'][0][i]), set(np.argsort(-row)[:5]))
            cross = [j for j in np.argsort(-row) if platforms[j] != platforms[i]][:5]
            self.assertEqual(set(out['cross'][0][i]), set(cross))


class ContestTests(unittest.TestCase):
    def test_contests_and_problems_from_sources(self):
        bodies = {'codeforces_contests': json.dumps({'status': 'OK', 'result': [{'id': 4, 'name': 'Codeforces Beta Round 4 (Div. 2)', 'startTimeSeconds': 1268395200, 'durationSeconds': 7200}]}).encode(),
                  'atcoder_contests': json.dumps([{'id': 'abc300', 'start_epoch_second': 1682164800, 'duration_second': 6000, 'title': 'ABC 300'}]).encode(),
                  'atcoder_contest_problem': json.dumps([{'contest_id': 'abc300', 'problem_id': 'abc300_a', 'problem_index': 'A'}]).encode()}
        rows = [{'problem_key': 'codeforces:4A', 'platform': 'codeforces', 'contest': 4}, {'problem_key': 'atcoder:abc300_a', 'platform': 'atcoder', 'contest': 'abc300'}]
        contests = {c['contest_id']: c for c in R.attach_contest_problems(R.contests_from(bodies), rows, bodies)}
        self.assertEqual(contests['4']['kind'], 'div2')
        self.assertEqual(contests['4']['problems'], [('codeforces:4A', 'A')])
        self.assertEqual(contests['abc300']['start_at'], '2023-04-22T12:00:00Z')
        self.assertEqual(contests['abc300']['problems'], [('atcoder:abc300_a', 'A')])


class EndToEndTests(unittest.TestCase):
    def test_synthetic_refresh_with_removal_addition_and_manual_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prev = root / 'catalog-p' / 'run'
            prev.mkdir(parents=True)
            base = {'tags': ['dp'], 'eligibility': 'eligible', 'exclusion_reasons': [], 'status': 'active', 'first_seen_version': 'catalog-p',
                    'duplicate_candidate_group': None, 'encoder_flags': []}
            rows = [dict(base, problem_key='codeforces:1A', platform='codeforces', title='Alpha', url='https://codeforces.com/problemset/problem/1/A', native_difficulty=800, contest=1),
                    dict(base, problem_key='codeforces:1B', platform='codeforces', title='Beta', url='https://codeforces.com/problemset/problem/1/B', native_difficulty=900, contest=1),
                    dict(base, problem_key='leetcode:two-sum', platform='leetcode', title='Two Sum', url='https://leetcode.com/problems/two-sum/', native_difficulty='Easy', contest=None, tags=['Hash Table'])]
            for r in rows:
                r['representative_key'] = r['problem_key']
            (prev / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            keys = [r['problem_key'] for r in rows]
            (prev / 'keys.json').write_text(json.dumps(keys))
            (prev / 'embedded_text.jsonl').write_text(''.join(json.dumps({'problem_key': r['problem_key'], 'text': r['title'], 'source': 'identity'}) + '\n' for r in rows))
            (prev / 'vector_meta.jsonl').write_text(''.join(json.dumps({'problem_key': k, 'reliability': 'reliable', 'language_class': 'en'}) + '\n' for k in keys))
            vecs = np.eye(3, 4, dtype=np.float32)
            np.save(prev / 'vectors.npy', vecs)
            raw = root / 'raw'
            raw.mkdir()
            cf = {'status': 'OK', 'result': {'problems': [{'contestId': 1, 'index': 'A', 'name': 'Alpha', 'rating': 800, 'tags': ['dp']},
                                                         {'contestId': 2, 'index': 'A', 'name': 'Gamma', 'rating': 1000, 'tags': ['dp']}], 'problemStatistics': []}}
            bodies = {'codeforces_problems': json.dumps(cf).encode(), 'atcoder_problems': b'[]'}
            manifest = {}
            for name, body in bodies.items():
                (raw / f'{name}.raw').write_bytes(body)
                manifest[name] = {'status': 'ok', 'sha256': R.sha256_bytes(body)}
            (raw / 'manifest.json').write_text(json.dumps(manifest))
            manual = root / 'manual.json'
            manual.write_text(json.dumps({'rows': [{'problem_key': 'leetcode:new-problem', 'platform': 'leetcode', 'title': 'New Problem', 'tags': ['Dynamic Programming'],
                                                    'native_difficulty': 'Medium', 'url': 'https://leetcode.com/problems/new-problem/', 'contest': None}]}))
            encoded = []
            class Encoder:
                def encode(self, texts):
                    encoded.extend(texts)
                    return np.full((len(texts), 4), 0.5, dtype=np.float32)
            receipt = R.refresh('catalog-p', 'catalog-q', raw_dir=raw, encoder=Encoder(), translator=None, tokenizer=None, manual_additions=manual,
                                output_root=root, now='2026-10-17T00:00:00+00:00')
            self.assertEqual(receipt['delta'], {'added': 1, 'changed': 0, 'removed': 1, 'unchanged': 1})
            self.assertEqual(sorted(encoded), ['Gamma', 'New Problem'], 'only new titles are encoded')
            self.assertEqual(receipt['manual_additions'], ['leetcode:new-problem'])
            import sqlite3
            con = sqlite3.connect(f"file:{root / 'catalog-q' / 'catalog-q.sqlite3'}?mode=ro", uri=True)
            got = {k: (s, f, p) for k, s, f, p in con.execute('SELECT problem_key, status, first_seen_version, provenance FROM problems')}
            self.assertEqual(got['codeforces:1B'][0], 'removed_upstream', 'upstream removals are kept')
            self.assertEqual(got['codeforces:2A'][1], 'catalog-q')
            self.assertEqual(got['codeforces:1A'][1], 'catalog-p')
            self.assertTrue(got['leetcode:new-problem'][2].startswith('user_entered'))
            self.assertTrue(got['leetcode:two-sum'][2].startswith('catalog'))
            with self.assertRaises(FileExistsError):
                R.refresh('catalog-p', 'catalog-q', raw_dir=raw, encoder=Encoder(), output_root=root)


if __name__ == '__main__':
    unittest.main()
