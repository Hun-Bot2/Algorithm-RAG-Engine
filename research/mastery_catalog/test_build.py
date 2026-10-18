import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.mastery_catalog import build as builder
from research.mastery_catalog import judgments


def write(path, value):
    path.write_text(value if isinstance(value, str) else json.dumps(value))
    return path


class CatalogFixture:
    def __init__(self, root):
        self.root = Path(root)
        rows = [
            {'problem_key': 'leetcode:two-sum', 'platform': 'leetcode', 'title': 'Two Sum', 'url': 'https://leetcode.com/problems/two-sum/',
             'native_difficulty': 'Easy', 'tags': ['Hash Table'], 'contest': None, 'eligibility': 'eligible', 'exclusion_reasons': [],
             'content': 'MUST NOT BE READ'},
            {'problem_key': 'codeforces:4A', 'platform': 'codeforces', 'title': 'Watermelon', 'url': 'https://codeforces.com/problemset/problem/4/A',
             'native_difficulty': 800, 'tags': ['math', 'brute force'], 'contest': 4, 'eligibility': 'eligible', 'exclusion_reasons': []},
            {'problem_key': 'atcoder:abc300_a', 'platform': 'atcoder', 'title': 'N-choice', 'url': 'https://atcoder.jp/contests/abc300/tasks/abc300_a',
             'native_difficulty': None, 'tags': None, 'contest': 'abc300', 'eligibility': 'eligible', 'exclusion_reasons': []},
            {'problem_key': 'atcoder:agc010_a', 'platform': 'atcoder', 'title': 'Addition', 'url': 'https://atcoder.jp/contests/agc010/tasks/agc010_a',
             'native_difficulty': None, 'tags': None, 'contest': 'agc010', 'eligibility': 'excluded', 'exclusion_reasons': ['fixture']},
        ]
        self.catalog = write(self.root / 'catalog.jsonl', '\n'.join(json.dumps(r) for r in rows) + '\n')
        keys = [r['problem_key'] for r in rows]
        self.keys = write(self.root / 'keys.json', keys)
        self.meta = write(self.root / 'meta.jsonl', '\n'.join(json.dumps({'problem_key': k, 'reliability': 'reliable', 'language_class': 'en'}) for k in keys))
        nd = self.root / 'neighbors'
        nd.mkdir()
        idx = np.array([[1, 2, -1], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int32)
        score = np.linspace(0.9, 0.1, 12, dtype=np.float32).reshape(4, 3)
        receipt = {}
        for name in ('any', 'cross'):
            np.save(nd / f'{name}-idx.npy', idx)
            np.save(nd / f'{name}-score.npy', score)
            receipt[name] = {f'{p}_sha256': hashlib.sha256((nd / f'{name}-{p}.npy').read_bytes()).hexdigest() for p in ('idx', 'score')}
        write(nd / 'receipt.json', receipt)
        self.models = write(self.root / 'models.json', {'abc300_a': {'difficulty': -1000, 'discrimination': 0.0044794, 'is_experimental': False},
                                                         'agc010_a': {'difficulty': -10000, 'discrimination': 0.0044794}})
        self.aliases = write(self.root / 'aliases.json', {'aliases': [{'legacy_key': 'atcoder:abc049_c', 'native_key': 'atcoder:arc065_a', 'evidence': 'fixture'}]})
        self.concepts = write(self.root / 'concepts.json', {'concepts': [{'id': 'arrays-hashing', 'label': 'a'}, {'id': 'search-enumeration', 'label': 'b'}]})
        self.cmap = write(self.root / 'cmap.json', {'schema': 'mastery-concept-map.v1', 'entries': [
            {'platform': 'leetcode', 'native_tag': 'Hash Table', 'judgment_ref': 'r1', 'concepts': [{'concept_id': 'arrays-hashing', 'evidence_status': 'ai_reviewed'}]},
            {'platform': 'codeforces', 'native_tag': 'brute force', 'judgment_ref': 'r2', 'concepts': [{'concept_id': 'search-enumeration', 'evidence_status': 'abstained'}]}]})

    def inputs(self):
        return {'catalog': self.catalog, 'keys': self.keys, 'vector_meta': self.meta, 'neighbors_dir': self.root / 'neighbors',
                'atcoder_models': self.models,
                'aliases': self.aliases, 'concepts': self.concepts}


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = CatalogFixture(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def build(self, name='out'):
        return builder.build(Path(self.tmp.name) / name, 'catalog-test', self.fx.inputs(), concept_map=self.fx.cmap,
                             built_at='2026-10-17T00:00:00+00:00', verify_atcoder=False)

    def test_build_contents_and_receipt(self):
        receipt = self.build()
        db = Path(self.tmp.name) / 'out' / 'catalog-test.sqlite3'
        self.assertEqual(receipt['output_sha256'], hashlib.sha256(db.read_bytes()).hexdigest())
        con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        rows = {r[0]: r for r in con.execute('SELECT problem_key, difficulty_kind, difficulty_value, access, eligible FROM problems')}
        self.assertEqual(rows['leetcode:two-sum'][1:], ('lc_level', 1.0, 'free_at_collection', 1))
        self.assertEqual(rows['codeforces:4A'][1:4], ('cf_rating', 800.0, 'free'))
        self.assertEqual(rows['atcoder:abc300_a'][1:3], ('atc_irt', -1000.0))
        self.assertEqual(rows['atcoder:agc010_a'][1], 'none', 'sentinel difficulty becomes unknown')
        self.assertEqual(rows['atcoder:agc010_a'][4], 0)
        dump = '\n'.join(con.iterdump())
        self.assertNotIn('MUST NOT BE READ', dump, 'statement fields are never loaded')
        channels = dict(con.execute('SELECT channel, COUNT(*) FROM neighbors GROUP BY channel'))
        self.assertEqual(channels['title_v2'], 11, 'negative indices skipped')
        self.assertEqual(set(channels), {'title_v2', 'title_v2_cross'})
        concepts = set(con.execute('SELECT problem_key, concept_id, evidence_status FROM problem_concepts'))
        self.assertEqual(concepts, {('leetcode:two-sum', 'arrays-hashing', 'ai_reviewed'), ('codeforces:4A', 'search-enumeration', 'abstained')})
        self.assertEqual(con.execute("SELECT value FROM catalog_meta WHERE key='statement_fields_read'").fetchone()[0], 'false')
        self.assertEqual(receipt['counts']['problems_with_concepts'], 1)
        with self.assertRaises(sqlite3.OperationalError):
            con.execute('DELETE FROM problems')

    def test_refuses_overwrite_and_hash_drift(self):
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()
        (self.fx.root / 'neighbors' / 'any-idx.npy').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.build('other')

    def test_duplicate_keys_rejected(self):
        with self.fx.catalog.open('a') as handle:
            handle.write(self.fx.catalog.read_text().splitlines()[0] + '\n')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.build('dup')


class JudgmentTests(unittest.TestCase):
    vocab = {'arrays-hashing', 'sorting'}
    expected = {('leetcode', 'Hash Table'), ('leetcode', 'Bucket Sort')}

    def entries(self, **override):
        base = [{'platform': 'leetcode', 'native_tag': 'Hash Table', 'concepts': ['arrays-hashing'], 'rationale': 'lookup'},
                {'platform': 'leetcode', 'native_tag': 'Bucket Sort', 'concepts': ['sorting'], 'rationale': 'sorting'}]
        base[0].update(override)
        return base

    def test_validator_rejections(self):
        judgments.validate_crosswalk(self.entries(), self.expected, self.vocab)
        for bad in ({'rationale': ' '}, {'concepts': ['nope']}, {'concepts': ['sorting'] * 2}, {'concepts': ['sorting', 'arrays-hashing', 'x', 'y']}):
            with self.assertRaises(ValueError):
                judgments.validate_crosswalk(self.entries(**bad), self.expected, self.vocab)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            judgments.validate_crosswalk(self.entries()[:1], self.expected, self.vocab)

    def test_merge_pairs_and_records(self):
        draft = judgments.validate_crosswalk(self.entries(), self.expected, self.vocab)
        review = judgments.validate_crosswalk([dict(e, concepts=['arrays-hashing', 'sorting'] if e['native_tag'] == 'Bucket Sort' else e['concepts']) for e in self.entries()], self.expected, self.vocab)
        meta = {'draft': {'model': 'm1'}, 'review': {'model': 'm2'}, 'run_date': '2026-10-17', 'prompt_sha256': 'a' * 64, 'input_sha256': 'b' * 64}
        doc = judgments.merge(draft, review, meta)
        bucket = next(e for e in doc['entries'] if e['native_tag'] == 'Bucket Sort')
        self.assertEqual(bucket['concepts'], [{'concept_id': 'sorting', 'evidence_status': 'ai_reviewed'}, {'concept_id': 'arrays-hashing', 'evidence_status': 'abstained'}])
        self.assertEqual(doc['stats'], {'pairs_agreed': 2, 'pairs_abstained': 1, 'tags_identical': 1, 'tags_both_empty': 0})
        self.assertTrue(doc['not_gold'])
        with self.assertRaisesRegex(ValueError, 'empty'):
            judgments.validate_record(dict(doc['judgments'][0], rationale=''))
        with self.assertRaisesRegex(ValueError, 'prompt_sha256'):
            judgments.validate_record(dict(doc['judgments'][0], prompt_sha256='short'))


if __name__ == '__main__':
    unittest.main()
