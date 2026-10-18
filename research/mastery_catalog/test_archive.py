import copy
import json
import unittest
from pathlib import Path

from research.mastery_catalog import archive as A

EXAMPLE = json.loads((Path(__file__).resolve().parents[2] / 'assets/mastery/archive.example.v1.json').read_text())


class ArchiveTests(unittest.TestCase):
    def mutate(self, index, **changes):
        doc = copy.deepcopy(EXAMPLE)
        for key, value in changes.items():
            if '.' in key:
                outer, inner = key.split('.')
                doc['records'][index][outer][inner] = value
            else:
                doc['records'][index][key] = value
        return doc

    def test_example_is_valid_and_compiles_with_history(self):
        seen, successors = A.validate(EXAMPLE)
        self.assertEqual(len(seen), 3)
        self.assertEqual(successors, {'example-grad-2024-s1-q3@1'})
        rows = A.compile_rows(EXAMPLE)
        flags = {(r['row'][0], r['row'][1]): r['row'][12] for r in rows}
        self.assertEqual(flags[('example-grad-2024-s1-q3', 1)], 1, 'earlier version kept and marked superseded')
        self.assertEqual(flags[('example-grad-2024-s1-q3', 2)], 0)

    def test_rejections(self):
        cases = [
            (self.mutate(0, statement_text='copied exam text'), 'statement text requires'),
            (self.mutate(2, confidence='high'), 'recollection cannot have high confidence'),
            (self.mutate(0, category='rumor'), 'invalid category'),
            (self.mutate(0, **{'source.url': 'http://insecure.example'}), 'HTTPS'),
            (self.mutate(0, **{'source.citation': ' '}), 'citation required'),
            (self.mutate(1, supersedes='example-grad-2024-s1-q3@9'), 'dangling'),
            (self.mutate(1, supersedes=None), 'must supersede'),
        ]
        same = copy.deepcopy(EXAMPLE)
        same['records'][0]['links'] = [{'problem_key': 'codeforces:4A', 'relation': 'same_problem', 'evidence': ['one citation'], 'status': 'approved'}]
        cases.append((same, 'two evidence'))
        ai = copy.deepcopy(EXAMPLE)
        ai['records'][0]['recorded_by'] = 'ai_draft'
        ai['records'][0]['links'] = [{'problem_key': 'codeforces:4A', 'relation': 'same_problem', 'evidence': ['a', 'b'], 'status': 'ai_reviewed'}]
        cases.append((ai, 'cannot come from AI'))
        branch = copy.deepcopy(EXAMPLE)
        extra = copy.deepcopy(branch['records'][1])
        extra['version'] = 3
        branch['records'].append(extra)
        cases.append((branch, 'branching'))
        dup = copy.deepcopy(EXAMPLE)
        dup['records'].append(copy.deepcopy(dup['records'][0]))
        cases.append((dup, 'duplicate record version'))
        for doc, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                A.validate(doc)


if __name__ == '__main__':
    unittest.main()
