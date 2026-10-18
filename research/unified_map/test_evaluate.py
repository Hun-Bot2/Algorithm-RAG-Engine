import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from research.unified_map import evaluate


class CohortTests(unittest.TestCase):
    def test_full_semantic_and_matched_cohorts_are_separate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'nodes.jsonl').write_text('{"query":"q"}\n')
            (root/'manifest.json').write_text(json.dumps({'output_sha256':{},'taxonomy_sha256':hashlib.sha256((evaluate.ROOT/'research/algorithm_system/taxonomy.v1.json').read_bytes()).hexdigest()}))
            for name in ('semantic','matched_semantic','tag','hybrid'):
                (root/name).mkdir();(root/name/'top50.jsonl').write_text('{"query":"q"}\n')
            manifest=json.loads((root/'manifest.json').read_text())
            manifest['output_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'nodes.jsonl',*root.glob('*/top50.jsonl')]}
            (root/'manifest.json').write_text(json.dumps(manifest))
            seen=[]
            def fake(corpus,methods,labels,**kw):
                seen.append(set(methods));return {'methods':list(methods)}
            with patch.object(evaluate,'evaluate_rankings',fake):
                evaluate.run(root)
            self.assertEqual(seen,[{'semantic'},{'matched_semantic','tag','hybrid'}])
            with self.assertRaises(FileExistsError):
                with patch.object(evaluate,'evaluate_rankings',fake):evaluate.run(root)
