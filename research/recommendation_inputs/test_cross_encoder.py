import json
from pathlib import Path
import tempfile
import unittest
from research.recommendation_inputs.cross_encoder import download_approved,load_local,rerank_bundle,source_query


class CrossEncoderTests(unittest.TestCase):
    def test_download_rejects_wrong_revision_before_creating_model(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaisesRegex(ValueError,'identity/revision'):
                download_approved(root/'model',root/'manifest.json',fetcher=lambda url,limit:json.dumps({'id':'wrong','sha':'wrong'}).encode())
            self.assertFalse((root/'model').exists())

    def test_local_loader_rejects_unreviewed_manifest_without_model_access(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=root/'manifest.json';m.write_text('{}')
            with self.assertRaisesRegex(ValueError,'unreviewed'):
                load_local(root/'missing',m)

    def bundle(self):
        return {'problems':{key:{'input_views':{'idea_rationale':key,'idea_structured':key+' structure'}} for key in ['a','b','c']},
                'rankings':{'idea_rationale':{'a':[['b',.9]],'b':[['a',.8]],'c':[['a',.4]]},
                            'idea_structured':{'a':[['c',.8]],'b':[['c',.7]],'c':[['b',.6]]}}}

    def test_query_modes_change_only_query_text(self):
        p={'input_views':{'idea_rationale':'A complement lookup.'}}
        self.assertEqual(source_query(p,'plain'),'A complement lookup.')
        self.assertTrue(source_query(p,'instruction').startswith('Which problem'))
        with self.assertRaises(ValueError):source_query(p,'unknown')

    def test_rerank_preserves_pool_and_labels_scores_as_logits(self):
        seen=[]
        def score(pairs):seen.extend(pairs);return list(range(len(pairs)))
        rows,receipt=rerank_bundle(self.bundle(),score)
        self.assertEqual(receipt['pair_count'],6)
        self.assertEqual(rows['a'][0],['c',1.])
        self.assertTrue(all(k!=source for source,targets in rows.items() for k,_ in targets))
        self.assertTrue(seen[0][0].startswith('Which problem'))
        self.assertIn('not probability',receipt['score_kind'])
        with self.assertRaises(ValueError):rerank_bundle(self.bundle(),lambda pairs:[float('nan')]*len(pairs))


if __name__=='__main__':unittest.main()
