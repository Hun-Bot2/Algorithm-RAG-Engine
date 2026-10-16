import unittest
import types
from codecontests import normalize,source_names,SOURCE_NAMES

class PublisherSchemaTests(unittest.TestCase):
    def test_sidecar_required_and_hf_enum(self):
        schema=types.SimpleNamespace(metadata=None)
        with self.assertRaises(ValueError):source_names(schema)
        side={'deepmind--code_contests':{'features':{'source':{'names':SOURCE_NAMES,'num_classes':7}}}}
        self.assertEqual(source_names(schema,side)[5],'ATCODER')
    def test_native_cf_and_no_fuzzy_atcoder_join(self):
        row={'source':2,'description':'Find a sum.','name':'A','cf_contest_id':123,'cf_index':'A'}
        a=normalize(row,'ignored','train',0,SOURCE_NAMES,{'codeforces:123A'},'shard1')
        b=normalize(row,'ignored','train',0,SOURCE_NAMES,set(),'shard2')
        self.assertEqual(a['native_key'],'codeforces:123A')
        self.assertNotEqual(a['dataset_key'],b['dataset_key'])
        row['source']=5
        self.assertIsNone(normalize(row,'ignored','train',0,SOURCE_NAMES,set())['native_key'])
    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):normalize({'source':7},'', '',0,SOURCE_NAMES,set())

if __name__=='__main__':unittest.main()

class RangeSafetyTests(unittest.TestCase):
    def test_whole_shard_and_ignored_range_rejected(self):
        import tempfile
        from unittest.mock import patch
        from codecontests import RangeReader
        with tempfile.TemporaryDirectory() as d:
            r=RangeReader('https://example.test/file',100,d,{})
            with self.assertRaises(ValueError):r.read()
            response=types.SimpleNamespace(status=200)
            class Context:
                def __enter__(self):return response
                def __exit__(self,*args):pass
            with patch('codecontests.urlopen',return_value=Context()):
                with self.assertRaisesRegex(ValueError,'ignored Range'):r.read(10)
