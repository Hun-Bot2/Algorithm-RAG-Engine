"""Runner tests use authored fixtures and an isolated private directory."""
import json,unittest
from unittest.mock import patch
from research.identity_resolution import test_build as fixtures
from research.identity_resolution.run import run

class RunTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.EvidenceBuilderTests();self.fixture.setUp();self.addCleanup(self.fixture.tearDown)
        self.wb=self.fixture.make_workbench();self.root=self.fixture.root
        self.evidence=self.root/'evidence.json';self.evidence.write_text(json.dumps([self.fixture.row]))
        self.human=dict(pair_id='pair-fixture',reviewer='fixture-user',role='human',kind='relationship',independent=False,
            similarity=None,prerequisite='unknown',transfer='unknown',difficulty='unknown',
            rationale='Authored human fixture only; no actual research review.',evidence=['Authored fixture evidence.'])
        self.wb.append(self.human)
    def execute(self,out,append=False):
        with patch('research.identity_resolution.run.ROOT',self.root):
            return run(self.evidence,self.root/'pilot.json',self.wb.db_path,self.root/'private-learning'/out,append=append)
    def test_preview_append_and_idempotent_import_preserve_human_records(self):
        preview=self.execute('preview');self.assertFalse(preview['appended']);self.assertEqual(len(self.wb.records()),1)
        result=self.execute('append',True)
        self.assertEqual(result['new_records'],1);self.assertEqual(result['human_records_before'],1);self.assertEqual(result['human_records_after'],1)
        self.assertEqual(result['accepted_overlay_after'],0);self.assertEqual(result['preserved_existing_records'],1)
        replay=self.execute('replay',True);self.assertEqual(replay['new_records'],0);self.assertEqual(replay['duplicate_records'],1)
        with self.assertRaises(FileExistsError):self.execute('append',True)
    def test_invalid_evidence_creates_no_output_or_record(self):
        row=dict(self.fixture.row,source_content_sha256='0'*64);self.evidence.write_text(json.dumps([row]))
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.execute('invalid',True)
        self.assertEqual(len(self.wb.records()),1);self.assertFalse((self.root/'private-learning/invalid').exists())
    def test_output_must_stay_private(self):
        with patch('research.identity_resolution.run.ROOT',self.root),self.assertRaisesRegex(ValueError,'private output'):
            run(self.evidence,self.root/'pilot.json',self.wb.db_path,self.root/'public-out',append=True)
        self.assertEqual(len(self.wb.records()),1)

if __name__=='__main__':unittest.main()
