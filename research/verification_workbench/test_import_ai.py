"""AI import tests use authored fixtures, not actual pilot judgments."""
import unittest
import test_core as fixtures
from import_ai_reviews import import_ai

class AIImportTests(unittest.TestCase):
    def setUp(self):fixtures.WorkbenchTest.setUp(self)
    def batch(self):
        return [fixtures.relationship(p,'fixture-ai',role='ai',independent=False) for p in self.w.pairs]
    def test_exact_coverage_and_idempotent_ai_only_import(self):
        records=self.batch();result=import_ai(self.w,records)
        self.assertEqual(result['new_records'],2);self.assertEqual(result['accepted_overlay_count'],0)
        replay=import_ai(self.w,records)
        self.assertEqual(replay['new_records'],0);self.assertEqual(replay['duplicate_records'],2)
        self.assertEqual(self.w.audit()['records'],2)
    def test_bad_roles_missing_pairs_and_invalid_records_never_partially_write(self):
        for mutate in ('human','missing','invalid'):
            records=self.batch()
            if mutate=='human':records[-1]['role']='human'
            elif mutate=='missing':records.pop()
            else:records[-1]['rationale']='short'
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):import_ai(self.w,records)
            self.assertEqual(self.w.audit()['records'],0)

if __name__=='__main__':unittest.main()
