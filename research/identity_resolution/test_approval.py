import unittest
from copy import deepcopy
from unittest.mock import patch
from research.identity_resolution.apply_approval import approved_overlay, mapping_for


class ApprovalTests(unittest.TestCase):
    def test_metadata_only_preserves_flags_and_unresolved_and_input(self):
        rows = [dict(source_key='train', source_content_sha256='abc', selected_native_key='atcoder:abc084_c', status='ai_proposed', canonical_url='url', quality_flags=['official_en_ja_constraint_disagreement', 'sample_exceeds_english_bound']),
                dict(source_key='p03510', source_content_sha256='def', selected_native_key=None, status='unresolved')]
        original = deepcopy(rows)
        with patch('research.identity_resolution.apply_approval.validate_evidence') as validate:
            overlay = approved_overlay(rows, {}, {}, ['train', 'p03510'], 'hash')
            validate.assert_called_once()
        self.assertEqual(rows, original)
        self.assertEqual(overlay['unresolved']['p03510']['native_key'], None)
        self.assertIsNone(mapping_for(overlay, 'p03510', 'def'))
        result = mapping_for(overlay, 'train', 'abc')
        self.assertEqual(result['status'], 'user_approved')
        self.assertEqual(result['quality_flags'], rows[0]['quality_flags'])
        result['quality_flags'].clear()
        self.assertEqual(len(overlay['mappings']['train']['quality_flags']), 2)
        with self.assertRaises(ValueError):
            mapping_for(overlay, 'train', 'changed')
        self.assertFalse(overlay['approval']['independent_adjudication'])

    def test_invalid_evidence_is_not_approved(self):
        with self.assertRaises(ValueError):
            approved_overlay([], {}, {}, ['missing'], 'hash')
