import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from research.unified_map.build import _concepts, _rrf, enrich_catalog, jaccard_block, tag_features
from research.full_catalog.retrieve import Rows


class UnifiedMapBuilderTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = {
            "schema": "algorithm-taxonomy.v1", "version": "test",
            "concepts": [{"id": "array", "parents": []}, {"id": "search", "parents": []}],
            "tag_mappings": [
                {"platform": "codeforces", "tag": "arrays", "concept_id": "array",
                 "relation": "equivalent", "confidence": .9, "rationale": "test"},
                {"platform": "leetcode", "tag": "binary-search", "concept_id": "search",
                 "relation": "related", "confidence": 1, "rationale": "test"},
            ],
        }

    def test_only_high_confidence_equivalent_tags_become_features(self):
        rows = [
            {"platform": "codeforces", "tags": ["arrays"]},
            {"platform": "leetcode", "tags": ["binary-search"]},
            {"platform": "atcoder", "tags": None},
            {"platform": "leetcode", "tags": []},
        ]
        features = tag_features(rows, self.taxonomy)
        self.assertEqual(features, [{"array"}, set(), None, set()])
        self.assertEqual(_concepts({"mapped_concepts": [
            {"concept_id": "array", "relation": "equivalent", "confidence": .84},
            {"concept_id": "search", "relation": "equivalent", "confidence": .85},
            {"concept_id": "array", "relation": "narrower", "confidence": 1},
        ]}), {"search"})

    def test_unknown_tags_are_not_imputed_in_jaccard(self):
        score = jaccard_block([{"array"}, None, set()])(0, 3)
        self.assertEqual(score[0, 0], 1)
        self.assertEqual(score[0, 1], 0)
        self.assertEqual(score[0, 2], 0)
        self.assertTrue(np.isfinite(score).all())

    def test_node_records_preserve_all_nodes_and_modality_missingness(self):
        catalog = [{"problem_key": "atcoder:x", "platform": "atcoder", "tags": None,
                    "eligibility": "eligible", "native_difficulty": None}]
        aliases = {"by_native": {}}
        rows = enrich_catalog(catalog, [], aliases, self.taxonomy)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["a1_embedding_available"])
        self.assertIsNone(rows[0]["source_tags"])
        self.assertIsNone(rows[0]["native_difficulty"])
        self.assertEqual(rows[0]["statement_availability"]["status"],
                         "not_available_in_admitted_corpus")
        self.assertEqual(rows[0]["normalized_taxonomy"]["tag_status"], "unavailable")

    def test_rrf_does_not_reward_zero_jaccard_tag_ties(self):
        rows = Rows(["codeforces:a", "codeforces:b", "leetcode:x", "leetcode:y"],
                    ["codeforces", "codeforces", "leetcode", "leetcode"],
                    ["codeforces:a", "codeforces:b", "leetcode:x", "leetcode:y"],
                    [None] * 4)
        semantic_scores = np.array([
            [0, 0, .9, .1], [0, 0, .2, .8],
            [.7, .3, 0, 0], [.4, .6, 0, 0],
        ], dtype=np.float32)
        zero_tags = lambda start, stop: np.zeros((stop - start, 4), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "hybrid"
            receipt = _rrf(rows, lambda start, stop: semantic_scores[start:stop],
                           zero_tags, output, block_size=2)
            self.assertEqual(receipt["fusion"]["zero_jaccard_candidates_receive_no_taxonomy_rank_or_weight"], True)
            components = [json.loads(line) for line in
                          (output / "score-components.jsonl").read_text().splitlines()]
            self.assertTrue(components)
            self.assertTrue(all(row["tag_rank"] is None and row["tag_jaccard"] is None
                                and row["tag_supported"] is False for row in components))

    def test_approved_overlay_quality_flags_are_joined_without_mutating_source(self):
        row = {"problem_key": "atcoder:abc084_c", "platform": "atcoder", "tags": None,
               "eligibility": "eligible"}
        source_record = {"dataset_key": "source:abc084", "content_sha256": "a" * 64,
                         "identity_basis": "unresolved_dataset_identity"}
        aliases = {"by_native": {"atcoder:abc084_c": [source_record]},
                   "mapping_by_native": {"atcoder:abc084_c": [{"quality_flags": ["constraint_disagreement"]}]}}
        result = enrich_catalog([row], [], aliases, self.taxonomy)[0]
        self.assertEqual(result["statement_availability"]["quality_flags"], ["constraint_disagreement"])
        self.assertEqual(source_record["identity_basis"], "unresolved_dataset_identity")


if __name__ == "__main__":
    unittest.main()
