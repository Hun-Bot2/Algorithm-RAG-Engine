import json
import unittest
from pathlib import Path

from taxonomy import load_taxonomy, normalize_tags, validate_taxonomy


class TaxonomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = load_taxonomy()
        cls.concepts = {c["id"] for c in cls.taxonomy["concepts"]}

    def test_schema_and_hierarchy_validate(self):
        self.assertEqual(validate_taxonomy(self.taxonomy), [])

    def test_observed_tag_vocabulary_is_fully_covered(self):
        mappings = {(m["platform"], m["tag"]) for m in self.taxonomy["tag_mappings"]}
        self.assertEqual(len([x for x in mappings if x[0] == "leetcode"]), 72)
        self.assertEqual(len([x for x in mappings if x[0] == "codeforces"]), 38)
        self.assertFalse(any(x[0] == "atcoder" for x in mappings))

    def test_broad_categories_do_not_infer_specific_structures(self):
        result = normalize_tags("codeforces", ["data structures"] , self.taxonomy)
        self.assertEqual([m["concept_id"] for m in result["mapped_concepts"]], ["data-structures"])
        self.assertNotIn("segment-tree", result["ancestors"])

    def test_false_friends_and_bitmask_distinction_are_preserved(self):
        matrices = normalize_tags("codeforces", ["matrices"], self.taxonomy)
        lc_matrix = normalize_tags("leetcode", ["Matrix"], self.taxonomy)
        cf_bits = normalize_tags("codeforces", ["bitmasks"], self.taxonomy)
        lc_bitops = normalize_tags("leetcode", ["Bit Manipulation"], self.taxonomy)
        self.assertNotEqual(matrices["mapped_concepts"][0]["concept_id"], lc_matrix["mapped_concepts"][0]["concept_id"])
        self.assertNotEqual(cf_bits["mapped_concepts"][0]["concept_id"], lc_bitops["mapped_concepts"][0]["concept_id"])

    def test_native_tags_and_ancestor_features_are_separate(self):
        result = normalize_tags("leetcode", ["Segment Tree"], self.taxonomy)
        self.assertEqual(result["original_tags"], ["Segment Tree"])
        self.assertEqual(result["mapped_concepts"][0]["concept_id"], "segment-tree")
        self.assertEqual(result["ancestors"], ["data-structures"])

    def test_related_mapping_does_not_imply_ancestors(self):
        tax = json.loads(json.dumps(self.taxonomy))
        tax["tag_mappings"].append({"platform": "fixture", "tag": "overlap",
                                    "concept_id": "enumeration", "relation": "related",
                                    "confidence": 0.5, "rationale": "Fixture overlap."})
        result = normalize_tags("fixture", ["overlap"], tax)
        self.assertEqual(result["ancestors"], [])

    def test_suffix_structures_do_not_assert_suffix_array(self):
        result = normalize_tags("codeforces", ["string suffix structures"], self.taxonomy)
        self.assertEqual(result["mapped_concepts"][0]["concept_id"], "string-suffix-structures")
        self.assertNotIn("suffix-array", result["ancestors"])

    def test_unavailable_empty_unknown_and_unmapped_states(self):
        unavailable = normalize_tags("atcoder", None, self.taxonomy)
        empty = normalize_tags("codeforces", [], self.taxonomy)
        unknown = normalize_tags("leetcode", ["Future Tag"], self.taxonomy)
        marker = normalize_tags("codeforces", ["*special"], self.taxonomy)
        self.assertEqual(unavailable["tag_status"], "unavailable")
        self.assertIsNone(unavailable["original_tags"])
        self.assertEqual(empty["tag_status"], "empty")
        self.assertEqual(unknown["unknown_tags"], ["Future Tag"])
        self.assertEqual(unknown["tag_status"], "unknown_tags")
        self.assertEqual(marker["mapped_concepts"][0]["relation"], "unmapped")
        self.assertIsNone(marker["mapped_concepts"][0]["concept_id"])

    def test_catalog_tag_document_frequencies_match_frozen_catalog(self):
        root = Path(__file__).resolve().parents[2]
        catalog = root / "private-learning/full-catalog/catalog-v1/catalog.v1.jsonl"
        if not catalog.exists():
            self.skipTest("private full-catalog input unavailable")
        counts = {p: {} for p in ("leetcode", "codeforces", "atcoder")}
        records = {p: 0 for p in counts}
        nulls = {p: 0 for p in counts}
        empties = {p: 0 for p in counts}
        with catalog.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                platform, tags = row["platform"], row.get("tags")
                records[platform] += 1
                if tags is None:
                    nulls[platform] += 1
                elif not tags:
                    empties[platform] += 1
                else:
                    for tag in set(tags):
                        counts[platform][tag] = counts[platform].get(tag, 0) + 1
        for platform, expected in self.taxonomy["platform_coverage"].items():
            self.assertEqual(records[platform], expected["records"])
            self.assertEqual(nulls[platform], expected["null_tag_records"])
            self.assertEqual(empties[platform], expected["empty_list_records"])
            self.assertEqual(counts[platform], expected["tag_document_frequency"])


if __name__ == "__main__":
    unittest.main()
