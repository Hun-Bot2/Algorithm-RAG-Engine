import unittest

from difficulty import (
    availability_receipt,
    midrank_empirical_cdf,
    parse_models,
    prepare_native_rows,
    quantile_baseline,
)


class DifficultyTests(unittest.TestCase):
    def test_midrank_ties_return_closed_ecdf_interval(self):
        self.assertEqual(midrank_empirical_cdf([1, 1, 2, 4]), {
            1: (0.0, 0.5, 0.25),
            2: (0.5, 0.75, 0.625),
            4: (0.75, 1.0, 0.875),
        })

    def test_missing_and_constant_values_are_well_defined(self):
        self.assertEqual(midrank_empirical_cdf([None, None]), {})
        self.assertEqual(midrank_empirical_cdf([5, None, 5]), {5: (0.0, 1.0, 0.5)})
        rows = quantile_baseline([
            {"problem_key": "atcoder:a", "native_difficulty": None},
            {"problem_key": "atcoder:b", "native_difficulty": 5},
            {"problem_key": "atcoder:c", "native_difficulty": 5},
        ])
        self.assertIsNone(rows[0]["empirical_cdf_interval"])
        self.assertEqual(rows[1]["empirical_cdf_interval"], [0.0, 1.0, 0.5])

    def test_native_join_preserves_ids_negative_values_and_nulls(self):
        catalog = [
            {"problem_key": "atcoder:abc001_a", "platform": "atcoder"},
            {"problem_key": "atcoder:abc001_b", "platform": "atcoder"},
            {"problem_key": "codeforces:1A", "platform": "codeforces"},
        ]
        rows = prepare_native_rows(catalog, {"abc001_a": -12.5})
        self.assertEqual(rows[0]["task_id"], "abc001_a")
        self.assertEqual(rows[0]["native_difficulty"], -12.5)
        self.assertEqual(rows[1]["native_difficulty"], None)
        self.assertEqual([row["problem_key"] for row in rows], ["atcoder:abc001_a", "atcoder:abc001_b"])

    def test_parser_rejects_guessed_or_malformed_values(self):
        parsed = parse_models(b'{"task": {"difficulty": -3, "other": true}}')
        self.assertEqual(parsed, {"task": -3})
        with self.assertRaises(ValueError):
            parse_models(b'{"task": {"difficulty": "hard"}}')
        with self.assertRaises(ValueError):
            parse_models(b'[]')

    def test_receipt_counts_partial_and_unmatched_coverage(self):
        catalog_rows = prepare_native_rows([
            {"problem_key": "atcoder:a", "platform": "atcoder"},
            {"problem_key": "atcoder:b", "platform": "atcoder"},
        ], {"a": 0})
        receipt = availability_receipt({"a": 0, "stale": 4}, catalog_rows, "abc", "now", 200, 3)
        self.assertEqual(receipt["matched_catalog_records"], 1)
        self.assertEqual(receipt["matched_with_null_difficulty"], 0)
        self.assertEqual(receipt["catalog_records_without_dataset_match"], 1)
        self.assertEqual(receipt["dataset_records_without_catalog_match"], 1)
        self.assertEqual(receipt["cross_platform_equivalence"], "not_established")
        self.assertEqual(receipt["codeforces_native_rating_nulls"], 3)


if __name__ == "__main__":
    unittest.main()
