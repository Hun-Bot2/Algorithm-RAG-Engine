import unittest
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

from calibration import (
    bootstrap_landmarks,
    cdf_summary,
    make_split_plan,
    topic_rank_diagnostics,
    write_cdf_figure,
)


class CalibrationTests(unittest.TestCase):
    def test_cdf_preserves_numeric_ties_and_missingness(self):
        cdf = cdf_summary([1, 1, 2, None])
        self.assertAlmostEqual(cdf["1"]["lower"], 0.0)
        self.assertAlmostEqual(cdf["1"]["upper"], 2 / 3)
        self.assertAlmostEqual(cdf["1"]["mid"], 1 / 3)
        self.assertAlmostEqual(cdf["2"]["lower"], 2 / 3)
        self.assertAlmostEqual(cdf["2"]["upper"], 1.0)
        self.assertAlmostEqual(cdf["2"]["mid"], 5 / 6)
        self.assertNotIn("None", cdf)

    def test_leetcode_category_order_is_easy_medium_hard_with_ties(self):
        cdf = cdf_summary(["Hard", "Easy", "Medium", "Medium"])
        self.assertEqual(list(cdf), ["Easy", "Medium", "Hard"])
        self.assertEqual(cdf["Easy"], {"lower": 0.0, "upper": 0.25, "mid": 0.125})
        self.assertEqual(cdf["Medium"], {"lower": 0.25, "upper": 0.75, "mid": 0.5})
        self.assertEqual(cdf["Hard"], {"lower": 0.75, "upper": 1.0, "mid": 0.875})

    def test_bootstrap_is_bounded_and_deterministic(self):
        first = bootstrap_landmarks([None, 2, 2, 8, 9], seed=17, resamples=30)
        second = bootstrap_landmarks([None, 2, 2, 8, 9], seed=17, resamples=30)
        self.assertEqual(first, second)
        self.assertEqual(first["p50"]["estimate"], 2)
        self.assertEqual(first["p50"]["resamples"], 30)
        self.assertLessEqual(first["p10"]["bootstrap_95_percentile_interval"][0],
                             first["p10"]["bootstrap_95_percentile_interval"][1])

    def test_topic_primary_and_ancestor_diagnostics_are_separate(self):
        taxonomy = {
            "version": "fixture",
            "concepts": [
                {"id": "array", "parents": ["data-structures"]},
                {"id": "data-structures", "parents": []},
            ],
            "tag_mappings": [{"platform": "leetcode", "tag": "Array", "concept_id": "array",
                              "relation": "equivalent", "confidence": 0.97,
                              "rationale": "fixture"}],
        }
        rows = [{"platform": "leetcode", "difficulty": value,
                 "catalog_row": {"platform": "leetcode", "tags": ["Array"]}}
                for value in range(20)]
        result = topic_rank_diagnostics(rows, taxonomy, min_n=20)["leetcode"]
        self.assertEqual(result["primary_concept"]["array"]["n"], 20)
        self.assertEqual(result["ancestor"]["data-structures"]["n"], 20)
        self.assertNotIn("data-structures", result["primary_concept"])

    def test_small_topic_cohort_is_marked_unavailable(self):
        taxonomy = {"version": "fixture", "concepts": [{"id": "array", "parents": []}],
                    "tag_mappings": [{"platform": "leetcode", "tag": "Array", "concept_id": "array",
                                      "relation": "equivalent", "confidence": 0.97, "rationale": "fixture"}]}
        rows = [{"platform": "leetcode", "difficulty": i,
                 "catalog_row": {"platform": "leetcode", "tags": ["Array"]}} for i in range(19)]
        self.assertEqual(topic_rank_diagnostics(rows, taxonomy, min_n=20)["leetcode"]
                         ["primary_concept"]["array"],
                         {"n": 19, "status": "below_planning_minimum"})

    def test_split_is_disjoint_groups_evaluation_only_and_complete(self):
        rows = [
            {"problem_key": "codeforces:1A", "platform": "codeforces", "representative_key": "codeforces:1A"},
            {"problem_key": "leetcode:one", "platform": "leetcode", "representative_key": "codeforces:1A"},
            {"problem_key": "atcoder:abc_a", "platform": "atcoder", "representative_key": "atcoder:abc_a"},
            {"problem_key": "atcoder:abc_b", "platform": "atcoder", "representative_key": "atcoder:abc_b"},
        ]
        plan = make_split_plan(rows, {"leetcode:one"}, seed=17)
        parts = plan["partitions"]
        self.assertEqual(parts["legacy_eval_only"]["records"], 2)
        self.assertEqual(plan["catalog_key_coverage"], 4)
        self.assertTrue(plan["all_catalog_keys_partitioned_once"])
        self.assertEqual(plan["disjoint_record_overlap_count"], 0)
        self.assertEqual(plan["fit_status"], "no supervised fit performed")

    def test_figure_is_valid_svg_with_three_separate_native_panels(self):
        result = {"platforms": {
            "leetcode": {"native_value_cdf_midrank_intervals": {
                "Easy": {"lower": 0, "upper": 0.5, "mid": 0.25},
                "Hard": {"lower": 0.5, "upper": 1, "mid": 0.75}},
                "observed_native_values": 2, "missing_native_values": 0},
            "codeforces": {"native_value_cdf_midrank_intervals": {
                "800": {"lower": 0, "upper": 0.5, "mid": 0.25},
                "1200": {"lower": 0.5, "upper": 1, "mid": 0.75}},
                "observed_native_values": 2, "missing_native_values": 1},
            "atcoder": {"native_value_cdf_midrank_intervals": {
                "-100": {"lower": 0, "upper": 0.5, "mid": 0.25},
                "200": {"lower": 0.5, "upper": 1, "mid": 0.75}},
                "observed_native_values": 2, "missing_native_values": 2},
        }}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figure.svg"
            write_cdf_figure(result, path)
            root = ET.parse(path).getroot()
            text = " ".join(element.text or "" for element in root.iter())
        self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
        self.assertIn("separate scales", text)
        self.assertIn("LeetCode categories", text)
        self.assertIn("Codeforces ratings", text)
        self.assertIn("AtCoder Problems estimates", text)
        self.assertIn("Easy", text)


if __name__ == "__main__":
    unittest.main()
