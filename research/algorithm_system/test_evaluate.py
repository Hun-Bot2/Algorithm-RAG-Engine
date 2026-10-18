import json
import tempfile
import unittest
from pathlib import Path

from evaluate import evaluate, run


TAXONOMY = {
    "version": "fixture", "concepts": [{"id": "array", "parents": []}],
    "tag_mappings": [
        {"platform": "codeforces", "tag": "array", "concept_id": "array",
         "relation": "equivalent", "confidence": .95, "rationale": "fixture"},
    ],
}
CORPUS = [
    {"problem_key": "codeforces:1A", "native_key": "codeforces:1A", "platform": "codeforces", "source_tags": ["array"], "native_difficulty": 800},
    {"problem_key": "codeforces:2A", "native_key": "codeforces:2A", "platform": "codeforces", "source_tags": ["array"], "native_difficulty": 1000},
    {"problem_key": "codeforces:3A", "native_key": "codeforces:3A", "platform": "codeforces", "source_tags": [], "native_difficulty": None},
    {"problem_key": "leetcode:x", "native_key": "leetcode:x", "platform": "leetcode", "source_tags": None, "native_difficulty": "Easy"},
]
LABELS = [
    {"query": "codeforces:1A", "target": "codeforces:2A", "label": 3, "label_source": "ai_reviewed", "deceptive": False},
    {"query": "codeforces:1A", "target": "codeforces:3A", "label": 0, "label_source": "ai_reviewed", "deceptive": False},
    {"query": "codeforces:1A", "target": "leetcode:x", "label": None, "label_source": "ai_reviewed", "deceptive": None},
]
RUNS = {
    "semantic": [{"query": "codeforces:1A", "results": [["codeforces:2A", .9], ["leetcode:x", .8], ["codeforces:3A", .7]]}],
    "lexical": [{"query": "codeforces:1A", "results": [["codeforces:3A", .9], ["codeforces:2A", .8], ["leetcode:x", .7]]}],
}


class EvaluateTests(unittest.TestCase):
    def test_judged_denominators_bounds_and_common_set_ndcg(self):
        result = evaluate(CORPUS, RUNS, LABELS, TAXONOMY, resamples=30)
        sem = result["metrics"]["10"]["semantic"]
        lex = result["metrics"]["10"]["lexical"]
        self.assertEqual((sem["retrieved_slots"], sem["judged_slots"], sem["unjudged_slots"]), (3, 2, 1))
        self.assertEqual(sem["observed_precision_judged"], .5)
        self.assertEqual(sem["ai_deceptive_flagged_judged_slots"], 0)
        self.assertEqual(sem["precision_bounds"], [1 / 3, 2 / 3])
        self.assertEqual(sem["ndcg_common_judged_query_count"], 1)
        self.assertGreater(sem["query_bootstrap"]["ndcg_common_judged"]["mean"],
                           lex["query_bootstrap"]["ndcg_common_judged"]["mean"])
        self.assertEqual(sem["tag_consistency"], {"eligible_pairs": 1, "mean_primary_concept_jaccard": 1.0})
        self.assertEqual(sem["native_difficulty_distance"]["codeforces"]["mean_absolute_native_distance"], 200)
        self.assertEqual(sem["cross_platform_difficulty_distance"],
                         "unavailable; native scales are not equated")

    def test_review_queue_includes_only_unjudged_and_assigns_no_label(self):
        result = evaluate(CORPUS, RUNS, LABELS, TAXONOMY, resamples=20)
        queue = result["review_queue"]
        self.assertEqual(queue["candidate_pairs"], 1)
        self.assertEqual(queue["selected_pairs"], 1)
        self.assertEqual(queue["items"][0]["native_target"], "leetcode:x")
        self.assertIsNone(queue["items"][0]["old_label"])
        self.assertIn("method", queue["items"][0]["stratum"])
        self.assertTrue(result["review_queue_sampling"]["sampling_applies_only_to_review_queues"])
        duplicate = result["ambiguous_duplicate_review_queue"]
        self.assertEqual(duplicate["selected_pairs"], 1)
        self.assertEqual(duplicate["items"][0]["target"], "leetcode:x")
        self.assertIn("human_duplicate_review_pending", duplicate["items"][0]["status"])

    def test_review_query_sampling_does_not_change_full_metrics(self):
        sampled = evaluate(CORPUS, RUNS, LABELS, TAXONOMY, resamples=5,
                           review_queries_per_platform=0)
        full = evaluate(CORPUS, RUNS, LABELS, TAXONOMY, resamples=5,
                        review_queries_per_platform=100)
        self.assertEqual(sampled["metrics"], full["metrics"])
        self.assertEqual(sampled["review_queue"]["candidate_pairs"], 0)
        self.assertEqual(full["review_queue"]["candidate_pairs"], 1)

    def test_old_abstention_is_unknown_and_unlisted_pair_is_unknown(self):
        labels = LABELS[:2]
        result = evaluate(CORPUS, RUNS, labels, TAXONOMY, resamples=10)
        self.assertEqual(result["metrics"]["20"]["semantic"]["unjudged_slots"], 1)
        self.assertEqual(result["review_queue"]["candidate_pairs"], 1)

    def test_duplicate_native_target_is_rejected(self):
        corpus = CORPUS + [{**CORPUS[1], "problem_key": "codecontests:duplicate"}]
        bad = {"x": [{"query": "codeforces:1A", "results": [["codeforces:2A", 1], ["codecontests:duplicate", .8]]}]}
        with self.assertRaisesRegex(ValueError, "duplicate_native_target"):
            evaluate(corpus, bad, LABELS, TAXONOMY, resamples=5)

    def test_cli_run_hashes_inputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, labels = root / "corpus.jsonl", root / "labels.jsonl"
            methods = root / "methods"
            methods.mkdir()
            for path, rows in ((corpus, CORPUS), (labels, LABELS)):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            for method, rows in RUNS.items():
                folder = methods / method
                folder.mkdir()
                (folder / "top50.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            output = root / "evaluation.json"
            result = run(corpus, methods, output, labels, resamples=5)
            self.assertEqual(len(result["inputs"]["corpus"]["sha256"]), 64)
            self.assertEqual(set(result["inputs"]["runs"]), set(RUNS))
            with self.assertRaises(FileExistsError):
                run(corpus, methods, output, labels, resamples=5)


if __name__ == "__main__":
    unittest.main()
