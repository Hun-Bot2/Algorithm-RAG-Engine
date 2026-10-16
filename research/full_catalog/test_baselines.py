"""Fixture-only tests for inputs.py and baselines.py (research .venv; no private data)."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import baselines  # noqa: E402
import inputs  # noqa: E402
import retrieve  # noqa: E402

CROSSWALK = json.loads((HERE / "tag_crosswalk.v1.json").read_text(encoding="utf-8"))


def row(key, title, tags=None, eligible=True, flags=(), group=None):
    return {"problem_key": key, "platform": key.split(":")[0], "title": title, "tags": tags,
            "eligibility": "eligible" if eligible else "excluded", "encoder_flags": list(flags),
            "representative_key": key, "duplicate_candidate_group": group}


CATALOG = [
    row("leetcode:climbing-stairs", "Climbing Stairs", ["Math", "Dynamic Programming", "Memoization"]),
    row("leetcode:two-sum", "Two Sum", ["Array", "Hash Table"]),
    row("leetcode:sql", "Combine Two Tables", ["Database"], eligible=False),
    row("codeforces:1A", "Theatre Square", ["math"]),
    row("codeforces:2B", "Stairs and Steps", ["dp", "*special"]),
    row("codeforces:3C", "Контрольная сумма", ["math"], flags=["non_latin_title"]),
    row("codeforces:4D", "Untagged Sum", []),
    row("atcoder:abc001_a", "Two Sum Again", None),
    row("atcoder:abc002_b", "階段", None, flags=["non_latin_title"]),
]


class Inputs(unittest.TestCase):
    def keys(self, variant):
        rows, _ = inputs.select(CATALOG, variant)
        return [r["problem_key"] for r in rows]

    def test_variant_row_sets(self):
        self.assertEqual(self.keys("A1"), ["atcoder:abc001_a", "codeforces:1A", "codeforces:2B",
                                           "codeforces:4D", "leetcode:climbing-stairs", "leetcode:two-sum"])
        self.assertIn("atcoder:abc002_b", self.keys("A1-all"))
        self.assertIn("codeforces:3C", self.keys("A1-all"))
        self.assertEqual(self.keys("A2"), ["codeforces:1A", "codeforces:2B",
                                           "leetcode:climbing-stairs", "leetcode:two-sum"])

    def test_a2_text_template_drops_special_tag(self):
        rows, texts = inputs.select(CATALOG, "A2")
        self.assertEqual(texts[1], "Stairs and Steps. Topics: dp.")
        self.assertEqual(texts[2], "Climbing Stairs. Topics: Math, Dynamic Programming, Memoization.")

    def test_selection_is_order_independent_and_digest_stable(self):
        a = inputs.select(CATALOG, "A1")
        b = inputs.select(CATALOG[::-1], "A1")
        self.assertEqual(inputs.input_digest(*a), inputs.input_digest(*b))
        with self.assertRaises(ValueError):
            inputs.select(CATALOG, "B")


class Tfidf(unittest.TestCase):
    def test_sparse_block_matches_dense_cosine_and_brute_force_topk(self):
        rows, texts = inputs.select(CATALOG, "A1")
        matrix, info = baselines.tfidf_matrix(texts, "tfidf_word")
        dense = matrix.toarray()
        full = dense @ dense.T
        block = baselines.sparse_cosine(matrix)
        np.testing.assert_allclose(np.vstack([block(0, 2), block(2, len(rows))]), full, atol=1e-6)
        self.assertEqual(info["zero_rows"], 0)
        universe = retrieve.Rows.from_catalog(CATALOG, [r["problem_key"] for r in rows])
        results = dict(retrieve.topk(universe, block, 3, "cross", block_size=2))
        two_sum = [r["problem_key"] for r in rows].index("leetcode:two-sum")
        top = [universe.keys[t] for t, _ in results[two_sum]]
        self.assertEqual(top[0], "atcoder:abc001_a")  # shares "two sum"

    def test_char_ngrams_handle_non_latin_titles(self):
        rows, texts = inputs.select(CATALOG, "A1-all")
        matrix, info = baselines.tfidf_matrix(texts, "tfidf_char")
        self.assertEqual(info["zero_rows"], 0)
        self.assertEqual(matrix.shape[0], len(rows))


class TagJaccard(unittest.TestCase):
    def test_crosswalk_concepts_and_scores(self):
        rows, _ = inputs.select(CATALOG, "A2")
        concepts = baselines.concept_sets(rows, CROSSWALK)
        by_key = dict(zip([r["problem_key"] for r in rows], concepts))
        self.assertEqual(by_key["leetcode:climbing-stairs"], {"math", "dp"})
        self.assertEqual(by_key["leetcode:two-sum"], set())  # Array/Hash Table are unmapped
        self.assertEqual(by_key["codeforces:2B"], {"dp"})
        block, info = baselines.jaccard_block(concepts)
        scores = block(0, len(rows))
        keys = [r["problem_key"] for r in rows]
        stairs, cf2b, cf1a = keys.index("leetcode:climbing-stairs"), keys.index("codeforces:2B"), keys.index("codeforces:1A")
        self.assertAlmostEqual(float(scores[stairs, cf2b]), 0.5)
        self.assertAlmostEqual(float(scores[stairs, cf1a]), 0.5)
        self.assertEqual(float(scores[keys.index("leetcode:two-sum"), cf1a]), 0.0)
        self.assertEqual(info["empty_concept_rows"], 1)

    def test_related_relations_are_not_concepts(self):
        rows = [row("leetcode:x", "X", ["Sliding Window"])]
        self.assertEqual(baselines.concept_sets(rows, CROSSWALK), [set()])

    def test_rejects_atcoder_rows(self):
        with self.assertRaises(ValueError):
            baselines.concept_sets([row("atcoder:a", "A", None)], CROSSWALK)


class RandomBaseline(unittest.TestCase):
    def test_deterministic_and_block_size_independent(self):
        block = baselines.random_block(40)
        whole = block(0, 40)
        np.testing.assert_array_equal(np.vstack([block(0, 7), block(7, 40)]), whole)
        np.testing.assert_array_equal(baselines.random_block(40)(0, 40), whole)
        self.assertFalse(np.array_equal(baselines.random_block(40, seed=1)(0, 40), whole))


class Coverage(unittest.TestCase):
    def test_every_variant_method_is_buildable(self):
        for variant, methods in baselines.METHODS_BY_VARIANT.items():
            rows, texts = inputs.select(CATALOG, variant)
            for method in methods:
                block, _ = baselines.method_block(method, rows, texts, CROSSWALK)
                self.assertEqual(block(0, len(rows)).shape, (len(rows), len(rows)), (variant, method))


if __name__ == "__main__":
    unittest.main()
