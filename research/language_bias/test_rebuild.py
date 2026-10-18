import unittest

import numpy as np

from research.language_bias import rebuild as R


class Tiers(unittest.TestCase):
    def test_provenance_and_unknown_share_decide_the_tier(self):
        self.assertEqual(R.reliability("identity", 0.0), "reliable")
        self.assertEqual(R.reliability("parenthetical", 0.05), "reliable")
        self.assertEqual(R.reliability("identity", 0.10), "degraded")
        self.assertEqual(R.reliability("machine", 0.0), "degraded")
        self.assertEqual(R.reliability("machine", 0.6), "unreliable")
        self.assertEqual(R.reliability("untranslated", 0.0), "unreliable")
        self.assertEqual(R.reliability("machine_empty_fallback", 0.0), "unreliable")

    def test_meta_rows_carry_provenance_and_language(self):
        rows = R.meta_rows(["a", "b"], ["en", "ja_pure"], [{"source": "identity"}, {"source": "machine"}], [0.0, 0.2], [3, 4])
        self.assertEqual([r["reliability"] for r in rows], ["reliable", "degraded"])
        self.assertEqual(rows[1]["embedded_source"], "machine")
        self.assertEqual(rows[1]["language_class"], "ja_pure")
        self.assertTrue(all(r["vector_source"] == "c2_minilm_on_translated_title" for r in rows))


class Island(unittest.TestCase):
    def test_share_of_queries_with_mostly_same_language_neighbors(self):
        groups = np.array(["ja"] * 4 + ["en"] * 4)
        nb = np.array([[1, 2, 3, 4], [0, 2, 3, 5], [0, 1, 3, 4], [0, 1, 2, 5]] + [[0, 1, 2, 3]] * 4)
        self.assertAlmostEqual(R.island_index(nb, groups, np.array([0, 1, 2, 3]), k=3, share=0.9), 1.0)
        self.assertAlmostEqual(R.island_index(nb, groups, np.array([0, 1, 2, 3]), k=4, share=0.9), 0.0)
        self.assertAlmostEqual(R.island_index(nb, groups, np.array([0, 1, 2, 3]), k=4, share=0.7), 1.0)
        self.assertTrue(np.isnan(R.island_index(nb, groups, np.array([], dtype=int))))


if __name__ == "__main__":
    unittest.main()
