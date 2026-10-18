import unittest

import numpy as np
from sklearn.manifold import trustworthiness

from research.full_catalog.retrieve import Rows
from research.unified_map import full_quality as fq


def small_rows(n):
    keys = [f"{('atcoder', 'codeforces', 'leetcode')[i % 3]}:{i:03d}" for i in range(n)]
    keys.sort()
    return Rows(keys, [k.split(":")[0] for k in keys], list(keys), [None] * n)


class NeighborsAndBlocks(unittest.TestCase):
    def test_euclidean_block_ranking_matches_brute_force_and_pads_short_rows(self):
        rng = np.random.default_rng(0)
        coords = rng.normal(size=(40, 2)).astype(np.float32)
        rows = small_rows(40)
        idx, score = fq.neighbor_arrays(rows, fq.euclidean_score_block(coords), 8, "all", block_size=7)
        for q in (0, 13, 39):
            d = ((coords - coords[q]) ** 2).sum(axis=1)
            d[q] = np.inf
            order = np.lexsort((np.arange(40), d))[:8]
            self.assertEqual(idx[q].tolist(), order.tolist())
            np.testing.assert_allclose(-score[q], d[order], rtol=1e-4, atol=1e-5)
        short, short_score = fq.neighbor_arrays(small_rows(5), fq.euclidean_score_block(coords[:5]), 8, "all")
        self.assertTrue((short[:, 4:] == -1).all())
        self.assertTrue(np.isnan(short_score[:, 4:]).all())

    def test_all_mode_excludes_only_self_and_exact_duplicates(self):
        keys = ["atcoder:a", "atcoder:b", "atcoder:old", "codeforces:x", "leetcode:y"]
        rows = Rows(keys, [k.split(":")[0] for k in keys],
                    ["atcoder:a", "atcoder:b", "atcoder:a", "codeforces:x", "leetcode:y"],  # old -> a
                    ["g1", "g1", None, None, None])
        flat = lambda a, b: np.ones((b - a, 5))
        idx, _ = fq.neighbor_arrays(rows, flat, 5, "all")
        names = lambda q: [keys[j] for j in idx[q] if j >= 0]
        self.assertEqual(names(0), ["atcoder:b", "codeforces:x", "leetcode:y"])   # candidate-group peer kept, duplicate dropped
        self.assertEqual(names(2), ["atcoder:b", "codeforces:x", "leetcode:y"])
        self.assertEqual(names(3), ["atcoder:a", "atcoder:b", "atcoder:old", "leetcode:y"])
        cross, _ = fq.neighbor_arrays(rows, flat, 5, "cross")
        self.assertEqual([keys[j] for j in cross[0] if j >= 0], ["codeforces:x", "leetcode:y"])

    def test_non_finite_blocks_are_rejected(self):
        rows = small_rows(4)
        with self.assertRaises(ValueError):
            fq.neighbor_arrays(rows, lambda a, b: np.full((b - a, 4), np.nan), 2, "all")

    def test_preservation_counts_ignore_padding_and_order(self):
        ref = np.array([[1, 2, 3, -1], [4, 5, 6, 7]])
        cand = np.array([[3, 9, 1, -1], [7, 6, 5, 4]])
        self.assertEqual(fq.preservation_counts(ref, cand, k=4).tolist(), [2, 4])
        self.assertEqual(fq.preservation_counts(ref, cand, k=2).tolist(), [0, 0])


class TrustContinuity(unittest.TestCase):
    def test_matches_scikit_learn_on_all_queries(self):
        rng = np.random.default_rng(3)
        hi = rng.normal(size=(70, 6)).astype(np.float64)
        lo = rng.normal(size=(70, 2)).astype(np.float64)
        hi_block = fq.euclidean_score_block(hi.astype(np.float32))
        lo_block = fq.euclidean_score_block(lo.astype(np.float32))
        mine = fq.trust_continuity(hi_block, lo_block, np.arange(70), 70, k=5, block=16)
        with np.errstate(all="ignore"):  # spurious Accelerate FP-status warnings
            expected_trust = trustworthiness(hi, lo, n_neighbors=5)
            expected_cont = trustworthiness(lo, hi, n_neighbors=5)
        self.assertAlmostEqual(mine["trustworthiness"], expected_trust, places=6)
        self.assertAlmostEqual(mine["continuity"], expected_cont, places=6)

    def test_perfect_embedding_scores_one_and_sample_is_reported(self):
        x = np.random.default_rng(1).normal(size=(50, 3)).astype(np.float32)
        block = fq.euclidean_score_block(x)
        result = fq.trust_continuity(block, block, np.array([0, 10, 20]), 50, k=5)
        self.assertEqual((result["trustworthiness"], result["continuity"], result["queries"]), (1.0, 1.0, 3))


class Diagnostics(unittest.TestCase):
    def test_same_platform_fraction(self):
        platforms = np.array([0, 0, 1, 1])
        neighbors = np.array([[1, 2], [0, 2], [3, 0], [2, -1]])
        self.assertAlmostEqual(fq.same_platform_fraction(neighbors, platforms, k=2), 4 / 7)

    def test_tag_agreement_skips_untagged_and_reports_coverage(self):
        concepts = [{"dp"}, {"dp", "greedy"}, None, {"math"}, set()]
        neighbors = np.array([[1, 2, 3], [0, 3, 4], [0, 1, 3], [0, 1, 2], [0, 1, 3]])
        result = fq.tag_agreement(neighbors, concepts, k=3)
        self.assertEqual(result["queries"], 3)
        self.assertEqual(result["tagged_neighbor_slots"], 6)
        self.assertAlmostEqual(result["share_with_concept"], 2 / 6)
        self.assertAlmostEqual(result["tagged_slot_coverage"], 6 / 9)

    def test_random_baseline_is_seeded_and_in_range(self):
        concepts = [{"a"}, {"a"}, {"b"}, None, {"b", "c"}]
        first = fq.random_tag_agreement(concepts, seed=5, draws=5000)
        self.assertEqual(first, fq.random_tag_agreement(concepts, seed=5, draws=5000))
        self.assertTrue(0 < first < 1)

    def test_matched_baseline_uses_neighbor_platform_and_is_seeded(self):
        concepts = [{"a"}, {"a"}, {"b"}, {"a"}, {"b"}, None]
        platforms = np.array([0, 0, 1, 1, 1, 1])
        neighbors = np.array([[2, 3, 5], [3, 4, 2], [0, 1, 3], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
        first = fq.matched_random_tag_agreement(neighbors, concepts, platforms, k=3, seed=1)
        self.assertEqual(first, fq.matched_random_tag_agreement(neighbors, concepts, platforms, k=3, seed=1))
        # 14 tagged slots exist (untagged neighbor 5 is skipped); draws equal to the query are skipped.
        self.assertTrue(10 <= first["slots"] <= 14)
        self.assertTrue(0 <= first["share_with_concept"] <= 1)
        only = fq.matched_random_tag_agreement(np.array([[-1, -1], [-1, -1]]), [{"a"}, {"a"}], np.array([0, 0]), k=2)
        self.assertEqual(only, {"slots": 0, "share_with_concept": None})

    def test_procrustes_is_invariant_to_rotation_and_scale(self):
        pts = np.random.default_rng(2).normal(size=(30, 2))
        theta = 0.7
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.assertAlmostEqual(fq.procrustes_disparity(pts, 3.0 * pts @ rot + 5), 0.0, places=10)
        self.assertGreater(fq.procrustes_disparity(pts, np.random.default_rng(9).normal(size=(30, 2))), 0.1)


if __name__ == "__main__":
    unittest.main()
