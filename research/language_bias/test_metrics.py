import unittest

import numpy as np

from research.language_bias import metrics as M


def unit(rows):
    a = np.asarray(rows, dtype=np.float64)
    return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)


class Bootstrap(unittest.TestCase):
    def test_deterministic_and_brackets_mean(self):
        v = np.random.default_rng(0).uniform(size=200)
        a, b = M.bootstrap(v), M.bootstrap(v)
        self.assertEqual(a, b)
        self.assertTrue(a["lo"] <= a["mean"] <= a["hi"])
        self.assertEqual(M.bootstrap([])["mean"], None)

    def test_ratio(self):
        r = M.bootstrap_ratio(np.array([1.0, 3.0]), np.array([2.0, 2.0]))
        self.assertAlmostEqual(r["value"], 1.0)
        self.assertEqual(M.bootstrap_ratio(np.array([1.0]), np.array([0.0]))["value"], None)


class Lift(unittest.TestCase):
    def test_identical_terms_give_one_and_signal_gives_more(self):
        s = np.array([2.0, 1.0, 3.0, 2.0] * 30)
        t = np.array([4.0, 4.0, 4.0, 4.0] * 30)
        flat = M.bootstrap_lift(s, t, s, t)
        self.assertAlmostEqual(flat["value"], 1.0)
        self.assertTrue(flat["lo"] <= 1.0 <= flat["hi"])
        strong = M.bootstrap_lift(s * 2, t, s, t)
        self.assertAlmostEqual(strong["value"], 2.0)
        self.assertGreater(strong["lo"], 1.5)
        self.assertEqual(M.bootstrap_lift([], [], [], [])["value"], None)
        self.assertEqual(M.bootstrap_lift([1.0], [0.0], [1.0], [1.0])["value"], None)
        self.assertEqual(M.bootstrap_lift(s, t, s, t), M.bootstrap_lift(s, t, s, t))


class Homophily(unittest.TestCase):
    def test_observed_and_chance(self):
        groups = np.array(["ja", "ja", "en", "en", "en", "en"])
        nb = np.array([[1, 2], [0, 3], [3, 4], [2, 0], [2, 3], [2, 3]])
        h = M.homophily(nb, groups, np.array([0, 1]), k=2)
        self.assertAlmostEqual(h["observed"], 0.5)            # (1/2 + 1/2) / 2
        self.assertAlmostEqual(h["chance"], 1 / 5)            # (2-1)/(6-1)
        self.assertEqual(h["queries"], 2)
        h2 = M.homophily(nb, groups, np.array([2]), k=2)
        self.assertAlmostEqual(h2["chance"], 3 / 5)

    def test_platform_composition(self):
        comp = M.platform_composition(np.array([[1, 2], [0, 2], [0, 1]]), np.array([0, 1, 1]), np.array([0, 1, 2]), k=2)
        self.assertAlmostEqual(sum(comp.values()), 1.0)
        self.assertAlmostEqual(comp["1"], 4 / 6)


class Translation(unittest.TestCase):
    def test_perfect_chance_and_ties(self):
        c = unit(np.eye(6) + 0.01)
        r = M.translation_retrieval(c[:3], c, np.arange(3))
        self.assertEqual(r["recall@1"]["mean"], 1.0)
        self.assertEqual(r["mrr"]["mean"], 1.0)
        self.assertAlmostEqual(r["chance_mrr"], np.mean(1 / np.arange(1, 7)))
        worst = M.translation_retrieval(unit([[1, 0, 0]]), unit([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), np.array([0]))
        self.assertEqual(worst["median_rank"], 2.0)          # one candidate scores strictly higher
        self.assertEqual(worst["recall@1"]["mean"], 0.0)
        tie = M.translation_retrieval(unit([[1, 0]]), unit([[1, 0], [1, 0], [0, 1]]), np.array([0]))
        self.assertEqual(tie["recall@1"]["mean"], 1.0)       # equal-vector duplicates do not outrank the truth


class Consistency(unittest.TestCase):
    def test_overlap_excludes_own_nodes_and_matches_known_case(self):
        index = unit([[1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1], [0.1, 0.9]])
        a = unit([[1, 0]])
        out = M.neighbor_consistency(index, a, a, [[0]], k=2)
        self.assertEqual(out["overlap"]["mean"], 1.0)
        b = unit([[0, 1]])
        out = M.neighbor_consistency(index, a, b, [[0]], k=2)
        self.assertEqual(out["overlap"]["mean"], 0.0)
        self.assertAlmostEqual(out["chance"], 2 / 4)
        nb = M.top_neighbors(index, a, [[0]], k=2)
        self.assertNotIn(0, nb[0].tolist())

    def test_paired_difference(self):
        d = M.paired_difference([0.5, 0.6, 0.7], [0.1, 0.1, 0.1])
        self.assertGreater(d["lo"], 0.3)


class Coherence(unittest.TestCase):
    def setUp(self):
        self.concepts = [None, {"dp"}, {"dp"}, {"graph"}, {"graph"}, {"dp"}, {"graph"}, {"math"}]
        self.platforms = np.array([0, 1, 1, 1, 1, 2, 2, 2])

    def test_pair_counts_and_lift_direction(self):
        coherent = np.array([[1, 2, 5, 0], [3, 4, 6, 0]])
        s, t, tagged, slots = M.shared_pairs(np.vstack([coherent, np.zeros((6, 4), dtype=int)]), self.concepts, [0, 1], k=3)
        self.assertEqual(t.tolist(), [3.0, 3.0])
        self.assertEqual(s.tolist(), [3.0, 3.0])
        self.assertEqual((tagged, slots), (6, 6))
        neighbors = np.zeros((8, 3), dtype=int)
        neighbors[0] = [1, 2, 5]
        neighbors[7] = [1, 3, 5]
        good = M.concept_coherence(neighbors, self.concepts, self.platforms, [0], k=3)
        bad = M.concept_coherence(neighbors, self.concepts, self.platforms, [7], k=3)
        self.assertGreater(good["lift"], bad["lift"])
        self.assertEqual(good["tagged_slot_coverage"], 1.0)

    def test_unassessable_when_no_tagged_neighbors(self):
        out = M.concept_coherence(np.zeros((8, 3), dtype=int), self.concepts, self.platforms, [3], k=3)
        self.assertEqual(out["queries"], 1)
        self.assertIsNotNone(out["tagged_slot_coverage"])


class Centroids(unittest.TestCase):
    def test_removes_group_means_and_renormalizes(self):
        v = unit([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
        g = np.array(["a", "a", "b", "b"])
        out = M.remove_group_centroids(v, g)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
        for grp in ("a", "b"):
            np.testing.assert_allclose(out[g == grp].mean(axis=0), 0.0, atol=0.2)
        self.assertEqual(out.dtype, np.float32)


class Selection(unittest.TestCase):
    def table(self, **over):
        base = {"english_retention": 1.0, "mrr": 0.02, "consistency": 0.10, "coherence_lift": 1.1, "coherence_coverage": 0.3}
        t = {"C0": dict(base), "random": dict(base, mrr=0.001, consistency=0.0, english_retention=0.5, coherence_lift=1.0),
             "C3": dict(base, mrr=0.6, consistency=0.45, consistency_diff_vs_baseline_lo=0.2, consistency_halfwidth=0.03),
             "C4": dict(base, mrr=0.5, consistency=0.40, consistency_diff_vs_baseline_lo=0.15, consistency_halfwidth=0.03)}
        for k, v in over.items():
            t[k] = dict(t[k], **v)
        return t

    def decide(self, t):
        return M.apply_selection_rule(t, baseline="C0", controls=("random",), sizes={"C3": 2, "C4": 1})

    def test_clear_winner_replaces_baseline(self):
        d = self.decide(self.table())
        self.assertEqual((d["winner"], d["replaced"]), ("C3", True))

    def test_guard_failures_remove_candidates_and_keep_baseline(self):
        d = self.decide(self.table(C3={"english_retention": 0.9}, C4={"mrr": 0.2}))
        self.assertIsNone(d["winner"])
        self.assertFalse(d["replaced"])
        self.assertIn("C3", d["ineligible_reasons"])
        self.assertIn("C4", d["ineligible_reasons"])

    def test_small_or_insignificant_gain_keeps_baseline(self):
        d = self.decide(self.table(C3={"consistency": 0.15, "consistency_diff_vs_baseline_lo": 0.01},
                                C4={"consistency": 0.12, "consistency_diff_vs_baseline_lo": 0.0}))
        self.assertFalse(d["replaced"])
        d = self.decide(self.table(C3={"consistency": 0.40, "consistency_diff_vs_baseline_lo": -0.01}))
        self.assertFalse(d["replaced"])

    def test_ties_go_to_higher_mrr_then_smaller_model(self):
        d = self.decide(self.table(C4={"consistency": 0.44, "mrr": 0.7}))
        self.assertEqual(d["winner"], "C4")
        d = self.decide(self.table(C4={"consistency": 0.44, "mrr": 0.6}))
        self.assertEqual(d["winner"], "C4")      # equal MRR: smaller model

    def test_coherence_below_chance_fails_g3_only_when_assessable(self):
        d = self.decide(self.table(C3={"coherence_lift": 0.8}))
        self.assertIn("C3", d["ineligible_reasons"])
        d = self.decide(self.table(C3={"coherence_lift": 0.8, "coherence_coverage": 0.05}))
        self.assertNotIn("C3", d["ineligible_reasons"])

    def test_homophily_and_mixing_are_never_inputs(self):
        t = self.table()
        t["C3"]["homophily"] = 99
        t["C4"]["platform_mixing"] = 1.0
        self.assertEqual(self.decide(t)["winner"], "C3")


if __name__ == "__main__":
    unittest.main()
