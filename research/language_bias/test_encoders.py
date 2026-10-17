import unittest

import numpy as np

from research.language_bias import encoders as E


def unit(rows):
    return E.normalize(np.asarray(rows, dtype=np.float64))


class RandomControl(unittest.TestCase):
    def test_deterministic_per_text_unit_norm_and_text_sensitive(self):
        a = E.random_hash_vectors(["Race", "Race", "Game"], dim=16)
        np.testing.assert_array_equal(a[0], a[1])
        self.assertFalse(np.allclose(a[0], a[2]))
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)
        self.assertEqual(a.dtype, np.float32)
        self.assertFalse(np.allclose(a, E.random_hash_vectors(["Race", "Race", "Game"], dim=16, salt="other")))

    def test_normalize_rejects_zero_and_nan(self):
        with self.assertRaises(ValueError):
            E.normalize(np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            E.normalize(np.array([[np.nan, 1.0]]))


class Centroids(unittest.TestCase):
    def test_removes_group_means_and_leaves_small_groups_alone(self):
        rng = np.random.default_rng(0)
        a = unit(rng.normal(size=(40, 6)) + np.array([3, 0, 0, 0, 0, 0]))
        b = unit(rng.normal(size=(40, 6)) + np.array([0, 3, 0, 0, 0, 0]))
        lone = unit([[0, 0, 1, 0, 0, 0]])
        vectors = np.vstack([a, b, lone])
        groups = np.array(["a"] * 40 + ["b"] * 40 + ["tiny"])
        model = E.CentroidRemoval(vectors, groups)
        self.assertEqual(sorted(model.centroids), ["a", "b"])
        out = model.transform(vectors, groups)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
        np.testing.assert_array_equal(out[-1], lone[0])
        centred = out[:40].astype(np.float64)
        before = vectors[:40].astype(np.float64)
        self.assertLess(np.abs(centred.mean(axis=0) - 0).max(), np.abs(before.mean(axis=0)).max())

    def test_query_transform_uses_index_centroids(self):
        rng = np.random.default_rng(1)
        v = unit(rng.normal(size=(60, 4)) + 2)
        g = np.array(["x"] * 30 + ["y"] * 30)
        model = E.CentroidRemoval(v, g)
        np.testing.assert_allclose(model.transform(v[:3], ["x"] * 3), model.transform(v, g)[:3], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
