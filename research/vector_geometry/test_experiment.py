"""Numerical checks for private diagnostics; run with the research environment."""
import unittest
import numpy as np
from research.vector_geometry.experiment import (topk, partial_topk, normalized,
    cosine_neighbors, euclidean_neighbors, overlap, benchmark_search)
import faiss
from threadpoolctl import threadpool_limits


class NumericalChecks(unittest.TestCase):
    def test_exact_search_and_partial_selection(self):
        rng = np.random.default_rng(44)
        values = rng.normal(size=30).astype(np.float32)
        np.testing.assert_array_equal(topk(values), partial_topk(values))
        vectors = rng.normal(size=(32, 16)).astype(np.float32)
        with threadpool_limits(limits=1):
            faiss.omp_set_num_threads(1)
            report = benchmark_search(vectors, np.array([1, 5, 12]))
        self.assertEqual(report['numpy_faiss_identical_order_fraction'], 1)
        self.assertEqual(report['numpy_dot_identical_top10_set_fraction'], 1)
        self.assertLess(report['max_abs_top10_squared_distance_difference'], 1e-4)

    def test_projection_geometry_is_translation_invariant(self):
        values = np.random.default_rng(7).normal(size=(40, 2)).astype(np.float64)
        original = euclidean_neighbors(values)
        shifted = euclidean_neighbors(values + [100, -200])
        self.assertEqual(overlap(original, shifted), 1)
        self.assertTrue(all(i not in row for i, row in enumerate(original)))

    def test_cosine_scaling_and_invalid_input(self):
        values = np.random.default_rng(5).normal(size=(30, 5)).astype(np.float32)
        self.assertEqual(overlap(cosine_neighbors(values), cosine_neighbors(values * 2)), 1)
        with self.assertRaises(ValueError):
            normalized(np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            overlap(np.zeros((3, 2)), np.zeros((2, 2)))


if __name__ == '__main__':
    unittest.main()
