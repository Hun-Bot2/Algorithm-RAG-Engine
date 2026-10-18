import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from research.full_catalog.retrieve import Rows
from research.unified_map.reduction import (
    compression_models,
    exact_neighbors,
    sampled_2d_neighbors,
    _verify_embedding_source,
    stratified_sample,
    top10_preservation,
    sha256_file,
)
from research.full_catalog.inputs import input_digest, select


class ReductionFixtureTests(unittest.TestCase):
    def test_stratified_sample_is_reproducible_sorted_and_capped(self):
        keys = [f"k{i:02}" for i in range(10)]
        platforms = ["atcoder"] * 5 + ["codeforces"] * 3 + ["leetcode"] * 2
        first = stratified_sample(keys, platforms, cap=7, seed=17)
        second = stratified_sample(keys, platforms, cap=7, seed=17)
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(first), 7)
        self.assertEqual(first.tolist(), sorted(first.tolist()))
        observed = [platforms[i] for i in first]
        self.assertEqual({p: observed.count(p) for p in set(platforms)},
                         {"atcoder": 3, "codeforces": 2, "leetcode": 2})

    def test_exact_cross_platform_neighbors_obey_candidate_exclusion(self):
        keys = ["a:1", "a:2", "b:1", "b:2", "c:1", "c:2"]
        platforms = ["atcoder", "atcoder", "codeforces", "codeforces", "leetcode", "leetcode"]
        representatives = keys[:]
        candidates = ["dup", "dup", "dup", None, None, None]
        rows = Rows(keys, platforms, representatives, candidates)
        vectors = np.asarray([
            [1, 0], [0.99, 0.01], [1, 0], [0, 1], [0.9, 0.1], [0.1, 0.9],
        ], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        neighbors = exact_neighbors(rows, vectors, block_size=2)
        self.assertEqual(neighbors[0].tolist(), [4, 5, 3])
        self.assertEqual(neighbors[1].tolist(), [4, 5, 3])
        self.assertTrue(all(platforms[index] != platforms[query]
                            for query, values in enumerate(neighbors) for index in values))

    def test_top10_overlap_and_2d_ranking(self):
        expected = [np.array([2, 3]), np.array([0, 3])]
        self.assertEqual(top10_preservation(expected, expected)["mean_top10_overlap"], 1.0)
        coords = np.asarray([[0, 0], [5, 5], [0.1, 0], [5.1, 5]], dtype=np.float32)
        keys = ["a:1", "a:2", "b:1", "b:2"]
        rows = Rows(keys, ["atcoder", "atcoder", "codeforces", "codeforces"], keys,
                    [None, None, None, None])
        neighbors = sampled_2d_neighbors(coords, rows)
        self.assertEqual(neighbors[0].tolist(), [2, 3])
        self.assertEqual(neighbors[1].tolist(), [3, 2])

    def test_compression_models_return_normalized_float32_fixture(self):
        rng = np.random.default_rng(3)
        vectors = rng.normal(size=(24, 8)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        outputs, fit = compression_models(vectors, seed=17, dimensions=(4, 2))
        self.assertEqual(set(outputs), {"pca4", "pca2", "svd4", "svd2"})
        for name, projected in outputs.items():
            self.assertEqual(projected.dtype, np.float32)
            self.assertEqual(projected.shape, (24, int(name.removeprefix("pca").removeprefix("svd"))))
            np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 1.0, atol=1e-5)
        self.assertTrue(all(record["seconds"] >= 0 for record in fit.values()))

    def test_embedding_source_guard_checks_artifact_hashes_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path, vectors_path, keys_path, receipt_path = (
                root / name for name in ("catalog.jsonl", "vectors.npy", "keys.json", "receipt.json"))
            catalog = [{"problem_key": "atcoder:a", "platform": "atcoder", "eligibility": "eligible",
                        "encoder_flags": [], "title": "A", "tags": []}]
            catalog_path.write_text("\n".join(json.dumps(row) for row in catalog) + "\n")
            keys = ["atcoder:a"]
            keys_path.write_text(json.dumps(keys) + "\n")
            vectors = np.ones((1, 3), dtype=np.float32)
            with vectors_path.open("wb") as handle:
                np.save(handle, vectors, allow_pickle=False)
            chosen, texts = select(catalog, "A1")
            receipt = {"schema": "embedding-receipt.v1", "variant": "A1", "rows": 1,
                       "model": {"dimensions": 3},
                       "inputs": {"catalog": sha256_file(catalog_path),
                                  "variant_input": input_digest(chosen, texts)},
                       "outputs": {"keys.json": sha256_file(keys_path),
                                   "vectors.npy": sha256_file(vectors_path)}}
            receipt_path.write_text(json.dumps(receipt))
            verified = _verify_embedding_source(receipt_path, catalog_path, vectors_path, keys_path,
                                                catalog, keys, vectors)
            self.assertEqual(verified["variant"], "A1")
            receipt["outputs"]["vectors.npy"] = "0" * 64
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaises(SystemExit):
                _verify_embedding_source(receipt_path, catalog_path, vectors_path, keys_path,
                                         catalog, keys, vectors)


if __name__ == "__main__":
    unittest.main()
