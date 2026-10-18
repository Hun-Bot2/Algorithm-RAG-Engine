import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import compress
from research.full_catalog.retrieve import Rows, topk, dense_cosine, write_results


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_run(root: Path):
    run_dir = root / "statement-run-v1"
    (run_dir / "minilm_chunk").mkdir(parents=True)
    corpus = [
        {"problem_key": "codeforces:a", "platform": "codeforces", "content_sha256": "1" * 64},
        {"problem_key": "codeforces:b", "platform": "codeforces", "content_sha256": "2" * 64},
        {"problem_key": "leetcode:a", "platform": "leetcode", "content_sha256": "3" * 64},
        {"problem_key": "leetcode:b", "platform": "leetcode", "content_sha256": "4" * 64},
    ]
    corpus_path = run_dir / "corpus.jsonl"
    corpus_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in corpus), encoding="utf-8")
    vectors = np.zeros((4, 384), dtype=np.float32)
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0
    vectors[2, 0] = 1.0
    vectors[3, 0] = 1.0
    vector_path = run_dir / "minilm_chunk.npy"
    np.save(vector_path, vectors, allow_pickle=False)
    rows = Rows([r["problem_key"] for r in corpus], [r["platform"] for r in corpus],
                [r["content_sha256"] for r in corpus], [None] * len(corpus))
    top_path = run_dir / "minilm_chunk" / "top50.jsonl"
    generated_dir = run_dir / "generated-baseline"
    write_results(rows, topk(rows, dense_cosine(vectors), 50, "cross"), generated_dir)
    top_path.write_bytes((generated_dir / "top50.jsonl").read_bytes())
    manifest = {
        "schema": "statement-run.v1",
        "baseline": {
            "status": "complete", "method": "minilm_chunk", "dimensions": 384,
            "corpus_sha256": digest(corpus_path), "vector_sha256": digest(vector_path),
            "top50_sha256": digest(top_path), "top50_path": "minilm_chunk/top50.jsonl",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


class CompressionTests(unittest.TestCase):
    def test_projection_outputs_are_unit_normalized_and_seeded(self):
        rng = np.random.default_rng(3)
        vectors = rng.normal(size=(8, 384)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        pca, diagnostics = compress.pca_project(vectors, 3)
        rp_a, rp_info_a = compress.gaussian_project(vectors, 3, seed=17)
        rp_b, rp_info_b = compress.gaussian_project(vectors, 3, seed=17)
        self.assertTrue(np.allclose(np.linalg.norm(pca, axis=1), 1.0, atol=1e-6))
        self.assertTrue(np.allclose(np.linalg.norm(rp_a, axis=1), 1.0, atol=1e-6))
        self.assertTrue(np.array_equal(rp_a, rp_b))
        self.assertEqual(diagnostics["rank"], 7)
        self.assertEqual(rp_info_a["transform_bytes"], 384 * 3 * 4)
        self.assertEqual({k: v for k, v in rp_info_a.items() if k != "fit_seconds"},
                         {k: v for k, v in rp_info_b.items() if k != "fit_seconds"})

    def test_uncentered_svd_is_distinct_from_centered_pca(self):
        rng = np.random.default_rng(41)
        vectors = rng.normal(size=(10, 384)).astype(np.float32)
        vectors += np.linspace(0.1, 2.0, 384, dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        projected, info = compress.svd_project(vectors, 4)
        pca, _ = compress.pca_project(vectors, 4)
        self.assertTrue(np.allclose(np.linalg.norm(projected, axis=1), 1.0, atol=1e-6))
        self.assertEqual(info["rank"], 10)
        self.assertFalse(np.allclose(projected @ projected.T, pca @ pca.T, atol=1e-3))
        with self.assertRaisesRegex(ValueError, "insufficient_rows_for_svd"):
            compress.svd_project(np.ones((3, 384), dtype=np.float32), 4)
        with self.assertRaisesRegex(ValueError, "insufficient_rank_for_svd"):
            compress.svd_project(np.ones((5, 384), dtype=np.float32), 2)

    def test_deterministic_npz_persists_exact_transform_arrays_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            components = np.arange(12, dtype=np.float32).reshape(4, 3)
            mean = np.arange(4, dtype=np.float32)
            paths = [root / "first.npz", root / "second.npz"]
            receipts = [compress._deterministic_npz(path, {"components": components, "mean": mean})
                        for path in paths]
            self.assertEqual(receipts[0], receipts[1])
            with np.load(paths[0], allow_pickle=False) as saved:
                self.assertTrue(np.array_equal(saved["components"], components))
                self.assertTrue(np.array_equal(saved["mean"], mean))

    def test_pca_and_random_projection_refuse_insufficient_rows_or_rank(self):
        vectors = np.zeros((5, 384), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "insufficient_rows_for_pca"):
            compress.pca_project(vectors, 5)
        with self.assertRaisesRegex(ValueError, "pca_input_has_zero_rank"):
            compress.pca_project(vectors, 2)
        with self.assertRaisesRegex(ValueError, "insufficient_rows_for_random_projection"):
            compress.gaussian_project(np.ones((4, 384), dtype=np.float32), 4)

    def test_topk_ties_are_stable_and_overlap_is_per_query_fraction(self):
        keys = ["cf:a", "cf:b", "lc:a", "lc:b"]
        rows = Rows(keys, ["cf", "cf", "lc", "lc"], keys, [None] * 4)
        vectors = np.ones((4, 3), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        result = list(topk(rows, dense_cosine(vectors), 50, "cross"))
        self.assertEqual([rows.keys[i] for i, _ in result[0][1]], ["lc:a", "lc:b"])
        baseline = {rows.keys[query]: [rows.keys[i] for i, _ in ranked] for query, ranked in result}
        overlaps = {k: [] for k in compress.KS}
        wrapped = list(compress._overlap_stream(rows, result, baseline, overlaps))
        self.assertEqual(len(wrapped), len(keys))
        self.assertEqual({k: set(values) for k, values in overlaps.items()}, {10: {1.0}, 20: {1.0}, 50: {1.0}})

    def test_manifest_hashes_gate_full_384d_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = fixture_run(Path(temp))
            self.assertEqual(compress._validate_manifest(run_dir)[1:], (
                run_dir / "corpus.jsonl", run_dir / "minilm_chunk.npy", run_dir / "minilm_chunk/top50.jsonl"))
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["baseline"]["top50_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "baseline_manifest_hash_mismatch:top50_sha256"):
                compress._validate_manifest(run_dir)

    def test_synthetic_run_skips_unestimable_dims_and_keeps_original_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = fixture_run(root)
            output = root / "private-learning" / "algorithm-system" / "compression-v1"
            with patch.object(compress, "ROOT", root):
                receipt = compress.run(run_dir, output, block_size=2)
            self.assertEqual(receipt["rows"], 4)
            self.assertEqual(receipt["methods"]["original384"]["neighbor_overlap"]["top50"]["mean_fraction"], 1.0)
            self.assertEqual(receipt["methods"]["pca256"]["status"], "not_estimable")
            self.assertEqual(receipt["methods"]["rp256_seed17"]["status"], "not_estimable")
            self.assertEqual(receipt["methods"]["svd256"]["status"], "not_estimable")
            self.assertEqual(receipt["svd_fit"]["centering"], "none")
            self.assertTrue(receipt["no_model_promotion"])
            self.assertTrue((output / "receipt.json").is_file())


if __name__ == "__main__":
    unittest.main()
