"""Fixture-only tests for retrieve.py (requires numpy; run with the research .venv)."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("full_catalog_retrieve", HERE / "retrieve.py")
retrieve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retrieve)

PLATFORMS = ("atcoder", "codeforces", "leetcode")


def make_rows(n, seed=0, with_groups=True):
    rng = np.random.default_rng(seed)
    keys = [f"{PLATFORMS[i % 3]}:{i:05d}" for i in range(n)]
    order = sorted(range(n), key=lambda i: keys[i])
    keys = [keys[i] for i in order]
    platforms = [k.split(":")[0] for k in keys]
    reps = list(keys)
    groups = [None] * n
    if with_groups:
        for i in rng.choice(n, size=n // 10, replace=False):
            groups[i] = f"g{int(i) % 5}-{platforms[i]}"
        reps[1] = reps[0] if platforms[0] == platforms[1] else reps[1]
    return retrieve.Rows(keys, platforms, reps, groups), rng


def brute_force(rows, matrix, k, mode, exclude_candidates):
    out = []
    for q in range(len(rows)):
        allowed = []
        for t in range(len(rows)):
            same = rows.platform[q] == rows.platform[t]
            if (mode == "cross" and same) or (mode == "same" and not same):
                continue
            if rows.representative[q] == rows.representative[t]:
                continue
            if exclude_candidates and rows.candidate[q] >= 0 and rows.candidate[q] == rows.candidate[t]:
                continue
            allowed.append((-matrix[q, t], rows.keys[t], t))
        out.append([t for _, _, t in sorted(allowed)[:k]])
    return out


class BruteForceEquality(unittest.TestCase):
    def test_ranking_matches_brute_force_for_all_settings(self):
        rows, rng = make_rows(120)
        matrix = np.round(rng.normal(size=(120, 120)), 2)  # rounding creates real ties
        matrix = (matrix + matrix.T) / 2
        for mode in ("cross", "same"):
            for exclude in (True, False):
                expected = brute_force(rows, matrix, 15, mode, exclude)
                for block in (1, 7, 120, 500):
                    got = [[t for t, _ in r] for _, r in retrieve.topk(
                        rows, lambda a, b: matrix[a:b], 15, mode, block_size=block, exclude_candidates=exclude)]
                    self.assertEqual(got, expected, (mode, exclude, block))

    def test_dense_blockwise_matches_full_matrix(self):
        rows, rng = make_rows(300, seed=3, with_groups=False)
        vectors = rng.normal(size=(300, 32)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        with np.errstate(all="ignore"):
            full = vectors @ vectors.T
        expected = brute_force(rows, full, 20, "cross", True)
        results = list(retrieve.topk(rows, retrieve.dense_cosine(vectors), 20, "cross", block_size=64))
        self.assertEqual([[t for t, _ in r] for _, r in results], expected)
        for q, ranked in results:
            np.testing.assert_allclose([s for _, s in ranked], full[q, [t for t, _ in ranked]], atol=1e-6)

    def test_rejects_unnormalized_vectors(self):
        with self.assertRaises(ValueError):
            retrieve.dense_cosine(np.ones((3, 4), dtype=np.float32))


class TieOrder(unittest.TestCase):
    def test_equal_scores_order_by_key_including_boundary_ties(self):
        keys = ["atcoder:q", "leetcode:a", "leetcode:b", "leetcode:c", "leetcode:d"]
        rows = retrieve.Rows(keys, [k.split(":")[0] for k in keys], keys, [None] * 5)
        scores = np.array([[1.0, 0.5, 0.9, 0.5, 0.5]] * 5)
        _, ranked = next(iter(retrieve.topk(rows, lambda a, b: scores[a:b], 2, "cross")))
        self.assertEqual([keys[t] for t, _ in ranked], ["leetcode:b", "leetcode:a"])
        _, ranked = next(iter(retrieve.topk(rows, lambda a, b: scores[a:b], 4, "cross")))
        self.assertEqual([keys[t] for t, _ in ranked], ["leetcode:b", "leetcode:a", "leetcode:c", "leetcode:d"])


class Exclusions(unittest.TestCase):
    def setUp(self):
        self.keys = ["atcoder:a", "atcoder:b", "atcoder:old", "codeforces:x", "leetcode:y"]
        platforms = [k.split(":")[0] for k in self.keys]
        reps = ["atcoder:a", "atcoder:b", "atcoder:a", "codeforces:x", "leetcode:y"]  # old -> a
        groups = ["g1", "g1", None, None, None]
        self.rows = retrieve.Rows(self.keys, platforms, reps, groups)
        self.scores = np.ones((5, 5))

    def targets(self, query, mode, exclude=True):
        results = dict(retrieve.topk(self.rows, lambda a, b: self.scores[a:b], 10, mode,
                                     exclude_candidates=exclude))
        return [self.keys[t] for t, _ in results[self.keys.index(query)]]

    def test_cross_mode_excludes_self_and_same_platform(self):
        self.assertEqual(self.targets("atcoder:a", "cross"), ["codeforces:x", "leetcode:y"])

    def test_same_mode_excludes_self_exact_duplicates_and_candidates(self):
        self.assertEqual(self.targets("atcoder:a", "same"), [])
        self.assertEqual(self.targets("atcoder:a", "same", exclude=False), ["atcoder:b"])
        self.assertEqual(self.targets("atcoder:old", "same"), ["atcoder:b"])

    def test_rows_require_sorted_unique_keys(self):
        with self.assertRaises(ValueError):
            retrieve.Rows(["b", "a"], ["x", "x"], ["b", "a"], [None, None])
        with self.assertRaises(ValueError):
            retrieve.Rows(["a", "a"], ["x", "x"], ["a", "a"], [None, None])

    def test_bad_block_shape_or_non_finite_scores_fail(self):
        with self.assertRaises(ValueError):
            list(retrieve.topk(self.rows, lambda a, b: np.ones((1, 2)), 2, "cross"))
        with self.assertRaises(ValueError):
            list(retrieve.topk(self.rows, lambda a, b: np.full((b - a, 5), np.nan), 2, "cross"))


class Output(unittest.TestCase):
    def test_files_are_prefixes_deterministic_and_never_overwritten(self):
        rows, rng = make_rows(60, seed=5)
        matrix = rng.normal(size=(60, 60))
        with tempfile.TemporaryDirectory() as tmp:
            hashes = []
            for name in ("a", "b"):
                results = retrieve.topk(rows, lambda a, b: matrix[a:b], 50, "cross", block_size=16)
                digest, counts = retrieve.write_results(rows, results, Path(tmp) / name)
                hashes.append(digest)
            self.assertEqual(hashes[0], hashes[1])
            self.assertEqual(counts["queries"], 60)
            top10 = [json.loads(l) for l in (Path(tmp) / "a/top10.jsonl").read_text().splitlines()]
            top50 = [json.loads(l) for l in (Path(tmp) / "a/top50.jsonl").read_text().splitlines()]
            for short, long in zip(top10, top50):
                self.assertEqual(short["results"], long["results"][:10])
            with self.assertRaises(FileExistsError):
                retrieve.write_results(rows, iter(()), Path(tmp) / "a")


if __name__ == "__main__":
    unittest.main()
