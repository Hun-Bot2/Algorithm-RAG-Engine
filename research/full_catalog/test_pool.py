"""Fixture-only tests for pool.py (stdlib; no private data)."""

import importlib.util
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("full_catalog_pool", HERE / "pool.py")
pool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pool)


def row(key, difficulty=None, title=None):
    return {"problem_key": key, "platform": key.split(":")[0], "native_difficulty": difficulty,
            "title": title or key, "url": f"https://example.invalid/{key}", "tags": None}


class Strata(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(pool.stratum(row("leetcode:a", "Medium")), "lc-medium")
        self.assertEqual(pool.stratum(row("codeforces:1A", None)), "cf-unrated")
        self.assertEqual(pool.stratum(row("codeforces:1A", 800)), "cf<1200")
        self.assertEqual(pool.stratum(row("codeforces:1A", 1999)), "cf1600-1999")
        self.assertEqual(pool.stratum(row("codeforces:1A", 3500)), "cf>=2400")
        self.assertEqual(pool.stratum(row("atcoder:abc212_a")), "at-abc")
        self.assertEqual(pool.stratum(row("atcoder:practice2_a")), "at-other")

    def test_allocation_sums_respects_minimum_and_capacity(self):
        alloc = pool.allocate({"a": 1000, "b": 50, "c": 1, "d": 300}, 50)
        self.assertEqual(sum(alloc.values()), 50)
        self.assertEqual(alloc["c"], 1)
        self.assertGreaterEqual(alloc["b"], 2)
        self.assertGreater(alloc["a"], alloc["d"])
        self.assertEqual(pool.allocate({"a": 3, "b": 2}, 10), {"a": 3, "b": 2})
        with self.assertRaises(ValueError):
            pool.allocate({str(i): 5 for i in range(10)}, 10)


class Sampling(unittest.TestCase):
    def setUp(self):
        self.rows = ([row(f"leetcode:p{i:03d}", ["Easy", "Medium", "Hard"][i % 3]) for i in range(60)]
                     + [row(f"codeforces:{i}A", 800 + 100 * (i % 20)) for i in range(80)]
                     + [row(f"atcoder:{['abc', 'arc', 'joi'][i % 3]}{i:03d}_a") for i in range(40)])
        a1 = {r["problem_key"] for r in self.rows}
        a2 = {k for k in a1 if not k.startswith("atcoder") and not k.endswith("7A")}
        self.universes = {"A1": a1, "A2": a2}

    def test_sample_is_deterministic_stratified_and_a2_restricted(self):
        first = pool.sample_queries(self.rows, self.universes, 12)
        second = pool.sample_queries(list(reversed(self.rows)), self.universes, 12)
        self.assertEqual(first, second)
        by_platform = {}
        for key, platform, _ in first:
            by_platform[platform] = by_platform.get(platform, 0) + 1
            if platform != "atcoder":
                self.assertIn(key, self.universes["A2"])
        self.assertEqual(by_platform, {"atcoder": 12, "codeforces": 12, "leetcode": 12})
        self.assertEqual({s for _, p, s in first if p == "leetcode"}, {"lc-easy", "lc-medium", "lc-hard"})


class Pairs(unittest.TestCase):
    def setUp(self):
        self.queries = [("leetcode:q", "leetcode", "lc-easy")]
        self.lists = {
            "A1/minilm": {"leetcode:q": ["atcoder:x", "codeforces:y", "codeforces:z"]},
            "A2/tfidf_word": {"leetcode:q": ["codeforces:z", "codeforces:w"]},
            "A1/random": {"leetcode:q": ["atcoder:r", "atcoder:x", "atcoder:s"]},
            "A2/minilm": {},
        }

    def test_union_ranks_min_rank_and_random_depth(self):
        pairs = {p["target"]: p for p in pool.pool_pairs(self.queries, self.lists, depth=2, random_depth=2)}
        self.assertEqual(sorted(pairs), ["atcoder:r", "atcoder:x", "codeforces:w", "codeforces:y", "codeforces:z"])
        self.assertEqual(pairs["atcoder:x"]["ranks"], {"A1/minilm": 1, "A1/random": 2})
        self.assertEqual(pairs["atcoder:x"]["min_rank"], 1)
        self.assertTrue(pairs["atcoder:r"]["random_only"])
        self.assertIsNone(pairs["atcoder:r"]["min_rank"])
        self.assertEqual(pairs["codeforces:z"]["min_rank"], 1)

    def test_label_sheet_is_blind(self):
        pairs = pool.pool_pairs(self.queries, self.lists, depth=3, random_depth=1)
        catalog = [row(k) for k in ("leetcode:q", "atcoder:x", "codeforces:y", "codeforces:z",
                                    "codeforces:w", "atcoder:r")]
        sheet = pool.label_sheet(pairs, catalog, {("leetcode:q", "codeforces:y")})
        text = json.dumps(sheet)
        for leak in ("minilm", "tfidf", "random", "ranks", "min_rank", "score"):
            self.assertNotIn(leak, text)
        self.assertEqual(len(sheet), len(pairs))
        self.assertEqual(sum(r["has_prior_judgment"] for r in sheet), 1)
        self.assertTrue(all(r["label"] is None for r in sheet))
        self.assertEqual({r["tier"] for r in sheet}, {1})


if __name__ == "__main__":
    unittest.main()
