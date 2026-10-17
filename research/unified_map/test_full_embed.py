import unittest

import numpy as np

from research.unified_map import full_embed as fe


def node(key, platform="atcoder", eligibility="eligible", flags=()):
    return {"problem_key": key, "platform": platform, "eligibility": eligibility, "encoder_flags": list(flags)}


class Reliability(unittest.TestCase):
    def test_tiers_and_non_latin_cap(self):
        self.assertEqual(fe.reliability_tier(0.0), "reliable")
        self.assertEqual(fe.reliability_tier(0.09), "reliable")
        self.assertEqual(fe.reliability_tier(0.10), "degraded")
        self.assertEqual(fe.reliability_tier(0.49), "degraded")
        self.assertEqual(fe.reliability_tier(0.5), "unreliable")
        self.assertEqual(fe.reliability_tier(0.0, non_latin=True), "degraded")
        self.assertEqual(fe.reliability_tier(0.9, non_latin=True), "unreliable")

    def test_unknown_share(self):
        self.assertEqual(fe.unknown_share([5, 100, 100, 7], 100), 0.5)
        self.assertEqual(fe.unknown_share([], 100), 1.0)


class Assembly(unittest.TestCase):
    def test_orders_by_key_and_tracks_sources(self):
        reused = {"b": np.array([1.0, 0.0], dtype=np.float32), "d": np.array([0.0, 1.0], dtype=np.float32)}
        encoded = {"a": np.array([0.6, 0.8], dtype=np.float32), "c": np.array([0.8, 0.6], dtype=np.float32)}
        vectors, sources = fe.assemble_vectors(["a", "b", "c", "d"], reused, encoded)
        self.assertEqual(sources, ["new_title_encode", "a1_reused", "new_title_encode", "a1_reused"])
        np.testing.assert_array_equal(vectors[1], reused["b"])
        np.testing.assert_array_equal(vectors[2], encoded["c"])
        self.assertEqual(vectors.dtype, np.float32)

    def test_overlap_and_missing_fail(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            fe.assemble_vectors(["a"], {"a": v}, {"a": v})
        with self.assertRaises(ValueError):
            fe.assemble_vectors(["a", "z"], {"a": v}, {})


class EncodeAndCoverage(unittest.TestCase):
    def test_batches_preserve_order_and_validate(self):
        seen = []

        def encode(batch):
            seen.append(list(batch))
            return np.array([[1.0, 0.0] for _ in batch])

        out = fe.encode_in_batches(encode, [str(i) for i in range(7)], batch_size=3)
        self.assertEqual([len(b) for b in seen], [3, 3, 1])
        self.assertEqual(out.shape, (7, 2))
        with self.assertRaises(ValueError):
            fe.encode_in_batches(lambda b: np.ones((len(b), 2)), ["a"])
        with self.assertRaises(ValueError):
            fe.encode_in_batches(lambda b: np.full((len(b), 2), np.nan), ["a"])

    def test_coverage_table_counts_every_node_once(self):
        nodes = [node("a"), node("b", flags=["non_latin_title"]), node("c", eligibility="excluded"),
                 node("d", "leetcode")]
        table = fe.coverage_table(nodes, {"a", "d"})
        self.assertEqual(sum(row["rows"] for row in table), 4)
        gap = [row for row in table if not row["had_a1_vector"]]
        self.assertEqual(sum(row["rows"] for row in gap), 2)


if __name__ == "__main__":
    unittest.main()
