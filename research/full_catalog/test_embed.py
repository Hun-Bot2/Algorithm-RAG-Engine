"""Fixture-only tests for embed.py helpers (no model load, no private data)."""

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import embed  # noqa: E402


def fake_encoder(dim=4):
    calls = []

    def encode(batch):
        calls.append(list(batch))
        out = np.array([[len(t), 1.0, i % 3, 0.5] for i, t in enumerate(batch)], dtype=np.float32)[:, :dim]
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    return encode, calls


class EncodeAll(unittest.TestCase):
    def test_fixed_order_batches_and_normalized_output(self):
        encode, calls = fake_encoder()
        texts = [f"t{i}" * (i + 1) for i in range(10)]
        vectors = embed.encode_all(encode, texts, batch_size=4)
        self.assertEqual([len(c) for c in calls], [4, 4, 2])
        self.assertEqual(sum(calls, []), texts)
        self.assertEqual(vectors.shape, (10, 4))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)

    def test_rejects_invalid_vectors(self):
        with self.assertRaises(ValueError):
            embed.encode_all(lambda b: np.ones((len(b), 3)), ["a", "b"])
        with self.assertRaises(ValueError):
            embed.encode_all(lambda b: np.full((len(b), 2), np.nan), ["a"])
        with self.assertRaises(ValueError):
            embed.encode_all(lambda b: np.ones((1, 2)) / np.sqrt(2), ["a", "b"], batch_size=2)

    def test_array_digest_depends_on_values_only(self):
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.assertEqual(embed.array_digest(a), embed.array_digest(np.asfortranarray(a)))
        self.assertNotEqual(embed.array_digest(a), embed.array_digest(a + 1))


class TokenStats(unittest.TestCase):
    def test_truncation_and_distribution(self):
        stats = embed.token_stats([3, 5, 7, 300], 256)
        self.assertEqual(stats["truncated"], 1)
        self.assertEqual(stats["truncation_rate"], 0.25)
        self.assertEqual((stats["min"], stats["max"], stats["count"]), (3, 300, 4))
        with self.assertRaises(ValueError):
            embed.token_stats([], 256)

    def test_only_minilm_comparable_variants_are_embedded(self):
        self.assertEqual(embed.EMBED_VARIANTS, ("A1", "A2"))


if __name__ == "__main__":
    unittest.main()
