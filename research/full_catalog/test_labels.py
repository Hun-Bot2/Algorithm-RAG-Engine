"""Fixture-only tests for labels.py (stdlib)."""

import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("full_catalog_labels", HERE / "labels.py")
labels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(labels)


def out(pid, label, deceptive=False, **kw):
    row = {"pair_id": pid, "label": label, "deceptive": None if label is None else deceptive,
           "basis": "metadata_only", "confidence": "low", "rationale": "both greedy",
           "abstain_reason": "insufficient_metadata" if label is None else None}
    row.update(kw)
    return row


def sheet_row(pid, query, tier=1):
    card = lambda k: {"key": k, "title": k, "tags": None, "difficulty": None, "url": "u"}
    return {"pair_id": pid, "query": card(query), "target": card("t" + pid), "tier": tier,
            "has_prior_judgment": False, "label": None}


class Batches(unittest.TestCase):
    def test_tier1_only_queries_kept_together_and_balanced(self):
        rows = ([sheet_row(f"a{i}", "q1") for i in range(4)] + [sheet_row(f"b{i}", "q2") for i in range(3)]
                + [sheet_row(f"c{i}", "q3") for i in range(2)] + [sheet_row("z", "q3", tier=2)])
        batches = labels.make_batches(rows, 2)
        self.assertEqual(sorted(len(b) for b in batches), [4, 5])
        for batch in batches:
            self.assertEqual(len({r["query"]["key"] for r in batch}), len({r["query"]["key"] for r in batch}))
        self.assertNotIn("z", [r["pair_id"] for b in batches for r in b])
        self.assertNotIn("tier", batches[0][0])
        query_batches = {}
        for i, batch in enumerate(batches):
            for r in batch:
                query_batches.setdefault(r["query"]["key"], set()).add(i)
        self.assertTrue(all(len(v) == 1 for v in query_batches.values()))


class Validate(unittest.TestCase):
    def test_accepts_valid_and_reports_problems(self):
        self.assertEqual(labels.validate([out("p1", 2), out("p2", None)], ["p1", "p2"]), [])
        self.assertTrue(labels.validate([out("p1", 2)], ["p1", "p2"]))
        self.assertTrue(labels.validate([out("p1", 3, deceptive=True)], ["p1"]))
        self.assertTrue(labels.validate([out("p1", 4)], ["p1"]))
        self.assertTrue(labels.validate([{**out("p1", None), "deceptive": False}], ["p1"]))
        self.assertTrue(labels.validate([out("p1", 1, rationale="x " * 30)], ["p1"]))
        self.assertTrue(labels.validate([{**out("p1", 1), "extra": 1}], ["p1"]))


class Resolve(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(labels.resolve(out("p", 2), out("p", 2)), (2, False, "agreement"))
        self.assertEqual(labels.resolve(out("p", None), out("p", None)), (None, None, "agreement"))
        self.assertEqual(labels.resolve(out("p", 2), out("p", 1)), (1, False, "conservative_min"))
        self.assertIsNone(labels.resolve(out("p", 3), out("p", 1)))
        self.assertIsNone(labels.resolve(out("p", 1), out("p", None)))
        self.assertIsNone(labels.resolve(out("p", 1, deceptive=True), out("p", 0)))

    def test_kappa(self):
        self.assertAlmostEqual(labels.quadratic_kappa([(0, 0), (1, 1), (2, 2), (3, 3)]), 1.0)
        self.assertLess(labels.quadratic_kappa([(0, 3), (3, 0), (1, 2), (2, 1)]), 0)
        self.assertIsNone(labels.quadratic_kappa([]))

    def test_finalize_requires_adjudication(self):
        merged = [{"pair_id": "p", "query": "q", "target": "t", "draft": out("p", 3), "review": out("p", 0),
                   "label": None, "deceptive": None, "resolution": "pending_adjudication"}]
        with self.assertRaises(ValueError):
            labels.finalize(merged, [])
        final = labels.finalize(merged, [out("p", 1)])
        self.assertEqual((final[0]["label"], final[0]["resolution"]), (1, "adjudicated"))
        self.assertEqual(final[0]["label_source"], "ai_reviewed")


if __name__ == "__main__":
    unittest.main()
