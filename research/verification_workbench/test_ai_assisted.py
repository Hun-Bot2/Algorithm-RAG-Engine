"""Synthetic checks for scoped AI-assisted review exposure."""
import unittest
import json
import tempfile
from pathlib import Path

from core import Workbench, pair_id, prepare


def relationship(pid, reviewer, *, role="human", independent=True, similarity=None):
    return dict(pair_id=pid, reviewer=reviewer, role=role, kind="relationship",
                rationale="Synthetic fixture rationale for workflow validation.",
                evidence=["Synthetic fixture evidence, not a real judgment."],
                similarity=similarity, prerequisite="unknown", transfer="unknown",
                difficulty="unknown", independent=independent)


class AIAssistedViewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        queue, corpus, catalog, pilot = (root / name for name in ("queue.json", "corpus.jsonl", "catalog.jsonl", "pilot.json"))
        corpus.write_text("".join(json.dumps(row) + "\n" for row in [
            {"problem_key": "dataset:a", "platform": "atcoder", "title": "p00001 AtCoder Beginner Contest100 - A + B"},
            {"problem_key": "dataset:b", "platform": "codeforces", "title": "4A Watermelon"},
        ]))
        catalog.write_text("".join(json.dumps(row) + "\n" for row in [
            {"problem_key": "atcoder:abc100_a", "platform": "atcoder", "title": "A + B", "url": "https://atcoder.jp/contests/abc100/tasks/abc100_a"},
            {"problem_key": "codeforces:4A", "platform": "codeforces", "title": "Watermelon", "url": "https://codeforces.com/problemset/problem/4/A"},
        ]))
        queue.write_text(json.dumps({"review_queue": {"items": [
            {"query": "dataset:a", "target": "dataset:b", "methods": {"m": 1}, "stratum": {"method": "m"}},
        ]}}))
        prepare(queue, corpus, catalog, pilot, size=1)
        self.w = Workbench(pilot, root / "reviews.sqlite")
        self.pid = pair_id("dataset:a", "dataset:b")

    def test_ai_assisted_view_shows_ai_and_own_records_only_and_locks_independence(self):
        ai = self.w.append(relationship(self.pid, "draft", role="ai"))["id"]
        own = self.w.append(relationship(self.pid, "judge", independent=False))["id"]
        other = self.w.append(relationship(self.pid, "other"))["id"]

        view = self.w.pair(self.pid, "judge", mode="ai_assisted")

        self.assertEqual({r["id"] for r in view["records"]}, {ai, own})
        self.assertNotIn("methods", view["pair"])
        self.assertNotIn("stratum", view["pair"])
        self.assertTrue(view["blind"])
        self.assertTrue(view["ai_assisted"])
        self.assertFalse(view["unblinded_for_adjudication"])
        self.assertEqual(len(self.w.exposures()), 1)

        self.w.append(relationship(self.pid, "judge", independent=False, similarity=1))
        with self.assertRaisesRegex(ValueError, "unblinded"):
            self.w.append(relationship(self.pid, "judge", independent=True, similarity=0))

        # The distinct human's record remains unavailable in AI-assisted mode.
        self.assertNotIn(other, {r["id"] for r in self.w.pair(self.pid, "judge", mode="ai_assisted")["records"]})

    def test_ai_assisted_requires_trimmed_reviewer(self):
        for reviewer in ("", " judge "):
            with self.subTest(reviewer=reviewer), self.assertRaisesRegex(ValueError, "trimmed reviewer"):
                self.w.pair(self.pid, reviewer, mode="ai_assisted")


if __name__ == "__main__":
    unittest.main()
