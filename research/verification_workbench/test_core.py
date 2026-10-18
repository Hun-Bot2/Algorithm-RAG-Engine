"""Synthetic workflow tests; fixture records are not human research judgments."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from core import Workbench, pair_id, prepare, restore, title_core


def relationship(pid, reviewer, *, role="human", independent=True, similarity=None):
    return dict(pair_id=pid, reviewer=reviewer, role=role, kind="relationship",
                rationale="Synthetic fixture rationale for workflow validation.",
                evidence=["Synthetic fixture evidence, not a real judgment."],
                similarity=similarity, prerequisite="unknown", transfer="unknown",
                difficulty="unknown", independent=independent)


def identity(pid, reviewer, subject, native, url, *, role="human", independent=True):
    return dict(pair_id=pid, reviewer=reviewer, role=role, kind="identity",
                rationale="Synthetic fixture rationale for identity validation.",
                evidence=[url], subject_key=subject, native_key=native,
                status="proposed", independent=independent)


def adjudication(pid, reviewer, task, ids, decision, *, outcome="accept", evidence=None):
    return dict(pair_id=pid, reviewer=reviewer, role="human", kind="adjudication",
                rationale="Synthetic fixture adjudication for gate validation.",
                evidence=evidence or ["Synthetic fixture evidence, not a real judgment."],
                task=task, review_ids=ids, outcome=outcome, decision=decision)


class WorkbenchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.queue = self.root / "queue.json"
        self.corpus = self.root / "corpus.jsonl"
        self.catalog = self.root / "catalog.jsonl"
        self.pilot = self.root / "pilot.json"
        self.db = self.root / "reviews.sqlite"
        rows = [
            dict(problem_key="dataset:a", platform="atcoder", title="p00001 AtCoder Beginner Contest100 - A + B"),
            dict(problem_key="dataset:b", platform="codeforces", title="4A Watermelon"),
            dict(problem_key="dataset:c", platform="atcoder", title="p00002 AtCoder Beginner Contest100 - Other"),
        ]
        native = [
            dict(problem_key="atcoder:abc100_a", platform="atcoder", title="A + B", url="https://atcoder.jp/contests/abc100/tasks/abc100_a"),
            dict(problem_key="atcoder:arc100_a", platform="atcoder", title="A + B", url="https://atcoder.jp/contests/arc100/tasks/arc100_a"),
            dict(problem_key="codeforces:4A", platform="codeforces", title="Watermelon", url="https://codeforces.com/problemset/problem/4/A"),
            dict(problem_key="atcoder:abc100_b", platform="atcoder", title="Other", url="https://atcoder.jp/contests/abc100/tasks/abc100_b"),
        ]
        self.corpus.write_text("".join(json.dumps(x) + "\n" for x in rows))
        self.catalog.write_text("".join(json.dumps(x) + "\n" for x in native))
        self.queue.write_text(json.dumps({"review_queue": {"items": [
            dict(query="dataset:a", target="dataset:b", methods={"m": 1}, stratum={"method": "m", "platform_pair": "a>b"}),
            dict(query="dataset:c", target="dataset:b", methods={"n": 2}, stratum={"method": "n", "platform_pair": "a>b"}),
        ]}}))
        self.frozen = prepare(self.queue, self.corpus, self.catalog, self.pilot, size=40, seed=17)
        self.w = Workbench(self.pilot, self.db)
        self.pid = pair_id("dataset:a", "dataset:b")

    def test_pilot_is_deterministic_bounded_and_hash_bound(self):
        other = self.root / "pilot2.json"
        self.assertEqual(self.frozen, prepare(self.queue, self.corpus, self.catalog, other, size=40, seed=17))
        self.assertEqual(self.frozen["actual_pairs"], 2)
        self.assertEqual(sum(self.frozen["strata"].values()), 2)
        self.assertEqual({x["pair_id"] for x in self.frozen["items"]},
                         {pair_id("dataset:a", "dataset:b"), pair_id("dataset:c", "dataset:b")})
        with self.assertRaises(FileExistsError):
            prepare(self.queue, self.corpus, self.catalog, self.pilot)
        self.corpus.write_text(self.corpus.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "source drift"):
            Workbench(self.pilot, self.root / "drift.sqlite")

    def test_candidates_are_only_unverified_and_blind_view_hides_rank(self):
        view = self.w.pair(self.pid, "h1")
        self.assertTrue(view["blind"])
        self.assertNotIn("methods", view["pair"])
        self.assertNotIn("stratum", view["pair"])
        candidate = view["identity_candidates"]["query"][0]
        self.assertEqual(candidate["problem_key"], "atcoder:abc100_a")
        self.assertEqual(candidate["status"], "unverified_candidate")
        # A title duplicate from a different contest remains only a candidate.
        candidates = view["identity_candidates"]["query"]
        self.assertEqual({row["problem_key"] for row in candidates},
                         {"atcoder:abc100_a", "atcoder:arc100_a"})
        self.assertTrue(all(row["status"] == "unverified_candidate" for row in candidates))
        self.assertEqual(self.w.progress()["accepted_adjudications"], 0)
        self.assertEqual(self.w.export(self.root / "candidate-export.json")["accepted_overlay"], [])
        self.w.append(relationship(self.pid, "h1"))
        hidden = self.w.pair(self.pid, "h2")
        self.assertEqual(hidden["records"], [])
        visible = self.w.pair(self.pid, "h1")
        self.assertFalse(visible["blind"])
        self.assertIn("methods", visible["pair"])
        self.assertEqual(len(visible["records"]), 1)

    def test_title_normalization_preserves_numeric_title_prefixes(self):
        self.assertEqual(title_core("p12345 AtCoder Beginner Contest100 - 3 Steps"), "3 steps")
        self.assertEqual(title_core("p67890 AtCoder Beginner Contest200 - 01 Unbalanced"), "01 unbalanced")
        self.assertEqual(title_core("p00001 AtCoder Beginner Contest100 - A + B"), "a b")

    def test_dimensions_keep_unknown_distinct_and_reject_invalid(self):
        rec = relationship(self.pid, "h1")
        self.w.append(rec)
        stored = self.w.records()[0]
        self.assertIsNone(stored["similarity"])
        self.assertEqual(stored["transfer"], "unknown")
        for change in ({"similarity": -1}, {"similarity": "unknown"},
                       {"prerequisite": "maybe"}, {"transfer": "maybe"},
                       {"difficulty": "maybe"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.w.append(dict(rec, reviewer="h2", **change))
        with self.assertRaises(ValueError):
            self.w.append(dict(rec, reviewer="h2", rationale="short"))

    def test_relationship_adjudication_requires_distinct_independent_humans(self):
        decision = {k: relationship(self.pid, "x")[k]
                    for k in ("similarity", "prerequisite", "transfer", "difficulty")}
        ai = self.w.append(relationship(self.pid, "bot", role="ai"))["id"]
        h1 = self.w.append(relationship(self.pid, "h1"))["id"]
        nonind = self.w.append(relationship(self.pid, "h2", independent=False))["id"]
        for ids, who in (([ai, h1], "h3"), ([h1, nonind], "h3"), ([h1, h1], "h3"), ([h1, ai], "h1")):
            with self.subTest(ids=ids, who=who), self.assertRaises(ValueError):
                self.w.append(adjudication(self.pid, who, "relationship", ids, decision))
        h2 = self.w.append(relationship(self.pid, "h2"))["id"]
        with self.assertRaisesRegex(ValueError, "third distinct"):
            self.w.append(adjudication(self.pid, "h1", "relationship", [h1, h2], decision))
        with self.assertRaisesRegex(ValueError, "only first review"):
            self.w.append(relationship(self.pid, "h1", similarity=0))
        self.w.append(adjudication(self.pid, "h3", "relationship", [h1, h2], decision))
        self.assertEqual(self.w.progress()["accepted_adjudications"], 1)
        self.assertFalse(self.w.progress()["human_identity_authenticated"])

    def test_unblinding_exposure_prevents_later_independent_claim(self):
        original = relationship(self.pid, "h1")
        self.w.append(original)
        with self.assertRaises(ValueError):
            self.w.pair(self.pid, " judge ", mode="adjudicate")
        shown = self.w.pair(self.pid, "judge", mode="adjudicate")
        self.assertTrue(shown["unblinded_for_adjudication"])
        self.assertEqual(len(shown["records"]), 1)
        self.assertEqual(len(self.w.exposures()), 1)
        self.assertEqual(len(self.w.export(self.root / "export.json")["exposures"]), 1)
        with sqlite3.connect(self.db) as con:
            for sql in ("UPDATE exposures SET reviewer='someone else'", "DELETE FROM exposures"):
                with self.subTest(sql=sql), self.assertRaises(sqlite3.IntegrityError):
                    con.execute(sql)
        with self.assertRaisesRegex(ValueError, "unblinded"):
            self.w.append(relationship(self.pid, "judge"))
        self.w.pair(self.pid, "h1", mode="adjudicate")
        self.assertTrue(self.w.append(original)["duplicate"])
        with self.assertRaisesRegex(ValueError, "unblinded"):
            self.w.append(relationship(self.pid, "h1", similarity=0))

    def test_identity_requires_subject_native_url_and_two_agreeing_reviews(self):
        url = "https://atcoder.jp/contests/abc100/tasks/abc100_a"
        first = identity(self.pid, "h1", "dataset:a", "atcoder:abc100_a", url)
        for change in ({"subject_key": "dataset:c"}, {"native_key": "codeforces:4A"},
                       {"evidence": ["wrong URL"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.w.append(dict(first, **change))
        h1 = self.w.append(first)["id"]
        h2 = self.w.append(identity(self.pid, "h2", "dataset:a", "atcoder:abc100_a", url))["id"]
        decision = {"subject_key": "dataset:a", "native_key": "atcoder:abc100_a"}
        with self.assertRaisesRegex(ValueError, "canonical adjudication URL"):
            self.w.append(adjudication(self.pid, "h3", "identity", [h1, h2], decision))
        accepted = adjudication(self.pid, "h3", "identity", [h1, h2], decision, evidence=[url])
        self.w.append(accepted)
        self.assertEqual(self.w.export(self.root / "export.json")["accepted_overlay"][0]["decision"], decision)

    def test_identity_blindness_is_scoped_to_reviewed_subject(self):
        a_url = "https://atcoder.jp/contests/abc100/tasks/abc100_a"
        b_url = "https://codeforces.com/problemset/problem/4/A"
        self.w.append(identity(self.pid, "h1", "dataset:a", "atcoder:abc100_a", a_url))
        self.w.append(identity(self.pid, "h2", "dataset:a", "atcoder:abc100_a", a_url))
        self.w.append(identity(self.pid, "h2", "dataset:b", "codeforces:4A", b_url))
        visible = self.w.pair(self.pid, "h1")["records"]
        self.assertEqual(len(visible), 2)
        self.assertEqual({r["subject_key"] for r in visible}, {"dataset:a"})

    def test_idempotency_triggers_chain_and_export(self):
        rec = relationship(self.pid, "h1")
        first = self.w.append(rec)
        second = self.w.append(rec)
        self.assertEqual(first["id"], second["id"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(self.w.audit()["records"], 1)
        with sqlite3.connect(self.db) as con:
            for sql in ("UPDATE records SET payload='{}'", "DELETE FROM records"):
                with self.subTest(sql=sql), self.assertRaises(sqlite3.IntegrityError):
                    con.execute(sql)
        exported = self.w.export(self.root / "export.json")
        self.assertEqual(exported["records"], self.w.records())
        self.assertEqual(exported["audit"], self.w.audit())
        with sqlite3.connect(self.db) as con:
            con.execute("DROP TRIGGER no_update")
            con.execute("UPDATE records SET payload='{}'")
        with self.assertRaisesRegex(ValueError, "chain corrupted"):
            self.w.audit()

    def test_export_restore_round_trip_and_pilot_mismatch_cleanup(self):
        self.w.append(relationship(self.pid, "h1"))
        self.w.pair(self.pid, "judge", mode="adjudicate")
        exported = self.w.export(self.root / "export.json")
        restored = restore(self.pilot, self.root / "export.json", self.root / "restored.sqlite")
        self.assertEqual(restored.audit(), self.w.audit())
        self.assertEqual(restored.records(), self.w.records())
        self.assertEqual(restored.exposures(), self.w.exposures())
        with self.assertRaises(FileExistsError):
            restore(self.pilot, self.root / "export.json", self.root / "restored.sqlite")
        changed = dict(exported)
        changed["audit"] = dict(exported["audit"], pilot_sha256="0" * 64)
        bad_export = self.root / "bad-export.json"
        bad_export.write_text(json.dumps(changed))
        failed_db = self.root / "failed.sqlite"
        with self.assertRaises(ValueError):
            restore(self.pilot, bad_export, failed_db)
        self.assertFalse(failed_db.exists())

    def test_later_correction_suspends_accepted_overlay(self):
        initial = relationship(self.pid, "h1")
        h1 = self.w.append(initial)["id"]
        h2 = self.w.append(relationship(self.pid, "h2"))["id"]
        decision = {k: initial[k] for k in ("similarity", "prerequisite", "transfer", "difficulty")}
        self.w.append(adjudication(self.pid, "h3", "relationship", [h1, h2], decision))
        self.assertEqual(len(self.w.export(self.root / "before.json")["accepted_overlay"]), 1)
        correction = dict(initial, similarity=0, independent=False)
        self.w.append(correction)
        self.assertEqual(self.w.export(self.root / "after.json")["accepted_overlay"], [])
        self.assertEqual(self.w.progress()["accepted_adjudications"], 0)


if __name__ == "__main__":
    unittest.main()
