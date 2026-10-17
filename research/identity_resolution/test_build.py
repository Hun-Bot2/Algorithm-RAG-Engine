import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research.identity_resolution.build import (
    REVIEWER,
    proposal_records,
    report_markdown,
    validate_evidence,
)
from research.verification_workbench.core import Workbench, sha


def hashed(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class EvidenceBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_key = "atcoder:dataset:entry-1"
        self.native_key = "atcoder:abc123_a"
        self.text = "authored fixture source text"
        self.corpus = {
            self.source_key: {
                "problem_key": self.source_key,
                "platform": "atcoder",
                "title": "Same title",
                "statement_text": self.text,
                "content_sha256": hashed(self.text),
            }
        }
        self.catalog = {
            self.native_key: {
                "problem_key": self.native_key,
                "platform": "atcoder",
                "title": "Same title",
                "url": "https://atcoder.jp/contests/arc999/tasks/abc123_a",
            }
        }
        self.row = {
            "identity_position": 1,
            "source_key": self.source_key,
            "source_content_sha256": hashed(self.text),
            "publisher_title": "Same title",
            "selected_native_key": self.native_key,
            "canonical_url": "https://atcoder.jp/contests/arc999/tasks/abc123_a",
            "official_url": "https://www.atcoder.jp/contests/abc123/tasks/abc123_a",
            "catalog_member": True,
            "official_access": {"status": "read_via_web_tool", "observed_date": "2026-10-17"},
            "comparison": {
                "contest": "match",
                "semantics": {"same": "match"},
                "constraints": {"same": "match"},
                "samples": {"same": "unavailable"},
            },
            "notes": ["The stated objective and contest context match."],
            "rejected_candidates": [],
            "status": "ai_proposed",
            "extra_receipt": {"kept": True},
        }

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, rows=None, corpus=None, catalog=None):
        return validate_evidence(rows or [self.row], corpus or self.corpus,
                                 catalog or self.catalog, [self.source_key])

    def test_hash_is_recomputed_from_actual_source_text(self):
        changed = dict(self.row, source_content_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.validate([changed])

    def test_same_title_with_wrong_contest_cannot_be_promoted(self):
        changed = dict(self.row)
        changed["comparison"] = dict(self.row["comparison"], contest="conflict")
        with self.assertRaisesRegex(ValueError, "contest and semantic"):
            self.validate([changed])

    def test_task_id_must_match_selected_native_key_even_across_contests(self):
        self.assertEqual(self.validate()["records"], 1)
        changed = dict(self.row, official_url="https://atcoder.jp/contests/abc123/tasks/abc123_b")
        with self.assertRaisesRegex(ValueError, "task ID mismatch"):
            self.validate([changed])

    def test_official_access_allows_provenance_fields_but_requires_iso_date(self):
        changed = dict(self.row)
        changed["official_access"] = dict(self.row["official_access"], scope="public task page")
        self.assertEqual(self.validate([changed])["records"], 1)
        changed["official_access"] = dict(changed["official_access"], observed_date="17-10-2026")
        with self.assertRaisesRegex(ValueError, "observed_date"):
            self.validate([changed])

    def test_ai_proposal_requires_official_url_catalog_url_title_and_notes(self):
        cases = [
            (dict(self.row, official_url=None), "official task URL"),
            (dict(self.row, canonical_url="https://atcoder.jp/contests/abc123/tasks/abc123_a"), "frozen catalog URL"),
            (dict(self.row, publisher_title="Different title"), "publisher title"),
            (dict(self.row, notes=[]), "substantive authored notes"),
        ]
        for row, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.validate([row])

    def test_task_url_rejects_extra_prefix_query_credentials_and_port(self):
        bad_urls = [
            "https://atcoder.jp/extra/contests/abc123/tasks/abc123_a",
            "https://atcoder.jp/contests/abc123/tasks/abc123_a?x=1",
            "https://atcoder.jp/contests/abc123/tasks/abc123_a?",
            "https://atcoder.jp/contests//abc123/tasks/abc123_a",
            "https://atcoder.jp/contests/abc123/tasks/abc123_a/",
            "https://user@atcoder.jp/contests/abc123/tasks/abc123_a",
            "https://atcoder.jp:443/contests/abc123/tasks/abc123_a",
        ]
        for url in bad_urls:
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "task ID mismatch"):
                self.validate([dict(self.row, official_url=url)])

    def test_official_task_url_accepts_only_supported_language_query(self):
        row = dict(self.row, official_url=
                   "https://atcoder.jp/contests/abc123/tasks/abc123_a?lang=en")
        self.assertEqual(self.validate([row])["records"], 1)
        row["official_url"] = "https://atcoder.jp/contests/abc123/tasks/abc123_a?lang=fr"
        with self.assertRaisesRegex(ValueError, "task ID mismatch"):
            self.validate([row])

    def test_task_url_accepts_hyphenated_contest_slug(self):
        key = "atcoder:cf16-final_a"
        row = dict(self.row, selected_native_key=key,
                   canonical_url="https://atcoder.jp/contests/cf16-final/tasks/cf16-final_a",
                   official_url="https://atcoder.jp/contests/cf16-final/tasks/cf16-final_a?lang=en")
        catalog = {key: dict(self.catalog[self.native_key], problem_key=key,
                             url=row["canonical_url"])}
        self.assertEqual(self.validate([row], catalog=catalog)["records"], 1)

    def test_missing_catalog_identity_is_evidence_only_and_not_proposed(self):
        candidate = "atcoder:arc888_b"
        row = dict(self.row, selected_native_key=candidate,
                   canonical_url=None,
                   official_url="https://atcoder.jp/contests/arc888/tasks/arc888_b",
                   catalog_member=False, status="source_identified_missing_catalog")
        self.assertEqual(self.validate([row])["status_counts"], {"source_identified_missing_catalog": 1})
        wb = self.make_workbench()
        self.assertEqual(proposal_records([row], wb), [])

    def test_proposal_requires_constraint_or_sample_match(self):
        changed = dict(self.row)
        changed["comparison"] = dict(self.row["comparison"], constraints="unavailable",
                                     samples="unavailable")
        with self.assertRaisesRegex(ValueError, "constraint or sample"):
            self.validate([changed])

    def test_coverage_is_unique_complete_and_dynamic_for_fixture_subset(self):
        self.assertEqual(self.validate()["positions"], 1)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_evidence([self.row, self.row], self.corpus, self.catalog,
                              [self.source_key, self.source_key])
        changed = dict(self.row, identity_position=2)
        with self.assertRaisesRegex(ValueError, "required-key order"):
            self.validate([changed])

    def test_positions_are_bound_to_required_key_order(self):
        other_key = "atcoder:dataset:entry-2"
        other_text = "second authored fixture source"
        other_source = dict(self.corpus[self.source_key], problem_key=other_key,
                            title="Other source", statement_text=other_text,
                            content_sha256=hashed(other_text))
        first = dict(self.row, identity_position=2)
        second = dict(self.row, identity_position=1, source_key=other_key,
                      publisher_title="Other source", source_content_sha256=hashed(other_text))
        with self.assertRaisesRegex(ValueError, "required-key order"):
            validate_evidence([first, second],
                              {self.source_key: self.corpus[self.source_key], other_key: other_source},
                              self.catalog, [self.source_key, other_key])

    def make_workbench(self):
        corpus_path = self.root / "corpus.jsonl"
        catalog_path = self.root / "catalog.jsonl"
        corpus_path.write_text(json.dumps(next(iter(self.corpus.values()))) + "\n")
        catalog_path.write_text(json.dumps(next(iter(self.catalog.values()))) + "\n")
        pilot = {
            "schema_version": 1,
            "sources": {
                "corpus": {"path": str(corpus_path), "sha256": sha(corpus_path.read_bytes())},
                "catalog": {"path": str(catalog_path), "sha256": sha(catalog_path.read_bytes())},
            },
            "items": [{"pair_id": "pair-fixture", "query": self.source_key,
                       "target": self.native_key}],
        }
        pilot_path = self.root / "pilot.json"
        pilot_path.write_text(json.dumps(pilot))
        return Workbench(pilot_path, self.root / "reviews.sqlite3")

    def test_drafts_are_ai_nonindependent_and_only_for_eligible_occurrences(self):
        wb = self.make_workbench()
        records = proposal_records([self.row], wb)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual((record["reviewer"], record["role"], record["kind"]),
                         (REVIEWER, "ai", "identity"))
        self.assertFalse(record["independent"])
        self.assertEqual(record["status"], "proposed")
        self.assertIn(self.catalog[self.native_key]["url"], record["evidence"])
        self.assertIn("SHA-256", record["evidence"][-1])
        self.assertNotIn(self.text, json.dumps(records))
        self.assertEqual(wb.records(), [])

    def test_proposal_builder_revalidates_complete_evidence_coverage(self):
        wb = self.make_workbench()
        with self.assertRaisesRegex(ValueError, "cover every required"):
            proposal_records([], wb)

    def test_report_contains_authored_notes_but_no_statement_text(self):
        report = report_markdown([self.row])
        self.assertIn("The stated objective and contest context match.", report)
        self.assertIn("[official page](<https://www.atcoder.jp/contests/abc123/tasks/abc123_a>)", report)
        self.assertNotIn(self.text, report)

    def test_report_finishes_overview_before_problem_sections(self):
        second = dict(self.row, identity_position=2, source_key="atcoder:dataset:entry-2",
                      publisher_title="3 Steps | follow-up\nvariant", status="unresolved",
                      selected_native_key=None, canonical_url=None, official_url=None,
                      catalog_member=False, quality_flags=["needs | review"],
                      notes=["Tried https://atcoder.jp/contests/cf16-final/tasks/cf16-final_a."])
        report = report_markdown([second, self.row])
        overview = report.split("\n### ", 1)[0]
        self.assertEqual(sum(line.startswith("| 1 |") for line in overview.splitlines()), 1)
        self.assertEqual(sum(line.startswith("| 2 |") for line in overview.splitlines()), 1)
        self.assertIn("3 Steps \\| follow-up<br>variant", overview)
        self.assertIn("needs \\| review", overview)
        self.assertLess(overview.index("| 2 |"), report.index("\n### 1."))
        self.assertIn("### 2. 3 Steps \\| follow-up variant", report)
        self.assertIn("Attempted candidate URLs:", report)
        self.assertIn("[https://atcoder.jp/contests/cf16-final/tasks/cf16-final_a](<https://atcoder.jp/contests/cf16-final/tasks/cf16-final_a>)", report)

    def test_unresolved_report_includes_attempted_official_urls(self):
        unresolved = dict(self.row, status="unresolved", selected_native_key=None,
                          canonical_url=None, official_url=None, catalog_member=False,
                          attempted_official_urls=[
                              "https://atcoder.jp/contests/jsc2019-qual/tasks/jsc2019_qual_a?lang=en"
                          ])
        report = report_markdown([unresolved])
        self.assertIn("Attempted candidate URLs:", report)
        self.assertIn("[https://atcoder.jp/contests/jsc2019-qual/tasks/jsc2019_qual_a?lang=en]", report)
        self.assertNotIn("None recorded.", report)


if __name__ == "__main__":
    unittest.main()
