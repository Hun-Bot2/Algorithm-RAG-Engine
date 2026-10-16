import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.learning.catalog_sources import FetchResult, collect, parse_atcoder, parse_codeforces


def cf_bytes(problems):
    return json.dumps({"status": "OK", "result": {"problems": problems}}).encode()


class CatalogSourceTests(unittest.TestCase):
    def test_codeforces_missing_rating_and_tags_are_preserved(self):
        rows, rejects = parse_codeforces(cf_bytes([{"contestId": 4, "index": "A", "name": "Watermelon"}]))
        self.assertEqual(rejects, {})
        self.assertEqual(rows[0]["problem_key"], "codeforces:4A")
        self.assertIsNone(rows[0]["native_difficulty"])
        self.assertIsNone(rows[0]["tags"])
        self.assertIsNone(rows[0]["full_statement"])

    def test_atcoder_metadata_does_not_invent_rating_or_tags(self):
        rows, rejects = parse_atcoder(json.dumps([{"id": "abc086_a", "contest_id": "abc086", "name": "Product"}]).encode())
        self.assertEqual(rejects, {})
        self.assertEqual(rows[0]["problem_key"], "atcoder:abc086_a")
        self.assertIsNone(rows[0]["native_difficulty"])
        self.assertIsNone(rows[0]["tags"])

    def test_atcoder_hyphenated_contest_id_is_a_valid_url_component(self):
        rows, rejects = parse_atcoder(b'[{"id":"past201912_open_a","name":"Task","contest_id":"past201912-open"}]')
        self.assertEqual(rejects, {})
        self.assertEqual(rows[0]["url"], "https://atcoder.jp/contests/past201912-open/tasks/past201912_open_a")

    def test_api_failure_and_html_are_schema_errors(self):
        with self.assertRaisesRegex(ValueError, "status_not_OK"):
            parse_codeforces(json.dumps({"status": "FAILED"}).encode())
        with self.assertRaisesRegex(ValueError, "invalid_json"):
            parse_atcoder(b"<html>blocked</html>")

    def test_duplicate_identity_is_counted_as_rejection(self):
        problems = [{"contestId": 4, "index": "A", "name": "One"},
                    {"contestId": 4, "index": "A", "name": "Duplicate"}]
        rows, rejects = parse_codeforces(cf_bytes(problems))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rejects, {"duplicate_identity": 1})

    def test_unsafe_ids_and_negative_or_nonfinite_rating_are_rejected(self):
        rows, rejects = parse_codeforces(cf_bytes([
            {"contestId": 4, "index": "../A", "name": "Unsafe"},
            {"contestId": 4, "index": "B", "name": "Bad rating", "rating": -1},
            {"contestId": 4, "index": "C", "name": "Infinite", "rating": float("inf")},
        ]))
        self.assertEqual(rows, [])
        self.assertEqual(rejects["missing_or_invalid_index"], 1)
        self.assertEqual(rejects["invalid_rating"], 2)
        rows, rejects = parse_atcoder(b'[{"id":"../bad","name":"Unsafe"}]')
        self.assertEqual(rows, [])
        self.assertEqual(rejects["missing_or_invalid_task_id"], 1)

    def test_existing_output_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            output.mkdir()
            marker = output / "keep"
            marker.write_text("unchanged")
            with self.assertRaises(FileExistsError):
                collect(output)
            self.assertEqual(marker.read_text(), "unchanged")

    def test_fetch_failure_preserves_partial_raw_receipts_and_other_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = iter([FetchResult(b"", None, "timeout"), FetchResult(b'[{"id":"abc_a","name":"A","contest_id":"abc"}]', 200)])
            with patch("src.learning.catalog_sources.time.sleep"):
                manifest = collect(Path(tmp) / "run", fetch=True, transport=lambda _url: next(calls))
            self.assertEqual(manifest["codeforces"]["status"], "failed")
            self.assertEqual(manifest["codeforces"]["error"], "timeout")
            self.assertEqual(manifest["atcoder"]["status"], "ok")
            self.assertTrue((Path(tmp) / "run" / "codeforces.raw").exists())

    def test_replay_checks_raw_hash_before_normalizing(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            body = cf_bytes([{"contestId": 4, "index": "A", "name": "Watermelon"}])
            (source / "codeforces.raw").write_bytes(body)
            (source / "atcoder.raw").write_bytes(b'[{"id":"abc_a","name":"A","contest_id":"abc"}]')
            manifest = {"codeforces": {"byte_count": len(body), "sha256": hashlib.sha256(body).hexdigest(), "http_status": 200},
                        "atcoder": {"byte_count": len((source / "atcoder.raw").read_bytes()), "sha256": hashlib.sha256((source / "atcoder.raw").read_bytes()).hexdigest(), "http_status": 200}}
            for source_name in ("codeforces", "atcoder"):
                manifest[source_name]["retrieved_at"] = "2026-10-16T00:00:00+00:00"
            (source / "manifest.json").write_text(json.dumps(manifest))
            output = Path(tmp) / "replay"
            result = collect(output, replay=source)
            self.assertEqual(result["codeforces"]["normalized_count"], 1)
            self.assertEqual(result["codeforces"]["retrieved_at"], "2026-10-16T00:00:00+00:00")
            self.assertIn("replayed_from", result["codeforces"])
            (source / "atcoder.raw").write_bytes(b"[ ]")
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                collect(Path(tmp) / "replay-bad", replay=source)

    def test_http_429_is_failed_not_empty_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = iter([FetchResult(b"rate limited", 429), FetchResult(b'[{"id":"abc_a","name":"A"}]', 200)])
            with patch("src.learning.catalog_sources.time.sleep"):
                manifest = collect(Path(tmp) / "run", fetch=True, transport=lambda _url: next(calls))
            self.assertEqual(manifest["codeforces"]["status"], "failed")
            self.assertEqual(manifest["codeforces"]["error"], "http_error_429")


class IdentityCoverageTests(unittest.TestCase):
    def test_reviewed_alias_preserves_raw_identity_and_direct_missingness(self):
        from src.learning.catalog_sources import curriculum_coverage
        native = {"atcoder:arc089_a", "codeforces:4A"}
        result = curriculum_coverage(["atcoder:abc086_c", "codeforces:4A"], native,
                                    [{"legacy_key": "atcoder:abc086_c", "native_key": "atcoder:arc089_a"}])
        self.assertEqual(result["atcoder"]["missing"], ["atcoder:abc086_c"])
        self.assertEqual(result["atcoder"]["unresolved"], [])
        self.assertEqual(result["atcoder"]["matched_via_reviewed_alias"][0]["native_key"], "atcoder:arc089_a")
        self.assertNotIn("atcoder:abc086_c", native)

    def test_absent_native_target_is_not_counted_as_coverage(self):
        from src.learning.catalog_sources import curriculum_coverage
        result = curriculum_coverage(["atcoder:abc086_c"], set(),
                                    [{"legacy_key": "atcoder:abc086_c", "native_key": "atcoder:arc089_a"}])
        self.assertEqual(result["atcoder"]["unresolved"], ["atcoder:abc086_c"])

    def test_ambiguous_or_cross_platform_alias_is_rejected(self):
        from src.learning.catalog_sources import curriculum_coverage
        bad = [{"legacy_key": "atcoder:abc086_c", "native_key": "codeforces:4A"}]
        with self.assertRaises(ValueError):
            curriculum_coverage([], set(), bad)
        alias = {"legacy_key": "atcoder:abc086_c", "native_key": "atcoder:arc089_a"}
        with self.assertRaises(ValueError):
            curriculum_coverage([], set(), [alias, alias])


if __name__ == "__main__":
    unittest.main()
