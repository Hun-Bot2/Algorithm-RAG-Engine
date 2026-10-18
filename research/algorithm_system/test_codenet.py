import base64
import csv
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import codenet


def make_tar(path: Path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, content, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                data = content if isinstance(content, bytes) else content.encode("utf-8")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            else:
                info.type = kind
                archive.addfile(info)


class CodeNetImporterTests(unittest.TestCase):
    def test_html_text_preserves_math_and_preformatted_sample_whitespace(self):
        raw = (b"<html><head><title>Ignored</title><script>bad()</script></head>"
               b"<body><h2>Statement</h2><p>Compute &alpha; + x.</p>"
               b"<p>Constraints</p><pre>1  2\n  3\t4\n</pre>"
               b"<p>Sample Input</p><pre>5  7\n</pre><p>Sample Output</p>"
               b"<pre>12\n</pre></body></html>")
        text = codenet.extract_text(raw)
        self.assertIn("Compute α + x.", text)
        self.assertIn("1  2\n  3\t4", text)
        self.assertNotIn("bad()", text)
        fields = codenet.field_availability(text)
        self.assertEqual(fields["constraints"], "heuristic_present")
        self.assertEqual(fields["samples"], "heuristic_present")

    def test_language_is_script_heuristic_and_reports_ambiguity(self):
        self.assertEqual(codenet.language_script_heuristic("問題です")["label"], "ja_script_present")
        self.assertEqual(codenet.language_script_heuristic("漢字")["label"], "han_script_present_language_ambiguous")
        self.assertIn("heuristic", codenet.language_script_heuristic("Problem")["basis"])

    def test_native_identity_and_catalog_join_require_explicit_url(self):
        catalog = {"https://atcoder.jp/contests/abc001/tasks/abc001_a": "atcoder:abc001_a"}
        index = codenet._catalog_url_index([{"url": k, "problem_key": v} for k, v in catalog.items()])
        raw = b'<p>Sample</p><a href="https://atcoder.jp/contests/abc001/tasks/abc001_a">source</a>'
        explicit = codenet.make_record("p00001", raw, {"dataset": "AtCoder"}, index)
        missing = codenet.make_record("p00002", b"<p>Sample</p>", {"dataset": "AtCoder", "name": "ABC A"}, index)
        self.assertEqual(explicit["native_key"], "atcoder:abc001_a")
        self.assertEqual(explicit["problem_key"], "atcoder:abc001_a")
        self.assertIsNone(missing["native_key"])
        self.assertIsNone(missing["problem_key"])
        self.assertEqual(missing["dataset_record_key"], "codenet:p00002")
        self.assertEqual(base64.b64decode(explicit["raw_html_base64"]), raw)

    def test_metadata_free_records_keep_platform_unknown_without_url(self):
        record = codenet.make_record("p00003", "<p>題目</p>".encode("utf-8"), {})
        self.assertIsNone(record["source_platform"])
        self.assertEqual(record["source_platform_basis"], "unavailable")
        self.assertIsNone(record["native_key"])
        self.assertIsNone(record["problem_key"])
        self.assertEqual(record["dataset_record_key"], "codenet:p00003")

    def test_private_import_records_hashes_counts_and_license_caveat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private_root = root / "private-learning"
            private_root.mkdir()
            archive_path = root / "descriptions.tar.gz"
            make_tar(archive_path, [("problem_descriptions/p00001.html", "<p>Title</p><pre>2 3</pre>", "file")])
            metadata_path = root / "problem_list.csv"
            with metadata_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "name", "dataset", "time_limit", "memory_limit", "rating", "tags", "complexity"])
                writer.writeheader()
                writer.writerow({"id": "p00001", "dataset": "AtCoder", "name": "A"})
            codenet.ROOT = root
            source_commit = codenet.PINNED_SOURCE_COMMIT
            archive_blob_sha = codenet.git_blob_sha1_file(archive_path)
            with patch.object(codenet, "PINNED_ARCHIVE_GIT_BLOB_SHA1", archive_blob_sha):
                receipt = codenet.write_private_import(private_root / "run-a", archive_path, metadata_path, source_commit)
            record_path = private_root / "run-a" / "records.jsonl"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema"], "codenet-import-receipt.v1")
            self.assertEqual(receipt["summary"]["description_members"], 1)
            self.assertEqual(receipt["summary"]["descriptions_without_metadata"], 0)
            self.assertIn("separate rights", receipt["license"]["scope_caveat"])
            self.assertEqual(record["raw_html_sha256"], hashlib.sha256(base64.b64decode(record["raw_html_base64"])).hexdigest())
            with self.assertRaises(FileExistsError):
                with patch.object(codenet, "PINNED_ARCHIVE_GIT_BLOB_SHA1", archive_blob_sha):
                    codenet.write_private_import(private_root / "run-a", archive_path, metadata_path, source_commit)

    def test_archive_rejects_traversal_links_and_expansion_over_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, kind, limit, expected in [
                ("../escape.html", "file", codenet.MAX_EXPANDED_BYTES, "unsafe_archive_path"),
                ("p00001.html", tarfile.SYMTYPE, codenet.MAX_EXPANDED_BYTES, "unsupported_archive_entry"),
                ("p00001.html", "file", 2, "archive_expanded_size_limit"),
            ]:
                archive_path = root / f"{len(list(root.iterdir()))}.tar.gz"
                make_tar(archive_path, [(name, "12345", kind)])
                with self.assertRaisesRegex(ValueError, expected):
                    list(codenet._archive_description_members(archive_path, max_expanded_bytes=limit))

    def test_catalog_url_index_rejects_ambiguous_exact_url(self):
        rows = [
            {"url": "https://atcoder.jp/contests/a/tasks/a", "problem_key": "atcoder:a"},
            {"url": "https://atcoder.jp/contests/a/tasks/a?lang=en", "problem_key": "atcoder:a_alias"},
        ]
        with self.assertRaisesRegex(ValueError, "catalog_url_not_unique"):
            codenet._catalog_url_index(rows)


if __name__ == "__main__":
    unittest.main()
