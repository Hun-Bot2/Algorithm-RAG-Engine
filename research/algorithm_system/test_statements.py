"""Authored, synthetic tests for the statement corpus helpers."""

import hashlib
import unittest

from statements import build_corpus, duplicate_audit, feature_tags, jaccard_block, token_windows


_SHARED_HASH = hashlib.sha256(b"authored fixture only").hexdigest()

# All statement text below is authored test text, not copied from platform sources.
AUTHORED_SOURCE = [
    {
        "dataset_key": "publisher-a:fixture:long",
        "native_key": "atcoder:fixture_task",
        "platform": "atcoder",
        "title": "Shared Fixture Name",
        "statement_text": "Authored fixture statement, longer selected version.",
        "content_sha256": _SHARED_HASH,
        "source_tags": ["graph"],  # Must not fill the catalog's unavailable AtCoder tags.
    },
    {
        "dataset_key": "publisher-b:fixture:short",
        "native_key": "atcoder:fixture_task",
        "platform": "atcoder",
        "title": "Shared Fixture Name",
        "statement_text": "Short fixture.",
        "content_sha256": hashlib.sha256(b"short fixture").hexdigest(),
        "source_tags": ["graph"],
    },
    {
        "dataset_key": "publisher-a:fixture:cf",
        "native_key": "codeforces:4A",
        "platform": "codeforces",
        "title": "Shared Fixture Name",
        "statement_text": "Another authored fixture statement.",
        "content_sha256": _SHARED_HASH,
        "source_tags": ["untrusted-source-tag"],
    },
]

AUTHORED_CATALOG = [
    {"problem_key": "atcoder:fixture_task", "platform": "atcoder",
     "tags": None, "native_difficulty": None},
    {"problem_key": "codeforces:4A", "platform": "codeforces",
     "tags": ["math"], "native_difficulty": 800},
]


class StatementHelpersTests(unittest.TestCase):
    def test_corpus_preserves_versions_and_does_not_impute_atcoder_tags(self):
        corpus, alternatives = build_corpus(AUTHORED_SOURCE, AUTHORED_CATALOG)
        atcoder = next(row for row in corpus if row["platform"] == "atcoder")
        codeforces = next(row for row in corpus if row["platform"] == "codeforces")

        self.assertEqual(atcoder["problem_key"], "atcoder:fixture_task")
        self.assertEqual(atcoder["dataset_key"], "publisher-a:fixture:long")
        self.assertEqual(atcoder["publisher_versions"], [
            "publisher-a:fixture:long", "publisher-b:fixture:short",
        ])
        self.assertEqual(atcoder["statement_text"], "Authored fixture statement, longer selected version.")
        self.assertIsNone(atcoder["source_tags"])
        self.assertEqual(atcoder["taxonomy"]["tag_status"], "unavailable")
        self.assertIsNone(atcoder["taxonomy"]["original_tags"])
        self.assertIsNone(atcoder["native_difficulty"])
        self.assertEqual(codeforces["source_tags"], ["math"])
        self.assertEqual(codeforces["native_difficulty"], 800)
        self.assertEqual(len(corpus), 2)
        self.assertEqual(alternatives[0]["selected"], "publisher-a:fixture:long")
        self.assertEqual(alternatives[0]["versions"], atcoder["publisher_versions"])

    def test_duplicate_audit_reports_hash_and_title_candidates_without_dropping_rows(self):
        corpus, _ = build_corpus(AUTHORED_SOURCE, AUTHORED_CATALOG)
        exact, ambiguous = duplicate_audit(corpus)
        self.assertEqual(exact, [["atcoder:fixture_task", "codeforces:4A"]])
        self.assertEqual(len(ambiguous), 1)
        self.assertEqual(ambiguous[0]["status"], "pending_human_review")
        self.assertEqual(ambiguous[0]["basis"], "normalized title equality; not verified semantic duplicates")
        self.assertEqual(len(corpus), 2)

    def test_tag_features_keep_missing_atcoder_distinct_from_codeforces(self):
        corpus, _ = build_corpus(AUTHORED_SOURCE, AUTHORED_CATALOG)
        raw = feature_tags(corpus)
        hierarchy = feature_tags(corpus, hierarchy=True)
        self.assertEqual(raw[0], set())
        self.assertEqual(hierarchy[0], set())
        self.assertEqual(raw[1], {"math"})
        # Taxonomy v1 keeps the direct concept and its broader ancestor separate
        # in the normalized record, then feature_tags unions both for hierarchy.
        self.assertEqual(hierarchy[1], {"math", "mathematics"})

    def test_jaccard_blocks_match_exact_small_reference(self):
        features = [{"dp", "graphs"}, {"dp"}, set()]
        score = jaccard_block(features)(0, 3)
        self.assertEqual(score.dtype.name, "float32")
        self.assertEqual(score.tolist(), [
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

    def test_token_windows_preserve_every_original_token_in_order(self):
        token_ids = [list(range(1, 256)), [500, 501], list(range(1000, 1510))]
        windows = token_windows(token_ids, window=254)
        self.assertTrue(all(1 <= len(part) <= 254 for _, part in windows))
        for owner, original in enumerate(token_ids):
            flattened = [token for index, part in windows if index == owner for token in part]
            self.assertEqual(flattened, original)

    def test_token_windows_reject_zero_and_oversize_windows(self):
        with self.assertRaises(ValueError):
            token_windows([[1]], window=0)
        with self.assertRaises(ValueError):
            token_windows([[1]], window=255)


if __name__ == "__main__":
    unittest.main()
