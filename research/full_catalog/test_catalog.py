"""Fixture-only tests for catalog.py (no private data, no network)."""

import importlib.util
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("full_catalog_catalog", HERE / "catalog.py")
catalog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(catalog)

RULES = json.loads((HERE / "exclusions.v1.json").read_text(encoding="utf-8"))["rules"]
ALIASES = [{"legacy_key": "atcoder:abc049_c", "native_key": "atcoder:arc065_a"}]


def lc_line(slug, title, tags, **extra):
    return json.dumps({"id": "1", "slug": slug, "title": title, "difficulty": "Easy", "tags": tags,
                       "url": f"https://leetcode.com/problems/{slug}/", "content_clean": "SECRET",
                       "embedding_text": "SECRET", **extra})


def ext_line(key, platform, title, tags=None, url=None, difficulty=None):
    return json.dumps({"problem_key": key, "platform": platform, "title": title, "tags": tags,
                       "url": url, "native_difficulty": difficulty, "full_statement": None,
                       "statement_status": "not_provided"})


def at(task, contest, title):
    return ext_line(f"atcoder:{task}", "atcoder", title,
                    url=f"https://atcoder.jp/contests/{contest}/tasks/{task}")


def cf(key, title, tags):
    return ext_line(f"codeforces:{key}", "codeforces", title, tags=tags, difficulty=1500)


FIXTURE_LC = [
    lc_line("two-sum", "Two Sum", ["Array", "Hash Table"], content="<p>SECRET</p>"),
    lc_line("combine-two-tables", "Combine Two Tables", ["Database"]),
    lc_line("print-in-order", "Print in Order", ["Concurrency"]),
]
FIXTURE_EXT = [
    cf("1000A", "Game", ["games"]),
    cf("1001C", "Game", ["games"]),
    cf("1275E1", "Контрольная сумма", ["*special"]),
    cf("1275E2", "Контрольная сумма", ["*special"]),
    cf("500B", "Game", ["greedy"]),
    cf("4A", "Two Sum", ["math"]),
    at("APG4bPython_a", "APG4bPython", "1.00.はじめに"),
    at("practice_1", "practice", "Welcome to AtCoder"),
    at("practice2_a", "practice2", "Disjoint Set Union"),
    at("ahc001_a", "ahc001", "AtCoder Ad"),
    at("future_contest_2019_qual_b", "future-contest-2019-qual", "ビジュアライザ"),
    at("abc212_a", "adt_easy_20260908_1", "Alloy"),
    at("arc065_a", "abc049", "Daydream"),
]


def build():
    return catalog.build_catalog(catalog.load_leetcode(FIXTURE_LC),
                                 catalog.load_external(FIXTURE_EXT), RULES, ALIASES)


class LeetCodeQuarantine(unittest.TestCase):
    def test_statement_fields_are_dropped_at_parse_time(self):
        rows = catalog.load_leetcode(FIXTURE_LC)
        self.assertEqual(len(rows), 3)
        serialized = json.dumps(rows)
        self.assertNotIn("SECRET", serialized)
        for row in rows:
            self.assertTrue(set(row).isdisjoint(catalog.FORBIDDEN_LC_FIELDS))


class Exclusions(unittest.TestCase):
    def setUp(self):
        rows, _, _ = build()
        self.by_key = {r["problem_key"]: r for r in rows}

    def reasons(self, key):
        return self.by_key[key]["exclusion_reasons"]

    def test_rules_match_expected_codes(self):
        self.assertEqual(self.reasons("leetcode:combine-two-tables"), ["lc_non_algorithmic_tag"])
        self.assertEqual(self.reasons("leetcode:print-in-order"), ["lc_non_algorithmic_tag"])
        self.assertEqual(self.reasons("codeforces:1275E1"), ["cf_special_problem"])
        self.assertEqual(self.reasons("atcoder:APG4bPython_a"), ["at_tutorial_contest"])
        self.assertEqual(self.reasons("atcoder:practice_1"), ["at_practice_contest"])
        self.assertEqual(self.reasons("atcoder:ahc001_a"), ["at_heuristic_contest"])
        self.assertEqual(self.reasons("atcoder:future_contest_2019_qual_b"), ["at_non_problem_page"])

    def test_algorithmic_rows_stay_eligible(self):
        for key in ("leetcode:two-sum", "atcoder:practice2_a", "atcoder:abc212_a", "codeforces:4A"):
            self.assertEqual(self.by_key[key]["eligibility"], "eligible", key)

    def test_nothing_is_removed(self):
        self.assertEqual(len(self.by_key), len(FIXTURE_LC) + len(FIXTURE_EXT))

    def test_language_is_a_flag_not_an_exclusion(self):
        row = self.by_key["atcoder:APG4bPython_a"]
        self.assertEqual(row["encoder_flags"], ["non_latin_title"])
        self.assertEqual(self.by_key["leetcode:two-sum"]["encoder_flags"], [])


class DuplicateCandidates(unittest.TestCase):
    def setUp(self):
        self.rows, groups, self.cross = build()
        self.groups = {tuple(g["keys"]): g for g in groups}

    def test_same_title_groups_are_candidates_with_structural_evidence(self):
        game = self.groups[("codeforces:1000A", "codeforces:1001C", "codeforces:500B")]
        self.assertEqual(game["kind"], "possible_division_mirror")
        self.assertEqual(game["evidence"], {"adjacent_contest_pairs": [[1000, 1001]]})
        self.assertEqual(game["status"], "candidate_unverified")
        versions = self.groups[("codeforces:1275E1", "codeforces:1275E2")]
        self.assertEqual(versions["kind"], "easy_hard_versions")

    def test_candidates_never_change_eligibility_or_representative(self):
        for row in self.rows:
            self.assertEqual(row["representative_key"], row["problem_key"])
        self.assertEqual(next(r for r in self.rows if r["problem_key"] == "codeforces:1000A")["eligibility"],
                         "eligible")

    def test_cross_platform_title_match_is_evidence_only(self):
        self.assertEqual(self.cross, [["codeforces:4A", "leetcode:two-sum"]])
        self.assertNotIn(("codeforces:4A", "leetcode:two-sum"), self.groups)

    def test_alias_representative_only_when_both_keys_present(self):
        extra = catalog.load_external([at("abc049_c", "abc049", "Daydream")])
        rows, _, _ = catalog.build_catalog([], catalog.load_external(FIXTURE_EXT) + extra, RULES, ALIASES)
        rep = {r["problem_key"]: r["representative_key"] for r in rows}
        self.assertEqual(rep["atcoder:abc049_c"], "atcoder:arc065_a")
        self.assertEqual(rep["atcoder:arc065_a"], "atcoder:arc065_a")

    def test_repeated_key_fails(self):
        with self.assertRaises(ValueError):
            catalog.build_catalog([], catalog.load_external(FIXTURE_EXT[:1] * 2), RULES, ALIASES)


class Determinism(unittest.TestCase):
    def test_input_order_does_not_change_output(self):
        a = build()
        b = catalog.build_catalog(catalog.load_leetcode(FIXTURE_LC[::-1]),
                                  catalog.load_external(FIXTURE_EXT[::-1]), RULES, ALIASES)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_title_normalization(self):
        self.assertEqual(catalog.normalize_title("  Querying  Multiset "), "querying multiset")
        self.assertEqual(catalog.normalize_title("ＡＢＣ"), "abc")


class Crosswalk(unittest.TestCase):
    def test_validation_reports_unknown_and_unlisted_tags(self):
        walk = {"cf_to_lc": [{"cf": "dp", "lc": [{"tag": "Dynamic Programming", "relation": "equivalent"}]},
                             {"cf": "ghost", "lc": []}],
                "cf_ignored": {}, "lc_unmapped": ["Array"]}
        errors = catalog.validate_crosswalk(walk, {"dp", "greedy"}, {"Dynamic Programming", "Array", "Trie"})
        self.assertEqual(errors, ["cf_tag_not_in_data:ghost", "cf_tag_unlisted:greedy", "lc_tag_unlisted:Trie"])

    def test_committed_crosswalk_is_well_formed(self):
        walk = json.loads((HERE / "tag_crosswalk.v1.json").read_text(encoding="utf-8"))
        relations = {m["relation"] for e in walk["cf_to_lc"] for m in e["lc"]}
        self.assertLessEqual(relations, {"equivalent", "narrower", "related"})
        cf_tags = [e["cf"] for e in walk["cf_to_lc"]]
        self.assertEqual(len(cf_tags), len(set(cf_tags)))


if __name__ == "__main__":
    unittest.main()
