import json
import re
import unittest
from pathlib import Path

import numpy as np

from research.unified_map import full_html as fh

TEMPLATE = Path(fh.TEMPLATE)


def node(key, platform, title, tags=None, contest=None, url=None, eligibility="eligible", reasons=(),
         group=None, concepts=(), stmt="not_available_in_admitted_corpus", difficulty=None):
    return {"problem_key": key, "platform": platform, "title": title, "title_normalized": title.lower(),
            "eligibility": eligibility, "exclusion_reasons": list(reasons), "native_difficulty": difficulty,
            "contest": contest, "url": url if url is not None else fh.derive_url(
                {"problem_key": key, "platform": platform, "contest": contest}),
            "tags": tags, "duplicate_candidate_group": group,
            "normalized_taxonomy": {"mapped_concepts": [
                {"concept_id": c, "relation": "equivalent", "confidence": 0.97, "tag": c} for c in concepts]},
            "statement_availability": {"status": stmt}}


def fixture(n=12):
    base = [node("atcoder:abc001_a", "atcoder", "Hello </script><b>x", contest="abc001"),
            node("codeforces:1000A", "codeforces", "Game", tags=["greedy"], concepts=["greedy"], group="g1",
                 stmt="publisher_statement_available", difficulty=800),
            node("codeforces:1001C", "codeforces", "Game", tags=["greedy", "dp"], concepts=["greedy", "dp"], group="g1"),
            node("leetcode:two-sum", "leetcode", "Two Sum", tags=["Array"], concepts=[]),
            node("leetcode:sql", "leetcode", "Combine Two Tables", tags=["Database"], eligibility="excluded",
                 reasons=["lc_non_algorithmic_tag"])]
    while len(base) < n:
        base.append(node(f"atcoder:arc{len(base):03d}_a", "atcoder", f"Task {len(base)}", contest=f"arc{len(base):03d}"))
    return sorted(base, key=lambda x: x["problem_key"])


def build(n=12):
    nodes = fixture(n)
    rng = np.random.default_rng(0)
    meta = [{"problem_key": x["problem_key"], "reliability": "reliable", "unk_share": 0.0,
             "vector_source": "a1_reused"} for x in nodes]
    meta[0]["reliability"], meta[0]["unk_share"], meta[0]["vector_source"] = "degraded", 0.3, "new_title_encode"
    idx = np.array([[(i + 1 + j) % n for j in range(10)] for i in range(n)], dtype=np.int32)
    score = np.sort(rng.uniform(0.2, 0.9, size=(n, 10)), axis=1)[:, ::-1].astype(np.float32)
    coords = {name: rng.normal(size=(n, 2)).astype(np.float32) for name in fh.PROJECTIONS}
    pres = {name: rng.integers(0, 11, size=n).astype(np.uint8) for name in fh.PROJECTIONS}
    keys = [x["problem_key"] for x in nodes]
    labels = [{"query": keys[1], "target": keys[2], "label": 1, "deceptive": True},
              {"query": keys[3], "target": keys[1], "label": None, "deceptive": None}]
    history = {"judgments": [
        {"source": "atcoder:abc049_c", "target": keys[3], "type": "similarity", "relevance": 2, "evidence": "same trick"},
        {"source": "leetcode:missing", "target": keys[3], "type": "transfer", "relevance": 1, "evidence": "x"}]}
    aliases = {"atcoder:abc049_c": keys[0]}
    quality = {"projections": {}, "stability": {}, "high_dimensional": {}, "trust_query_sample": {"size": 4}}
    payload = fh.build_payload(nodes, meta, idx, score, idx, score, coords, pres, quality, labels, history, aliases,
                               {"model": "fixture"})
    return nodes, payload, idx, score, coords, pres


class Packing(unittest.TestCase):
    def test_round_trip_all_dtypes_and_byte_order(self):
        cases = {"u8": np.array([[0, 255], [7, 8]]), "i16": np.array([-32768, -1, 0, 32767]),
                 "u16": np.array([0, 65535, 300]), "u32": np.array([0, 4_000_000_000]),
                 "i32": np.array([-5, 2_000_000_000]), "f32": np.array([[0.25, -1.5], [3.0, 1e-3]])}
        for dtype, array in cases.items():
            record = fh.pack_array(array, dtype)
            self.assertEqual(record["shape"], list(np.asarray(array).shape))
            np.testing.assert_array_equal(fh.unpack_array(record), np.asarray(array).astype(fh.DTYPES[dtype]))
        self.assertEqual(fh.pack_array(np.array([1], dtype=np.uint16), "u16")["b64"], "AQA=")  # little-endian

    def test_scores_quantize_clip_and_reject_nan(self):
        self.assertEqual(fh.quantize_scores(np.array([0.5, 1.2, -1.5, 0.12346])).tolist(), [5000, 10000, -10000, 1235])
        with self.assertRaises(ValueError):
            fh.quantize_scores(np.array([np.nan]))

    def test_coordinates_normalize_per_axis_and_handle_constant_axis(self):
        out = fh.normalize_coordinates(np.array([[1.0, 5.0], [3.0, 5.0], [2.0, 5.0]]))
        self.assertEqual(out[:, 0].tolist(), [0.0, 1.0, 0.5])
        self.assertEqual(out[:, 1].tolist(), [0.0, 0.0, 0.0])


class Derivations(unittest.TestCase):
    def test_urls_match_each_platform_and_exceptions_are_detected(self):
        self.assertEqual(fh.derive_url(node("leetcode:two-sum", "leetcode", "T")), "https://leetcode.com/problems/two-sum/")
        self.assertEqual(fh.derive_url(node("codeforces:1275E1", "codeforces", "T")),
                         "https://codeforces.com/problemset/problem/1275/E1")
        self.assertEqual(fh.derive_url(node("atcoder:abc084_c", "atcoder", "T", contest="abc084")),
                         "https://atcoder.jp/contests/abc084/tasks/abc084_c")
        self.assertEqual(fh.derive_url(node("atcoder:arc065_a", "atcoder", "T", contest="abc049")),
                         "https://atcoder.jp/contests/abc049/tasks/arc065_a")
        nodes, payload, *_ = build()
        self.assertEqual(payload["url_exceptions"], {})

    def test_title_and_duplicate_groups(self):
        nodes = fixture()
        tg = fh.title_groups(nodes)
        keys = [x["problem_key"] for x in nodes]
        game = [i for i, k in enumerate(keys) if k.startswith("codeforces:100")]
        self.assertEqual(len({int(tg[i]) for i in game}), 1)
        self.assertTrue(tg[game[0]] >= 0)
        self.assertEqual(int(tg[keys.index("leetcode:two-sum")]), -1)
        ids, members = fh.duplicate_groups(nodes)
        self.assertEqual(members, [game])
        self.assertEqual([int(ids[i]) for i in game], [0, 0])

    def test_pair_and_history_resolution_uses_reviewed_aliases(self):
        keys = {"a": 0, "b": 1, "native": 2}
        rows = fh.judged_pairs([{"query": "a", "target": "b", "label": None, "deceptive": None},
                                {"query": "b", "target": "a", "label": 2, "deceptive": False}], keys)
        self.assertEqual(rows, [[0, 1, 255, 255], [1, 0, 2, 0]])
        history = {"judgments": [{"source": "legacy", "target": "b", "type": "transfer", "relevance": 3, "evidence": "e"},
                                 {"source": "zzz", "target": "b", "type": "similarity", "relevance": 0, "evidence": "e"}]}
        out, unresolved = fh.history_pairs(history, {"legacy": "native"}, keys)
        self.assertEqual(out, [[2, 1, 2, 3, "e", 1]])
        self.assertEqual(unresolved, 1)


class Payload(unittest.TestCase):
    def test_schema_every_node_once_and_arrays_decode(self):
        nodes, payload, idx, score, coords, pres = build()
        n = len(nodes)
        self.assertEqual(payload["n"], n)
        self.assertEqual(payload["keys"], [x["problem_key"] for x in nodes])
        self.assertEqual(len(set(payload["keys"])), n)
        self.assertEqual(fh.unpack_array(payload["platform"]).tolist(),
                         [fh.PLATFORMS.index(x["platform"]) for x in nodes])
        np.testing.assert_array_equal(fh.unpack_array(payload["neighbors"]["any_idx"]), idx.astype(np.uint16))
        np.testing.assert_allclose(fh.unpack_array(payload["neighbors"]["any_score"]) / 1e4, score, atol=1e-4)
        for name in fh.PROJECTIONS:
            xy = fh.unpack_array(payload["projections"][name]["coords"])
            self.assertEqual(xy.shape, (n, 2))
            self.assertTrue(((xy >= 0) & (xy <= 1)).all())
            np.testing.assert_array_equal(fh.unpack_array(payload["projections"][name]["preserved_top10"]), pres[name])

    def test_tags_concepts_flags_and_judgments(self):
        nodes, payload, *_ = build()
        keys = payload["keys"]
        off, ids = fh.unpack_array(payload["tags"]["offsets"]), fh.unpack_array(payload["tags"]["ids"])
        i = keys.index("codeforces:1001C")
        self.assertEqual(sorted(payload["tags"]["vocab"][t] for t in ids[off[i]:off[i + 1]]), ["dp", "greedy"])
        a = keys.index("atcoder:abc001_a")
        self.assertEqual(off[a], off[a + 1])                      # AtCoder: no tags at all
        self.assertEqual(payload["concepts"]["mapped"][a], [])
        eligible = fh.unpack_array(payload["eligible"])
        self.assertEqual(int(eligible[keys.index("leetcode:sql")]), 0)
        self.assertEqual(payload["excluded_reasons"]["by_node"][str(keys.index("leetcode:sql"))], [0])
        self.assertEqual(fh.unpack_array(payload["reliability"])[a], 1)
        self.assertEqual(fh.unpack_array(payload["vector_source"])[a], 1)
        self.assertEqual(fh.unpack_array(payload["statement_available"])[keys.index("codeforces:1000A")], 1)
        self.assertEqual(len(payload["judged_pairs"]), 2)
        self.assertEqual(len(payload["history"]), 1)
        self.assertEqual(payload["history_unresolved"], 1)

    def test_payload_has_no_statement_fields(self):
        _, payload, *_ = build()
        text = json.dumps(payload)
        for forbidden in ("content_clean", "full_statement", "statement_text", "embedding_text"):
            self.assertNotIn(forbidden, text)
        self.assertFalse({"statement", "statements", "content"} & set(payload))

    def test_meta_order_mismatch_is_rejected(self):
        nodes = fixture()
        meta = [{"problem_key": x["problem_key"], "reliability": "reliable", "unk_share": 0, "vector_source": "a1_reused"}
                for x in reversed(nodes)]
        with self.assertRaises(ValueError):
            fh.build_payload(nodes, meta, np.zeros((12, 10), dtype=np.int32), np.zeros((12, 10), dtype=np.float32),
                             np.zeros((12, 10), dtype=np.int32), np.zeros((12, 10), dtype=np.float32), {}, {}, {}, [],
                             {"judgments": []}, {}, {})


class Extras(unittest.TestCase):
    def test_extras_are_merged_and_cannot_overwrite_core_fields(self):
        nodes = fixture()
        meta = [{"problem_key": x["problem_key"], "reliability": "reliable", "unk_share": 0.0, "vector_source": "a1_reused"}
                for x in nodes]
        n = len(nodes)
        idx = np.array([[(i + 1 + j) % n for j in range(10)] for i in range(n)], dtype=np.int32)
        score = np.full((n, 10), 0.5, dtype=np.float32)
        coords = {name: np.zeros((n, 2), dtype=np.float32) for name in fh.PROJECTIONS}
        pres = {name: np.zeros(n, dtype=np.uint8) for name in fh.PROJECTIONS}
        args = (nodes, meta, idx, score, idx, score, coords, pres, {}, [], {"judgments": []}, {}, {})
        payload = fh.build_payload(*args, extras={"language": {"vocab": ["en"]}})
        self.assertEqual(payload["language"], {"vocab": ["en"]})
        with self.assertRaises(ValueError):
            fh.build_payload(*args, extras={"keys": []})

    def test_template_guards_every_optional_field(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        for guard in ("P.language ?", "P.embedded ?", "P.representation ||", "const lb = P.language_bias; if (!lb) return ''",
                      "if (LANG) $('optLanguage').hidden = false", "if (REP && REP.banner)"):
            self.assertIn(guard, text)


class Html(unittest.TestCase):
    def test_render_embeds_json_safely_and_round_trips(self):
        _, payload, *_ = build()
        html = fh.render_html(payload, "<script id='payload' type='application/json'>" + fh.PLACEHOLDER + "</script>")
        body = re.search(r"<script id='payload' type='application/json'>(.*)</script>", html, re.S).group(1)
        self.assertNotIn("</script", body)
        parsed = json.loads(body)
        self.assertEqual(parsed["titles"][0], payload["titles"][0])
        self.assertIn("</script><b>", parsed["titles"][0] + "</script><b>")  # original hostile text survives
        with self.assertRaises(ValueError):
            fh.render_html(payload, "no placeholder")

    def test_real_template_is_offline_and_states_the_caveats(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        self.assertEqual(text.count(fh.PLACEHOLDER), 1)
        self.assertIsNone(re.search(r"<script[^>]+src=", text))
        self.assertIsNone(re.search(r"<link[^>]+href=", text))
        self.assertIsNone(re.search(r"fetch\(|XMLHttpRequest|importScripts|WebSocket", text))
        for phrase in ("title only", "AtCoder publishes no", "not that two problems share an algorithm",
                       "never used to place points", "unknown, not negative", "Tag evidence", "Semantic title similarity"):
            self.assertIn(phrase, text)
        self.assertIsNone(re.search(r"same algorithm\b(?!, invariant)(?! or)", text.replace("share an algorithm, invariant", "")),
                          "no unqualified claim of algorithm equivalence")


if __name__ == "__main__":
    unittest.main()
