import unittest

from research.language_bias import html_v2 as H


def candidate(obs, chance, cons, mrr, retention=1.0):
    return {"homophily": {"ja_pure": {"observed": obs, "chance": chance}},
            "cross_lingual_neighbor_consistency": {"overlap": {"mean": cons}},
            "translation_retrieval": {"mrr": {"mean": mrr}}, "english_quality_retention": retention}


def maps(island_before, island_after):
    side = lambda v: {"maps": {m: {"island_index_ja_pure_2d": v} for m in ("umap", "tsne")}}
    return {"runs": {"v1_minilm_raw_title": side(island_before), "v2_minilm_translated_title": side(island_after)}}


class Summary(unittest.TestCase):
    def test_rows_report_before_after_chance_and_caveats(self):
        evaluation = {"candidates": {"C0": candidate(0.988, 0.091, 0.016, 0.039), "C2": candidate(0.252, 0.091, 0.506, 0.664, 1.008)}}
        s = H.language_bias_summary(evaluation, maps(0.97, 0.10))
        by_label = {r["label"].strip(): r for r in s["rows"]}
        self.assertEqual(by_label["Pure-Japanese titles: neighbors that are also Japanese (384-d)"]["before"], "98.8%")
        self.assertEqual(by_label["Pure-Japanese titles: neighbors that are also Japanese (384-d)"]["after"], "25.2%")
        self.assertEqual(by_label["language-blind chance level"]["before"], by_label["language-blind chance level"]["after"])
        self.assertEqual(by_label["UMAP: pure-Japanese titles in a Japanese-only 2D neighborhood"]["before"], "97.0%")
        self.assertEqual(by_label["English-title neighbor quality retained (tag lift ratio)"]["after"], "1.008")
        self.assertTrue(any("random control" in c for c in s["caveats"]))
        self.assertTrue(any("Russian and Korean" in c for c in s["caveats"]))

    def test_pct_handles_missing_values(self):
        self.assertEqual(H.pct(None), "n/a")
        self.assertEqual(H.pct(0.0505), "5.1%")


if __name__ == "__main__":
    unittest.main()
