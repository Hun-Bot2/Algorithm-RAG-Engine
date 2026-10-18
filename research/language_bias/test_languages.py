import unittest

from research.language_bias import languages as L


def node(key, title):
    return {"problem_key": key, "title": title}


class Classification(unittest.TestCase):
    def test_classes(self):
        cases = {"Two Sum": "en", "Контрольная сумма": "ru", "한국어": "ko", "テスト用": "ja_pure",
                 "9.99.for文・演算子": "ja_other", "ミカン (Orange)": "ja_parallel", "ミカン（Orange）": "ja_parallel",
                 "山の高さ ( The height of a hill )": "ja_parallel", "X足すY課題": "ja_other"}
        for title, expected in cases.items():
            self.assertEqual(L.language_class(title), expected, title)

    def test_group_is_coarse(self):
        self.assertEqual(L.language_group("ミカン (Orange)"), "ja")
        self.assertEqual(L.language_group("テスト用"), "ja")
        self.assertEqual(L.language_group("Game"), "en")


class Parallel(unittest.TestCase):
    def test_extracts_halves_and_rejects_non_parallel(self):
        self.assertEqual(L.parse_parallel("伝説の寿司職人 (legendary sushi chef)"), ("伝説の寿司職人", "legendary sushi chef"))
        self.assertEqual(L.parse_parallel("月の満ち欠け（Phases of the Moon）"), ("月の満ち欠け", "Phases of the Moon"))
        self.assertEqual(L.parse_parallel("  模様替え ( Redecoration ) "), ("模様替え", "Redecoration"))
        for bad in ("Orange (ミカン)", "Two Sum", "テスト用", "ミカン (R)", "ミカン (日本語)", "ミカン (Orange) 2", "(Orange)"):
            self.assertIsNone(L.parse_parallel(bad), bad)


class Benchmark(unittest.TestCase):
    def setUp(self):
        self.nodes = [node("a:1", "ミカン (Orange)"), node("a:2", "ミカン (Orange)"), node("a:3", "染め (Dyeing)"),
                      node("a:4", "テスト用")] + [node(f"e:{i}", f"English {i}") for i in range(30)]

    def test_pairs_dedupe_and_keep_all_nodes(self):
        b = L.build_benchmark(self.nodes, distractors=10)
        self.assertEqual([(p["ja"], p["en"]) for p in b["pairs"]], [("ミカン", "Orange"), ("染め", "Dyeing")])
        self.assertEqual(b["pairs"][0]["nodes"], [0, 1])
        self.assertEqual(b["pair_node_indices"], [0, 1, 2])

    def test_distractors_are_english_nonpair_seeded_and_stable(self):
        b = L.build_benchmark(self.nodes, distractors=10)
        self.assertEqual(len(b["distractor_node_indices"]), 10)
        self.assertTrue(all(L.language_class(self.nodes[i]["title"]) == "en" for i in b["distractor_node_indices"]))
        self.assertEqual(b, L.build_benchmark(list(self.nodes), distractors=10))
        self.assertNotEqual(b["distractor_node_indices"], L.build_benchmark(self.nodes, seed=1, distractors=10)["distractor_node_indices"])
        self.assertTrue(set(b["distractor_node_indices"]).isdisjoint(b["pair_node_indices"]))

    def test_class_index_partitions_nodes(self):
        idx = L.class_index(self.nodes)
        self.assertEqual(sum(len(v) for v in idx.values()), len(self.nodes))
        self.assertEqual(idx["ja_pure"], [3])


if __name__ == "__main__":
    unittest.main()
