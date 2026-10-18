import unittest

from research.language_bias import translate as T


def fake_mt(texts):
    return [f"EN({t})" for t in texts]


class Provenance(unittest.TestCase):
    def test_each_class_uses_the_expected_source(self):
        self.assertEqual(T.translate_title("Two Sum", "en", None), ("Two Sum", "identity"))
        self.assertEqual(T.translate_title("ミカン (Orange)", "ja_parallel", None), ("Orange", "parenthetical"))
        self.assertEqual(T.translate_title("テスト用", "ja_pure", fake_mt), ("EN(テスト用)", "machine"))
        self.assertEqual(T.translate_title("X足すY課題", "ja_other", fake_mt), ("EN(X足すY課題)", "machine"))
        self.assertEqual(T.translate_title("Контрольная сумма", "ru", None), ("Контрольная сумма", "untranslated"))
        self.assertEqual(T.translate_title("한국어", "ko", None), ("한국어", "untranslated"))

    def test_machine_translation_is_required_for_japanese_only_titles(self):
        with self.assertRaises(ValueError):
            T.translate_title("テスト用", "ja_pure", None)

    def test_empty_machine_output_falls_back_and_is_marked(self):
        self.assertEqual(T.translate_title("テスト用", "ja_pure", lambda t: ["  "]), ("テスト用", "machine_empty_fallback"))


class Batch(unittest.TestCase):
    def test_mt_runs_once_per_distinct_japanese_title_in_fixed_order(self):
        titles = ["テスト用", "Game", "ミカン (Orange)", "テスト用", "あ"]
        classes = ["ja_pure", "en", "ja_parallel", "ja_pure", "ja_pure"]
        calls = []

        def mt(batch):
            calls.append(list(batch))
            return fake_mt(batch)

        rows = T.translate_all(titles, classes, mt, batch_size=2)
        self.assertEqual(sum(calls, []), sorted({"テスト用", "あ"}))
        self.assertEqual([r["text"] for r in rows], ["EN(テスト用)", "Game", "Orange", "EN(テスト用)", "EN(あ)"])
        self.assertEqual([r["source"] for r in rows], ["machine", "identity", "parenthetical", "machine", "machine"])
        counts = T.provenance_counts(rows, classes)
        self.assertEqual(counts["by_source"], {"identity": 1, "machine": 3, "parenthetical": 1})

    def test_no_machine_calls_when_none_needed_and_length_mismatch_fails(self):
        self.assertEqual(T.translate_all(["Game"], ["en"], None)[0]["source"], "identity")
        with self.assertRaises(ValueError):
            T.translate_all(["あ", "い"], ["ja_pure", "ja_pure"], lambda b: ["x"])
        with self.assertRaises(ValueError):
            T.translate_all(["あ"], ["ja_pure"], None)


if __name__ == "__main__":
    unittest.main()
