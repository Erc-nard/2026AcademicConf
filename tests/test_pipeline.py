import tempfile
import unittest
import unicodedata
from pathlib import Path

from kvsr.data import build_manifest, read_catalog, select_subset
from kvsr.evaluation import evaluate


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = self.root / "list.txt"
        self.catalog.write_text(
            "# id domain participants seconds split\n"
            "aaaaaaaaaaa issu 2 10 train\n"
            "bbbbbbbbbbb issu 2 20 test\n"
            "ccccccccccc inte 3 30 train license_error\n"
            "ddddddddddd issu 2 0 train\n"
            "eeeeeeeeeee inte 2 40 train\n", encoding="utf-8")

    def test_selection_preserves_split_and_excludes_flags(self):
        rows = read_catalog(self.catalog)
        selected = select_subset(rows, "train", 2, 2026)
        self.assertEqual({r["video_id"] for r in selected}, {"aaaaaaaaaaa", "eeeeeeeeeee"})
        self.assertEqual(selected, select_subset(list(reversed(rows)), "train", 2, 2026))
        with self.assertRaises(ValueError):
            select_subset(rows, "train", 3, 2026)

    def test_duplicate_catalog_id_fails(self):
        with self.catalog.open("a", encoding="utf-8") as handle:
            handle.write("aaaaaaaaaaa issu 2 10 test\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_catalog(self.catalog)

    def test_manifest_rejects_unsafe_or_duplicate_subset(self):
        for rows in [[dict(video_id="../outside", split="train")],
                     [dict(video_id="aaaaaaaaaaa", split="train")] * 2]:
            with self.assertRaises(ValueError):
                build_manifest(rows, self.root)

    def test_manifest_order_no_cross_video_context_and_face_label(self):
        records = read_catalog(self.catalog)[:2]
        for record in records:
            folder = self.root / record["video_id"] / "utts"
            folder.mkdir(parents=True)
            for index in [10, 2]:
                (folder / f"{index:06}.txt").write_text("안녕하세요", encoding="utf-8")
                # File-index fixture only; deliberately not a decodable video.
                (folder / f"{index:06}.mp4").write_bytes(b"synthetic fixture")
        rows = build_manifest(records, self.root)
        self.assertEqual([r["utterance_index"] for r in rows], [2, 10, 2, 10])
        self.assertIsNone(rows[0]["previous_available_id"])
        self.assertIsNone(rows[2]["previous_available_id"])
        self.assertEqual(rows[1]["previous_available_id"], rows[0]["id"])
        self.assertIsNone(rows[0]["mouth_roi"])
        Path(rows[0]["face_video"]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            build_manifest(records, self.root)


class EvaluationTests(unittest.TestCase):
    def test_corpus_weighting_and_known_edits(self):
        result = evaluate([dict(id="1", reference="가나다라"), dict(id="2", reference="마")],
                          [dict(id="1", hypothesis="가나다라"), dict(id="2", hypothesis="바")])
        self.assertEqual(result["cer"], 1 / 5)
        self.assertEqual(result["wer"], 1 / 2)

    def test_hangul_nfc_and_whitespace(self):
        result = evaluate([dict(id="1", reference="학교 가요")],
                          [dict(id="1", hypothesis=unicodedata.normalize("NFD", "학교  가요"))])
        self.assertEqual(result["cer"], 0)
        self.assertEqual(result["wer"], 0)

    def test_insertions_can_exceed_one(self):
        self.assertEqual(evaluate([dict(id="1", reference="가")],
                                 [dict(id="1", hypothesis="가나다라")])["cer"], 3)

    def test_empty_hypothesis_counts_deletions(self):
        self.assertEqual(evaluate([dict(id="1", reference="학교")],
                                 [dict(id="1", hypothesis="")])["cer"], 1)

    def test_ids_must_match_without_duplicates(self):
        refs = [dict(id="1", reference="가")]
        for predictions in [[dict(id="2", hypothesis="가")],
                            [dict(id="1", hypothesis="가")] * 2, []]:
            with self.assertRaises(ValueError):
                evaluate(refs, predictions)

    def test_empty_reference_corpus_fails(self):
        with self.assertRaises(ValueError):
            evaluate([dict(id="1", reference="")], [dict(id="1", hypothesis="가")])


if __name__ == "__main__":
    unittest.main()
