import tempfile
import unittest
import zipfile
from pathlib import Path

from kvsr.asd import metadata_rows, parse_asd, read_asd_archive


SAMPLE = """# Script: 안녕: 반가워
# ImageSize: 1280 720
# FPS: 25
# Audio Duration(ms): 40 160
# Index X Y W H
000002 10 20 30 40
000004 14 24 34 44
"""


class AsdTests(unittest.TestCase):
    def test_header_colon_frame_mapping_and_interpolation(self):
        item = parse_asd(SAMPLE, "aaaaaaaaaaa", "000001")
        self.assertEqual(item.reference, "안녕: 반가워")
        self.assertEqual(item.frame_range, (1, 4))
        self.assertEqual(item.box_at(3), (12, 22, 32, 42))
        self.assertEqual(item.box_at(1), (10, 20, 30, 40))
        with self.assertRaises(ValueError):
            item.box_at(10)

    def test_fractional_timing_covers_last_frame(self):
        item = parse_asd(SAMPLE.replace("40 160", "41 161"), "aaaaaaaaaaa", "000001")
        self.assertEqual(item.frame_range, (1, 5))

    def test_reject_reversed_times_and_duplicate_frames(self):
        for text in [SAMPLE.replace("40 160", "160 40"), SAMPLE.replace("000004", "000002")]:
            with self.assertRaises(ValueError):
                parse_asd(text, "aaaaaaaaaaa", "000001")

    def test_archive_filter_does_not_extract_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asd.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("kmsav_asd_v0.2/aaaaaaaaaaa/000001.txt", SAMPLE)
                archive.writestr("../../aaaaaaaaaaa/000002.txt", SAMPLE)
            self.assertEqual(len(read_asd_archive(path, "aaaaaaaaaaa")), 1)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_context_excludes_overlap_and_large_gap(self):
        items = [parse_asd(SAMPLE.replace("40 160", interval), "aaaaaaaaaaa", f"{i:06}")
                 for i, interval in enumerate(["40 160", "140 300", "320 400", "20000 21000"])]
        rows = metadata_rows(items, "train")
        self.assertEqual([r["context_time_valid"] for r in rows], [False, False, True, False])
        self.assertEqual(rows[2]["previous_available_id"], rows[1]["id"])


if __name__ == "__main__":
    unittest.main()
