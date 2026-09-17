import importlib.util
import unittest

from kvsr.batch import summarize, paired_comparisons
from kvsr.olkavs import prepare_frames, chunk_frontend


class CoverageTests(unittest.TestCase):
    def test_pairwise_matching_survives_an_all_failed_condition(self):
        refs = [dict(id="a", reference="가나"), dict(id="b", reference="다라")]
        predictions = dict(first=[dict(id="a", hypothesis="가나"), dict(id="b", hypothesis="다라")],
                           second=[dict(id="a", hypothesis="가다")], failed=[])
        pairs = paired_comparisons(refs, predictions)
        self.assertEqual(pairs["first__vs__second"]["ids"], ["a"])
        self.assertEqual(pairs["first__vs__second"]["metrics"]["second"]["cer"], .5)
        self.assertIsNone(pairs["first__vs__failed"]["metrics"])
    def test_failures_are_not_silently_excluded(self):
        refs = [dict(id="a", reference="가나"), dict(id="b", reference="다라")]
        result = summarize(refs, [dict(id="a", hypothesis="가나")], [dict(id="b", error="missing face")])
        self.assertEqual(result["coverage"], .5)
        self.assertEqual(result["successful_only"]["cer"], 0)
        self.assertEqual(result["all_selected_failure_as_empty"]["cer"], .5)

    def test_missing_duplicate_or_both_failed_and_successful_ids_rejected(self):
        with self.assertRaises(ValueError):
            summarize([], [], [])
        refs = [dict(id="a", reference="가")]
        for predictions, failures in [([], []), ([dict(id="a", hypothesis="가")] * 2, []),
                                      ([dict(id="a", hypothesis="가")], [dict(id="a")])]:
            with self.assertRaises(ValueError):
                summarize(refs, predictions, failures)


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"), "Video deps required")
class ResamplingTests(unittest.TestCase):
    def test_native_rate_is_identity(self):
        import numpy as np
        frames = np.arange(12, dtype=np.uint8)[:, None, None, None] * np.ones((12, 96, 96, 3), dtype=np.uint8)
        actual, fps = prepare_frames(frames, 25, 96, None)
        np.testing.assert_array_equal(frames, actual)
        self.assertEqual(fps, 25)

    def test_duration_preserved_and_values_are_source_frames(self):
        import numpy as np
        frames = np.arange(25, dtype=np.uint8)[:, None, None, None] * np.ones((25, 96, 96, 3), dtype=np.uint8)
        actual, fps = prepare_frames(frames, 25, 224, 30)
        self.assertEqual(actual.shape, (30, 224, 224, 3))
        self.assertEqual(len(actual) / fps, len(frames) / 25)
        self.assertEqual(actual[0, 0, 0, 0], 0)
        self.assertEqual(actual[-1, 0, 0, 0], 24)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
class ChunkTests(unittest.TestCase):
    def test_temporal_halo_matches_full_frontend_including_boundaries(self):
        import torch
        class Frontend(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv3d(3, 4, (5, 3, 3), padding=(2, 1, 1))
            def forward(self, x):
                return self.conv(x).mean(dim=(-2, -1)).transpose(1, 2)
        torch.manual_seed(0)
        model = Frontend().eval()
        video = torch.randn(1, 3, 17, 8, 8)
        with torch.inference_mode():
            torch.testing.assert_close(chunk_frontend(model, video, chunk=6), model(video))
        model.train()
        with self.assertRaises(ValueError):
            chunk_frontend(model, video)


if __name__ == "__main__":
    unittest.main()
