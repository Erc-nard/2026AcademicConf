import importlib.util
import tempfile
import unittest
from pathlib import Path

from kvsr.asd import parse_asd
from kvsr.preprocessing import extract_faces, square_crop, roll_transform


HAS_VIDEO = all(importlib.util.find_spec(name) for name in ["numpy", "cv2", "imageio_ffmpeg"])


@unittest.skipUnless(HAS_VIDEO, "Install requirements-preprocessing.txt for video integration tests")
class PreprocessingTests(unittest.TestCase):
    def test_eye_rotation_levels_line_and_marks_padding(self):
        import cv2
        import numpy as np
        frame = np.full((100, 100, 3), 160, np.uint8)
        eyes = np.array([[25., 35.], [75., 55.]])
        rotated, valid, matrix, angle = roll_transform(frame, *eyes)
        transformed = cv2.transform(eyes[None], matrix)[0]
        self.assertAlmostEqual(transformed[0, 1], transformed[1, 1], places=5)
        self.assertGreater(angle, 0)
        self.assertEqual(rotated.shape, frame.shape)
        self.assertEqual(valid[50, 50], 255)
        self.assertEqual(valid.min(), 0)

    def test_zero_roll_preserves_pixels_and_invalid_eyes_fail(self):
        import numpy as np
        frame = np.full((100, 100, 3), 160, np.uint8)
        rotated, valid, _, angle = roll_transform(frame, (20., 30.), (80., 30.))
        np.testing.assert_array_equal(frame, rotated)
        self.assertEqual(valid.min(), 255)
        self.assertEqual(angle, 0)
        with self.assertRaises(ValueError):
            roll_transform(frame, (20., 30.), (20., 30.))

    def test_frame_timing_and_face_box_on_decodable_video(self):
        import cv2
        import imageio_ffmpeg
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "source.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
            self.assertTrue(writer.isOpened())
            try:
                for number in range(10):
                    writer.write(np.full((48, 64, 3), number * 20, dtype=np.uint8))
            finally:
                writer.release()
            text = "# Script: 가\n# ImageSize: 64 48\n# FPS: 25\n# Audio Duration(ms): 80 200\n"
            text += "3 10 10 20 20\n4 10 10 20 20\n5 10 10 20 20\n"
            utterance = parse_asd(text, "aaaaaaaaaaa", "000001")
            faces, track = extract_faces(video, utterance, imageio_ffmpeg.get_ffmpeg_exe())
            self.assertEqual(len(faces), 3)
            self.assertEqual([r["source_frame_one_based"] for r in track], [3, 4, 5])
            for frame, expected in zip(faces, [40, 60, 80]):
                self.assertLess(abs(float(frame.mean()) - expected), 3)
            from dataclasses import replace
            gappy = replace(utterance, start_ms=0, end_ms=400,
                            boxes=((6,10,10,20,20),(7,10,10,20,20)))
            masked, trace = extract_faces(video, gappy, imageio_ffmpeg.get_ffmpeg_exe(), missing_policy="mask")
            self.assertEqual([r["source_frame_one_based"] for r in trace], list(range(1,11)))
            self.assertEqual([r["available"] for r in trace], [False]*3+[True]*6+[False])
            for index, frame in enumerate(masked):
                if trace[index]["available"]:
                    self.assertLess(abs(float(frame.mean()) - index*20), 3)
                else:
                    self.assertTrue((frame == 127).all())
                    self.assertEqual(trace[index]["reason"], "track_missing")
            with self.assertRaises(ValueError):
                extract_faces(video, gappy, imageio_ffmpeg.get_ffmpeg_exe())

    def test_crop_rejects_border_and_preserves_center_pixels(self):
        import numpy as np
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[40:60, 40:60] = 255
        crop = square_crop(frame, (50, 50), 20, 96)
        self.assertEqual(crop.shape, (96, 96, 3))
        self.assertEqual(float(crop.mean()), 255)
        with self.assertRaises(ValueError):
            square_crop(frame, (0, 0), 20, 96)


if __name__ == "__main__":
    unittest.main()
