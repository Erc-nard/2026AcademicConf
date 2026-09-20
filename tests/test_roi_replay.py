import importlib.util
import unittest

from kvsr.roi_replay import mouth_rectangle, source_rectangle, replay_frame


class GeometryTests(unittest.TestCase):
    def test_pixel_edges_map_with_independent_horizontal_vertical_scales(self):
        rect, error = source_rectangle([10,20,448,112], (28,56,140,168), 640,480)
        self.assertEqual(rect, (66,48,290,104))
        self.assertEqual(error, 0)
        rect, error = source_rectangle([10,20,113,157], (1,2,98,99), 640,480)
        self.assertEqual(rect, (10,21,60,90))
        self.assertGreater(error, 0)
        self.assertLess(error, 1)

    def test_bad_geometry_and_source_bounds_are_rejected(self):
        for center, side in [([float('nan'),20],10), ([10,10],0), ([0,0],20), ([223,223],20)]:
            with self.assertRaises(ValueError):
                mouth_rectangle(center, side)
        for box in [[-1,0,224,224], [0,0,0,224], [600,0,224,224]]:
            with self.assertRaises(ValueError):
                source_rectangle(box, (0,0,96,96), 640,480)


@unittest.skipUnless(all(importlib.util.find_spec(k) for k in ('numpy','cv2')), 'optional video dependencies')
class ReplayTests(unittest.TestCase):
    def test_control_and_direct_paths_use_same_integer_mouth_bounds(self):
        import cv2
        import numpy as np
        from kvsr.preprocessing import square_crop
        frame = np.random.default_rng(2026).integers(0,256,(300,400,3),dtype=np.uint8)
        geometry = dict(center=[111.8,157.3], side=71.4, align='none')
        paths, original, error = replay_frame(frame, [50,30,224,224], geometry)
        face = frame[30:254,50:274]
        expected = square_crop(face, geometry['center'], geometry['side'],224)
        np.testing.assert_array_equal(original, expected)
        np.testing.assert_array_equal(paths['control'], cv2.resize(expected,(96,96),interpolation=cv2.INTER_AREA))
        np.testing.assert_array_equal(paths['face_direct'], paths['source_direct'])
        self.assertEqual(error, 0)
        self.assertFalse(np.array_equal(paths['control'], paths['face_direct']))

    def test_missing_geometry_does_not_invent_or_reuse_mouth_pixels(self):
        import numpy as np
        paths, original, error = replay_frame(np.zeros((300,400,3),np.uint8), None, dict(center=None))
        self.assertEqual(error, 0)
        self.assertTrue((original==127).all())
        for value in paths.values():
            self.assertEqual(value.shape, (96,96,3))
            self.assertTrue((value==127).all())

    def test_rotated_and_interpolated_geometry_cannot_be_replayed_as_plain_crop(self):
        import numpy as np
        frame = np.zeros((300,400,3),np.uint8)
        for extra in [dict(align='eyes'),dict(geometry_interpolated=True)]:
            with self.assertRaises(ValueError):
                replay_frame(frame, [0,0,224,224], dict(center=[100,100],side=50,**extra))
