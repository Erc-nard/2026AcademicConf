import importlib.util
import unittest

from kvsr.preprocessing import fill_geometry_gaps


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional numpy dependency')
class TrackingGapTests(unittest.TestCase):
    def test_bounded_coordinate_interpolation_preserves_observations(self):
        import numpy as np
        left, right = (np.array([2., 4.]), 10.), (np.array([8., 10.]), 16.)
        source = [left, None, None, right]
        filled = fill_geometry_gaps(source, 2)
        self.assertIs(filled[0], left)
        self.assertIs(filled[-1], right)
        self.assertTrue(np.array_equal(filled[1][0], [4., 6.]))
        self.assertTrue(np.array_equal(filled[2][0], [6., 8.]))
        self.assertEqual(filled[1][1], 12.)
        self.assertIsNone(source[1])

    def test_rejects_boundary_long_and_disabled_gaps(self):
        observed = ([2., 4.], 10.)
        for source, gap in [([None, observed], 3), ([observed, None], 3),
                            ([observed, None, None, observed], 1),
                            ([observed, None, observed], 0), ([None], 3)]:
            with self.assertRaises(ValueError):
                fill_geometry_gaps(source, gap)
