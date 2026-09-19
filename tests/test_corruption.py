import importlib.util
import unittest

from kvsr.corruption import corrupt_frames


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional numpy dependency')
class CorruptionTests(unittest.TestCase):
    def test_exact_spatial_mask_and_source_unchanged(self):
        import numpy as np
        source = np.zeros((8, 8, 8, 3), dtype=np.uint8)
        for name, height in [('spatial25', 2), ('spatial50', 4)]:
            changed, info = corrupt_frames(source, name)
            self.assertEqual(np.count_nonzero(changed), 8*height*8*3)
            self.assertEqual(info['masked_fraction'], height/8)
            self.assertTrue((changed[:, (8-height)//2:(8+height)//2] == 127).all())
        self.assertFalse(source.any())

    def test_temporal_mask_no_frame_reordering_and_clean_identity(self):
        import numpy as np
        source = np.arange(8, dtype=np.uint8)[:, None, None, None] * np.ones((8, 4, 4, 3), dtype=np.uint8)
        changed, info = corrupt_frames(source, 'temporal25')
        self.assertTrue((changed[3:5] == 127).all())
        self.assertTrue((changed[:3] == source[:3]).all())
        self.assertTrue((changed[5:] == source[5:]).all())
        self.assertEqual(info['masked_fraction'], .25)
        clean, _ = corrupt_frames(source, 'none')
        self.assertTrue(np.array_equal(clean, source))
        self.assertFalse(np.shares_memory(clean, source))
