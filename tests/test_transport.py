import importlib.util
from pathlib import Path
import tempfile
import unittest

from kvsr.transport_experiment import transform


@unittest.skipUnless(all(importlib.util.find_spec(k) for k in ('numpy', 'cv2')), 'optional video dependencies')
class TransportTests(unittest.TestCase):
    def test_control_matches_existing_input_and_linear_actually_differs(self):
        import numpy as np
        from kvsr.olkavs import prepare_frames
        frames = np.random.default_rng(2026).integers(0, 256, (3,224,224,3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as root:
            area, _ = transform(frames, 25., 'area', 'npz', Path(root))
            linear, _ = transform(frames, 25., 'linear', 'npz', Path(root))
        expected, _ = prepare_frames(frames, 25., 96, None)
        np.testing.assert_array_equal(area, expected)
        self.assertFalse(np.array_equal(linear, area))

    def test_codec_preserves_temporal_order_and_color_channels(self):
        import numpy as np
        # Distinct time steps and channels expose reversal, frame loss and RGB swaps.
        frames = np.stack([np.full((224,224,3), [20+i*20,70+i*20,120+i*20], dtype=np.uint8) for i in range(5)])
        with tempfile.TemporaryDirectory() as root:
            decoded, report = transform(frames, 25., 'area', 'mp4', Path(root))
        self.assertEqual(report['frames'], 5)
        np.testing.assert_allclose(decoded.mean(axis=(1,2)), frames.mean(axis=(1,2)), atol=8, rtol=0)
        self.assertTrue(report['timeline_preserved'])

    def test_invalid_inputs_fail_before_encoding(self):
        import numpy as np
        valid = np.zeros((2,224,224,3), dtype=np.uint8)
        for frames, fps, interpolation, transport in [
            (valid, 0, 'area', 'npz'), (valid[:0], 25, 'area', 'npz'),
            (valid.astype(float), 25, 'area', 'npz'), (valid, 25, 'cubic', 'npz'),
            (valid, 25, 'area', 'avi')]:
            with self.assertRaises(ValueError):
                transform(frames, fps, interpolation, transport, Path('.'))
