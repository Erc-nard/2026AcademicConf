import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kvsr.missing import normalize_available, resample_availability
from kvsr.preprocessing import extract_mouths
from kvsr.batch import summarize_decisions

HAS_ARRAY = all(importlib.util.find_spec(k) for k in ('numpy', 'cv2'))
HAS_TORCH = HAS_ARRAY and importlib.util.find_spec('torch') is not None


class DecisionTests(unittest.TestCase):
    def test_abstention_counts_in_corpus_but_not_inference(self):
        refs = [dict(id='a',reference='가'),dict(id='b',reference='나')]
        rows = [dict(id='a',hypothesis='가',inference_performed=True),
                dict(id='b',hypothesis='',status='abstained_no_usable_roi',inference_performed=False)]
        report = summarize_decisions(refs, rows, [])
        self.assertEqual(report['inference_performed'], 1)
        self.assertEqual(report['abstained'], 1)
        self.assertEqual(report['all_selected_failure_as_empty']['cer'], .5)
        self.assertEqual(report['inferred_only']['cer'], 0.)
        rows[1]['hypothesis'] = '나'
        with self.assertRaises(ValueError):
            summarize_decisions(refs, rows, [])


@unittest.skipUnless(HAS_ARRAY, 'optional video dependencies')
class AvailabilityTests(unittest.TestCase):
    def test_resampling_uses_frame_centers_and_rejects_bad_mask(self):
        import numpy as np
        mask = np.array([True, False, True, False, True])
        expected = mask[np.floor((np.arange(6)+.5)*25/30).astype(int)]
        np.testing.assert_array_equal(resample_availability(mask, 5, 25, 6, 30), expected)
        np.testing.assert_array_equal(resample_availability(mask, 5, 25, 5, 25), mask)
        for bad in ([1,0,1,0,1], [True]):
            with self.assertRaises(ValueError):
                resample_availability(bad, 5, 25, 5, 25)

    def test_no_landmarks_preserves_frames_and_reasons_without_geometry(self):
        import numpy as np
        fake = MagicMock()
        detector = fake.tasks.vision.FaceLandmarker.create_from_options.return_value.__enter__.return_value
        detector.detect_for_video.return_value = SimpleNamespace(face_landmarks=[])
        frames = [np.zeros((224,224,3), dtype=np.uint8)] * 3
        with tempfile.TemporaryDirectory() as root, patch.dict('sys.modules', mediapipe=fake):
            model = Path(root)/'fake.task'
            model.write_bytes(b'test')
            mouths, trace = extract_mouths(frames, model, 25, missing_policy='mask',
                                           face_available=[False, True, True])
            self.assertEqual(mouths.shape, (3,96,96,3))
            self.assertTrue((mouths == 127).all())
            self.assertEqual(detector.detect_for_video.call_count, 2)
            self.assertEqual([r['reason'] for r in trace], ['track_missing','landmark_missing','landmark_missing'])
            self.assertTrue(all(r['center'] is None and not r['geometry_interpolated'] for r in trace))
            with self.assertRaises(ValueError):
                extract_mouths(frames, model, 25)
            with self.assertRaises(ValueError):
                extract_mouths(frames, model, 25, missing_policy='mask', landmark_max_gap=3)


@unittest.skipUnless(HAS_TORCH, 'optional torch/video dependencies')
class MaskedNormalizationTests(unittest.TestCase):
    def test_placeholders_cannot_change_observed_values_and_timeline_survives(self):
        import numpy as np
        import torch
        frames = np.arange(3*4*4*3, dtype=np.uint8).reshape(3,4,4,3)
        mask = np.array([True, False, True])
        first = normalize_available(frames, mask)
        frames[1] = 255
        second = normalize_available(frames, mask)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(first.shape, (1,3,3,4,4))
        self.assertTrue((first[:,:,1] == 0).all())
        with self.assertRaises(ValueError):
            normalize_available(frames, [False]*3)

    def test_all_observed_is_bitwise_identical_to_legacy_normalization(self):
        import numpy as np
        import torch
        frames = np.arange(4*4*4*3,dtype=np.uint8).reshape(4,4,4,3)
        values = torch.from_numpy(frames).float()
        legacy = ((values-values.mean())/values.std()).permute(3,0,1,2).unsqueeze(0)
        torch.testing.assert_close(normalize_available(frames, [True]*4), legacy, rtol=0, atol=0)

    def test_all_unavailable_abstains_before_loading_weights(self):
        import numpy as np
        from kvsr.olkavs import infer_olkavs
        builder = MagicMock()
        vocabulary = MagicMock()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root/'avsr/utils').mkdir(parents=True)
            (root/'avsr/utils/model_builder.py').write_text('')
            (root/'vocabulary').mkdir()
            (root/'vocabulary/kor_grapheme.csv').write_text('id,char\n'+'\n'.join(
                f'{i},'+('<unk>' if i == 3 else 'x') for i in range(72)), encoding='utf-8')
            checkpoint = root/'weights.pt'
            checkpoint.write_bytes(b'not real weights; must not load')
            mouth = root/'mouth.npz'
            np.savez(mouth, frames=np.full((2,96,96,3),127,np.uint8), fps=25,
                     missing_policy='mask', roi_observed=np.array([False,False]))
            with patch.dict('sys.modules', {'avsr.utils.model_builder':builder,'vocabulary.utils':vocabulary}), \
                 patch('kvsr.olkavs.subprocess.run', return_value=SimpleNamespace(stdout='test-revision')), \
                 patch('torch.load') as load:
                row = infer_olkavs(mouth, checkpoint, root, 'test', root/'output.jsonl')
                for decoder in ('ctc_beam', 'hybrid_rescore'):
                    deferred = infer_olkavs(mouth, checkpoint, root, 'test', root/'nbest.jsonl', decoder=decoder)
                    self.assertEqual(deferred['candidates'][0]['hypothesis'], '')
                    self.assertIsNone(deferred['candidates'][0]['log_mass'])
                    self.assertFalse(deferred['inference_performed'])
            load.assert_not_called()
            builder.build_model.assert_not_called()
            self.assertEqual(row['hypothesis'], '')
            self.assertFalse(row['inference_performed'])
            self.assertEqual(row['status'], 'abstained_no_usable_roi')
