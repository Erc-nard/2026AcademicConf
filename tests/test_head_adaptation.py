import importlib.util
from pathlib import Path
import tempfile
import unittest

from kvsr.head_adaptation import require_ctc_alignment,build_targets,read_cached_feature
from kvsr.preprocessing import sha256


class CTCTargetTests(unittest.TestCase):
    def test_repeated_labels_require_intervening_blank_frame(self):
        self.assertEqual(require_ctc_alignment([5,5,2],4),4)
        with self.assertRaises(ValueError):
            require_ctc_alignment([5,5,2],3)

    def test_empty_or_special_training_targets_are_rejected(self):
        for tokens in [[],[0,2],[1,2],[3,2]]:
            with self.assertRaises(ValueError):
                require_ctc_alignment(tokens,20)

    def test_normalization_precedes_tokenization_and_eos_is_retained(self):
        target=build_targets('가!',lambda text:text.replace('!',''),lambda text:list(text),{'가':5},2)
        self.assertEqual(target,[5,2])
        with self.assertRaises(ValueError):
            build_targets('!',lambda text:'',list,{},20)

    def test_unknown_labels_are_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            build_targets('가?',lambda text:text,list,{'가':5},20)


@unittest.skipUnless(importlib.util.find_spec('numpy'),'optional numpy dependency')
class FeatureCacheTests(unittest.TestCase):
    def test_cache_requires_matching_checkpoint_input_and_feature_bytes(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            (root/'features').mkdir()
            path=root/'features/video_000001.npy'
            expected=np.ones((3,256),dtype=np.float32)
            np.save(path,expected)
            summary=dict(plan=dict(checkpoint_sha256='model',inputs=[dict(id='video/000001',mouth_sha256='mouth')]),
                         feature_hashes={path.name:sha256(path)})
            actual=read_cached_feature(root,summary,'video/000001','mouth','model',3)
            np.testing.assert_array_equal(actual,expected)
            self.assertIsNone(read_cached_feature(root,summary,'video/000002','new','model',3))
            for mouth,model,frames in [('changed','model',3),('mouth','changed',3),('mouth','model',4)]:
                with self.assertRaises(ValueError):
                    read_cached_feature(root,summary,'video/000001',mouth,model,frames)
            np.save(path,expected+1)
            with self.assertRaises(ValueError):
                read_cached_feature(root,summary,'video/000001','mouth','model',3)

    def test_nonfinite_and_wrong_dtype_features_are_rejected_even_with_matching_hash(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            (root/'features').mkdir()
            path=root/'features/video_000001.npy'
            for data in [np.full((3,256),np.nan,np.float32),np.ones((3,256),np.float64)]:
                np.save(path,data)
                summary=dict(plan=dict(checkpoint_sha256='model',inputs=[dict(id='video/000001',mouth_sha256='mouth')]),
                             feature_hashes={path.name:sha256(path)})
                with self.assertRaises(ValueError):
                    read_cached_feature(root,summary,'video/000001','mouth','model',3)
