import unittest

from kvsr.english_occlusion import mask_band, paired_counts


class PairedCountsTests(unittest.TestCase):
    def test_counts_and_mismatched_ids(self):
        clean={'details':[dict(id=str(i),word_errors=e) for i,e in enumerate([1,2,3])]}
        masked={'details':[dict(id=str(i),word_errors=e) for i,e in enumerate([2,2,1])]}
        self.assertEqual(paired_counts(clean,masked),dict(improved=1,unchanged=1,worsened=1))
        with self.assertRaises(ValueError): paired_counts(clean,{'details':[]})


class MaskTests(unittest.TestCase):
    def test_exact_nested_area_all_frames_and_input_preserved(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch optional')
        x=torch.ones(3,1,88,88)
        self.assertTrue(torch.equal(mask_band(x,0.),x))
        quarter=mask_band(x,.25)
        half=mask_band(x,.5)
        self.assertEqual(int((quarter==0).sum()),3*22*88)
        self.assertEqual(int((half==0).sum()),3*44*88)
        self.assertTrue(torch.all(quarter[:,:,33:55,:]==0))
        self.assertTrue(torch.all(half[:,:,22:66,:]==0))
        self.assertTrue(torch.all(x==1))
        self.assertTrue(torch.all(half[quarter==0]==0))
        with self.assertRaises(ValueError): mask_band(x,.3)
