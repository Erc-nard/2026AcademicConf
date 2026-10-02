import unittest
from pathlib import Path

from kvsr import meld_experiment as base
from kvsr.meld_heldout_roi import configure_base,point_at,validate_annotations


class HeldoutRoiTests(unittest.TestCase):
    def test_adapter_restores_historical_paths_even_after_error(self):
        before=(base.ROOT,base.SOURCE,base.PAIRS,base.WRONG,base.anchor)
        with self.assertRaises(RuntimeError):
            with configure_base(Path('new'),Path('source'),{'a':'p','b':'q'}):
                self.assertEqual(base.WRONG,{'a':'q','b':'p'})
                raise RuntimeError('interrupted')
        self.assertEqual((base.ROOT,base.SOURCE,base.PAIRS,base.WRONG,base.anchor),before)

    def test_ambiguous_frames_have_no_other_person_fallback(self):
        row=dict(frames=40,video_sha256='abc',qa_approved=True,note='visual review',
                 segments=[dict(start=5,end=20,nose=[.3,.4])])
        meta={'x':dict(frames=40,video_sha256='abc')}
        validate_annotations({'x':row},meta)
        self.assertIsNone(point_at(row,0,40));self.assertIsNone(point_at(row,20,40))
        self.assertEqual(point_at(row,19,40),(.3,.4))
        with self.assertRaises(ValueError):point_at(row,19,41)
        with self.assertRaises(ValueError):validate_annotations({'x':dict(row,qa_approved=False)},meta)
        with self.assertRaises(ValueError):validate_annotations({'x':dict(row,video_sha256='changed')},meta)
        with self.assertRaises(ValueError):validate_annotations({'x':dict(row,segments=row['segments']*2)},meta)
