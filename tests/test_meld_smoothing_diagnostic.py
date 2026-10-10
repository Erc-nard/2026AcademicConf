import unittest
from kvsr.meld_smoothing_diagnostic import segments


class SegmentsTests(unittest.TestCase):
    def test_no_smoothing_through_missing_frames_or_shot_changes(self):
        self.assertEqual(segments([True,True,False,True,True,True],[(.2,.2),(.2,.2),None,(.2,.2),(.6,.2),(.6,.2)]),[(0,2),(3,4),(4,6)])
        self.assertEqual(segments([False,False],[None,None]),[])
        self.assertEqual(segments([True],[(.2,.2)]),[(0,1)])
