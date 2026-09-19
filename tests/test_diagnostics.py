import unittest

from kvsr.asd import Utterance
from kvsr.diagnostics import audit_track, contiguous_runs


class AuditTests(unittest.TestCase):
    def test_boundary_four_is_explicit_and_never_extends_past_four(self):
        u = Utterance('abcdefghijk', '000001', 'unused', 100, 100, 25, 0, 400,
                      ((6,1,2,20,20),(7,1,2,20,20)))
        with self.assertRaises(ValueError):
            u.box_at(2)
        self.assertEqual(u.box_at(2, boundary_limit=4), (1,2,20,20))
        with self.assertRaises(ValueError):
            u.box_at(1, boundary_limit=4)
        with self.assertRaises(ValueError):
            u.box_at(6, boundary_limit=100)

    def test_runs_are_inclusive_and_deduplicated(self):
        self.assertEqual(contiguous_runs([5, 2, 3, 3]),
                         [dict(start=2,end=3,length=2), dict(start=5,end=5,length=1)])
        self.assertEqual(contiguous_runs([]), [])

    def test_track_counts_boundary_extension_and_missing(self):
        utterance = Utterance('abcdefghijk', '000001', 'unused', 100, 100, 25, 0, 400,
                              ((4,0,0,20,20),(5,0,0,20,20),(6,0,0,20,20)))
        report = audit_track(utterance)
        self.assertEqual(report['requested_frames'], 10)
        self.assertEqual(report['missing_frames'], 3)
        self.assertEqual(report['interpolated_or_extended_frames'], 4)
        self.assertEqual(report['missing_runs'],
                         [dict(start=1,end=1,length=1),dict(start=9,end=10,length=2)])
