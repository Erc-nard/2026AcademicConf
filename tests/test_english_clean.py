import unittest

from kvsr.english_clean import select_members


class EnglishSelectionTests(unittest.TestCase):
    def test_order_independent_excludes_seen_smoke_and_nonvideo(self):
        names=[f'WildVSR/videos/{i:04d}.mp4' for i in range(30)]+['WildVSR/labels.json']
        chosen=select_members(names)
        self.assertEqual(chosen,select_members(list(reversed(names))))
        self.assertEqual(len(set(chosen)),12)
        self.assertNotIn('WildVSR/videos/0000.mp4',chosen)
        self.assertNotIn('WildVSR/labels.json',chosen)

    def test_rejects_duplicate_and_insufficient_candidates(self):
        with self.assertRaises(ValueError):
            select_members(['a.mp4']*12)
        with self.assertRaises(ValueError):
            select_members(['a.mp4'])
