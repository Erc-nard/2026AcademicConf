from types import SimpleNamespace
import unittest

from kvsr.train_expansion import choose


class TrainingSelectionTests(unittest.TestCase):
    def test_selection_depends_only_on_order_and_duration(self):
        items=[SimpleNamespace(index=i,start_ms=i*20000,end_ms=i*20000+(11000 if i==1 else 10000)) for i in range(25)]
        selected=choose(items)
        self.assertEqual(len(selected),22)
        self.assertEqual([u.index for u in selected],[0]+list(range(2,23)))

    def test_short_video_keeps_all_eligible_samples(self):
        items=[SimpleNamespace(start_ms=0,end_ms=1000)]*4
        self.assertEqual(len(choose(items)),4)
