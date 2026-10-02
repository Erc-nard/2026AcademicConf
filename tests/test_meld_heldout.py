import unittest
from kvsr.meld_heldout import select_candidates


def pair(dialog,episode='2',overlap=False):
    return [dict(Dialogue_ID=str(dialog),Utterance_ID=str(i),Season='1',Episode=episode,
                 StartTime='0:00:00,000' if i==0 else ('0:00:00,500' if overlap else '0:00:01,000'),
                 EndTime='0:00:01,000' if i==0 else '0:00:02,000',Utterance='not selectable evidence') for i in [0,1]]


class HeldoutSelectionTests(unittest.TestCase):
    def test_excludes_development_episodes_and_overlaps_before_outputs(self):
        rows=pair(0)+pair(3,'1')+pair(4,overlap=True)+pair(5)+pair(6)
        result=select_candidates(list(reversed(rows)),{('1','1')},2)
        self.assertEqual([r['id'] for r in result],['dia5_utt1','dia6_utt1'])
        self.assertTrue(all('Utterance' not in r for r in result))
    def test_shortfall_duplicate_and_invalid_count_fail(self):
        with self.assertRaises(ValueError):select_candidates(pair(5),set(),2)
        with self.assertRaises(ValueError):select_candidates(pair(5)*2,set(),1)
        with self.assertRaises(ValueError):select_candidates(pair(5),set(),0)
