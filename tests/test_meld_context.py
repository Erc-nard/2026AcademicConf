import unittest

from kvsr.meld_context import oracle_variants, pairs_from_rows, select_rows, timestamp_ms


def row(dialog,utterance,start,end,episode='1'):
    return dict(Dialogue_ID=str(dialog),Utterance_ID=str(utterance),StartTime=start,EndTime=end,
                Season='1',Episode=episode,Utterance=f'text {dialog} {utterance}')


class MeldContextTests(unittest.TestCase):
    def test_oracle_controls_separate_target_and_wrong_dialogue(self):
        rows=[row(d,u,f'0:00:0{u*2},000',f'0:00:0{u*2+1},000') for d in range(3) for u in range(3)]
        pairs=pairs_from_rows(rows)
        variants=oracle_variants(pairs)
        self.assertEqual(len(variants),18)
        indexed={p['id']:p for p in pairs}
        for v in variants:
            self.assertNotIn('target_reference',v)
            self.assertIsNone(v['current_vsr'])
            self.assertFalse(v['ready_for_correction'])
            if v['condition']=='wrong_previous_oracle':
                self.assertNotEqual(v['context_dialogue_id'],indexed[v['id']]['dialogue_id'])

    def test_timestamp_formats(self):
        self.assertEqual(timestamp_ms('0:12:43,012'),763012)
        self.assertEqual(timestamp_ms('00:00:01.5'),1500)

    def test_previous_only_rejects_overlap_and_episode_mismatch(self):
        rows=[row(0,0,'0:00:01,000','0:00:02,000'),
              row(0,1,'0:00:02,000','0:00:04,000'),
              row(0,2,'0:00:03,000','0:00:05,000'),
              row(0,3,'0:00:06,000','0:00:07,000',episode='2')]
        pairs=pairs_from_rows(rows)
        self.assertEqual([p['status'] for p in pairs],['eligible','excluded','excluded'])
        self.assertEqual(pairs[0]['previous_text'],'text 0 0')
        self.assertEqual(pairs[0]['target_reference'],'text 0 1')
        self.assertTrue(all(p['previous_utterance']<p['target_utterance'] for p in pairs))

    def test_selection_order_independent_rejects_duplicates(self):
        rows=[row(d,u,'0:00:01,000','0:00:02,000') for d in range(4) for u in range(3)]
        self.assertEqual(select_rows(rows),select_rows(rows[::-1]))
        self.assertEqual(len(select_rows(rows)),9)
        with self.assertRaises(ValueError): select_rows(rows+[rows[0]])
