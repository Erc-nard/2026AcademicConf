import unittest

from kvsr.meld_experiment import PAIRS, WRONG, anchor, comparison, correction_input, normalize_english
from kvsr.evaluation import evaluate


class MeldExperimentTests(unittest.TestCase):
    def test_wrong_context_is_never_same_dialogue_or_current(self):
        assert set(PAIRS)==set(WRONG)
        for target,previous in WRONG.items():
            assert target.split('_')[0]!=previous.split('_')[0]
            assert previous in PAIRS.values()


    def test_prompt_payload_has_no_reference_or_scene_fields(self):
        assert correction_input('noisy VSR','earlier VSR')=={
            'current_subtitle':'noisy VSR','previous_utterance':'earlier VSR'}


    def test_scoring_retains_contractions_and_splits_hyphens(self):
        assert normalize_english('You must’ve. Hey-hey!')=="you must've hey hey"


    def test_shot_change_excludes_wrong_speaker(self):
        assert anchor('dia0_utt2',1,70) is None
        assert anchor('dia0_utt2',2,70) is not None
        assert anchor('dia1_utt1',31,53)!=anchor('dia1_utt1',32,53)


    def test_harm_and_correction_denominators_are_distinct(self):
        refs=[{'id':'a','reference':'yes'},{'id':'b','reference':'no'},{'id':'c','reference':'hello'}]
        before=evaluate(refs,[{'id':'a','hypothesis':'yes'},{'id':'b','hypothesis':'maybe'},{'id':'c','hypothesis':'bye'}])
        after=evaluate(refs,[{'id':'a','hypothesis':'no'},{'id':'b','hypothesis':'no'},{'id':'c','hypothesis':'bye'}])
        result=comparison(before,after)
        assert result['correction_rate']==.5
        assert result['exact_harm_rate']==1
        assert (result['improved'],result['unchanged'],result['worsened'])==(1,1,1)
        assert comparison(after,after)['exact_harm_rate']==0
        all_wrong=evaluate(refs,[{'id':r['id'],'hypothesis':''} for r in refs])
        assert comparison(all_wrong,after)['exact_harm_rate'] is None
        with self.assertRaises(ValueError):comparison(before,{'details':[]})
