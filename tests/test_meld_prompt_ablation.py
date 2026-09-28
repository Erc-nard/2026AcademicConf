import copy
import unittest

from kvsr.meld_prompt_ablation import ACTIVE_PROMPT, validate_one_factor


class PromptAblationTests(unittest.TestCase):
    def test_only_system_prompt_may_change(self):
        before=[dict(id='sample',mask=25,condition='none',request=dict(
            temperature=0,model='fixed',messages=[dict(role='system',content='old'),
                                                dict(role='user',content='same input')]))]
        after=copy.deepcopy(before)
        after[0]['request']['messages'][0]['content']=ACTIVE_PROMPT
        validate_one_factor(before,after)
        for field,value in [('temperature',.5),('model','other')]:
            changed=copy.deepcopy(after);changed[0]['request'][field]=value
            with self.assertRaises(ValueError):validate_one_factor(before,changed)
        changed=copy.deepcopy(after);changed[0]['request']['messages'][1]['content']='different input'
        with self.assertRaises(ValueError):validate_one_factor(before,changed)
        with self.assertRaises(ValueError):validate_one_factor(before,after+after)
        with self.assertRaises(ValueError):validate_one_factor(before,[])
