import copy
import unittest

from kvsr.validation_experiment import validate_completed_plan


class FrozenValidationTests(unittest.TestCase):
    def fixture(self):
        selected=[dict(id='video/000001',utterance='000001',start_ms=0,end_ms=1000)]
        assets=dict(catalog='c',archive='a',landmarker='l',checkpoint='m')
        actual=dict(split='valid',video_id='video',selected=selected,suite='baseline',missing_policy='mask',
            landmark_max_gap=0,track_boundary_limit=2,max_seconds=10,max_samples=6,
            profiles=dict(scale17_none=dict(size=96,target_fps=None,crop_scale=1.7,align='none',decoder='ctc_greedy')),
            hashes=dict(catalog='c',asd='a',landmarker='l',checkpoint='m'))
        return actual,dict(video_id='video',selected=selected),assets

    def test_changed_run_settings_cannot_be_reused_as_fixed_validation(self):
        actual,video,assets=self.fixture()
        validate_completed_plan(actual,video,assets)
        for key,value in [('split','train'),('video_id','other'),('missing_policy','strict'),
                          ('landmark_max_gap',3),('track_boundary_limit',4),('selected',[])]:
            changed=copy.deepcopy(actual)
            changed[key]=value
            with self.assertRaises(ValueError):
                validate_completed_plan(changed,video,assets)

    def test_changed_checkpoint_or_input_profile_is_rejected(self):
        for change in ['checkpoint','size']:
            actual,video,assets=self.fixture()
            if change=='checkpoint':
                actual['hashes']['checkpoint']='different'
            else:
                actual['profiles']['scale17_none']['size']=224
            with self.assertRaises(ValueError):
                validate_completed_plan(actual,video,assets)
