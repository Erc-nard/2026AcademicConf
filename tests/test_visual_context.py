import tempfile
import unittest
from pathlib import Path

from kvsr.visual_context import mask_rectangles, approved_image, validate_context, filter_context
from kvsr.context_comparison import build_input
from kvsr.meld_experiment import sha


class VisualContextTests(unittest.TestCase):
    def test_unsupported_personal_and_speech_cues_do_not_reach_corrector(self):
        raw=dict(scene='room',objects=['young man in a suit','desk','conversation'],
                 actions=['shared enjoyment','seated behind desk'],emotion=None,speaker_tracking_cue='his face')
        result=filter_context(raw)
        self.assertEqual(result['objects'],['person in a suit','desk'])
        self.assertEqual(result['actions'],['seated behind desk'])
        self.assertIsNone(result['speaker_tracking_cue'])
        self.assertEqual(raw['objects'][0],'young man in a suit')

    def test_black_union_clipping_and_source_preservation(self):
        import numpy as np
        original=np.full((10,10,3),255,dtype=np.uint8)
        masked,fraction=mask_rectangles(original,[[-.2,-.2,.5,.5],[.4,.4,1.2,1.2]])
        self.assertEqual(int((masked[:,:,0]==0).sum()),25+36-1)
        self.assertAlmostEqual(fraction,.6)
        self.assertTrue((original==255).all())
        with self.assertRaises(ValueError):mask_rectangles(original,[[.5,0,.1,1]])

    def test_unapproved_or_modified_image_never_reaches_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'masked.png';p.write_bytes(b'image')
            r=dict(image=p.name,face_masked=True,qa_approved=False,qa_image_sha256=sha(p),image_sha256=sha(p))
            with self.assertRaises(ValueError):approved_image(root,r)
            r['qa_approved']=True;self.assertEqual(approved_image(root,r),p)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):approved_image(root,r)

    def test_scene_schema_rejects_face_emotion_and_urls(self):
        v=dict(scene='room',objects=[],actions=[],emotion=None,speaker_tracking_cue=None)
        self.assertEqual(validate_context(v),v)
        with self.assertRaises(ValueError):validate_context(dict(v,emotion='happy'))
        with self.assertRaises(ValueError):validate_context(dict(v,scene='https://invented.example'))

    def test_context_channels_and_oracle_are_separate(self):
        args=('vsr','previous','wrong',{'scene':'room'},'oracle')
        self.assertEqual(build_input('C0',*args),dict(current_vsr='vsr',previous_dialogue=None,visual_context=None))
        self.assertEqual(build_input('C2',*args)['previous_dialogue'],None)
        self.assertEqual(build_input('C3',*args)['previous_dialogue'],'previous')
        self.assertEqual(build_input('C4',*args)['previous_dialogue'],'wrong')
        self.assertEqual(build_input('C5',*args)['previous_dialogue'],'oracle')
        self.assertNotIn('reference',build_input('C3',*args))
