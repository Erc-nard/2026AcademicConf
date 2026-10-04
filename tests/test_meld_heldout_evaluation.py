import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kvsr.meld_experiment import save,sha,read
from kvsr.meld_heldout_evaluation import validate_trials,evaluate
from kvsr.context_comparison import CONDITIONS
from kvsr.meld_heldout_roi import select


class HeldoutEvaluationTests(unittest.TestCase):
    def test_missing_duplicate_and_cross_target_conditions_rejected(self):
        pairs={'a':'p','b':'q'}
        rows=[dict(id=k,mask=m,condition=c) for k in pairs for m in [0,25,50] for c in CONDITIONS]
        validate_trials(rows,pairs)
        for bad in [rows[:-1],rows+[rows[0]],rows[:-1]+[dict(rows[-1],id='other')]]:
            with self.assertRaises(ValueError):validate_trials(bad,pairs)

    def test_incomplete_generation_never_opens_target_texts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'comparison').mkdir()
            save(root/'final_plan.json',dict(pairs={'a':'p'}));save(root/'comparison/outputs.json',[])
            with patch('kvsr.meld_heldout_evaluation.project_texts') as projection:
                with self.assertRaises(ValueError):evaluate(root)
                projection.assert_not_called()
            self.assertFalse((root/'target_references.json').exists())

    def test_wrong_person_crop_cannot_pass_on_detection_count_alone(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'roi').mkdir();keys=['a','p','b','q']
            save(root/'roi/crops.json',[dict(id=k,detected_frames=30,source_frames=30) for k in keys])
            save(root/'roi/speaker_annotations.json',{k:dict(qa_approved=True) for k in keys})
            save(root/'roi/crop_qa.json',dict(crops_sha256=sha(root/'roi/crops.json'),records={k:dict(qa_approved=k!='a',reason='wrong person') for k in keys}))
            save(root/'candidate_plan.json',dict(pairs=[dict(id='a',previous_id='p'),dict(id='b',previous_id='q')]))
            save(root/'approval.json',[dict(id=k,qa_approved=True) for k in ['a','b']])
            with patch('kvsr.visual_context.approved_image'):
                with self.assertRaises(ValueError):select(root,root/'approval.json',count=2)
            decision=read(root/'screening.json')[0]
            self.assertFalse(decision['usable']);self.assertIn('a: wrong person',decision['reasons'])
            self.assertFalse((root/'final_plan.json').exists())
