from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock

from kvsr import visual_context as vc
from kvsr.heldout_resume import resume
from kvsr.meld_experiment import save,read,sha


class ResumeTests(unittest.TestCase):
    def test_completed_failure_is_reused_and_only_missing_caption_called(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir();output=root/'resumed'
            rows=[]
            for key in ['a','b']:
                p=source/f'{key}.png';p.write_bytes(key.encode())
                rows.append(dict(id=key,image=p.name,image_sha256=sha(p),qa_image_sha256=sha(p),face_masked=True,qa_approved=True))
            cached=dict(**rows[0],caption=None,status='failed',error='Incomplete caption',vlm_raw_response={})
            save(source/'approval.json',rows);save(source/'captions.json',[cached]);save(source/'plan.json',{})
            response=Mock();response.json.return_value={'choices':[{'finish_reason':'stop','message':{'content':'A room.'}}]}
            def digest(path):
                if path.name==vc.MODEL:return vc.MODEL_HASH
                if path.name==vc.PROJECTOR:return vc.PROJECTOR_HASH
                return sha(path)
            with patch('kvsr.heldout_resume.sha',side_effect=digest),patch.object(vc,'local_server',return_value=nullcontext()),patch.object(vc,'structure_captions') as structure,patch('requests.post',return_value=response) as post:
                resume(source,output)
                self.assertEqual(post.call_count,1)
                self.assertEqual(post.call_args.kwargs['json']['max_tokens'],96)
                structure.assert_called_once_with(output,8767)
            result=read(output/'captions.json')
            self.assertEqual(result[0],cached);self.assertEqual(result[1]['caption'],'A room.')
            self.assertEqual(read(source/'captions.json'),[cached])
