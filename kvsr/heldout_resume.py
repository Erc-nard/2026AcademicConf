"""Resume interrupted caption calls while preserving every completed result."""
import argparse
import base64
from pathlib import Path
import shutil

from . import visual_context as vc
from .meld_experiment import read,save,sha


def resume(source,output,port=8767):
    import requests
    vc.configure('v2')
    frames=read(source/'approval.json');cached=read(source/'captions.json')
    if [r['id'] for r in cached]!=[r['id'] for r in frames[:len(cached)]]:
        raise ValueError('Saved captions must be an exact prefix of frozen inputs')
    if len(cached)>=len(frames):raise ValueError('No missing caption calls')
    model=Path('checkpoints')/vc.MODEL;projector=Path('checkpoints')/vc.PROJECTOR
    if sha(model)!=vc.MODEL_HASH or sha(projector)!=vc.PROJECTOR_HASH:raise ValueError('Model changed')
    output.mkdir(exist_ok=False)
    for row in frames:
        path=vc.approved_image(source,row);shutil.copyfile(path,output/path.name)
    save(output/'approval.json',frames)
    save(output/'plan.json',dict(source_plan=read(source/'plan.json'),source=str(source),
        source_captions_sha256=sha(source/'captions.json'),reused_caption_records=len(cached),
        remaining_caption_ids=[r['id'] for r in frames[len(cached):]],
        prompt=vc.CAPTION_PROMPT,max_tokens=96,temperature=0,seed=2026,
        note='Fresh server after interruption; completed successes and failures retained without retry'))
    captions=list(cached);save(output/'captions.json',captions)
    with vc.local_server(output,model,port,projector):
        for row in frames[len(cached):]:
            encoded=base64.b64encode(vc.approved_image(output,row).read_bytes()).decode('ascii')
            body=dict(model='context-model',messages=[dict(role='user',content=[dict(type='text',text=vc.CAPTION_PROMPT),
                dict(type='image_url',image_url=dict(url='data:image/png;base64,'+encoded))])],temperature=0,seed=2026,max_tokens=96)
            raw=None;caption=None;status='ok';error=None
            try:
                response=requests.post(f'http://127.0.0.1:{port}/v1/chat/completions',json=body,timeout=180)
                raw=response.json();response.raise_for_status()
                if raw['choices'][0]['finish_reason']!='stop':raise ValueError('Incomplete caption')
                caption=raw['choices'][0]['message']['content'].strip()
                if not caption:raise ValueError('Empty caption')
            except (requests.RequestException,ValueError,KeyError,TypeError) as exc:status='failed';error=str(exc)
            captions.append(dict(**row,caption=caption,status=status,error=error,vlm_raw_response=raw))
            save(output/'captions.json',captions);print('caption',row['id'],status,flush=True)
    vc.structure_captions(output,port)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8767)
    args=parser.parse_args();resume(args.source,args.output,args.port)
