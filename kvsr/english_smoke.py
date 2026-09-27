"""Auto-AVSR visual-only smoke inference on the first official WildVSR clip."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from .evaluation import evaluate
from .preprocessing import sha256


def run(output):
    import cv2
    import numpy as np
    import torch
    output.mkdir(parents=True, exist_ok=False)
    def save(name, data):
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    repo = Path('external/auto_avsr').resolve()
    checkpoint = Path('checkpoints/autoavsr_vsr_base.pth')
    archive = Path('data/raw/wildvsr.zip')
    with zipfile.ZipFile(archive) as z:
        member = sorted(n for n in z.namelist() if n.endswith('.mp4'))[0]
        # Fixed first lexical ID, selected before inference; never search by output quality.
        video_path = output/'input.mp4'
        video_path.write_bytes(z.read(member))
    commit = subprocess.check_output(['git','-c',f'safe.directory={repo.as_posix()}', '-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    plan = dict(model='Auto-AVSR vsr_trlrs2lrs3vox2avsp_base.pth',modality='video',
        dataset='WildVSR official author archive',sample=member,selection='lexicographically first MP4',
        repository_commit=commit,checkpoint_sha256=sha256(checkpoint),archive_sha256=sha256(archive),
        video_sha256=sha256(video_path),beam_size=40,ctc_weight=.1,lm_weight=0.,penalty=0.,
        device='cpu',threads=2,training=False,audio_read=False,reference_used_for_inference=False,
        scope='one development smoke sample, not benchmark performance',
        preprocessing='pre-cropped 96x96 ROI; RGB float /255; center crop 88; grayscale; (x-0.421)/0.165')
    save('plan.json',plan)
    sys.path.insert(0,str(repo))
    from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.scorers.length_bonus import LengthBonus
    units=(repo/'spm/unigram/unigram5000_units.txt').read_text(encoding='utf-8').splitlines()
    token_list=['<blank>']+[line.split()[0] for line in units]+['<eos>']
    model=E2E(len(token_list),'video').eval()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    del state
    print('Official visual-only checkpoint loaded strictly',flush=True)
    tree=ast.parse((repo/'lightning.py').read_text(encoding='utf-8'))
    definition=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_beam_search_decoder']
    assert len(definition)==1
    namespace=dict(BatchBeamSearch=BatchBeamSearch,LengthBonus=LengthBonus)
    exec(compile(ast.Module(body=definition,type_ignores=[]),'official_beam_search','exec'),namespace)
    beam=namespace['get_beam_search_decoder'](model,token_list)
    capture=cv2.VideoCapture(str(video_path))
    fps=capture.get(cv2.CAP_PROP_FPS)
    frames=[]
    while True:
        ok,frame=capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames or fps!=25 or frames[0].shape!=(96,96,3):
        raise ValueError('Unexpected WildVSR input format')
    x=torch.from_numpy(np.stack(frames)).permute(0,3,1,2).float()/255.
    x=x[:,:,4:92,4:92]
    # torchvision RGB-to-grayscale coefficients used by the official test transform.
    x=(x[:,0]*.2989+x[:,1]*.587+x[:,2]*.114).unsqueeze(1)
    x=(x-.421)/.165
    started=time.perf_counter()
    with torch.inference_mode():
        features=model.frontend(x.unsqueeze(0))
        encoded,_=model.encoder(model.proj_encoder(features),None)
        encoded_time=time.perf_counter()-started
        print(f'Encoded {len(frames)} frames in {encoded_time:.2f}s; official beam 40 decoding',flush=True)
        best=beam(encoded.squeeze(0))[0]
    elapsed=time.perf_counter()-started
    ids=[int(t) for t in best.yseq[1:]]
    hypothesis=''.join(token_list[t] for t in ids).replace('<space>',' ').replace('\u2581',' ').replace('<eos>','').strip()
    if not hypothesis or not any(c.isalpha() for c in hypothesis):
        raise ValueError('No English transcript generated')
    # Only now access the reference: it cannot influence the forward pass or beam.
    with zipfile.ZipFile(archive) as z:
        labels=json.loads(z.read('WildVSR/labels.json'))
    reference=labels[Path(member).name]
    metrics=evaluate([dict(id=member,reference=reference.lower())],[dict(id=member,hypothesis=hypothesis.lower())])
    metrics['normalization']='lowercase; collapse whitespace; WER whitespace words; CER removes whitespace; punctuation retained'
    result=dict(plan=plan,frames=len(frames),fps=fps,duration_seconds=len(frames)/fps,
        parameters=sum(p.numel() for p in model.parameters()),strict_checkpoint_load=True,
        encoder_seconds=encoded_time,inference_seconds=elapsed,metrics=metrics,
        versions=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__),
        input_tensor_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
        transcript_sha256=hashlib.sha256(hypothesis.encode()).hexdigest())
    save('summary.json',result)
    save('prediction.json',dict(reference=reference,hypothesis=hypothesis,token_ids=ids))
    (output/'sample_output.md').write_text('# English visual-only inference\n\n'
        f'Sample: `{member}`\n\nReference: {reference}\n\nPrediction: {hypothesis}\n\n'
        f'WER: {metrics["wer"]:.2%}; CER: {metrics["cer"]:.2%}\n',encoding='utf-8')
    print(json.dumps(dict(hypothesis=hypothesis,wer=metrics['wer'],cer=metrics['cer'],seconds=elapsed)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
