"""Paired spatial-occlusion experiment with frozen Auto-AVSR and WildVSR IDs."""
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


def mask_band(x, fraction):
    """Central horizontal band over all frames, filled with normalized mean zero."""
    if fraction not in (0., .25, .5) or tuple(x.shape[-2:]) != (88,88):
        raise ValueError('Expected 88x88 input and fixed 0/25/50 percent condition')
    result=x.clone()
    height=int(88*fraction)
    start=(88-height)//2
    result[...,start:start+height,:]=0
    return result


def paired_counts(clean, condition):
    baseline={r['id']:r['word_errors'] for r in clean['details']}
    current={r['id']:r['word_errors'] for r in condition['details']}
    if baseline.keys()!=current.keys():
        raise ValueError('Paired IDs differ')
    return dict(improved=sum(current[k]<v for k,v in baseline.items()),
                unchanged=sum(current[k]==v for k,v in baseline.items()),
                worsened=sum(current[k]>v for k,v in baseline.items()))


def run(output):
    import cv2
    import numpy as np
    import torch
    output.mkdir(parents=True,exist_ok=False)
    def save(name,data):
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    baseline_root=Path('results/english_clean_2026-09-27')
    baseline=json.loads((baseline_root/'summary.json').read_text(encoding='utf-8'))
    prior=json.loads((baseline_root/'predictions.json').read_text(encoding='utf-8'))
    entries={r['id']:r for r in baseline['records']}
    prior_map={r['id']:r['hypothesis'] for r in prior}
    frozen=baseline['plan']
    selected=frozen['selected']
    if len(selected)!=12 or set(prior_map)!=set(selected) or baseline['failed']:
        raise ValueError('Expected complete fixed clean baseline')
    if sha256(baseline_root/'plan.json')!=baseline['plan_sha256']:
        raise ValueError('Baseline plan changed')
    for member in selected:
        if hashlib.sha256(prior_map[member].encode()).hexdigest()!=entries[member]['transcript_sha256']:
            raise ValueError('Baseline prediction changed')
    repo=Path('external/auto_avsr').resolve()
    checkpoint=Path('checkpoints/autoavsr_vsr_base.pth')
    archive=Path('data/raw/wildvsr.zip')
    commit=subprocess.check_output(['git','-c',f'safe.directory={repo.as_posix()}','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if commit!=frozen['repository_commit'] or sha256(checkpoint)!=frozen['checkpoint_sha256'] or sha256(archive)!=frozen['archive_sha256']:
        raise ValueError('Frozen code, weights or archive changed')
    plan=dict(baseline_plan=frozen,baseline_summary_sha256=sha256(baseline_root/'summary.json'),
        conditions=[0,.25,.5],mask='central full-width horizontal band on post-crop 88x88 input, all frames',
        geometry={'25':dict(rows=[33,55],columns=[0,88]),'50':dict(rows=[22,66],columns=[0,88])},
        coordinates='zero-based half-open',fill='normalized zero, grayscale mean 0.421 before normalization',
        percent_denominator='area of 88x88 model input; not anatomical lip area',
        control='all 12 clean input tensors and predictions hash-checked; first sample zero-mask inference replay',
        training=False,reference_used_for_inference=False,scope='paired 12-sample development diagnostic, not held-out benchmark')
    save('plan.json',plan)
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    sys.path.insert(0,str(repo))
    from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.scorers.length_bonus import LengthBonus
    units=(repo/'spm/unigram/unigram5000_units.txt').read_text(encoding='utf-8').splitlines()
    tokens=['<blank>']+[s.split()[0] for s in units]+['<eos>']
    model=E2E(len(tokens),'video').eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    tree=ast.parse((repo/'lightning.py').read_text(encoding='utf-8'))
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_beam_search_decoder']
    assert len(funcs)==1
    namespace=dict(BatchBeamSearch=BatchBeamSearch,LengthBonus=LengthBonus)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'official_beam_search','exec'),namespace)
    def infer(x):
        beam=namespace['get_beam_search_decoder'](model,tokens)
        with torch.inference_mode():
            front=model.frontend(x.unsqueeze(0))
            encoded,_=model.encoder(model.proj_encoder(front),None)
            best=beam(encoded.squeeze(0))[0]
        return ''.join(tokens[int(t)] for t in best.yseq[1:]).replace('<space>',' ').replace('\u2581',' ').replace('<eos>','').strip()
    predictions={'0':prior,'25':[],'50':[]}
    records=[]
    clean_verified=[]
    for index,member in enumerate(selected,1):
        path=baseline_root/Path(member).name
        if sha256(path)!=frozen['video_hashes'][member]:
            raise ValueError('Input video changed')
        cap=cv2.VideoCapture(str(path))
        frames=[]
        fps=cap.get(cv2.CAP_PROP_FPS)
        try:
            while True:
                ok,frame=cap.read()
                if not ok: break
                frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if not frames or fps!=25 or any(f.shape!=(96,96,3) for f in frames):
            raise ValueError('Changed clean input format')
        x=torch.from_numpy(np.stack(frames)).permute(0,3,1,2).float()/255.
        x=x[:,:,4:92,4:92]
        x=(x[:,0]*.2989+x[:,1]*.587+x[:,2]*.114).unsqueeze(1)
        x=(x-.421)/.165
        if hashlib.sha256(x.numpy().tobytes()).hexdigest()!=entries[member]['input_tensor_sha256']:
            raise ValueError('Clean tensor mismatch')
        clean_verified.append(member)
        if index==1:
            if infer(mask_band(x,0.))!=prior_map[member]:
                raise ValueError('Zero-mask replay differs from baseline')
            print('Zero-mask sentinel replay exactly matches clean',flush=True)
        for key,fraction in [('25',.25),('50',.5)]:
            started=time.perf_counter()
            masked=mask_band(x,fraction)
            record=dict(id=member,condition=key,status='failed',input_tensor_sha256=hashlib.sha256(masked.numpy().tobytes()).hexdigest())
            hypothesis=''
            try:
                hypothesis=infer(masked)
                record.update(status='ok',transcript_sha256=hashlib.sha256(hypothesis.encode()).hexdigest())
            except (RuntimeError,ValueError,IndexError) as error:
                record['error']=str(error)
            record['inference_seconds']=time.perf_counter()-started
            predictions[key].append(dict(id=member,hypothesis=hypothesis))
            records.append(record)
            save('predictions.json',predictions)
            save('progress.json',records)
            print(f'{index}/12 {Path(member).name} mask={key}% {record["status"]} {record["inference_seconds"]:.1f}s',flush=True)
    with zipfile.ZipFile(archive) as z:
        labels=json.loads(z.read('WildVSR/labels.json'))
    refs=[dict(id=n,reference=labels[Path(n).name].lower()) for n in selected]
    metrics={key:evaluate(refs,[dict(id=r['id'],hypothesis=r['hypothesis'].lower()) for r in rows]) for key,rows in predictions.items()}
    for metric in metrics.values():
        metric['normalization']=baseline['metrics']['normalization']
    if metrics['0']!=baseline['metrics']:
        raise ValueError('Baseline metrics replay differs')
    comparisons={key:dict(paired_counts(metrics['0'],metrics[key]),wer_delta=metrics[key]['wer']-metrics['0']['wer'],
        exact_sentences=sum(r['word_errors']==0 for r in metrics[key]['details'])) for key in ['25','50']}
    save('summary.json',dict(plan=plan,plan_sha256=sha256(output/'plan.json'),metrics=metrics,comparisons=comparisons,
        clean_tensor_hashes_verified=clean_verified,zero_mask_sentinel_equal=True,records=records,
        successful=sum(r['status']=='ok' for r in records),failed=sum(r['status']!='ok' for r in records)))
    def cell(s): return s.replace('|','\\|').replace('\n',' ')
    lines=['# Paired English lip ROI occlusion','', 'Model input area masked by a central horizontal band. Development data only.','',
        '| ID | Reference | Clean | 25% | 50% |','|---|---|---|---|---|']
    for i,ref in enumerate(refs):
        lines.append('| '+' | '.join(cell(s) for s in [ref['id'],ref['reference']]+[predictions[k][i]['hypothesis'] for k in ['0','25','50']])+' |')
    (output/'sample_outputs.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({key:dict(wer=value['wer'],cer=value['cer']) for key,value in metrics.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
