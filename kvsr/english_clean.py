"""Frozen 12-sample WildVSR development baseline; no outcome-based filtering."""
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


def select_members(members, count=12):
    eligible=sorted(n for n in members if n.endswith('.mp4') and Path(n).name!='0000.mp4')
    if len(set(eligible))!=len(eligible) or len(eligible)<count:
        raise ValueError('Insufficient or duplicate sample IDs')
    return sorted(eligible,key=lambda n:hashlib.sha256(('2026:'+n).encode()).hexdigest())[:count]


def run(output):
    import cv2
    import numpy as np
    import torch
    output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    repo=Path('external/auto_avsr').resolve()
    checkpoint=Path('checkpoints/autoavsr_vsr_base.pth')
    archive=Path('data/raw/wildvsr.zip')
    original=json.loads(Path('experiments/english_smoke_2026-09-27.json').read_text(encoding='utf-8'))['plan']
    commit=subprocess.check_output(['git','-c',f'safe.directory={repo.as_posix()}','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if sha256(checkpoint)!=original['checkpoint_sha256'] or sha256(archive)!=original['archive_sha256'] or commit!=original['repository_commit']:
        raise ValueError('Frozen model, archive or code revision changed')
    with zipfile.ZipFile(archive) as z:
        selected=select_members(z.namelist())
        hashes={n:hashlib.sha256(z.read(n)).hexdigest() for n in selected}
    plan=dict(original)
    plan.pop('sample'); plan.pop('video_sha256')
    plan.update(selection='12 lowest SHA256(UTF8(2026: + full ZIP member path)); exclude prior smoke 0000.mp4; no duration or outcome filtering',
        selected=selected,video_hashes=hashes,scope='fixed development subset; not final held-out test or full benchmark',
        failure_policy='retain selected ID with empty prediction; no replacement',
        reference_used_for_inference=False)
    save('plan.json',plan)
    print('Frozen sample IDs: '+', '.join(Path(n).stem for n in selected),flush=True)
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
    predictions=[]
    records=[]
    for index,member in enumerate(selected,1):
        started=time.perf_counter()
        record=dict(id=member,status='failed')
        hypothesis=''
        try:
            path=output/(Path(member).name)
            with zipfile.ZipFile(archive) as z:
                path.write_bytes(z.read(member))
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
                raise ValueError('Unexpected video input format')
            x=torch.from_numpy(np.stack(frames)).permute(0,3,1,2).float()/255.
            x=x[:,:,4:92,4:92]
            x=(x[:,0]*.2989+x[:,1]*.587+x[:,2]*.114).unsqueeze(1)
            x=(x-.421)/.165
            record.update(frames=len(frames),duration_seconds=len(frames)/fps,input_tensor_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest())
            beam=namespace['get_beam_search_decoder'](model,tokens)
            infer_started=time.perf_counter()
            with torch.inference_mode():
                front=model.frontend(x.unsqueeze(0))
                encoded,_=model.encoder(model.proj_encoder(front),None)
                best=beam(encoded.squeeze(0))[0]
            ids=[int(t) for t in best.yseq[1:]]
            hypothesis=''.join(tokens[t] for t in ids).replace('<space>',' ').replace('\u2581',' ').replace('<eos>','').strip()
            record.update(status='ok',inference_seconds=time.perf_counter()-infer_started,
                transcript_sha256=hashlib.sha256(hypothesis.encode()).hexdigest())
        except (ValueError,RuntimeError,OSError,IndexError) as error:
            record['error']=str(error)
        record['elapsed_seconds']=time.perf_counter()-started
        predictions.append(dict(id=member,hypothesis=hypothesis))
        records.append(record)
        save('predictions.json',predictions)
        save('progress.json',records)
        print(f'{index}/12 {Path(member).name}: {record["status"]}, {record["elapsed_seconds"]:.1f}s',flush=True)
    # Labels are accessed only after every prediction is completed.
    with zipfile.ZipFile(archive) as z:
        labels=json.loads(z.read('WildVSR/labels.json'))
    refs=[dict(id=n,reference=labels[Path(n).name].lower()) for n in selected]
    metrics=evaluate(refs,[dict(id=r['id'],hypothesis=r['hypothesis'].lower()) for r in predictions])
    metrics['normalization']='lowercase; collapse whitespace; WER whitespace words; CER removes whitespace; punctuation retained'
    exact=sum(r['word_errors']==0 for r in metrics['details'])
    save('summary.json',dict(plan=plan,plan_sha256=sha256(output/'plan.json'),metrics=metrics,
        exact_sentences=exact,successful=sum(r['status']=='ok' for r in records),failed=sum(r['status']!='ok' for r in records),
        records=records,versions=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__)))
    lines=['# English clean development subset','', 'Fixed 12 samples; references read only after inference. Not a held-out test.','',
           '| ID | Reference | Prediction | Word errors / words |','|---|---|---|---|']
    def cell(s): return s.replace('|','\\|').replace('\n',' ')
    for ref,pred,detail in zip(refs,predictions,metrics['details']):
        lines.append('| '+' | '.join([cell(ref['id']),cell(ref['reference']),cell(pred['hypothesis']),f'{detail["word_errors"]}/{detail["reference_words"]}'])+' |')
    (output/'sample_outputs.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Corpus WER={metrics["wer"]:.4%}, CER={metrics["cer"]:.4%}, exact={exact}/12',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
