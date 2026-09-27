"""Acquire a fixed MELD conversation pilot and separate oracle context from targets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

from .preprocessing import sha256

URL='https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz'


def timestamp_ms(value):
    hours,minutes,seconds=value.strip().split(':')
    seconds,millis=seconds.replace('.',',').split(',')
    return ((int(hours)*60+int(minutes))*60+int(seconds))*1000+int(millis.ljust(3,'0'))


def select_rows(rows):
    indexed={(int(r['Dialogue_ID']),int(r['Utterance_ID'])):r for r in rows}
    if len(indexed)!=len(rows):
        raise ValueError('Duplicate dialogue/utterance IDs')
    dialogs=sorted({d for d,u in indexed if all((d,i) in indexed for i in range(3))})[:3]
    if len(dialogs)!=3:
        raise ValueError('Need three complete initial dialogue triples')
    return [indexed[(d,u)] for d in dialogs for u in range(3)]


def pairs_from_rows(rows):
    indexed={(int(r['Dialogue_ID']),int(r['Utterance_ID'])):r for r in rows}
    pairs=[]
    for (dialog,utterance),target in sorted(indexed.items()):
        if utterance==0: continue
        previous=indexed[(dialog,utterance-1)]
        gap=timestamp_ms(target['StartTime'])-timestamp_ms(previous['EndTime'])
        same_episode=(previous['Season'],previous['Episode'])==(target['Season'],target['Episode'])
        valid=gap>=0 and same_episode
        pairs.append(dict(id=f'dia{dialog}_utt{utterance}',previous_id=f'dia{dialog}_utt{utterance-1}',
            dialogue_id=dialog,previous_utterance=utterance-1,target_utterance=utterance,
            gap_ms=gap,status='eligible' if valid else 'excluded',
            reason=None if valid else 'overlap_or_episode_mismatch',
            previous_text=previous['Utterance'],target_reference=target['Utterance']))
    return pairs


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def oracle_variants(pairs):
    """Explicit upper-bound controls, not deployment-quality previous recognition."""
    eligible=[p for p in pairs if p['status']=='eligible']
    variants=[]
    for pair in eligible:
        wrong=next((p for p in eligible if p['dialogue_id']!=pair['dialogue_id']),None)
        if wrong is None: raise ValueError('Wrong-context control needs another dialogue')
        for kind,source in [('none',None),('previous_oracle',pair),('wrong_previous_oracle',wrong)]:
            variants.append(dict(id=pair['id'],condition=kind,
                context_text='' if source is None else source['previous_text'],
                context_source_id=None if source is None else source['previous_id'],
                context_dialogue_id=None if source is None else source['dialogue_id'],
                context_is_ground_truth=source is not None,current_vsr=None,
                ready_for_correction=False))
    return variants


def acquire(output):
    import requests
    output.mkdir(parents=True,exist_ok=False)
    repo=Path('external/meld').resolve()
    metadata=repo/'data/MELD/train_sent_emo.csv'
    with metadata.open(encoding='utf-8-sig',newline='') as f:
        selected=select_rows(list(csv.DictReader(f)))
    names=[f'dia{r["Dialogue_ID"]}_utt{r["Utterance_ID"]}.mp4' for r in selected]
    commit=subprocess.check_output(['git','-c',f'safe.directory={repo.as_posix()}','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    plan=dict(dataset='MELD official author distribution',split='train',url=URL,repository_commit=commit,
        metadata_sha256=sha256(metadata),selection='First 3 numeric dialogue IDs with utterance IDs 0,1,2; no visual or model-output filtering',
        files=names,scope='conversation-context feasibility pilot, not VSR benchmark or final test',
        maximum_stream_bytes=9*1024**3)
    save(output/'plan.json',plan)
    save(output/'selected_metadata.json',selected)
    (output/'videos').mkdir()
    needed=set(names)
    found={}
    started=time.perf_counter()
    class Counted:
        def __init__(self,raw): self.raw=raw; self.count=0; self.last=0
        def read(self,size=-1):
            data=self.raw.read(size)
            self.count+=len(data)
            if self.count>plan['maximum_stream_bytes']: raise ValueError('Stream size cap exceeded')
            if self.count-self.last>=256*1024**2:
                self.last=self.count
                print(f'Streamed {self.count/1024**3:.2f} GiB; clips {len(found)}/9',flush=True)
            return data
    with requests.get(URL,stream=True,timeout=(30,90)) as response:
        response.raise_for_status()
        stream=Counted(response.raw)
        with tarfile.open(fileobj=stream,mode='r|gz') as outer:
            for member in outer:
                if member.name!='MELD.Raw/train.tar.gz': continue
                with tarfile.open(fileobj=outer.extractfile(member),mode='r|gz') as inner:
                    for item in inner:
                        name=Path(item.name).name
                        if not item.isfile() or name not in needed: continue
                        if item.size>64*1024**2: raise ValueError('Unexpectedly large selected clip')
                        if name in found: raise ValueError('Duplicate selected clip in archive')
                        destination=output/'videos'/name
                        destination.write_bytes(inner.extractfile(item).read())
                        found[name]=dict(bytes=item.size,sha256=sha256(destination))
                        save(output/'download_progress.json',found)
                        print(f'Acquired {name} ({len(found)}/9)',flush=True)
                        if len(found)==len(needed): break
                break
    save(output/'acquisition.json',dict(plan=plan,downloaded=found,missing=sorted(needed-found.keys()),
        transferred_bytes=stream.count,seconds=time.perf_counter()-started))
    if needed-found.keys(): raise ValueError('Selected clips missing; no replacements')


def build(output):
    import cv2
    import numpy as np
    selected=json.loads((output/'selected_metadata.json').read_text(encoding='utf-8'))
    acquisition=json.loads((output/'acquisition.json').read_text(encoding='utf-8'))
    pairs=pairs_from_rows(selected)
    for pair in pairs:
        missing=[key for key in [pair['id'],pair['previous_id']]
                 if key+'.mp4' not in acquisition['downloaded']]
        pair['missing_video_ids']=missing
        if missing:
            pair['status']='excluded'
            pair['reason']='missing_video' if pair['reason'] is None else pair['reason']+';missing_video'
    video_records=[]
    thumbnails=[]
    (output/'contact_frames').mkdir(exist_ok=True)
    for name,entry in sorted(acquisition['downloaded'].items()):
        path=output/'videos'/name
        if sha256(path)!=entry['sha256']: raise ValueError('Video changed')
        cap=cv2.VideoCapture(str(path))
        fps=cap.get(cv2.CAP_PROP_FPS)
        expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames=[]
        while True:
            ok,frame=cap.read()
            if not ok: break
            frames.append(frame)
        cap.release()
        if not frames or fps<=0: raise ValueError('Unreadable selected video')
        # Local QA only: these unmasked images MUST NOT enter a scene-context model.
        indexes=sorted({0,len(frames)//2,len(frames)-1})
        for i in indexes:
            cv2.imwrite(str(output/'contact_frames'/f'{Path(name).stem}_{i}.jpg'),frames[i])
        thumb=cv2.resize(frames[len(frames)//2],(320,180))
        thumb=cv2.copyMakeBorder(thumb,24,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(thumb,Path(name).stem,(6,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        thumbnails.append(thumb)
        video_records.append(dict(id=Path(name).stem,frames=len(frames),container_frames=expected,
            fps=fps,width=frames[0].shape[1],height=frames[0].shape[0],duration_seconds=len(frames)/fps,
            decoded_all_expected_frames=len(frames)==expected))
    if len(thumbnails)==9:
        cv2.imwrite(str(output/'midpoint_contact_sheet.jpg'),np.concatenate([
            np.concatenate(thumbnails[i:i+3],axis=1) for i in range(0,9,3)],axis=0))
    # Oracle text is deliberately separate from model-ready input and target reference.
    save(output/'oracle_previous_context.json',[dict(id=p['id'],previous_id=p['previous_id'],
        previous_text=p['previous_text'],context_type='oracle_previous_transcript',
        status=p['status']) for p in pairs if p['status']=='eligible'])
    variants=oracle_variants(pairs)
    save(output/'oracle_context_variants.json',variants)
    save(output/'target_references.json',[dict(id=p['id'],reference=p['target_reference']) for p in pairs])
    save(output/'pending_model_inputs.json',[dict(id=p['id'],previous_id=p['previous_id'],
        target_vsr=None,previous_vsr=None,scene_context=None,status='not_model_ready',
        blocking_reasons=['active-speaker visual QA required','VSR not run','scene image needs all faces masked'])
        for p in pairs if p['status']=='eligible'])
    public_pairs=[{k:v for k,v in p.items() if k not in {'previous_text','target_reference'}} for p in pairs]
    save(output/'summary.json',dict(acquisition=acquisition,pairs=public_pairs,video_records=video_records,
        eligible_pairs=sum(p['status']=='eligible' for p in pairs),excluded_pairs=sum(p['status']!='eligible' for p in pairs),
        oracle_control_rows=len(variants),oracle_control_conditions=['none','previous_oracle','wrong_previous_oracle'],
        raw_scene_frames_local_only=True,correction_executed=False,vsr_executed=False,
        safeguards=['Only strictly prior non-overlapping utterances qualify',
            'Previous reference transcript is oracle-only, never realistic previous VSR',
            'Target reference stored separately from correction inputs',
            'Unmasked QA frames are not scene-context inputs']))
    lines=['# MELD context feasibility pilot','', 'Oracle previous transcript only. No VSR/correction performance measured.','',
        '| Target | Previous | Gap ms | Status |','|---|---|---:|---|']
    for p in public_pairs: lines.append(f'| {p["id"]} | {p["previous_id"]} | {p["gap_ms"]} | {p["status"]} |')
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Built {len(pairs)} pairs; eligible {sum(p["status"]=="eligible" for p in pairs)}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['acquire','build'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    (acquire if args.stage=='acquire' else build)(args.output)
