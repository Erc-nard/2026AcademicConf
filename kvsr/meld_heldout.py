"""Pre-output MELD candidate selection and selected-only official archive acquisition."""
import argparse
import csv
from pathlib import Path
import tarfile
import time

from .meld_context import timestamp_ms, URL
from .meld_experiment import read,save,sha


def select_candidates(metadata, excluded_episodes, count=30):
    """Metadata-only; one disjoint pair per new dialogue. No transcript/score access."""
    if type(count) is not int or count<1:raise ValueError('Positive candidate count required')
    index={(int(r['Dialogue_ID']),int(r['Utterance_ID'])):r for r in metadata}
    if len(index)!=len(metadata):raise ValueError('Duplicate metadata ID')
    selected=[]
    for d in sorted({d for d,u in index}):
        if d<3:continue
        for u in sorted(u for dia,u in index if dia==d and u>0):
            if (d,u-1) not in index:continue
            current,previous=index[(d,u)],index[(d,u-1)]
            episode=(current['Season'],current['Episode'])
            if episode in excluded_episodes or episode!=(previous['Season'],previous['Episode']):continue
            if timestamp_ms(current['StartTime'])<timestamp_ms(previous['EndTime']):continue
            durations=[timestamp_ms(r['EndTime'])-timestamp_ms(r['StartTime']) for r in [previous,current]]
            if not all(500<=ms<=12000 for ms in durations):continue
            selected.append(dict(id=f'dia{d}_utt{u}',previous_id=f'dia{d}_utt{u-1}',dialogue_id=d,
                                 season=current['Season'],episode=current['Episode'],durations_ms=durations,
                                 gap_ms=timestamp_ms(current['StartTime'])-timestamp_ms(previous['EndTime'])))
            break
        if len(selected)==count:break
    if len(selected)<count:raise ValueError('Not enough candidates under fixed metadata rules')
    return selected


def plan(output, count):
    output.mkdir(parents=True,exist_ok=False)
    path=Path('external/meld/data/MELD/train_sent_emo.csv')
    # A separate data-ingestion projection exposes only timing/ID fields to selection.
    columns=['Dialogue_ID','Utterance_ID','Season','Episode','StartTime','EndTime']
    with path.open(encoding='utf-8-sig',newline='') as stream:
        metadata=[{k:r[k] for k in columns} for r in csv.DictReader(stream)]
    dev=read(Path('results/meld_context_2026-09-27/selected_metadata.json'))
    excluded={(r['Season'],r['Episode']) for r in dev}
    selected=select_candidates(metadata,excluded,count)
    frozen=dict(split='train, project-held-out dialogues and episodes; not official MELD test split',
                selection='First numeric eligible pair per dialogue >=3, excluding all development episodes; durations .5-12s, non-overlap',
                excluded_episodes=sorted(excluded),pairs=selected,metadata_sha256=sha(path),url=URL,
                roi_screen='Before any inference, require at least 10 detected speaker frames and 25% coverage in each member; document every candidate',
                final_selection='First 20 candidates passing pre-inference ROI and all-face scene QA; no output-based replacement',
                inference_started=False,target_transcripts_in_plan=False)
    save(output/'candidate_plan.json',frozen)
    print('Frozen candidates:',[p['id'] for p in selected],flush=True)


def acquire(output):
    import requests
    frozen=read(output/'candidate_plan.json');destination=output/'videos';destination.mkdir(exist_ok=False)
    names={p[field]+'.mp4' for p in frozen['pairs'] for field in ['id','previous_id']}
    found={};started=time.perf_counter()
    class Counted:
        def __init__(self,raw):self.raw=raw;self.count=0;self.last=0
        def read(self,size=-1):
            data=self.raw.read(size);self.count+=len(data)
            if self.count>9*1024**3:raise ValueError('Official train stream byte cap exceeded')
            if self.count-self.last>128*1024**2:
                self.last=self.count;print(f'{self.count/1024**3:.2f} GiB; {len(found)}/{len(names)} clips',flush=True)
            return data
    with requests.get(frozen['url'],stream=True,timeout=(30,90)) as response:
        response.raise_for_status();stream=Counted(response.raw)
        with tarfile.open(fileobj=stream,mode='r|gz') as outer:
            for member in outer:
                if member.name!='MELD.Raw/train.tar.gz':continue
                with tarfile.open(fileobj=outer.extractfile(member),mode='r|gz') as inner:
                    for item in inner:
                        name=Path(item.name).name
                        if not item.isfile() or name not in names:continue
                        if item.size>64*1024**2 or name in found:raise ValueError('Unexpected or duplicate selected video')
                        path=destination/name;path.write_bytes(inner.extractfile(item).read())
                        found[name]=dict(bytes=item.size,sha256=sha(path));save(output/'acquisition_progress.json',found)
                        if len(found)==len(names):break
                break
    save(output/'acquisition.json',dict(downloaded=found,missing=sorted(names-found.keys()),streamed_bytes=stream.count,seconds=time.perf_counter()-started))
    if names-found.keys():raise ValueError('Missing selected clips; no replacements')


def inspect_videos(output):
    """Produce silent, transcript-free contact sheets before speaker ROI decisions."""
    import cv2
    import numpy as np
    frozen=read(output/'candidate_plan.json');acquisition=read(output/'acquisition.json')
    if acquisition['missing']:raise ValueError('Finish acquisition before candidate QA')
    folder=output/'video_qa';folder.mkdir(exist_ok=False)
    keys=[p[k] for p in frozen['pairs'] for k in ['previous_id','id']]
    strips=[];metadata=[]
    for key in keys:
        path=output/'videos'/f'{key}.mp4'
        if sha(path)!=acquisition['downloaded'][path.name]['sha256']:raise ValueError('Video changed')
        cap=cv2.VideoCapture(str(path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        indices=np.linspace(0,count-1,6).astype(int).tolist();images=[]
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
            if not ok:raise ValueError(f'Undecodable QA frame {key}/{index}')
            thumb=cv2.resize(frame,(240,135));thumb=cv2.copyMakeBorder(thumb,22,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
            cv2.putText(thumb,f'{key} f{index}',(3,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1);images.append(thumb)
        cap.release();strips.append(np.concatenate(images,axis=1))
        metadata.append(dict(id=key,frames=count,fps=fps,qa_indices=indices,video_sha256=sha(path)))
        if len(strips)==6:
            cv2.imwrite(str(folder/f'page_{len(metadata)//6:02d}.jpg'),np.concatenate(strips,axis=0));strips=[]
    if strips:cv2.imwrite(str(folder/f'page_{(len(metadata)+5)//6:02d}.jpg'),np.concatenate(strips,axis=0))
    save(folder/'metadata.json',metadata)
    print('QA sheets generated; no speaker selection or inference performed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['plan','acquire','inspect'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--count',type=int,default=30)
    args=parser.parse_args()
    if args.stage=='plan':plan(args.output,args.count)
    elif args.stage=='acquire':acquire(args.output)
    else:inspect_videos(args.output)
