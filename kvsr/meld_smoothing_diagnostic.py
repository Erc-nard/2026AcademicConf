"""One-factor development check: official landmark smoothing within observed shots."""
import argparse
from pathlib import Path
import sys

from . import meld_experiment as base
from .evaluation import evaluate

SOURCE=Path('results/meld_experiment_2026-09-28')
VIDEOS=Path('results/meld_context_2026-09-27/videos')


def segments(observed,anchors):
    """Never smooth across missing landmarks or a manually annotated shot change."""
    runs=[];start=None
    for i,valid in enumerate(observed):
        cut=i>0 and anchors[i] is not None and anchors[i-1] is not None and sum((a-b)**2 for a,b in zip(anchors[i],anchors[i-1]))>.1**2
        if start is not None and (not valid or cut):runs.append((start,i));start=None
        if valid and start is None:start=i
    if start is not None:runs.append((start,len(observed)))
    return runs


def prepare(root):
    import cv2
    import numpy as np
    sys.path.insert(0,str(Path('external/auto_avsr').resolve()))
    from preparation.detectors.mediapipe.video_process import VideoProcess
    root.mkdir(parents=True,exist_ok=False)
    sources=[SOURCE/'detections.json',SOURCE/'detections_short.json',SOURCE/'crops.json',SOURCE/'vsr.json',
             Path('kvsr/meld_experiment.py'),Path('external/auto_avsr/preparation/detectors/mediapipe/video_process.py')]
    sources += [VIDEOS/(k+'.mp4') for k in base.PAIRS]
    sources += [SOURCE/(k+'.npz') for k in base.PAIRS]
    base.save(root/'plan.json',dict(scope='Same five known development targets; one-factor diagnostic, not new held-out evaluation',
        pairs=base.PAIRS,variant='window_margin=12 within contiguous observed, manually shot-consistent runs',
        unchanged='Detections, anchors, missing frames, 25fps floor mapping, 96->88 crop, normalization, model, beam40, masks',
        interpolation=False,baseline='window_margin=1 per frame; reconstructed arrays must exactly match cached inputs',
        sources={str(p):base.sha(p) for p in sources},source_sha256=base.sha(Path(__file__)),
        model_calls=15,reference_access='Only after all fifteen outputs; never used to choose crop/segments'))
    full=base.read(SOURCE/'detections.json');short=base.read(SOURCE/'detections_short.json')
    originals={r['id']:r for r in base.read(SOURCE/'crops.json')};records=[];audit=[];sheets=[]
    for key in base.PAIRS:
        cap=cv2.VideoCapture(str(VIDEOS/(key+'.mp4')));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        cap.release();landmarks=[];anchors=[]
        for i in range(len(frames)):
            point=base.anchor(key,i,len(frames));anchors.append(point);chosen=None
            if point is not None:
                for detections in [full,short]:
                    candidates=detections[key][i]
                    if candidates:
                        best=min(candidates,key=lambda d:np.linalg.norm(np.array(d['center'])-point))
                        if np.linalg.norm(np.array(best['center'])-point)<=.10:chosen=np.array(best['landmarks']);break
            landmarks.append(chosen)
        patches=np.full((len(frames),96,96,3),107,dtype=np.uint8);observed=np.array([v is not None for v in landmarks])
        processor=VideoProcess(convert_gray=False,window_margin=1)
        for i,lm in enumerate(landmarks):
            if lm is not None:
                try:patches[i]=processor(np.asarray([frames[i]]),[lm.copy()])[0]
                except (ValueError,OverflowError,AssertionError):observed[i]=False
        indices=np.minimum((np.arange(round(len(frames)/fps*25))*fps/25).astype(int),len(frames)-1)
        cached=np.load(SOURCE/(key+'.npz'))
        if not np.array_equal(patches[indices],cached['rgb']) or not np.array_equal(observed[indices],cached['observed']):
            raise ValueError('Baseline reconstruction mismatch: '+key)
        smoothed=patches.copy();runs=segments(observed,anchors)
        processor=VideoProcess(convert_gray=False,window_margin=12)
        for start,end in runs:
            # crop_patch avoids interpolation or extending unobserved frames.
            smoothed[start:end]=processor.crop_patch(np.array(frames[start:end]),[v.copy() for v in landmarks[start:end]])
        target=root/(key+'.npz');np.savez_compressed(target,rgb=smoothed[indices],observed=observed[indices])
        records.append(dict(originals[key],crop_sha256=base.sha(target)))
        audit.append(dict(id=key,exact_baseline_replay=True,observed_source_frames=int(observed.sum()),
            source_frames=len(frames),fps=fps,model_frames=len(indices),segments=runs,
            changed_model_frames=int(np.any(smoothed[indices]!=patches[indices],axis=(1,2,3)).sum()),
            mean_absolute_pixel_change=float(np.abs(smoothed.astype(float)-patches.astype(float))[observed].mean())))
        selected=np.linspace(0,len(frames)-1,6).astype(int)
        for label,array in [('baseline',patches),('smooth12',smoothed)]:
            strip=np.concatenate([cv2.cvtColor(array[i],cv2.COLOR_RGB2BGR) for i in selected],axis=1)
            strip=cv2.copyMakeBorder(strip,22,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
            cv2.putText(strip,key+' '+label,(4,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1);sheets.append(strip)
        print('prepared',key,'baseline exact',flush=True)
    base.save(root/'crops.json',records);base.save(root/'input_audit.json',audit)
    cv2.imwrite(str(root/'crop_comparison.jpg'),np.concatenate(sheets,axis=0))


def infer(root):
    p=base.read(root/'plan.json')
    if base.sha(Path(__file__))!=p['source_sha256']:raise ValueError('Diagnostic source changed')
    for name,digest in p['sources'].items():
        if base.sha(Path(name))!=digest:raise ValueError('Frozen input changed')
    if (root/'vsr.json').exists():raise ValueError('Preserve existing inference')
    previous=base.ROOT
    try:base.ROOT=root;base.infer()
    finally:base.ROOT=previous


def score(root):
    rows=base.read(root/'vsr.json');expected={(k,m) for k in base.PAIRS for m in [0,25,50]}
    if len(rows)!=15 or {(r['id'],r['mask']) for r in rows}!=expected:raise ValueError('Incomplete trials')
    refs=[dict(id=r['id'],reference=base.normalize_english(r['reference'])) for r in base.read(Path('results/meld_context_2026-09-27/target_references.json')) if r['id'] in base.PAIRS]
    old=base.read(SOURCE/'vsr.json');scores={}
    for m in [0,25,50]:
        metrics={name:evaluate(refs,[dict(id=r['id'],hypothesis=base.normalize_english(r['hypothesis'])) for r in data if r['mask']==m and r['id'] in base.PAIRS]) for name,data in [('baseline',old),('smooth12',rows)]}
        scores[str(m)]=dict(**metrics,comparison=base.comparison(metrics['baseline'],metrics['smooth12']))
    report=dict(plan=base.read(root/'plan.json'),input_audit=base.read(root/'input_audit.json'),scores=scores,
        calls=len(rows),failures=sum(r['status']!='ok' for r in rows),output_sha256=base.sha(root/'vsr.json'))
    base.save(root/'summary.json',report);base.save(Path('experiments/meld_smoothing_2026-10-10.json'),report)
    print({m:{k:round(v[k]['wer']*100,2) for k in ['baseline','smooth12']} for m,v in scores.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','infer','score'])
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    {'prepare':prepare,'infer':infer,'score':score}[args.stage](args.output)
