"""Pre-inference held-out ROI QA, reusing the unchanged MELD detector/crop code."""
import argparse
from contextlib import contextmanager
from pathlib import Path

from . import meld_experiment as base


@contextmanager
def configure_base(root, source, pairs, anchors=None):
    """Scoped adapter; never edit the historical module or its output directory."""
    names=['ROOT','SOURCE','PAIRS','WRONG','anchor']
    saved={k:getattr(base,k) for k in names}
    keys=list(pairs)
    try:
        base.ROOT=root; base.SOURCE=source; base.PAIRS=pairs
        base.WRONG={key:pairs[keys[(i+1)%len(keys)]] for i,key in enumerate(keys)}
        if anchors is not None:
            base.anchor=lambda key,index,count: point_at(anchors[key],index,count)
        yield
    finally:
        for key,value in saved.items():setattr(base,key,value)


def point_at(annotation,index,count):
    if count!=annotation['frames']:raise ValueError('Annotation/frame count mismatch')
    for segment in annotation['segments']:
        if segment['start']<=index<segment['end']:
            return tuple(segment['nose'])
    return None  # Ambiguous/off-screen speaker frames remain missing, no fallback.


def validate_annotations(rows, metadata):
    if set(rows)!=set(metadata):raise ValueError('Annotate every candidate, including exclusions')
    for key,row in rows.items():
        meta=metadata[key]
        if row['video_sha256']!=meta['video_sha256'] or row['frames']!=meta['frames']:
            raise ValueError('Annotation source changed')
        if type(row['qa_approved']) is not bool or not isinstance(row['note'],str) or not row['note']:
            raise ValueError('Explicit QA decision and reason required')
        last=0
        for segment in row['segments']:
            start,end=segment['start'],segment['end'];nose=segment['nose']
            if type(start) is not int or type(end) is not int or not last<=start<end<=row['frames']:
                raise ValueError('Overlapping/out-of-range anchor segments')
            if len(nose)!=2 or any(not isinstance(x,(int,float)) or not 0<=x<=1 for x in nose):
                raise ValueError('Invalid normalized nose anchor')
            last=end
        if not row['qa_approved'] and row['segments']:raise ValueError('Rejected speaker cannot supply mouth frames')


def detect(output,resource_root):
    pairs={p['id']:p['previous_id'] for p in base.read(output/'candidate_plan.json')['pairs']}
    root=output/'roi'
    if root.exists():raise ValueError('Do not overwrite detections')
    if base.read(output/'acquisition.json')['missing']:raise ValueError('Incomplete acquisition')
    with configure_base(root,output,pairs):base.detect(resource_root)


def prepare(output,annotations):
    root=output/'roi'
    if (root/'crops.json').exists():raise ValueError('Do not overwrite crops')
    rows=base.read(annotations)
    metadata={r['id']:r for r in base.read(output/'video_qa/metadata.json')}
    validate_annotations(rows,metadata)
    pairs={p['id']:p['previous_id'] for p in base.read(output/'candidate_plan.json')['pairs']}
    base.save(root/'speaker_annotations.json',rows)
    with configure_base(root,output,pairs,rows):base.prepare()
    plan=base.read(root/'plan.json')
    plan.update(sample_scope='All preselected held-out candidates; no inference or target reference access',
                annotations_sha256=base.sha(annotations),stage='candidate ROI preparation only',
                scene_context_used=False)
    base.save(root/'plan.json',plan)


def select(output,scene_approval,count=20):
    """Freeze IDs only after ROI and all-face QA, before any inference."""
    if type(count) is not int or count<2:raise ValueError('At least two disjoint dialogues required')
    if (output/'final_plan.json').exists():raise ValueError('Final selection already frozen')
    crops={r['id']:r for r in base.read(output/'roi/crops.json')}
    annotations=base.read(output/'roi/speaker_annotations.json')
    crop_qa=base.read(output/'roi/crop_qa.json')
    if crop_qa['crops_sha256']!=base.sha(output/'roi/crops.json') or set(crop_qa['records'])!=set(crops):
        raise ValueError('Crop QA must match every prepared candidate')
    from .visual_context import approved_image
    scenes=base.read(scene_approval);scene_index={r['id']:r for r in scenes}
    candidates=base.read(output/'candidate_plan.json')['pairs']
    if len(scene_index)!=len(scenes) or set(scene_index)!={p['id'] for p in candidates}:
        raise ValueError('Record scene QA for every candidate')
    selected=[];decisions=[]
    for pair in candidates:
        reasons=[]
        for key in [pair['previous_id'],pair['id']]:
            row=crops[key]
            if not annotations[key]['qa_approved']:reasons.append(key+': speaker ambiguous or unavailable')
            if not crop_qa['records'][key]['qa_approved']:reasons.append(key+': '+crop_qa['records'][key]['reason'])
            if row['detected_frames']<10 or row['detected_frames']/row['source_frames']<.25:
                reasons.append(key+': mouth coverage below fixed threshold')
        scene=scene_index[pair['id']]
        if not scene['qa_approved']:reasons.append('all-face scene QA not approved')
        else:approved_image(scene_approval.parent,scene)
        usable=not reasons
        chosen=usable and len(selected)<count
        if chosen:selected.append(pair)
        decisions.append(dict(id=pair['id'],usable=usable,selected=chosen,reasons=reasons))
    base.save(output/'screening.json',decisions)
    if len(selected)!=count:raise ValueError('Fewer usable pairs than planned; retain all decisions, do not run inference')
    keys=[p['id'] for p in selected]
    plan=dict(pairs={p['id']:p['previous_id'] for p in selected},
              wrong={key:selected[(i+1)%len(selected)]['previous_id'] for i,key in enumerate(keys)},
              selected=selected,settings_sha256=base.sha(output/'settings_plan.json'),
              screening_sha256=base.sha(output/'screening.json'),scene_approval_sha256=base.sha(scene_approval),
              crop_qa_sha256=base.sha(output/'roi/crop_qa.json'),
              inference_started=False,selection='First passing pairs in candidate order; no model outputs or target references')
    base.save(output/'final_plan.json',plan)


def infer(output):
    """Run the unchanged pretrained VSR only on preselected pairs."""
    import shutil
    final=base.read(output/'final_plan.json');settings=base.read(output/'settings_plan.json')
    if base.sha(output/'settings_plan.json')!=final['settings_sha256']:raise ValueError('Frozen settings changed')
    for name,digest in settings['source_hashes'].items():
        if base.sha(Path(name))!=digest:raise ValueError('Frozen source changed: '+name)
    if base.sha(output/'screening.json')!=final['screening_sha256']:raise ValueError('Screening changed')
    root=output/'final_vsr';root.mkdir(exist_ok=False)
    keys=set(final['pairs'])|set(final['pairs'].values())
    records=[r for r in base.read(output/'roi/crops.json') if r['id'] in keys]
    if len(records)!=len(keys) or {r['id'] for r in records}!=keys:raise ValueError('Incomplete selected crops')
    for row in records:
        path=output/'roi'/f"{row['id']}.npz"
        if base.sha(path)!=row['crop_sha256']:raise ValueError('Crop changed')
        shutil.copyfile(path,root/path.name)
    base.save(root/'crops.json',records)
    base.save(root/'plan.json',dict(final_plan_sha256=base.sha(output/'final_plan.json'),settings=settings,
                                  pairs=final['pairs'],wrong=final['wrong'],target_reference_access=False))
    with configure_base(root,output,final['pairs']):base.infer()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['detect','prepare','select','infer'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resource-root',type=Path)
    parser.add_argument('--annotations',type=Path)
    parser.add_argument('--scene-approval',type=Path)
    args=parser.parse_args()
    if args.stage=='detect':detect(args.output,args.resource_root)
    elif args.stage=='prepare':prepare(args.output,args.annotations)
    elif args.stage=='select':select(args.output,args.scene_approval)
    else:infer(args.output)
