"""Freeze two official validation videos, then run the unchanged masked baseline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .asd import read_asd_archive
from .batch import summarize_decisions
from .data import read_catalog, read_jsonl, select_subset
from .preprocessing import sha256


def validate_completed_plan(actual, video, assets):
    expected_profile = dict(size=96,target_fps=None,crop_scale=1.7,align='none',decoder='ctc_greedy')
    if (actual['split']!='valid' or actual['video_id']!=video['video_id'] or actual['selected']!=video['selected']
            or actual['suite']!='baseline' or actual['missing_policy']!='mask'
            or actual['landmark_max_gap']!=0 or actual['track_boundary_limit']!=2
            or actual['max_seconds']!=10 or actual['max_samples']!=6
            or actual['profiles']!={'scale17_none':expected_profile}):
        raise ValueError('Per-video run differs from frozen validation plan')
    for local, frozen in [('catalog','catalog'),('asd','archive'),('landmarker','landmarker'),('checkpoint','checkpoint')]:
        if actual['hashes'][local]!=assets[frozen]:
            raise ValueError('Completed validation run uses different assets')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['catalog','archive','videos','model','checkpoint','repo','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--unavailable-video',action='append',default=[],
                        help='Frozen video ID whose acquisition failed; retain its six failures, never replace it')
    args = parser.parse_args()
    plan_path = args.output/'plan.json'
    assets = dict(catalog=sha256(args.catalog), archive=sha256(args.archive),
                  landmarker=sha256(args.model), checkpoint=sha256(args.checkpoint))
    if args.prepare_only:
        args.output.mkdir(parents=True,exist_ok=False)
        chosen = select_subset(read_catalog(args.catalog),'valid',2,2026)
        videos = []
        for row in chosen:
            selected = [u for u in read_asd_archive(args.archive,row['video_id']) if u.end_ms-u.start_ms<=10000][:6]
            if len(selected)!=6:
                raise ValueError('Selected video has fewer than six eligible utterances')
            videos.append(dict(video_id=row['video_id'], domain=row['domain'],
                selected=[dict(id=f'{u.video_id}/{u.index}',utterance=u.index,start_ms=u.start_ms,end_ms=u.end_ms) for u in selected]))
        plan = dict(split='valid',seed=2026,video_selection='first two unflagged official valid videos in seeded SHA256 order',
            utterance_selection='chronological first six utterances with duration <=10 seconds',
            videos=videos,hashes=assets,settings=dict(suite='baseline',missing_policy='mask',size=96,
                target_fps=None,crop_scale=1.7,align='none',landmark_max_gap=0,track_boundary_limit=2,decoder='ctc_greedy'),
            scope='new official validation videos; one fixed condition, not final held-out test',
            no_replacement_on_download_or_preprocessing_failure=True, references_used_in_decoding=False)
        plan_path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
        print('Frozen validation plan:',plan_path)
        return
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['hashes']!=assets:
        raise ValueError('Assets differ from frozen validation plan')
    if not set(args.unavailable_video) <= {v['video_id'] for v in plan['videos']}:
        raise ValueError('Unavailable ID is not in frozen selection')
    references, predictions, failures = [], [], []
    per_video = {}
    for video in plan['videos']:
        video_id = video['video_id']
        root = args.output/video_id
        items = {u.index:u for u in read_asd_archive(args.archive,video_id)}
        refs = [dict(id=row['id'],reference=items[row['utterance']].reference) for row in video['selected']]
        references.extend(refs)
        if video_id in args.unavailable_video:
            if (args.videos/(video_id+'.mp4')).exists():
                raise ValueError('Cannot mark a locally present video unavailable')
            unavailable = [dict(id=r['id'],stage='acquisition',error='Frozen source video unavailable; no replacement selected') for r in refs]
            failures.extend(unavailable)
            per_video[video_id] = dict(summary=summarize_decisions(refs,[],unavailable),
                                      acquisition_status='unavailable; no inference attempted')
            continue
        if not (root/'summary.json').exists():
            command = [sys.executable,'-m','kvsr.batch','--split','valid','--suite','baseline','--max-samples','6',
                '--missing-policy','mask','--video-id',video_id,'--video',str(args.videos/(video_id+'.mp4')),
                '--archive',str(args.archive),'--catalog',str(args.catalog),'--model',str(args.model),
                '--checkpoint',str(args.checkpoint),'--repo',str(args.repo),'--output',str(root)]
            result = subprocess.run(command,timeout=3600)
            if result.returncode:
                raise RuntimeError(f'Validation batch did not finish: {video_id}; keep fixed selection and inspect local logs')
        summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
        actual_plan = summary['plan']
        validate_completed_plan(actual_plan,video,assets)
        if summary['plan_sha256']!=sha256(root/'plan.json') or actual_plan['hashes']['video']!=sha256(args.videos/(video_id+'.mp4')):
            raise ValueError('Completed validation source or plan changed')
        result = summary['conditions']['scale17_none']
        per_video[video_id] = dict(summary=result, plan_sha256=sha256(root/'plan.json'),
            video_sha256=actual_plan['hashes']['video'], inference_provenance=summary['inference_provenance']['scale17_none'])
        predictions.extend(read_jsonl(root/'scale17_none/predictions.jsonl'))
        failures.extend(result['failures'])
    report = dict(plan=plan,plan_sha256=sha256(plan_path),per_video=per_video,
        overall=summarize_decisions(references,predictions,failures))
    (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Completed validation:',args.output/'summary.json')


if __name__=='__main__':
    main()
