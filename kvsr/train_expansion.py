"""Freeze and prepare 48 train utterances without expanding validation-based tuning."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .asd import read_asd_archive
from .batch import summarize_decisions
from .data import read_catalog,read_jsonl
from .preprocessing import sha256

VIDEO_IDS=('8r1k9NLdSvE','5TaVl7-lU4k','D-hzf2x_XO0')


def choose(items,limit=22):
    return [u for u in items if u.end_ms-u.start_ms<=10000][:limit]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['catalog','archive','videos','model','checkpoint','repo','previous','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    assets={key:sha256(getattr(args,key)) for key in ['catalog','archive','model','checkpoint']}
    plan_path=args.output/'plan.json'
    if args.prepare_only:
        args.output.mkdir(parents=True,exist_ok=False)
        catalog={r['video_id']:r for r in read_catalog(args.catalog)}
        videos=[]
        for video_id in VIDEO_IDS:
            if catalog[video_id]['split']!='train' or catalog[video_id]['flags']:
                raise ValueError('Only the original unflagged train videos are allowed')
            selected=choose(read_asd_archive(args.archive,video_id))
            videos.append(dict(video_id=video_id,video_sha256=sha256(args.videos/(video_id+'.mp4')),
                selected=[dict(id=f'{u.video_id}/{u.index}',utterance=u.index,start_ms=u.start_ms,end_ms=u.end_ms) for u in selected]))
        plan=dict(videos=videos,hashes=assets,split='train',selection='same three videos; chronological first up to 22 utterances <=10 seconds each',
            selected=sum(len(v['selected']) for v in videos),expected_counts=[4,22,22],
            preprocessing='unchanged mask baseline, 96px/native FPS/scale1.7/no alignment',
            adaptation=dict(steps=200,learning_rate=.001,seed=2026,trainable='CTC head only',checkpoint='fixed final step'),
            validation='reuse the same six available valid utterances; no new validation selection',
            limitations=['Adds utterances, not new videos or verified new speakers','Same validation has already been inspected; this is exploratory development'])
        if [len(v['selected']) for v in videos]!=plan['expected_counts']:
            raise ValueError('Unexpected fixed training counts')
        plan_path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
        print('Frozen 48-sample training plan:',plan_path)
        return
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['hashes']!=assets:
        raise ValueError('Assets changed after plan freeze')
    refs,rows,failures=[],[],[]
    per_video={}
    original_equal={}
    for video in plan['videos']:
        video_id=video['video_id']
        if sha256(args.videos/(video_id+'.mp4'))!=video['video_sha256']:
            raise ValueError('Source video changed')
        root=args.output/video_id
        previous=args.previous/video_id
        if video_id==VIDEO_IDS[0] and not root.exists():
            prior=json.loads((previous/'plan.json').read_text(encoding='utf-8'))
            if prior['selected']!=video['selected'] or prior['missing_policy']!='mask':
                raise ValueError('Reused first video does not match plan')
            root.mkdir()
            for name in ['plan.json','summary.json','references.jsonl']:
                shutil.copyfile(previous/name,root/name)
            (root/'scale17_none').mkdir()
            shutil.copyfile(previous/'scale17_none/predictions.jsonl',root/'scale17_none/predictions.jsonl')
            for item in video['selected']:
                crop=root/'crops/scale1.7_none'/item['utterance']
                crop.mkdir(parents=True)
                for name in ['mouth.npz','report.json']:
                    shutil.copyfile(previous/'crops/scale1.7_none'/item['utterance']/name,crop/name)
        if not (root/'summary.json').exists():
            command=[sys.executable,'-m','kvsr.batch','--split','train','--suite','baseline','--max-samples','22',
                '--missing-policy','mask','--video-id',video_id,'--video',str(args.videos/(video_id+'.mp4')),
                '--archive',str(args.archive),'--catalog',str(args.catalog),'--model',str(args.model),
                '--checkpoint',str(args.checkpoint),'--repo',str(args.repo),'--output',str(root)]
            subprocess.run(command,check=True,timeout=7200)
        summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
        actual=summary['plan']
        if actual['selected']!=video['selected'] or actual['split']!='train' or actual['missing_policy']!='mask':
            raise ValueError('Completed batch does not match frozen selection')
        for old,new in [('catalog','catalog'),('asd','archive'),('landmarker','model'),('checkpoint','checkpoint')]:
            if actual['hashes'][old]!=assets[new]:
                raise ValueError('Completed batch asset mismatch')
        result=summary['conditions']['scale17_none']
        current=read_jsonl(root/'scale17_none/predictions.jsonl')
        current_by_id={r['id']:r for r in current}
        old=read_jsonl(previous/'scale17_none/predictions.jsonl')
        original_equal[video_id]=all(r['id'] in current_by_id and r['token_ids']==current_by_id[r['id']]['token_ids'] for r in old)
        refs.extend(read_jsonl(root/'references.jsonl'))
        rows.extend(current)
        failures.extend(result['failures'])
        per_video[video_id]=result
    report=dict(plan=plan,plan_sha256=sha256(plan_path),per_video=per_video,overall=summarize_decisions(refs,rows,failures),
        previous_subset_tokens_equal=original_equal)
    (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not all(original_equal.values()):
        raise ValueError('Original subset predictions changed')
    print('Completed 48-sample preparation:',args.output/'summary.json')


if __name__=='__main__':
    main()
