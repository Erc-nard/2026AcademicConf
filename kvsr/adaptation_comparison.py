"""Compare small/expanded CTC heads on identical frozen feature sets."""
import argparse
import csv
import json
from pathlib import Path
import sys

from .data import read_jsonl,write_jsonl
from .evaluation import evaluate
from .preprocessing import sha256


def comparison_groups(old_inputs,new_inputs):
    if len({r['id'] for r in new_inputs})!=len(new_inputs):
        raise ValueError('Duplicate expanded IDs')
    old_train={r['id'] for r in old_inputs if r['split']=='train'}
    old_valid={r['id'] for r in old_inputs if r['split']=='valid'}
    new_train={r['id'] for r in new_inputs if r['split']=='train'}
    new_valid={r['id'] for r in new_inputs if r['split']=='valid'}
    if not old_train or not old_train<=new_train or old_valid!=new_valid or not new_valid or new_train&new_valid:
        raise ValueError('Training inclusion or validation identity changed')
    added=new_train-old_train
    if not added:
        raise ValueError('No training expansion')
    return dict(train_original=old_train,train_added=added,train_all=new_train,validation=new_valid)


def run(original,expanded,train,validation,repo,output):
    import numpy as np
    import torch
    from .olkavs import collapse_ctc
    if output.exists():
        raise ValueError('Output exists')
    output.mkdir(parents=True)
    old=json.loads((original/'summary.json').read_text(encoding='utf-8'))
    new=json.loads((expanded/'summary.json').read_text(encoding='utf-8'))
    if old['plan']['checkpoint_sha256']!=new['plan']['checkpoint_sha256'] or old['plan']['vocabulary_sha256']!=new['plan']['vocabulary_sha256']:
        raise ValueError('Model/vocabulary changed')
    if sha256(original/'ctc_head.pt')!=old['head_sha256']:
        raise ValueError('Original adapted head changed')
    groups=comparison_groups(old['plan']['inputs'],new['plan']['inputs'])
    sys.path.insert(0,str(repo.resolve()))
    from vocabulary.utils import grp2char
    with (repo/'vocabulary/kor_grapheme.csv').open(encoding='utf-8-sig',newline='') as stream:
        vocabulary={int(r['id']):r['char'] for r in csv.DictReader(stream)}
    if sha256(repo/'vocabulary/kor_grapheme.csv')!=new['plan']['vocabulary_sha256']:
        raise ValueError('Current vocabulary differs')
    torch.set_num_threads(2)
    state=torch.load(original/'ctc_head.pt',weights_only=True,map_location='cpu')
    small=[]
    absent={r['id'] for r in new.get('training_exclusions',[]) if r['reason']=='no usable features'}
    for row in new['plan']['inputs']:
        video_id,utterance=row['id'].split('/')
        path=expanded/'features'/f'{video_id}_{utterance}.npy'
        if row['id'] in absent:
            tokens=[]
        else:
            if sha256(path)!=new['feature_hashes'][path.name]:
                raise ValueError('Expanded feature cache changed')
            feature=torch.from_numpy(np.load(path,allow_pickle=False))
            with torch.inference_mode():
                logits=torch.nn.functional.linear(feature,state['fc.weight'],state['fc.bias'])
                tokens=collapse_ctc(logits.argmax(-1).tolist())
        small.append(dict(id=row['id'],hypothesis=grp2char(''.join(vocabulary[t] for t in tokens)),token_ids=tokens))
    original_predictions={r['id']:r for r in read_jsonl(original/'adapted_predictions.jsonl')}
    if not all(r['token_ids']==original_predictions[r['id']]['token_ids'] for r in small if r['id'] in original_predictions):
        raise ValueError('Original adapted head replay differs')
    write_jsonl(output/'small_head_predictions.jsonl',small)
    predictions=dict(baseline=read_jsonl(expanded/'baseline_predictions.jsonl'),small_head=small,
                     expanded_head=read_jsonl(expanded/'adapted_predictions.jsonl'))
    references=[]
    for video_id in sorted({r['id'].split('/')[0] for r in new['plan']['inputs']}):
        root=train if any(r['id'].startswith(video_id+'/') and r['split']=='train' for r in new['plan']['inputs']) else validation
        references.extend(read_jsonl(root/video_id/'references.jsonl'))
    metrics={}
    paired={}
    for group,ids in groups.items():
        refs=[r for r in references if r['id'] in ids]
        metrics[group]={name:evaluate(refs,[r for r in rows if r['id'] in ids]) for name,rows in predictions.items()}
        old_errors={r['id']:r['char_errors'] for r in metrics[group]['small_head']['details']}
        deltas=[r['char_errors']-old_errors[r['id']] for r in metrics[group]['expanded_head']['details']]
        paired[group]=dict(better=sum(d<0 for d in deltas),equal=sum(d==0 for d in deltas),worse=sum(d>0 for d in deltas))
    result=dict(original_summary_sha256=sha256(original/'summary.json'),expanded_summary_sha256=sha256(expanded/'summary.json'),
        original_head_replay_equal=True,group_ids={k:sorted(v) for k,v in groups.items()},metrics=metrics,
        expanded_vs_small_head=paired,
        interpretation='Matched evaluation sets. Added train samples are training data for expanded head, not held-out test data.',
        limitations=['Same three training videos, not more verified speakers','Same previously inspected validation video, six utterances',
            'Both heads use 200 full-batch updates; expanded training has greater total sample exposure'])
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print({g:{k:round(v['cer']*100,3) for k,v in m.items()} for g,m in metrics.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['original','expanded','train','validation','repo','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.original,args.expanded,args.train,args.validation,args.repo,args.output)
