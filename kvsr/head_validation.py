"""Evaluate fixed CTC heads on an additional validation video without training."""
import argparse
import csv
import json
from pathlib import Path
import sys

from .data import read_jsonl,write_jsonl
from .evaluation import evaluate
from .preprocessing import sha256


def run(batch,selection,small,expanded,checkpoint,repo,output):
    import numpy as np
    import torch
    from .missing import normalize_available
    from .olkavs import prepare_frames,chunk_frontend,collapse_ctc
    if output.exists():
        raise ValueError('Output exists')
    planned=json.loads(selection.read_text(encoding='utf-8'))
    source=json.loads((batch/'summary.json').read_text(encoding='utf-8'))
    actual=source['plan']
    if actual['split']!='valid' or actual['selected']!=planned['selected'] or actual['missing_policy']!='mask':
        raise ValueError('Validation selection or settings changed')
    base_hash=sha256(checkpoint)
    old=json.loads((small/'summary.json').read_text(encoding='utf-8'))
    new=json.loads((expanded/'summary.json').read_text(encoding='utf-8'))
    for root,summary in [(small,old),(expanded,new)]:
        if summary['plan']['checkpoint_sha256']!=base_hash or sha256(root/'ctc_head.pt')!=summary['head_sha256']:
            raise ValueError('Head checkpoint provenance mismatch')
        if any(r['id'].split('/')[0]==actual['video_id'] for r in summary['plan']['inputs']):
            raise ValueError('Additional video was already present in the adaptation experiment')
    if actual['hashes']['checkpoint']!=base_hash or source['plan_sha256']!=sha256(batch/'plan.json'):
        raise ValueError('Baseline checkpoint or plan mismatch')
    output.mkdir(parents=True)
    plan=dict(selection_sha256=sha256(selection),batch_summary_sha256=sha256(batch/'summary.json'),
        small_head_sha256=old['head_sha256'],expanded_head_sha256=new['head_sha256'],base_checkpoint_sha256=base_hash,
        video_id=actual['video_id'],selected=actual['selected'],training_performed=False,
        scope='additional official valid video, fixed heads; no checkpoint selection on these references')
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    torch.set_num_threads(2)
    sys.path.insert(0,str(repo.resolve()))
    from avsr.utils.model_builder import build_model
    from vocabulary.utils import grp2char
    with (repo/'vocabulary/kor_grapheme.csv').open(encoding='utf-8-sig',newline='') as stream:
        vocabulary={int(r['id']):r['char'] for r in csv.DictReader(stream)}
    for summary in [old,new]:
        if summary['plan']['vocabulary_sha256']!=sha256(repo/'vocabulary/kor_grapheme.csv'):
            raise ValueError('Vocabulary mismatch')
    model=build_model(72,0,architecture='video',loss_fn='hybrid',front_dim=512,
        encoder_n_layer=12,encoder_d_model=256,encoder_n_head=8,encoder_ff_dim=2048,encoder_dropout_p=.1,
        decoder_n_layer=6,decoder_d_model=256,decoder_n_head=8,decoder_ff_dim=2048,decoder_dropout_p=.1,
        pass_visual_frontend=False,verbose=False)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.eval()
    heads=dict(small_head=torch.load(small/'ctc_head.pt',weights_only=True,map_location='cpu'),
               expanded_head=torch.load(expanded/'ctc_head.pt',weights_only=True,map_location='cpu'))
    predictions={name:[] for name in ['baseline','small_head','expanded_head']}
    failures=source['conditions']['scale17_none']['failures']
    failed={r['id'] for r in failures}
    availability=[]
    for item in actual['selected']:
        feature=None
        observed_count=0
        count=None
        if item['id'] not in failed:
            with np.load(batch/'crops/scale1.7_none'/item['utterance']/'mouth.npz',allow_pickle=False) as data:
                frames,_=prepare_frames(data['frames'],float(data['fps']),96,None)
                observed=data['roi_observed']
            count=len(frames)
            observed_count=int(observed.sum())
            if observed.any():
                video=normalize_available(frames,observed)
                with torch.inference_mode():
                    front=chunk_frontend(model.encoder.front,video)
                    encoded=model.encoder.back(model.encoder.projection(front),torch.tensor([len(frames)]))
                    feature=model.medium(encoded)[0]
        availability.append(dict(id=item['id'],frames=count,usable_frames=observed_count,inference_performed=feature is not None))
        for name in predictions:
            tokens=[]
            if feature is not None:
                with torch.inference_mode():
                    logits=model.decoder.ctcdecoder(feature) if name=='baseline' else torch.nn.functional.linear(
                        feature,heads[name]['fc.weight'],heads[name]['fc.bias'])
                    tokens=collapse_ctc(logits.argmax(-1).tolist())
            predictions[name].append(dict(id=item['id'],hypothesis=grp2char(''.join(vocabulary[t] for t in tokens)),token_ids=tokens))
        print('Decoded fixed heads:',item['id'],flush=True)
    prior={r['id']:r for r in read_jsonl(batch/'scale17_none/predictions.jsonl')}
    if not all(r['token_ids']==prior[r['id']]['token_ids'] for r in predictions['baseline'] if r['id'] not in failed):
        raise ValueError('Recomputed baseline differs')
    for name,rows in predictions.items():
        write_jsonl(output/(name+'.jsonl'),rows)
    # Evaluation references do not enter feature extraction or any of the heads.
    references=read_jsonl(batch/'references.jsonl')
    report=dict(plan=plan,plan_sha256=sha256(output/'plan.json'),metrics={k:evaluate(references,v) for k,v in predictions.items()},
        availability=availability,failures=failures,baseline_tokens_equal=True,
        inference_performed=sum(r['inference_performed'] for r in availability),
        limitations=['One additional video with six selected utterances is not a final test set',
            'Previous inaccessible validation video remains recorded separately, not replaced'])
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print({k:(v['char_errors'],v['reference_chars'],v['cer']) for k,v in report['metrics'].items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['batch','selection','small','expanded','checkpoint','repo','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.batch,args.selection,args.small,args.expanded,args.checkpoint,args.repo,args.output)
