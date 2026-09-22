"""Actual CPU model experiment: matched head-only versus medium+head training."""
import argparse
import ast
import copy
import csv
import json
from pathlib import Path
import re
import sys

from .data import read_jsonl,write_jsonl
from .evaluation import evaluate
from .head_adaptation import build_targets
from .missing import normalize_available
from .olkavs import prepare_frames,chunk_frontend,collapse_ctc
from .preprocessing import sha256


def write_examples(output,references,predictions,ids):
    """Local-only reference/output comparison; never included in public metrics."""
    def cell(text):
        return text.replace('|','\\|').replace('\n',' ')
    mapped={name:{r['id']:r['hypothesis'] for r in rows} for name,rows in predictions.items()}
    lines=['# 실제 validation 자막 비교','',
        '빈 출력은 ∅로 표시한다. 실패한 입력도 포함하며, 정답은 평가 단계에서만 사용했다.','',
        '| 발화 | 정답 | 기존 모델 | 출력층 학습 | 중간층+출력층 학습 |',
        '|---|---|---|---|---|']
    for row in references:
        if row['id'] not in ids:
            continue
        values=[row['id'],row['reference']]+[mapped[name][row['id']] or '∅' for name in ['baseline','head_only','medium_head']]
        lines.append('| '+' | '.join(cell(v) for v in values)+' |')
    (output/'sample_outputs.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(output):
    import numpy as np
    import torch
    output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    torch.use_deterministic_algorithms(True)
    repo=Path('external/olkavs').resolve()
    checkpoint=Path('checkpoints/olkavs_multiview_lip.pt')
    sys.path.insert(0,str(repo))
    from avsr.utils.model_builder import build_model
    from vocabulary.utils import char2grp,grp2char
    with (repo/'vocabulary/kor_grapheme.csv').open(encoding='utf-8-sig',newline='') as stream:
        vocabulary={int(r['id']):r['char'] for r in csv.DictReader(stream)}
    char_to_id={v:k for k,v in vocabulary.items()}
    tree=ast.parse((repo/'preprocess.py').read_text(encoding='utf-8'))
    namespace=dict(re=re)
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'num2kor','unzip_groups','refine_transcription'}]
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(repo/'preprocess.py'),'exec'),namespace)
    roots=[Path('results/train48_2026-09-20')/v for v in ['8r1k9NLdSvE','5TaVl7-lU4k','D-hzf2x_XO0']]
    roots += [Path('results/validation_2026-09-20/rGR4FdaIvw8'),Path('results/validation_extension_2026-09-20/VYLaaU2kAsE')]
    rows=[]
    prior={}
    for root in roots:
        plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
        summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
        failed={r['id'] for r in summary['conditions']['scale17_none']['failures']}
        prior.update({r['id']:r for r in read_jsonl(root/'scale17_none/predictions.jsonl')})
        for item in plan['selected']:
            rows.append(dict(id=item['id'],split=plan['split'],root=root,utterance=item['utterance'],failed=item['id'] in failed))
    if len(rows)!=60 or sum(r['split']=='train' for r in rows)!=48 or len({r['id'] for r in rows})!=60:
        raise ValueError('Expected fixed 48 train and 12 validation samples')
    plan=dict(seed=2026,steps=400,batch_size=4,optimizer='Adam',learning_rate=.0001,gradient_clip=1.,
        conditions=['baseline','head_only','medium_head'],checkpoint_selection='fixed final step, no validation selection',
        batchnorm='eval mode, fixed running statistics in both conditions',encoder='frozen',
        checkpoint_sha256=sha256(checkpoint),source_plan_hashes={root.name:sha256(root/'plan.json') for root in roots},
        inputs=[dict(id=r['id'],split=r['split'],failed=r['failed'],mouth_sha256=None if r['failed'] else
                     sha256(r['root']/'crops/scale1.7_none'/r['utterance']/'mouth.npz')) for r in rows],
        scope='repeated train/valid development data; not held-out test',validation_used_for_training=False)
    save('plan.json',plan)
    model=build_model(72,0,architecture='video',loss_fn='hybrid',front_dim=512,
        encoder_n_layer=12,encoder_d_model=256,encoder_n_head=8,encoder_ff_dim=2048,encoder_dropout_p=.1,
        decoder_n_layer=6,decoder_d_model=256,decoder_n_head=8,decoder_ff_dim=2048,decoder_dropout_p=.1,
        pass_visual_frontend=False,verbose=False)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False)
    features={}
    predictions=dict(baseline=[])
    (output/'encoded').mkdir()
    for i,row in enumerate(rows):
        feature=None
        tokens=[]
        if not row['failed']:
            with np.load(row['root']/'crops/scale1.7_none'/row['utterance']/'mouth.npz',allow_pickle=False) as data:
                frames,_=prepare_frames(data['frames'],float(data['fps']),96,None)
                observed=data['roi_observed']
            if observed.any():
                video=normalize_available(frames,observed)
                with torch.inference_mode():
                    front=chunk_frontend(model.encoder.front,video)
                    encoded=model.encoder.back(model.encoder.projection(front),torch.tensor([len(frames)]))
                    logits=model.decoder.ctcdecoder(model.medium(encoded))[0]
                    tokens=collapse_ctc(logits.argmax(-1).tolist())
                feature=encoded[0].clone()
                np.save(output/'encoded'/(row['id'].replace('/','_')+'.npy'),feature.numpy(),allow_pickle=False)
            if tokens!=prior[row['id']]['token_ids']:
                raise ValueError('Recomputed baseline differs')
        features[row['id']]=feature
        predictions['baseline'].append(dict(id=row['id'],hypothesis=grp2char(''.join(vocabulary[t] for t in tokens)),token_ids=tokens))
        print(f'Video encoder {i+1}/60',flush=True)
    train_refs=[]
    for root in roots[:3]:
        train_refs.extend(read_jsonl(root/'references.jsonl'))
    eligible=[]
    exclusions=[]
    for row in train_refs:
        feature=features[row['id']]
        if feature is None:
            exclusions.append(dict(id=row['id'],reason='no usable input'))
            continue
        try:
            labels=build_targets(row['reference'],namespace['refine_transcription'],char2grp,char_to_id,len(feature))
        except ValueError as error:
            exclusions.append(dict(id=row['id'],reason=str(error)))
            continue
        eligible.append((feature,labels))
    if len(eligible)<4:
        raise ValueError('Too few eligible samples')
    schedule=[np.random.default_rng(2026+s).choice(len(eligible),4,replace=False).tolist() for s in range(400)]
    save('training_schedule.json',dict(eligible=[r['id'] for r in train_refs if r['id'] not in {e['id'] for e in exclusions}],batches=schedule,exclusions=exclusions))
    criterion=torch.nn.CTCLoss(blank=3,reduction='mean',zero_infinity=False)
    traces={}
    parameter_counts={}
    for condition in ['head_only','medium_head']:
        medium=copy.deepcopy(model.medium).eval().requires_grad_(condition=='medium_head')
        head=copy.deepcopy(model.decoder.ctcdecoder).requires_grad_(True)
        parameters=[p for module in [medium,head] for p in module.parameters() if p.requires_grad]
        parameter_counts[condition]=sum(p.numel() for p in parameters)
        optimizer=torch.optim.Adam(parameters,lr=.0001)
        traces[condition]=[]
        for step,indices in enumerate(schedule,1):
            batch=[eligible[i] for i in indices]
            x=torch.nn.utils.rnn.pad_sequence([f for f,t in batch],batch_first=True)
            lengths=torch.tensor([len(f) for f,t in batch])
            target_lengths=torch.tensor([len(t) for f,t in batch])
            targets=torch.tensor([v for f,t in batch for v in t])
            optimizer.zero_grad()
            loss=criterion(head(medium(x)).log_softmax(-1).transpose(0,1),targets,lengths,target_lengths)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step()
            if step==1 or step%50==0:
                traces[condition].append(dict(step=step,minibatch_loss=float(loss.detach())))
                print(f'{condition} {step}/400 loss={loss.item():.4f}',flush=True)
        torch.save(dict(medium=medium.state_dict(),head=head.state_dict()),output/(condition+'.pt'))
        predictions[condition]=[]
        with torch.inference_mode():
            for row in rows:
                feature=features[row['id']]
                tokens=collapse_ctc(head(medium(feature.unsqueeze(0)))[0].argmax(-1).tolist()) if feature is not None else []
                predictions[condition].append(dict(id=row['id'],hypothesis=grp2char(''.join(vocabulary[t] for t in tokens)),token_ids=tokens))
        write_jsonl(output/(condition+'.jsonl'),predictions[condition])
    write_jsonl(output/'baseline.jsonl',predictions['baseline'])
    # Both model conditions are finished before reading validation references.
    refs=list(train_refs)
    for root in roots[3:]:
        refs.extend(read_jsonl(root/'references.jsonl'))
    groups=dict(train={r['id'] for r in rows if r['split']=='train'},validation={r['id'] for r in rows if r['split']=='valid'})
    for root in roots[3:]:
        groups[root.name]={r['id'] for r in rows if r['root']==root}
    metrics={g:{name:evaluate([r for r in refs if r['id'] in ids],[r for r in values if r['id'] in ids])
        for name,values in predictions.items()} for g,ids in groups.items()}
    write_examples(output,refs,predictions,groups['validation'])
    save('summary.json',dict(plan=plan,plan_sha256=sha256(output/'plan.json'),metrics=metrics,
        trainable_parameters=parameter_counts,train_eligible=len(eligible),training_exclusions=exclusions,
        baseline_tokens_equal=True,loss_trace=traces,torch_version=torch.__version__,
        checkpoint_hashes={name:sha256(output/(name+'.pt')) for name in ['head_only','medium_head']},
        interpretation='Same 400 minibatch updates, identical sampling, fixed last checkpoint. Missing/failed samples count as empty outputs.'))
    print({g:{k:round(v['cer']*100,2) for k,v in m.items()} for g,m in metrics.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
