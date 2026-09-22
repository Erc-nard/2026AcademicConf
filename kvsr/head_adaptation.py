"""Fixed CPU pilot: train only the CTC linear head on cached, frozen video features.

This is a tiny adaptation/overfitting diagnostic, not a replacement baseline.
Validation predictions exist from a prior baseline check; no checkpoint selection
or hyperparameter tuning is performed on their references here.
"""
import argparse
import ast
import copy
import csv
import json
from pathlib import Path
import re
import sys

from .data import read_jsonl, write_jsonl
from .evaluation import evaluate,indexed
from .preprocessing import sha256


def require_ctc_alignment(tokens, frames):
    if not tokens or any(t in (0,1,3) for t in tokens):
        raise ValueError('CTC targets must be nonempty and exclude pad/SOS/blank')
    minimum = len(tokens)+sum(a==b for a,b in zip(tokens,tokens[1:]))
    if frames < minimum:
        raise ValueError('CTC alignment is impossible at the retained frame count')
    return minimum


def build_targets(reference, refine, decompose, char_to_id, frames):
    text=refine(reference)
    if not text.strip():
        raise ValueError('Empty normalized training target')
    graphemes=decompose(text)
    unknown=set(graphemes)-char_to_id.keys()
    if unknown:
        raise ValueError('Normalized training target contains unsupported vocabulary entries')
    tokens=[char_to_id[g] for g in graphemes]+[2]
    require_ctc_alignment(tokens,frames)
    return tokens


def read_cached_feature(root, summary, sample_id, mouth_hash, checkpoint_hash, frames):
    """Reject stale or corrupted caches; absence of an entry is a cache miss."""
    import numpy as np
    if summary['plan']['checkpoint_sha256']!=checkpoint_hash:
        raise ValueError('Feature cache checkpoint mismatch')
    entries={r['id']:r for r in summary['plan']['inputs']}
    if sample_id not in entries:
        return None
    if entries[sample_id]['mouth_sha256']!=mouth_hash:
        raise ValueError('Feature cache input mismatch')
    video_id,utterance=sample_id.split('/')
    name=f'{video_id}_{utterance}.npy'
    path=root/'features'/name
    expected=summary['feature_hashes'].get(name)
    if expected is None:
        return None
    if sha256(path)!=expected:
        raise ValueError('Feature cache bytes changed')
    feature=np.load(path,allow_pickle=False)
    if feature.dtype!=np.float32 or feature.shape!=(frames,256) or not np.isfinite(feature).all():
        raise ValueError('Feature cache shape, type or values invalid')
    return feature


def run(train, validation, checkpoint, repo, output, feature_cache=None):
    import numpy as np
    import torch
    from .missing import normalize_available
    from .olkavs import prepare_frames,chunk_frontend,collapse_ctc
    output.mkdir(parents=True,exist_ok=False)

    def save(name,value):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    torch.set_num_threads(2)
    torch.manual_seed(2026)
    torch.use_deterministic_algorithms(True)
    repo=repo.resolve()
    sys.path.insert(0,str(repo))
    from avsr.utils.model_builder import build_model
    from vocabulary.utils import char2grp,grp2char
    with (repo/'vocabulary/kor_grapheme.csv').open(encoding='utf-8-sig',newline='') as stream:
        vocabulary={int(r['id']):r['char'] for r in csv.DictReader(stream)}
    char_to_id={v:k for k,v in vocabulary.items()}
    # Reviewed official text-only functions, without importing audio/video dependencies.
    tree=ast.parse((repo/'preprocess.py').read_text(encoding='utf-8'))
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'num2kor','unzip_groups','refine_transcription'}]
    if len(selected)!=3:
        raise ValueError('Official transcription functions missing')
    namespace=dict(re=re)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(repo/'preprocess.py'),'exec'),namespace)
    rows=[]
    for video_id in ['8r1k9NLdSvE','5TaVl7-lU4k','D-hzf2x_XO0']:
        root=train/video_id
        plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
        if plan['split']!='train':
            raise ValueError('Only train references may be used for adaptation')
        for row in plan['selected']:
            rows.append(dict(id=row['id'],utterance=row['utterance'],video_id=video_id,split='train',root=root))
    valid_summary=json.loads((validation/'summary.json').read_text(encoding='utf-8'))
    for video in valid_summary['plan']['videos']:
        if valid_summary['per_video'][video['video_id']].get('acquisition_status'):
            continue
        root=validation/video['video_id']
        actual=json.loads((root/'plan.json').read_text(encoding='utf-8'))
        if actual['split']!='valid':
            raise ValueError('Validation source split mismatch')
        for row in actual['selected']:
            rows.append(dict(id=row['id'],utterance=row['utterance'],video_id=video['video_id'],split='valid',root=root))
    train_count=sum(r['split']=='train' for r in rows)
    if train_count not in (16,48) or len(rows)-train_count!=6 or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Expected frozen 16/48 train and six available validation samples')
    failed_inputs={}
    for root in {r['root'] for r in rows}:
        source_summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
        for failure in source_summary['conditions']['scale17_none']['failures']:
            failed_inputs[failure['id']]=failure['stage']
    inputs=[]
    for row in rows:
        mouth=row['root']/'crops/scale1.7_none'/row['utterance']/'mouth.npz'
        inputs.append(dict(id=row['id'],split=row['split'],mouth_sha256=sha256(mouth) if row['id'] not in failed_inputs else None))
    cached_summary=json.loads((feature_cache/'summary.json').read_text(encoding='utf-8')) if feature_cache else None
    frozen=dict(scope='small train adaptation pilot; one previously inspected validation video, not a held-out test',
        inputs=inputs,checkpoint_sha256=sha256(checkpoint),official_refinement_sha256=sha256(repo/'preprocess.py'),
        vocabulary_sha256=sha256(repo/'vocabulary/kor_grapheme.csv'),
        train_reference_hashes={v:sha256(train/v/'references.jsonl') for v in ['8r1k9NLdSvE','5TaVl7-lU4k','D-hzf2x_XO0']},
        validation_summary_sha256=sha256(validation/'summary.json'),
        feature_cache_summary_sha256=sha256(feature_cache/'summary.json') if feature_cache else None,
        excluded_training_policy='no features, empty/unsupported normalized target, or impossible CTC alignment; all retained in evaluation',
        settings=dict(seed=2026,steps=200,optimizer='Adam',learning_rate=.001,weight_decay=0.,gradient_clip_norm=1.,
            batch='all usable train utterances',loss='CTCLoss mean, true input lengths, blank=3, zero_infinity=False',
            target='official refine_transcription -> char2grp -> vocabulary, append EOS=2',
            trainable='only decoder.ctcdecoder.fc weight and bias; encoder/medium/attention frozen',
            checkpoint_selection='fixed final step; no early stopping or validation selection'),
        audio_used=False,validation_references_used_in_training=False,
        limitations=['A small set of train utterances from three videos is insufficient for a generalization claim',
            'Failed/all-missing samples are excluded from gradient updates but retained as empty output in evaluation',
            'Private validation video remains unavailable and is not used for training or model-quality scoring'])
    save('plan.json',frozen)
    model=build_model(72,0,architecture='video',loss_fn='hybrid',front_dim=512,
        encoder_n_layer=12,encoder_d_model=256,encoder_n_head=8,encoder_ff_dim=2048,encoder_dropout_p=.1,
        decoder_n_layer=6,decoder_d_model=256,decoder_n_head=8,decoder_ff_dim=2048,decoder_dropout_p=.1,
        pass_visual_frontend=False,verbose=False)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False)
    head=copy.deepcopy(model.decoder.ctcdecoder).requires_grad_(True)
    features={}
    baseline=[]
    cache_hits=0
    (output/'features').mkdir()
    for index,row in enumerate(rows,1):
        if row['id'] in failed_inputs:
            features[row['id']]=None
            baseline.append(dict(id=row['id'],hypothesis='',token_ids=[]))
            continue
        mouth=row['root']/'crops/scale1.7_none'/row['utterance']/'mouth.npz'
        with np.load(mouth,allow_pickle=False) as data:
            frames,_=prepare_frames(data['frames'],float(data['fps']),96,None)
            observed=data['roi_observed']
        if observed.any():
            cached=read_cached_feature(feature_cache,cached_summary,row['id'],sha256(mouth),frozen['checkpoint_sha256'],len(frames)) if cached_summary else None
            if cached is not None:
                features[row['id']]=torch.from_numpy(cached.copy())
                cache_hits+=1
            else:
                video=normalize_available(frames,observed)
                with torch.inference_mode():
                    front=chunk_frontend(model.encoder.front,video)
                    encoded=model.encoder.back(model.encoder.projection(front),torch.tensor([len(frames)]))
                    feature=model.medium(encoded)[0]
                features[row['id']]=feature.clone()  # Outside inference_mode: safe for head gradients.
            with torch.inference_mode():
                tokens=collapse_ctc(model.decoder.ctcdecoder(features[row['id']]).argmax(-1).tolist())
            np.save(output/'features'/f"{row['video_id']}_{row['utterance']}.npy",features[row['id']].numpy(),allow_pickle=False)
        else:
            features[row['id']]=None
            tokens=[]
        baseline.append(dict(id=row['id'],hypothesis=grp2char(''.join(vocabulary[t] for t in tokens)),token_ids=tokens))
        print(f'Frozen features {index}/{len(rows)}',flush=True)
    prior={}
    for root in {r['root'] for r in rows}:
        prior.update({r['id']:r for r in read_jsonl(root/'scale17_none/predictions.jsonl')})
    prior.update({sample_id:dict(token_ids=[]) for sample_id in failed_inputs})
    if not all(row['token_ids']==prior[row['id']]['token_ids'] for row in baseline):
        raise ValueError('Feature-cache baseline differs from recorded predictions')
    train_refs=[]
    for video_id in ['8r1k9NLdSvE','5TaVl7-lU4k','D-hzf2x_XO0']:
        train_refs.extend(read_jsonl(train/video_id/'references.jsonl'))
    if set(indexed(train_refs,'reference'))!={r['id'] for r in rows if r['split']=='train'}:
        raise ValueError('Training references must match the frozen training IDs exactly')
    selected_refs,labels,exclusions=[],[],[]
    for reference in train_refs:
        feature=features[reference['id']]
        if feature is None:
            exclusions.append(dict(id=reference['id'],reason='no usable features',stage=failed_inputs.get(reference['id'],'all_roi_missing')))
            continue
        try:
            target=build_targets(reference['reference'],namespace['refine_transcription'],char2grp,char_to_id,len(feature))
        except ValueError as error:
            exclusions.append(dict(id=reference['id'],reason=str(error),stage='training_target'))
            continue
        selected_refs.append(reference)
        labels.append(target)
    if not selected_refs:
        raise ValueError('No eligible training samples')
    save('training_eligibility.json',dict(selected=len(train_refs),used=len(selected_refs),excluded=exclusions))
    x=torch.nn.utils.rnn.pad_sequence([features[r['id']] for r in selected_refs],batch_first=True)
    lengths=torch.tensor([len(features[r['id']]) for r in selected_refs],dtype=torch.long)
    targets=torch.tensor([t for label in labels for t in label],dtype=torch.long)
    target_lengths=torch.tensor([len(label) for label in labels],dtype=torch.long)
    criterion=torch.nn.CTCLoss(blank=3,reduction='mean',zero_infinity=False)
    optimizer=torch.optim.Adam(head.parameters(),lr=.001,weight_decay=0.)
    losses=[]
    for step in range(201):
        loss=criterion(head(x).log_softmax(-1).transpose(0,1),targets,lengths,target_lengths)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss; no zero-infinity suppression')
        if step%20==0:
            losses.append(dict(step=step,loss=float(loss.detach())))
            print(f'Head step {step}/200, loss={loss.item():.5f}',flush=True)
        if step==200:
            break
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
    torch.save(head.state_dict(),output/'ctc_head.pt')
    adapted=[]
    with torch.inference_mode():
        for row in rows:
            feature=features[row['id']]
            tokens=collapse_ctc(head(feature).argmax(-1).tolist()) if feature is not None else []
            adapted.append(dict(id=row['id'],hypothesis=grp2char(''.join(vocabulary[t] for t in tokens)),token_ids=tokens))
    write_jsonl(output/'baseline_predictions.jsonl',baseline)
    write_jsonl(output/'adapted_predictions.jsonl',adapted)
    # Validation references are read only after all training and decoding finish.
    valid_refs=[]
    for root in {r['root'] for r in rows if r['split']=='valid'}:
        valid_refs.extend(read_jsonl(root/'references.jsonl'))
    metrics={}
    for name,refs in [('train_all',train_refs),('validation_available',valid_refs)]:
        ids={r['id'] for r in refs}
        metrics[name]={condition:evaluate(refs,[r for r in predictions if r['id'] in ids])
            for condition,predictions in [('baseline',baseline),('adapted',adapted)]}
    save('summary.json',dict(plan=frozen,plan_sha256=sha256(output/'plan.json'),metrics=metrics,loss_trace=losses,
        trainable_parameters=sum(p.numel() for p in head.parameters()),
        train_samples_with_features=sum(features[r['id']] is not None for r in train_refs),
        train_samples_used_for_updates=len(selected_refs),
        train_all_missing_retained_in_evaluation=sum(features[r['id']] is None and r['id'] not in failed_inputs for r in train_refs),
        training_exclusions=exclusions,feature_cache_hits=cache_hits,
        baseline_tokens_equal_previous=True,head_sha256=sha256(output/'ctc_head.pt'),torch_version=torch.__version__,
        frozen_parameter_tensors=sum(1 for p in model.parameters() if not p.requires_grad),
        feature_hashes={p.name:sha256(p) for p in sorted((output/'features').glob('*.npy'))}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['train','validation','checkpoint','repo','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--feature-cache',type=Path)
    args=parser.parse_args()
    run(args.train,args.validation,args.checkpoint,args.repo,args.output,args.feature_cache)
