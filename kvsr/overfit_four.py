"""Four-sample memorization diagnostic; no validation or generalization claim."""
import argparse
import ast
import csv
import json
from pathlib import Path
import re
import sys

from .data import read_jsonl, write_jsonl
from .evaluation import evaluate
from .head_adaptation import build_targets
from .olkavs import collapse_ctc
from .preprocessing import sha256


def run(output):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    repo = Path('external/olkavs').resolve()
    cache = Path('results/medium_experiment_2026-09-22')
    checkpoint = Path('checkpoints/olkavs_multiview_lip.pt')
    previous = json.loads((cache/'plan.json').read_text(encoding='utf-8'))
    if sha256(checkpoint) != previous['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    sys.path.insert(0, str(repo))
    from avsr.models.medium import MLPLayer
    from vocabulary.utils import char2grp, grp2char
    with (repo/'vocabulary/kor_grapheme.csv').open(encoding='utf-8-sig', newline='') as stream:
        vocab = {int(r['id']): r['char'] for r in csv.DictReader(stream)}
    tree = ast.parse((repo/'preprocess.py').read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'num2kor','unzip_groups','refine_transcription'}]
    if len(functions) != 3:
        raise ValueError('Missing official normalization functions')
    namespace = dict(re=re)
    exec(compile(ast.Module(body=functions,type_ignores=[]), 'official_text_functions', 'exec'), namespace)
    candidates = []
    for video in ['8r1k9NLdSvE','5TaVl7-lU4k','D-hzf2x_XO0']:
        root = Path('results/train48_2026-09-20')/video
        plan = json.loads((root/'plan.json').read_text(encoding='utf-8'))
        assert plan['split'] == 'train'
        refs = {r['id']: r for r in read_jsonl(root/'references.jsonl')}
        for item in plan['selected']:
            feature_path = cache/'encoded'/(item['id'].replace('/','_')+'.npy')
            if not feature_path.exists():
                continue
            mouth = root/'crops/scale1.7_none'/item['utterance']/'mouth.npz'
            entry = next(r for r in previous['inputs'] if r['id'] == item['id'])
            if sha256(mouth) != entry['mouth_sha256']:
                raise ValueError('Input changed')
            with np.load(mouth, allow_pickle=False) as data:
                if not data['roi_observed'].all():
                    continue
                frames = len(data['frames'])
            feature = np.load(feature_path, allow_pickle=False)
            if feature.shape != (frames,256) or feature.dtype != np.float32 or not np.isfinite(feature).all():
                raise ValueError('Invalid cached features')
            labels = build_targets(refs[item['id']]['reference'], namespace['refine_transcription'], char2grp,
                                   {v:k for k,v in vocab.items()}, frames)
            candidates.append(dict(id=item['id'], duration_ms=item['end_ms']-item['start_ms'],
                feature=feature, labels=labels, reference=refs[item['id']]['reference'],
                feature_sha256=sha256(feature_path), mouth_sha256=entry['mouth_sha256']))
    selected = sorted(candidates, key=lambda r:(r['duration_ms'],r['id']))[:4]
    if len(selected) != 4:
        raise ValueError('Need four fully observed train inputs')
    plan = dict(seed=2026, steps=1500, batch_size=4, learning_rate=.001, optimizer='Adam',
        selection='Four shortest train utterances with all ROI frames observed and feasible CTC targets; duration then ID; no prediction-based selection',
        encoder='frozen cached pre-medium features', batchnorm='fixed running statistics',
        checkpoint_sha256=sha256(checkpoint), source_plan_sha256=sha256(cache/'plan.json'),
        target='official normalized graphemes plus EOS', stopping='fixed 1500 steps; no early stopping',
        scope='memorization of four training samples only; no validation accessed',
        selected=[{k:r[k] for k in ['id','duration_ms','feature_sha256','mouth_sha256']} for r in selected])
    save('plan.json',plan)
    medium = MLPLayer(256,256).eval()
    head = torch.nn.Linear(256,72)
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    medium.load_state_dict({k.removeprefix('medium.'):v for k,v in state.items() if k.startswith('medium.')},strict=True)
    head.load_state_dict({k.removeprefix('decoder.ctcdecoder.fc.'):v for k,v in state.items() if k.startswith('decoder.ctcdecoder.fc.')},strict=True)
    x = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(r['feature']) for r in selected],batch_first=True)
    lengths = torch.tensor([len(r['feature']) for r in selected])
    labels = torch.tensor([t for r in selected for t in r['labels']])
    label_lengths = torch.tensor([len(r['labels']) for r in selected])
    raw_refs = [dict(id=r['id'],reference=r['reference']) for r in selected]
    normalized_refs = [dict(id=r['id'],reference=namespace['refine_transcription'](r['reference'])) for r in selected]
    prior = {r['id']:r['token_ids'] for r in read_jsonl(cache/'baseline.jsonl')}
    trace = []
    def measure(step):
        with torch.no_grad():
            logits = head(medium(x))
        rows = []
        for i,r in enumerate(selected):
            tokens = collapse_ctc(logits[i,:lengths[i]].argmax(-1).tolist())
            if step == 0 and tokens != prior[r['id']]:
                raise ValueError('Cached baseline replay mismatch')
            rows.append(dict(id=r['id'],hypothesis=grp2char(''.join(vocab[t] for t in tokens)),token_ids=tokens))
        value = dict(step=step, raw=evaluate(raw_refs,rows), normalized=evaluate(normalized_refs,rows))
        trace.append(value)
        write_jsonl(output/f'predictions_{step:04d}.jsonl',rows)
        print(f"step={step} raw_CER={value['raw']['cer']:.4f} normalized_CER={value['normalized']['cer']:.4f}",flush=True)
        return rows
    baseline = measure(0)
    params = list(medium.parameters())+list(head.parameters())
    optimizer = torch.optim.Adam(params,lr=.001)
    criterion = torch.nn.CTCLoss(blank=3,zero_infinity=False)
    losses = []
    for step in range(1,1501):
        optimizer.zero_grad()
        loss = criterion(head(medium(x)).log_softmax(-1).transpose(0,1),labels,lengths,label_lengths)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step%100 == 0:
            losses.append(dict(step=step,full_batch_loss_before_update=float(loss.detach())))
            final = measure(step)
    torch.save(dict(medium=medium.state_dict(),head=head.state_dict()),output/'final.pt')
    save('summary.json',dict(plan=plan,baseline_tokens_equal=True,trace=trace,loss_trace=losses,
        trainable_parameters=sum(p.numel() for p in params),checkpoint_sha256=sha256(output/'final.pt')))
    def cell(s):
        return s.replace('|','\\|').replace('\n',' ')
    lines = ['# 4개 학습 샘플 암기 실험','', '일반화 평가가 아닙니다. 정답은 이 네 샘플의 학습에 사용했습니다.','',
             '| ID | 원문 정답 | 학습용 정답 | 기존 출력 | 최종 출력 |','|---|---|---|---|---|']
    for r,n,b,f in zip(raw_refs,normalized_refs,baseline,final):
        lines.append('| '+' | '.join(cell(s) for s in [r['id'],r['reference'],n['reference'],b['hypothesis'],f['hypothesis']])+' |')
    (output/'sample_outputs.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
