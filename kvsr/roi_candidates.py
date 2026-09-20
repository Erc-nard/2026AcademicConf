"""Reference-assisted oracle across frozen ROI variants, never a deployable selector."""
import argparse
import json
from pathlib import Path

from .data import read_jsonl
from .nbest import evaluate_nbest
from .preprocessing import sha256


def run(source, references_root, output):
    if output.exists():
        raise ValueError('Output exists')
    summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if not summary['control_tokens_equal_previous'] or summary['coverage']['completed']!=48:
        raise ValueError('ROI replay control or coverage was not verified')
    conditions = summary['plan']['conditions']
    predictions, hashes = [], []
    for row in summary['plan']['inputs']:
        candidates, seen = [], set()
        for name in conditions:
            path = source/name/row['video_id']/row['utterance']/'prediction.jsonl'
            decoded = read_jsonl(path)
            if len(decoded)!=1 or decoded[0]['id']!=row['id']:
                raise ValueError('Prediction ID mismatch')
            hypothesis = decoded[0]['hypothesis']
            if hypothesis not in seen:
                candidates.append(dict(hypothesis=hypothesis))
                seen.add(hypothesis)
            hashes.append(dict(id=row['id'],condition=name,prediction_sha256=sha256(path)))
        predictions.append(dict(id=row['id'],candidates=candidates))
    references, ref_hashes = [], []
    for item in summary['plan']['sources']:
        path = references_root/item['video_id']/'references.jsonl'
        references.extend(read_jsonl(path))
        ref_hashes.append(dict(video_id=item['video_id'],sha256=sha256(path)))
    result = dict(source_summary_sha256=sha256(source/'summary.json'), input_hashes=hashes,
        reference_hashes=ref_hashes, condition_order=conditions,
        report=evaluate_nbest(references,predictions,ks=(1,2,3)),
        scope='same 16 train development utterances, including one all-ROI-missing abstention',
        duplicate_policy='deduplicate identical rendered hypotheses, preserve fixed condition order',
        inference_rerun=False, interpretation='Uses ground truth to select the least-CER whole-utterance candidate; not available at deployment')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print({k:(v['char_errors'],v['reference_chars'],v['cer']) for k,v in result['report']['oracle_by_k'].items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--references',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.references,args.output)
