"""Auxiliary punctuation sensitivity on frozen predictions; primary CER is unchanged."""
import argparse
from collections import Counter
import json
from pathlib import Path
import unicodedata

from .data import read_jsonl
from .evaluation import evaluate, normalize
from .preprocessing import sha256


def strip_punctuation(text):
    return ''.join(c for c in normalize(text) if not unicodedata.category(c).startswith('P'))


def audit(references, predictions):
    primary = evaluate(references, predictions)  # Includes strict ID validation.
    cleaned_refs = [dict(id=r['id'], reference=strip_punctuation(r['reference'])) for r in references]
    cleaned_hyps = [dict(id=r['id'], hypothesis=strip_punctuation(r['hypothesis'])) for r in predictions]
    auxiliary = evaluate(cleaned_refs, cleaned_hyps)
    auxiliary['normalization'] = 'NFC; delete Unicode category P characters on both sides; collapse whitespace; CER excludes whitespace'
    ref_categories = Counter(unicodedata.category(c) for r in references for c in normalize(r['reference'])
                             if unicodedata.category(c).startswith('P'))
    hyp_categories = Counter(unicodedata.category(c) for r in predictions for c in normalize(r['hypothesis'])
                             if unicodedata.category(c).startswith('P'))
    return dict(primary=primary, punctuation_removed_auxiliary=auxiliary,
        removed_reference_characters=sum(ref_categories.values()), removed_hypothesis_characters=sum(hyp_categories.values()),
        reference_punctuation_categories=dict(ref_categories), hypothesis_punctuation_categories=dict(hyp_categories),
        reference_characters_requiring_at_least_one_edit_if_hypotheses_have_no_punctuation=
            sum(ref_categories.values()) if not hyp_categories else None,
        limitations=['Auxiliary diagnostic only; primary scoring and predictions remain unchanged',
            'Changing the reference denominator means CER differences are not model error reductions',
            'Not an exact reproduction of upstream transcription refinement: digits, percent and alternative readings are untouched',
            'Removing punctuation can change the best edit alignment; error-count difference is not an attribution to individual errors'])


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    sources = []
    for video_id in ['8r1k9NLdSvE', '5TaVl7-lU4k', 'D-hzf2x_XO0']:
        root = source/video_id
        sources.append(dict(video_id=video_id, references_sha256=sha256(root/'references.jsonl'),
            predictions_sha256=sha256(root/'scale17_none/predictions.jsonl')))
    plan = dict(sources=sources, scope='same frozen 16 masked-baseline train development predictions',
        rule='Delete Unicode general-category P characters from references and hypotheses; retain all other scoring rules',
        model_rerun=False, primary_metric_changed=False, unicode_version=unicodedata.unidata_version)
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    references, predictions = [], []
    per_video = {}
    for item in sources:
        root = source/item['video_id']
        refs = read_jsonl(root/'references.jsonl')
        hyps = read_jsonl(root/'scale17_none/predictions.jsonl')
        references.extend(refs)
        predictions.extend(hyps)
        per_video[item['video_id']] = audit(refs, hyps)
    if len(references) != 16:
        raise ValueError('Expected the fixed 16 development utterances')
    result = dict(plan=plan, plan_sha256=sha256(output/'plan.json'), overall=audit(references,predictions), per_video=per_video)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result['overall'][k] for k in ['removed_reference_characters', 'removed_hypothesis_characters']}))
    for key in ['primary','punctuation_removed_auxiliary']:
        print(key, json.dumps({k:result['overall'][key][k] for k in ['char_errors','reference_chars','cer','wer']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source,args.output)
