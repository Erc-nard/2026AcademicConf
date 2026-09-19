"""Compare complete, frozen baselines against ROI availability masking."""

import argparse
import json
from pathlib import Path

from .batch import summarize_decisions
from .data import read_jsonl
from .evaluation import evaluate
from .preprocessing import sha256


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def predictions(path):
    return read_jsonl(path) if path.is_file() and path.stat().st_size else []


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, action='append', required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    all_refs, old_rows, new_rows, old_failures, new_failures = [], [], [], [], []
    videos, controls, frame_checks = {}, [], []
    for before_root in args.baseline:
        old = load(before_root/'summary.json')
        video = old['plan']['video_id']
        after_root = args.candidate/video
        new = load(after_root/'summary.json')
        before_refs = read_jsonl(before_root/'references.jsonl')
        after_refs = read_jsonl(after_root/'references.jsonl')
        if before_refs != after_refs or old['plan']['selected'] != new['plan']['selected']:
            raise ValueError('Candidate changed frozen IDs, times or references')
        before = predictions(before_root/'scale17_none/predictions.jsonl')
        after = predictions(after_root/'scale17_none/predictions.jsonl')
        bf = old['conditions']['scale17_none']['failures']
        af = new['conditions']['scale17_none']['failures']
        videos[video] = dict(baseline=summarize_decisions(before_refs, before, bf),
                            masked=summarize_decisions(after_refs, after, af),
                            baseline_plan_sha256=sha256(before_root/'plan.json'),
                            masked_plan_sha256=sha256(after_root/'plan.json'))
        before_by_id = {r['id']:r for r in before}
        for row in after:
            index = row['id'].split('/')[-1]
            after_crop = after_root/'crops/scale1.7_none'/index/'mouth.npz'
            with np.load(after_crop, allow_pickle=False) as archive:
                frames, observed, fps = archive['frames'], archive['roi_observed'], float(archive['fps'])
            item = next(r for r in new['plan']['selected'] if r['id'] == row['id'])
            expected = int(np.ceil(item['end_ms']*fps/1000)) - int(item['start_ms']*fps//1000)
            if len(frames) != expected or len(observed) != expected:
                raise ValueError('Changed timeline length')
            frame_checks.append(dict(id=row['id'], frames=len(frames), expected_frames=expected,
                                     usable_frames=int(observed.sum()), unavailable_frames=int((~observed).sum())))
            if row['id'] in before_by_id:
                with np.load(before_root/'crops/scale1.7_none'/index/'mouth.npz',allow_pickle=False) as archive:
                    same = np.array_equal(archive['frames'], frames)
                controls.append(dict(id=row['id'], all_observed=bool(observed.all()),
                    roi_pixels_identical=same, tokens_identical=row['token_ids'] == before_by_id[row['id']]['token_ids']))
        all_refs.extend(before_refs)
        old_rows.extend(before)
        new_rows.extend(after)
        old_failures.extend(bf)
        new_failures.extend(af)
    planned = {r['id'] for r in load(args.plan)['selected']}
    if len(all_refs) != len(planned) or {r['id'] for r in all_refs} != planned:
        raise ValueError('Combined references differ from preregistered plan')
    old_metric = summarize_decisions(all_refs, old_rows, old_failures)
    new_metric = summarize_decisions(all_refs, new_rows, new_failures)
    formerly_failed = {r['id'] for r in old_failures}
    newly_inferred = [r for r in new_rows if r['id'] in formerly_failed and r.get('inference_performed')]
    new_details = {r['id']:r for r in new_metric['all_selected_failure_as_empty']['details']}
    report = dict(scope='fixed train development; not held-out performance',
        plan_sha256=sha256(args.plan), plan=load(args.plan), baseline=old_metric, masked=new_metric,
        videos=videos, timeline_checks=frame_checks, controls=controls,
        newly_inferred_ids=[r['id'] for r in newly_inferred],
        newly_inferred_details=[dict(new_details[r['id']], availability=r['availability']) for r in newly_inferred],
        newly_inferred_metrics=evaluate([r for r in all_refs if r['id'] in {n['id'] for n in newly_inferred}],
                                       newly_inferred) if newly_inferred else None,
        abstained_ids=[r['id'] for r in new_rows if r.get('status') == 'abstained_no_usable_roi'],
        provenance={r['id']:{k:r[k] for k in ('checkpoint_sha256','input_sha256','upstream_commit',
                    'vocabulary_sha256','torch_version','input_frames','input_fps','input_size',
                    'audio_used','reference_used')} for r in new_rows},
        interpretation='Pipeline completion, model inference and abstention are separate. ROI availability is not ground-truth visibility. No mask channel was added to the model.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(baseline_cer=old_metric['all_selected_failure_as_empty']['cer'],
                         masked_cer=new_metric['all_selected_failure_as_empty']['cer'],
                         inferred=new_metric['inference_performed'], abstained=new_metric['abstained'],
                         controls_unchanged=all(c['roi_pixels_identical'] and c['tokens_identical'] for c in controls)),indent=2))


if __name__ == '__main__':
    main()
