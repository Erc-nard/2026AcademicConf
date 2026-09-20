"""Fixed 2x2 final-resize/MP4 sensitivity experiment; not training ROI reproduction."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .data import read_jsonl
from .evaluation import evaluate
from .preprocessing import sha256


CONDITIONS = [('area', 'npz'), ('linear', 'npz'), ('area', 'mp4'), ('linear', 'mp4')]


def transform(frames, fps, interpolation, transport, directory):
    """Resize the existing 224px ROI once, then optionally encode/decode mp4v."""
    import cv2
    import numpy as np
    if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (224, 224, 3)
            or len(frames) < 1 or not np.isfinite(fps) or fps <= 0):
        raise ValueError('Expected nonempty uint8 224px BGR ROIs and positive FPS')
    if (interpolation, transport) not in CONDITIONS:
        raise ValueError('Unknown condition')
    method = cv2.INTER_AREA if interpolation == 'area' else cv2.INTER_LINEAR
    resized = np.stack([cv2.resize(frame, (96, 96), interpolation=method) for frame in frames])
    decoded = resized
    if transport == 'mp4':
        path = directory / 'transport.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (96, 96))
        if not writer.isOpened():
            raise RuntimeError('mp4v encoder unavailable')
        try:
            for frame in resized:
                writer.write(frame)
        finally:
            writer.release()
        cap = cv2.VideoCapture(str(path))
        decoded = []
        try:
            stored_fps = cap.get(cv2.CAP_PROP_FPS)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                decoded.append(frame)
        finally:
            cap.release()
        if len(decoded) != len(resized) or not np.isclose(stored_fps, fps, atol=1e-6, rtol=0):
            raise ValueError('MP4 changed frame count or FPS')
        decoded = np.stack(decoded)
        if decoded.shape != resized.shape:
            raise ValueError('MP4 changed spatial shape')
    error = decoded.astype(np.float32)-resized.astype(np.float32)
    return decoded, dict(frames=len(decoded), fps=fps, shape=list(decoded.shape),
        transport_pixel_mae=float(np.abs(error).mean()), transport_pixel_mse=float((error**2).mean()),
        transport_pixel_max_abs=float(np.abs(error).max()), timeline_preserved=True)


def run(source, checkpoint, repo, output):
    import cv2
    import numpy as np
    source_plan = json.loads((source/'plan.json').read_text(encoding='utf-8'))
    if source_plan['split'] != 'train' or len(source_plan['selected']) != 4:
        raise ValueError('This experiment requires the fixed four train development clips')
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    inputs = [dict(id=row['id'], utterance=row['utterance'],
        sha256=sha256(source/'crops/scale1.7_none'/row['utterance']/'mouth.npz')) for row in source_plan['selected']]
    plan = dict(scope='four fixed utterances from one train video; exploratory paired sensitivity, no held-out claim',
        source_plan_sha256=sha256(source/'plan.json'), checkpoint_sha256=sha256(checkpoint), inputs=inputs,
        conditions=[dict(name=f'{a}_{b}', interpolation=a, transport=b) for a,b in CONDITIONS],
        fixed=dict(size=96, target_fps=None, crop_scale=1.7, alignment='none', decoder='ctc_greedy', threads=2),
        opencv_version=cv2.__version__, reference_used_in_decoding=False, audio_used=False,
        mp4_settings='OpenCV mp4v defaults; encode after final resize, retain source FPS',
        limitations=['Only final 224-to-96 resize changes; original face and mouth crop interpolation remains unchanged',
            'MP4 at 96px/native FPS is an isolated codec probe, not full official training preprocessing',
            'No adoption decision based on this single-video development set'])
    save('plan.json', plan)  # Freeze conditions before any inference or reference access.
    predictions, provenance = {}, {}
    environment = dict(os.environ, PYTHONIOENCODING='utf-8')
    for interpolation, transport in CONDITIONS:
        condition = f'{interpolation}_{transport}'
        predictions[condition], provenance[condition] = [], []
        for row in inputs:
            directory = output/condition/row['utterance']
            directory.mkdir(parents=True)
            with np.load(source/'crops/scale1.7_none'/row['utterance']/'mouth.npz', allow_pickle=False) as data:
                frames, info = transform(data['frames'], float(data['fps']), interpolation, transport, directory)
                np.savez_compressed(directory/'mouth.npz', frames=frames, fps=float(data['fps']),
                                    crop_scale=1.7, align='none')
            command = [sys.executable, '-m', 'kvsr', 'infer-olkavs', '--mouth', str(directory/'mouth.npz'),
                '--checkpoint', str(checkpoint), '--repo', str(repo), '--id', row['id'],
                '--output', str(directory/'prediction.jsonl'), '--decoder', 'ctc_greedy']
            with (directory/'inference.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
            decoded = read_jsonl(directory/'prediction.jsonl')
            if len(decoded) != 1 or decoded[0]['id'] != row['id']:
                raise ValueError('Mismatched prediction ID')
            prediction = decoded[0]
            predictions[condition].append(prediction)
            provenance[condition].append(dict(id=row['id'], input_sha256=sha256(directory/'mouth.npz'),
                transport=info, **{key: prediction[key] for key in
                ['checkpoint_sha256', 'upstream_commit', 'torch_version', 'input_frames', 'input_fps', 'input_size']}))
            save('progress.json', dict(completed=sum(map(len, predictions.values())), total=16, last_condition=condition))
            print(f'{condition}: {len(predictions[condition])}/4', flush=True)
    # No references enter preparation or decoder subprocesses.
    references = read_jsonl(source/'references.jsonl')
    prior = {r['id']: r for r in read_jsonl(source/'scale17_none/predictions.jsonl')}
    baseline = predictions['area_npz']
    baseline_equal = all(r['token_ids'] == prior[r['id']]['token_ids'] for r in baseline)
    metrics = {name: evaluate(references, rows) for name, rows in predictions.items()}
    pairs = {}
    baseline_errors = {r['id']: r['char_errors'] for r in metrics['area_npz']['details']}
    for name, rows in predictions.items():
        delta = [r['char_errors']-baseline_errors[r['id']] for r in metrics[name]['details']]
        pairs[name] = dict(better=sum(d<0 for d in delta), equal=sum(d==0 for d in delta), worse=sum(d>0 for d in delta),
            changed_token_sequences=sum(r['token_ids'] != b['token_ids'] for r,b in zip(rows, baseline)))
    save('summary.json', dict(plan=plan, plan_sha256=sha256(output/'plan.json'), metrics=metrics,
        paired_vs_area_npz=pairs, baseline_tokens_equal_previous=baseline_equal,
        coverage=dict(planned=16, completed=16, failed=0), provenance=provenance))
    if not baseline_equal:
        raise RuntimeError('Control differs from prior baseline; interpret no condition as an improvement')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'checkpoint', 'repo', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.checkpoint, args.repo, args.output)
