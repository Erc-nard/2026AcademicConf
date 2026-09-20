"""Replay frozen mouth geometry without rerunning landmark detection.

Three paired paths: the existing face224->mouth224->96 control, face224->96,
and a mapped integer source-image crop->96. Missing ROI availability is fixed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .data import read_jsonl
from .evaluation import evaluate
from .preprocessing import sha256, square_crop

CONDITIONS = ('control', 'face_direct', 'source_direct')


def mouth_rectangle(center, side):
    if (len(center) != 2 or not all(math.isfinite(v) for v in center)
            or not math.isfinite(side) or side < 2):
        raise ValueError('Invalid mouth geometry')
    left, top = [int(round(v-side/2)) for v in center]
    extent = math.ceil(side)
    if min(left, top) < 0 or max(left, top)+extent > 224:
        raise ValueError('Mouth geometry exceeds 224px face')
    return left, top, left+extent, top+extent


def source_rectangle(box, rectangle, width, height):
    """Map pixel edges, then expand outwards to integer source pixel edges."""
    x, y, w, h = box
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > width or y+h > height:
        raise ValueError('Face box exceeds source image')
    left, top, right, bottom = rectangle
    continuous = (x+left*w/224, y+top*h/224, x+right*w/224, y+bottom*h/224)
    rounded = (math.floor(continuous[0]), math.floor(continuous[1]),
               math.ceil(continuous[2]), math.ceil(continuous[3]))
    if min(rounded[:2]) < 0 or rounded[2] > width or rounded[3] > height:
        raise ValueError('Mapped mouth exceeds source image')
    return rounded, max(abs(a-b) for a,b in zip(continuous, rounded))


def replay_frame(frame, box, geometry):
    import cv2
    import numpy as np
    if geometry['center'] is None:
        return {name: np.full((96,96,3), 127, np.uint8) for name in CONDITIONS}, np.full((224,224,3), 127, np.uint8), 0.
    if box is None or geometry.get('align', 'none') != 'none' or geometry.get('geometry_interpolated', False):
        raise ValueError('Replay requires real, unrotated, uninterpolated geometry')
    rectangle = mouth_rectangle(geometry['center'], geometry['side'])
    source, rounding = source_rectangle(box, rectangle, frame.shape[1], frame.shape[0])
    x, y, w, h = box
    face = cv2.resize(frame[y:y+h, x:x+w], (224,224), interpolation=cv2.INTER_LINEAR)
    original = square_crop(face, geometry['center'], geometry['side'], 224)
    left, top, right, bottom = source
    variants = dict(control=cv2.resize(original, (96,96), interpolation=cv2.INTER_AREA),
        face_direct=square_crop(face, geometry['center'], geometry['side'], 96),
        source_direct=cv2.resize(frame[top:bottom, left:right], (96,96), interpolation=cv2.INTER_AREA))
    return variants, original, rounding


def replay_clip(video, report, stored):
    import cv2
    import imageio_ffmpeg
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if min(width, height) <= 0:
        raise ValueError('Unreadable source dimensions')
    track, geometry = report['face_track'], report['mouth_track']
    count = report['frames']
    if len(track) != count or len(geometry) != count or len(stored) != count:
        raise ValueError('Mismatched frame/geometry counts')
    first = report['start_ms']*report['fps']//1000
    stop = math.ceil(report['end_ms']*report['fps']/1000)
    if stop-first != count or [t['source_frame_one_based'] for t in track] != list(range(first+1, stop+1)):
        raise ValueError('Frozen track does not match the source timeline')
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-nostdin', '-i', str(video), '-an', '-sn',
        '-vf', f"fps={report['fps']},trim=start_frame={first}:end_frame={stop}", '-frames:v', str(count),
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
    variants = {name: [] for name in CONDITIONS}
    rounding = 0.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            for i in range(count):
                raw = process.stdout.read(width*height*3)
                if len(raw) != width*height*3:
                    raise ValueError(f'Source ended at frame {i}')
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(height,width,3)
                if geometry[i]['frame'] != i:
                    raise ValueError('Mouth geometry order changed')
                current, original, edge_error = replay_frame(frame, track[i]['box'], geometry[i])
                if not np.array_equal(original, stored[i]):
                    raise ValueError(f'Reconstructed 224px control differs at frame {i}')
                rounding = max(rounding, edge_error)
                for name in CONDITIONS:
                    variants[name].append(current[name])
            process.stdout.close()
            if process.wait(timeout=30):
                errors.seek(0)
                raise ValueError(errors.read().decode(errors='replace')[-1000:])
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
    observed = np.array([g['center'] is not None for g in geometry], dtype=np.bool_)
    return {k: np.stack(v) for k,v in variants.items()}, observed, rounding


def run(source, videos, checkpoint, repo, output):
    import cv2
    import numpy as np
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    inputs, plans = [], []
    for video_id in ['8r1k9NLdSvE', '5TaVl7-lU4k', 'D-hzf2x_XO0']:
        root = source/video_id
        plan = json.loads((root/'plan.json').read_text(encoding='utf-8'))
        if plan['split'] != 'train' or plan['video_id'] != video_id:
            raise ValueError('Expected frozen train video plans')
        plans.append(dict(video_id=video_id, plan_sha256=sha256(root/'plan.json'),
                          video_sha256=sha256(videos/(video_id+'.mp4'))))
        for row in plan['selected']:
            crop = root/'crops/scale1.7_none'/row['utterance']
            inputs.append(dict(id=row['id'], video_id=video_id, utterance=row['utterance'],
                crop_sha256=sha256(crop/'mouth.npz'), geometry_report_sha256=sha256(crop/'report.json')))
    if len(inputs) != 16 or len({r['id'] for r in inputs}) != 16:
        raise ValueError('Expected all 16 distinct frozen development utterances')
    frozen = dict(scope='same three train videos and 16 development utterances; not held-out evaluation',
        inputs=inputs, sources=plans, conditions=list(CONDITIONS), checkpoint_sha256=sha256(checkpoint),
        opencv_version=cv2.__version__, missing_policy='mask; same availability and timeline for all conditions',
        settings=dict(size=96, fps=25, decoder='ctc_greedy', crop_scale=1.7, alignment='none', threads=2),
        source_mapping='224px integer mouth rectangle mapped to original face box; floor left/top, ceil right/bottom',
        reference_used_in_decoding=False, audio_used=False, landmark_detection_rerun=False,
        limitations=['Reuses existing estimated geometry, not manually verified mouth boxes',
            'Source-direct condition also changes subpixel sampling and rounds mapped edges outwards by less than one source pixel',
            'Repeated development set; no statistical superiority or adoption claim'])
    save('plan.json', frozen)
    predictions = {name: [] for name in CONDITIONS}
    provenance = []
    environment = dict(os.environ, PYTHONIOENCODING='utf-8')
    for row in inputs:
        crop = source/row['video_id']/'crops/scale1.7_none'/row['utterance']
        report = json.loads((crop/'report.json').read_text(encoding='utf-8'))
        video_hash = next(p['video_sha256'] for p in plans if p['video_id'] == row['video_id'])
        if report['video_sha256'] != video_hash or report['id'] != row['id'] or report['align'] != 'none':
            raise ValueError('Source provenance mismatch')
        with np.load(crop/'mouth.npz', allow_pickle=False) as data:
            variants, observed, rounding = replay_clip(videos/(row['video_id']+'.mp4'), report, data['frames'])
            if not np.array_equal(observed, data['roi_observed']):
                raise ValueError('ROI availability changed')
        info = dict(id=row['id'], frames=len(observed), usable_frames=int(observed.sum()),
            reconstructed_224_pixels_equal=True, source_edge_rounding_max_pixels=rounding, variants={})
        for name in CONDITIONS:
            directory = output/name/row['video_id']/row['utterance']
            directory.mkdir(parents=True)
            np.savez_compressed(directory/'mouth.npz', frames=variants[name], fps=report['fps'],
                crop_scale=1.7, align='none', roi_observed=observed, missing_policy='mask')
            command = [sys.executable, '-m', 'kvsr', 'infer-olkavs', '--mouth', str(directory/'mouth.npz'),
                '--checkpoint', str(checkpoint), '--repo', str(repo), '--id', row['id'],
                '--output', str(directory/'prediction.jsonl'), '--decoder', 'ctc_greedy']
            with (directory/'inference.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
            decoded = read_jsonl(directory/'prediction.jsonl')
            if len(decoded) != 1 or decoded[0]['id'] != row['id']:
                raise ValueError('Prediction ID mismatch')
            predictions[name].append(decoded[0])
            diff = variants[name][observed].astype(np.float32)-variants['control'][observed].astype(np.float32)
            info['variants'][name] = dict(input_sha256=sha256(directory/'mouth.npz'),
                observed_pixel_mae_vs_control=float(np.abs(diff).mean()) if observed.any() else None,
                inference_performed=decoded[0]['inference_performed'])
        provenance.append(info)
        save('progress.json', dict(completed=len(provenance)*3, planned=48, last_id=row['id']))
        print(f"Completed {len(provenance)}/16 clips, {len(provenance)*3}/48 outputs", flush=True)
    # Read evaluation references and old predictions only after all model calls.
    references, prior = [], {}
    for p in plans:
        root = source/p['video_id']
        references.extend(read_jsonl(root/'references.jsonl'))
        prior.update({r['id']: r for r in read_jsonl(root/'scale17_none/predictions.jsonl')})
    control = predictions['control']
    control_equal = all(r['token_ids'] == prior[r['id']]['token_ids'] for r in control)
    metrics = {name: evaluate(references, rows) for name,rows in predictions.items()}
    per_video = {p['video_id']: {name: evaluate(
        [r for r in references if r['id'].startswith(p['video_id']+'/')],
        [r for r in rows if r['id'].startswith(p['video_id']+'/')]) for name,rows in predictions.items()} for p in plans}
    base_errors = {r['id']: r['char_errors'] for r in metrics['control']['details']}
    paired = {}
    for name, rows in predictions.items():
        deltas = [r['char_errors']-base_errors[r['id']] for r in metrics[name]['details']]
        paired[name] = dict(better=sum(d<0 for d in deltas), equal=sum(d==0 for d in deltas), worse=sum(d>0 for d in deltas),
            changed_token_sequences=sum(r['token_ids'] != b['token_ids'] for r,b in zip(rows,control)),
            inference_performed=sum(r['inference_performed'] for r in rows),
            abstained=sum(not r['inference_performed'] for r in rows))
    save('summary.json', dict(plan=frozen, plan_sha256=sha256(output/'plan.json'), metrics=metrics,
        per_video=per_video, paired_vs_control=paired, provenance=provenance,
        control_tokens_equal_previous=control_equal, coverage=dict(planned=48, completed=48, failed=0),
        model_provenance={k: control[0][k] for k in ['checkpoint_sha256', 'upstream_commit', 'vocabulary_sha256', 'torch_version']}))
    if not control_equal:
        raise RuntimeError('Control predictions differ from prior experiment; no paired accuracy claim')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'videos', 'checkpoint', 'repo', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.videos, args.checkpoint, args.repo, args.output)
