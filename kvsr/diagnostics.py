"""Audit failed preprocessing independently of reference transcripts."""

import argparse
import json
from pathlib import Path

from .asd import read_asd_archive
from .preprocessing import extract_faces, sha256


def contiguous_runs(indices):
    values = sorted(set(indices))
    runs = []
    for value in values:
        if runs and value == runs[-1]['end'] + 1:
            runs[-1]['end'] = value
            runs[-1]['length'] += 1
        else:
            runs.append(dict(start=value, end=value, length=1))
    return runs


def audit_track(utterance):
    first, stop = utterance.frame_range
    exact = {b[0] for b in utterance.boxes}
    missing, synthesized = [], []
    for number in range(first+1, stop+1):
        try:
            utterance.box_at(number)
            if number not in exact:
                synthesized.append(number)
        except ValueError:
            missing.append(number)
    return dict(frame_index='one-based source timeline; runs inclusive',
        requested_frames=stop-first, requested_first=first+1, requested_last=stop,
        available_first=utterance.boxes[0][0], available_last=utterance.boxes[-1][0],
        missing_frames=len(missing), missing_runs=contiguous_runs(missing),
        interpolated_or_extended_frames=len(synthesized))


def main():
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(Path('results/matplotlib').resolve()))
    import cv2
    import imageio_ffmpeg
    import mediapipe as mp
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--retry-confidence', type=float, choices=[.3, .5], default=.5)
    args = parser.parse_args()
    summaries = sorted(args.source.glob('*/summary.json'))
    failures = [f for p in summaries for f in json.loads(p.read_text(encoding='utf-8'))
                ['conditions']['scale17_none']['failures']]
    if not failures:
        raise ValueError('No failed baseline samples found')
    args.output.mkdir(parents=True, exist_ok=False)
    plan = dict(ids=[f['id'] for f in failures], selection='all failed baseline samples',
                modes='VIDEO; IMAGE retry only on VIDEO failures',
                video_thresholds=.5, image_detection_presence_thresholds=args.retry_confidence,
                reference_used=False, model_sha256=sha256(args.model), archive_sha256=sha256(args.archive),
                source_summary_sha256={p.parent.name:sha256(p) for p in summaries})
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    reports = []
    for failure in failures:
        video_id, index = failure['id'].split('/')
        utterance = next(u for u in read_asd_archive(args.archive, video_id) if u.index == index)
        row = dict(id=failure['id'], track=audit_track(utterance))
        if not row['track']['missing_frames']:
            faces, _ = extract_faces(args.raw/f'{video_id}.mp4', utterance, imageio_ffmpeg.get_ffmpeg_exe())
            def options(mode):
                return mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_buffer=args.model.read_bytes()),
                    running_mode=mode, num_faces=1,
                    min_face_detection_confidence=args.retry_confidence if mode == mp.tasks.vision.RunningMode.IMAGE else .5,
                    min_face_presence_confidence=args.retry_confidence if mode == mp.tasks.vision.RunningMode.IMAGE else .5)
            video_missing, recovered, remaining, retry_lips = [], [], [], {}
            with mp.tasks.vision.FaceLandmarker.create_from_options(options(mp.tasks.vision.RunningMode.VIDEO)) as video_detector, \
                 mp.tasks.vision.FaceLandmarker.create_from_options(options(mp.tasks.vision.RunningMode.IMAGE)) as image_detector:
                for i, frame in enumerate(faces):
                    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = video_detector.detect_for_video(img, round(i*1000/utterance.fps))
                    if len(result.face_landmarks) != 1:
                        video_missing.append(i)
                        retry = image_detector.detect(img)
                        (recovered if len(retry.face_landmarks) == 1 else remaining).append(i)
                        if len(retry.face_landmarks) == 1:
                            retry_lips[i] = [(int(p.x*224),int(p.y*224)) for k,p in enumerate(retry.face_landmarks[0])
                                            if k in (61,40,37,0,267,270,291,321,314,17,84,91)]
            row['landmarks'] = dict(frame_index='zero-based crop frame; runs inclusive',
                frames=len(faces), video_missing=len(video_missing), video_missing_runs=contiguous_runs(video_missing),
                image_recovered=len(recovered), image_recovered_runs=contiguous_runs(recovered),
                remaining_missing=len(remaining), remaining_runs=contiguous_runs(remaining))
            pool = sorted(set([0, len(faces)-1] + [i for run in contiguous_runs(video_missing)
                for i in (run['start'], (run['start']+run['end'])//2, run['end'])]))
            chosen = [pool[i] for i in np.linspace(0,len(pool)-1,min(12,len(pool)),dtype=int)]
            tiles = []
            for i in chosen:
                tile = faces[i].copy()
                for point in retry_lips.get(i, []):
                    cv2.circle(tile, point, 1, (0, 255, 255), -1)
                label = f'{i}: ' + ('IMAGE OK' if i in recovered else 'BOTH FAIL' if i in remaining else 'VIDEO OK')
                cv2.putText(tile, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 1)
                tiles.append(tile)
            ok, image = cv2.imencode('.jpg', np.hstack(tiles))
            if not ok:
                raise ValueError('Could not encode audit sheet')
            (args.output/f'{video_id}_{index}.jpg').write_bytes(image.tobytes())
        reports.append(row)
        (args.output/'summary.json').write_text(json.dumps(dict(plan=plan, samples=reports,
            complete=len(reports) == len(failures)), indent=2)+'\n', encoding='utf-8')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
