"""CPU preprocessing prototype. Its crops are NOT AV-HuBERT-aligned inputs."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

from .asd import read_asd_archive
from .data import write_jsonl


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def square_crop(frame, center, side, size):
    import cv2
    if not math.isfinite(side) or side < 2 or not all(math.isfinite(v) for v in center):
        raise ValueError("Invalid mouth geometry")
    x = int(round(center[0] - side / 2))
    y = int(round(center[1] - side / 2))
    side = int(math.ceil(side))
    h, w = frame.shape[:2]
    # Refuse silently padded mouths at image borders.
    if x < 0 or y < 0 or x + side > w or y + side > h:
        raise ValueError("Mouth crop crosses the face image boundary")
    return cv2.resize(frame[y:y + side, x:x + side], (size, size), interpolation=cv2.INTER_AREA)


def extract_faces(video, utterance, ffmpeg, track_boundary_limit=2, missing_policy="strict"):
    """Decode a bounded utterance; map official one-based boxes to a 25fps timeline."""
    import cv2
    import numpy as np
    if missing_policy not in {"strict", "mask"} or track_boundary_limit not in (2, 4):
        raise ValueError("Invalid missing policy or track boundary limit")
    first, stop = utterance.frame_range
    if (stop - first) / utterance.fps > 30:
        raise ValueError("Prototype accepts utterances of at most 30 seconds")
    # Require metadata resolution: arbitrary scaling would invalidate speaker boxes.
    cap = cv2.VideoCapture(str(video))
    try:
        dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
    if dimensions != (utterance.width, utterance.height):
        raise ValueError(f"Video dimensions {dimensions} differ from ASD {(utterance.width, utterance.height)}")
    command = [str(ffmpeg), "-v", "error", "-nostdin", "-i", str(video), "-an", "-sn",
               "-vf", f"fps={utterance.fps},trim=start_frame={first}:end_frame={stop}",
               "-frames:v", str(stop - first), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    # Drain stderr into a temporary file to avoid pipe deadlocks; stream frames.
    import tempfile
    faces, tracking = [], []
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            frame_bytes = utterance.width * utterance.height * 3
            for number in range(first, stop):
                raw = process.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    raise ValueError(f"Video ended before expected frame {number}")
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(utterance.height, utterance.width, 3)
                try:
                    x, y, w, h = utterance.box_at(number + 1, boundary_limit=track_boundary_limit)
                except ValueError:
                    if missing_policy == "strict":
                        raise
                    faces.append(np.full((224, 224, 3), 127, dtype=np.uint8))
                    tracking.append(dict(source_frame_one_based=number+1, box=None,
                                         boundary_extended=False, available=False, reason="track_missing"))
                    continue
                if x < 0 or y < 0 or x + w > utterance.width or y + h > utterance.height:
                    raise ValueError(f"ASD face box exceeds image at frame {number + 1}")
                faces.append(cv2.resize(frame[y:y + h, x:x + w], (224, 224)))
                tracking.append(dict(source_frame_one_based=number + 1, box=[x, y, w, h],
                    available=True,
                    boundary_extended=(number+1 < utterance.boxes[0][0] or number+1 > utterance.boxes[-1][0])))
            process.stdout.close()
            code = process.wait(timeout=30)
            if code:
                errors.seek(0)
                raise ValueError(f"FFmpeg failed: {errors.read().decode(errors='replace')[-1000:]}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
    return faces, tracking


def roll_transform(frame, eye_left, eye_right):
    """Rotate the eye line to horizontal; return a real-pixel validity mask."""
    import cv2
    import numpy as np
    left, right = sorted([np.asarray(eye_left), np.asarray(eye_right)], key=lambda p: p[0])
    delta = right - left
    if not np.isfinite(delta).all() or np.linalg.norm(delta) < 2:
        raise ValueError("Invalid eye geometry")
    angle = math.degrees(math.atan2(delta[1], delta[0]))
    h, w = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D(tuple((left + right) / 2), angle, 1)
    rotated = cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), matrix, (w, h), flags=cv2.INTER_LINEAR)
    return rotated, valid, matrix, angle


def fill_geometry_gaps(geometry, max_gap):
    """Interpolate bounded interior coordinate gaps only; never fabricate pixels."""
    import numpy as np
    if not isinstance(max_gap, int) or not 0 <= max_gap <= 3:
        raise ValueError("landmark-max-gap must be an integer from 0 to 3")
    output = list(geometry)
    index = 0
    while index < len(output):
        if output[index] is not None:
            index += 1
            continue
        start = index
        while index < len(output) and output[index] is None:
            index += 1
        if start == 0 or index == len(output) or index-start > max_gap:
            raise ValueError(f"Unrecoverable landmark gap at crop frame {start}; length {index-start}")
        left, right = geometry[start-1], geometry[index]
        for j in range(start, index):
            fraction = (j-start+1)/(index-start+1)
            output[j] = ((1-fraction)*np.asarray(left[0]) + fraction*np.asarray(right[0]),
                         (1-fraction)*left[1] + fraction*right[1])
    return output


def extract_mouths(faces, model, fps, size=96, crop_scale=1.7, align="none", landmark_max_gap=0,
                   missing_policy="strict", face_available=None):
    import cv2
    import mediapipe as mp
    import numpy as np
    if missing_policy not in {"strict", "mask"}:
        raise ValueError("Unknown missing policy")
    if missing_policy == "mask" and (align != "none" or landmark_max_gap):
        raise ValueError("Mask policy requires align=none and no landmark interpolation")
    available = [True]*len(faces) if face_available is None else list(face_available)
    if len(available) != len(faces) or not len(faces):
        raise ValueError("Face availability must match nonempty frames")
    if not math.isfinite(crop_scale) or not 1 <= crop_scale <= 3 or align not in {"none", "eyes"}:
        raise ValueError("crop-scale must be 1..3 and align must be none/eyes")
    if not isinstance(landmark_max_gap, int) or not 0 <= landmark_max_gap <= 3:
        raise ValueError("landmark-max-gap must be 0..3")
    if landmark_max_gap and align != "none":
        raise ValueError("Landmark gap interpolation currently requires align=none")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=Path(model).read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1)
    geometry, transformed, masks, rotations, matrices = [], [], [], [], []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for i, frame in enumerate(faces):
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                round(i * 1000 / fps)) if available[i] else None
            if result is None or len(result.face_landmarks) != 1:
                if not landmark_max_gap and missing_policy == "strict":
                    raise ValueError(f"Face landmark detection failed at crop frame {i}; no synthetic fill applied")
                geometry.append(None)
                transformed.append(frame)
                masks.append(np.full(frame.shape[:2], 255, np.uint8))
                rotations.append(0.)
                matrices.append([[1., 0., 0.], [0., 1., 0.]])
                continue
            points = result.face_landmarks[0]
            # MediaPipe's outer lip ring. Fixed median size avoids per-frame scale pumping.
            ring = [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91]
            xy = np.array([(points[k].x * frame.shape[1], points[k].y * frame.shape[0]) for k in ring])
            if not np.isfinite(xy).all():
                raise ValueError("Non-finite face landmarks")
            angle = 0.0
            matrix = np.array([[1., 0., 0.], [0., 1., 0.]])
            valid = np.full(frame.shape[:2], 255, np.uint8)
            if align == "eyes":
                eyes = [(points[k].x * frame.shape[1], points[k].y * frame.shape[0]) for k in [33, 263]]
                frame, valid, matrix, angle = roll_transform(frame, *eyes)
                xy = cv2.transform(xy[None].astype(np.float64), matrix)[0]
            transformed.append(frame)
            masks.append(valid)
            rotations.append(angle)
            matrices.append(matrix.tolist())
            low, high = xy.min(axis=0), xy.max(axis=0)
            geometry.append((0.5 * (low + high), float(max(high - low) * crop_scale)))
    detected = [g is not None for g in geometry]
    if missing_policy == "strict":
        geometry = fill_geometry_gaps(geometry, landmark_max_gap)
    observed_geometry = [g for g in geometry if g is not None]
    side = float(np.median([g[1] for g in observed_geometry])) if observed_geometry else None
    mouths, trace = [], []
    for i, item in enumerate(geometry):
        if item is None:
            mouths.append(np.full((size, size, 3), 127, dtype=np.uint8))
            trace.append(dict(frame=i, center=None, side=None, detected=False, geometry_interpolated=False,
                              reason="track_missing" if not available[i] else "landmark_missing", align=align))
            continue
        center, _ = item
        # Translate the context window inside the face crop, without fabricated pixels.
        extent = math.ceil(side)
        height, width = faces[i].shape[:2]
        if extent > min(height, width):
            raise ValueError("Mouth context window exceeds the entire face crop")
        bounded = np.clip(center, extent / 2 + .5,
                          np.array([width, height]) - extent / 2 - .5)
        if square_crop(masks[i], bounded, side, math.ceil(side)).min() < 255:
            raise ValueError(f"Aligned mouth contains rotation padding at frame {i}")
        mouths.append(square_crop(transformed[i], bounded, side, size))
        trace.append(dict(frame=i, center=bounded.tolist(), detected_center=center.tolist(),
                          shifted=bool(np.any(bounded != center)), side=side, detected=detected[i],
                          geometry_interpolated=not detected[i],
                          angle_degrees=rotations[i], affine_matrix=matrices[i], align=align))
    return np.stack(mouths), trace


def preprocess_sample(video, archive, video_id, utterance_index, model, output, split, size=96,
                      crop_scale=1.7, align="none", landmark_max_gap=0, track_boundary_limit=2,
                      missing_policy="strict"):
    import cv2
    import imageio_ffmpeg
    import numpy as np
    if size not in {96, 224}:
        raise ValueError("Mouth size must be 96 or 224")
    utterances = read_asd_archive(archive, video_id)
    utterance = next((u for u in utterances if u.index == utterance_index), None)
    if utterance is None:
        raise ValueError("Requested utterance is absent from ASD archive")
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    if not Path(video).is_file() or not Path(model).is_file():
        raise ValueError("Video or face landmark model is missing")
    faces, track = extract_faces(video, utterance, imageio_ffmpeg.get_ffmpeg_exe(), track_boundary_limit, missing_policy)
    mouths, trace = extract_mouths(faces, model, utterance.fps, size=size, crop_scale=crop_scale, align=align,
                                  landmark_max_gap=landmark_max_gap, missing_policy=missing_policy,
                                  face_available=[t["available"] for t in track])
    if align == "eyes":
        faces = [cv2.warpAffine(f, np.array(t["affine_matrix"]), (224, 224)) for f, t in zip(faces, trace)]
    output.mkdir(parents=True)
    np.savez_compressed(output / "mouth.npz", frames=mouths, fps=utterance.fps,
                        crop_scale=crop_scale, align=align, landmark_max_gap=landmark_max_gap,
                        interpolated_frames=sum(t["geometry_interpolated"] for t in trace),
                        roi_observed=np.array([t["detected"] for t in trace], dtype=np.bool_),
                        missing_policy=missing_policy,
                        track_boundary_limit=track_boundary_limit,
                        boundary_extended_frames=sum(t["boundary_extended"] for t in track))
    writer = cv2.VideoWriter(str(output / "face.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             utterance.fps, (224, 224))
    if not writer.isOpened():
        raise ValueError("Cannot open preview video writer")
    try:
        for frame in faces:
            writer.write(frame)
    finally:
        writer.release()
    # Contact sheet shows the same frames in the face and extracted mouth.
    tiles = []
    for index in np.linspace(0, len(faces) - 1, min(8, len(faces)), dtype=int):
        face = faces[index].copy()
        center, side = trace[index]["center"], trace[index]["side"]
        if center is not None:
            x, y = (int(round(v - side / 2)) for v in center)
            cv2.rectangle(face, (x, y), (x + math.ceil(side), y + math.ceil(side)), (0, 255, 0), 2)
        tile = np.vstack([face, cv2.resize(mouths[index], (224, 224))])
        cv2.putText(tile, f"frame {index}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        if center is None:
            cv2.putText(tile, trace[index]["reason"], (8, 260), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
        tiles.append(tile)
    encoded, image = cv2.imencode(".jpg", np.hstack(tiles))
    if not encoded:
        raise ValueError("Contact sheet encoding failed")
    (output / "contact_sheet.jpg").write_bytes(image.tobytes())
    report = dict(id=f"{video_id}/{utterance.index}", split=split, reference=utterance.reference,
                  start_ms=utterance.start_ms, end_ms=utterance.end_ms, fps=utterance.fps,
                  frames=len(mouths), mouth_roi=str((output / "mouth.npz").resolve()),
                  face_video=str((output / "face.mp4").resolve()),
                  color_order="BGR", shape=list(mouths.shape),
                  preprocessing="mediapipe-mouth-prototype-v1", avhubert_compatible=False,
                  crop_scale=crop_scale, align=align, face_preview_aligned=(align == "eyes"),
                  landmark_max_gap=landmark_max_gap,
                  track_boundary_limit=track_boundary_limit,
                  boundary_extended_frames=sum(t["boundary_extended"] for t in track),
                  interpolated_frames=sum(t["geometry_interpolated"] for t in trace),
                  missing_policy=missing_policy, observed_frames=sum(t["detected"] for t in trace),
                  missing_frames=sum(t["center"] is None for t in trace),
                  audio_used=False, model_sha256=sha256(model), video_sha256=sha256(video),
                  asd_sha256=sha256(archive), face_track=track, mouth_track=trace)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(output / "manifest.jsonl", [{k: v for k, v in report.items() if k not in {"face_track", "mouth_track"}}])
    return {k: v for k, v in report.items() if k not in {"face_track", "mouth_track", "reference"}}
