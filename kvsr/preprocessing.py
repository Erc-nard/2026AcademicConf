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


def extract_faces(video, utterance, ffmpeg):
    """Decode a bounded utterance; map official one-based boxes to a 25fps timeline."""
    import cv2
    import numpy as np
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
                x, y, w, h = utterance.box_at(number + 1)
                if x < 0 or y < 0 or x + w > utterance.width or y + h > utterance.height:
                    raise ValueError(f"ASD face box exceeds image at frame {number + 1}")
                faces.append(cv2.resize(frame[y:y + h, x:x + w], (224, 224)))
                tracking.append(dict(source_frame_one_based=number + 1, box=[x, y, w, h]))
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


def extract_mouths(faces, model, fps, size=96):
    import cv2
    import mediapipe as mp
    import numpy as np
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=Path(model).read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1)
    geometry = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for i, frame in enumerate(faces):
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                round(i * 1000 / fps))
            if len(result.face_landmarks) != 1:
                raise ValueError(f"Face landmark detection failed at crop frame {i}; no synthetic fill applied")
            points = result.face_landmarks[0]
            # MediaPipe's outer lip ring. Fixed median size avoids per-frame scale pumping.
            ring = [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91]
            xy = np.array([(points[k].x * frame.shape[1], points[k].y * frame.shape[0]) for k in ring])
            if not np.isfinite(xy).all():
                raise ValueError("Non-finite face landmarks")
            low, high = xy.min(axis=0), xy.max(axis=0)
            geometry.append((0.5 * (low + high), float(max(high - low) * 1.7)))
    side = float(np.median([g[1] for g in geometry]))
    mouths, trace = [], []
    for i, (center, _) in enumerate(geometry):
        # Translate the context window inside the face crop, without fabricated pixels.
        extent = math.ceil(side)
        height, width = faces[i].shape[:2]
        if extent > min(height, width):
            raise ValueError("Mouth context window exceeds the entire face crop")
        bounded = np.clip(center, extent / 2 + .5,
                          np.array([width, height]) - extent / 2 - .5)
        mouths.append(square_crop(faces[i], bounded, side, size))
        trace.append(dict(frame=i, center=bounded.tolist(), detected_center=center.tolist(),
                          shifted=bool(np.any(bounded != center)), side=side, detected=True))
    return np.stack(mouths), trace


def preprocess_sample(video, archive, video_id, utterance_index, model, output, split, size=96):
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
    faces, track = extract_faces(video, utterance, imageio_ffmpeg.get_ffmpeg_exe())
    mouths, trace = extract_mouths(faces, model, utterance.fps, size=size)
    output.mkdir(parents=True)
    np.savez_compressed(output / "mouth.npz", frames=mouths, fps=utterance.fps)
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
        x, y = (int(round(v - side / 2)) for v in center)
        cv2.rectangle(face, (x, y), (x + math.ceil(side), y + math.ceil(side)), (0, 255, 0), 2)
        tile = np.vstack([face, cv2.resize(mouths[index], (224, 224))])
        cv2.putText(tile, f"frame {index}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
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
                  audio_used=False, model_sha256=sha256(model), video_sha256=sha256(video),
                  asd_sha256=sha256(archive), face_track=track, mouth_track=trace)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(output / "manifest.jsonl", [{k: v for k, v in report.items() if k not in {"face_track", "mouth_track"}}])
    return {k: v for k, v in report.items() if k not in {"face_track", "mouth_track", "reference"}}
