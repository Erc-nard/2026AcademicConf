"""Adapters for the official KMSAV list and cropped utterance layout."""

import hashlib
import json
import math
import re
from pathlib import Path


def read_catalog(path):
    records = []
    seen = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Catalog line {number}: expected at least five columns")
        video_id, domain, participants, duration, split, *flags = fields
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError(f"Catalog line {number}: invalid YouTube ID")
        if video_id in seen:
            raise ValueError(f"Catalog line {number}: duplicate video ID {video_id}")
        seen.add(video_id)
        participants, duration = int(participants), float(duration)
        if participants < 1 or not math.isfinite(duration) or duration < 0:
            raise ValueError(f"Catalog line {number}: invalid numeric metadata")
        if split not in {"train", "valid", "test", "pretrain"}:
            raise ValueError(f"Catalog line {number}: unknown split {split}")
        records.append(dict(video_id=video_id, domain=domain, participants=participants,
                            duration_seconds=duration, split=split, flags=flags))
    if not records:
        raise ValueError("Catalog is empty")
    return records


def select_subset(records, split, count, seed):
    """Keep the official split; exclude flagged and zero-duration sources."""
    if count < 1:
        raise ValueError("count must be positive")
    eligible = [r for r in records if r["split"] == split and not r["flags"]
                and r["duration_seconds"] > 0]
    if len(eligible) < count:
        raise ValueError(f"Requested {count} videos, only {len(eligible)} eligible for {split}")
    return sorted(eligible, key=lambda r: hashlib.sha256(
        f"{seed}:{r['video_id']}".encode()).hexdigest())[:count]


def build_manifest(records, cropped_root):
    """Index face crops, NOT mouth ROIs. Link only available earlier utterance IDs.

    Text stays in a reference field; downstream inference must never consume it.
    File existence is checked, video decoding is a separate preprocessing step.
    """
    root = Path(cropped_root).resolve()
    rows = []
    seen = set()
    for record in records:
        video_id = record["video_id"]
        if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError("Subset contains an invalid video ID")
        if video_id in seen:
            raise ValueError(f"Duplicate subset video: {video_id}")
        if record["split"] not in {"train", "valid", "test"}:
            raise ValueError("Manifest requires a supervised official split")
        if record.get("flags"):
            raise ValueError(f"Flagged subset video: {video_id}")
        seen.add(video_id)
    if not seen:
        raise ValueError("Subset is empty")
    for record in sorted(records, key=lambda r: r["video_id"]):
        video_id = record["video_id"]
        folder = root / video_id / "utts"
        texts = list(folder.glob("*.txt"))
        if not texts:
            raise ValueError(f"No utterance transcripts: {folder}")
        if any(not p.stem.isdigit() for p in texts):
            raise ValueError(f"Non-numeric utterance filename in {folder}")
        if len({int(p.stem) for p in texts}) != len(texts):
            raise ValueError(f"Duplicate numeric utterance ID in {folder}")
        previous_id = None
        for transcript in sorted(texts, key=lambda p: int(p.stem)):
            clip = transcript.with_suffix(".mp4")
            if not clip.is_file() or clip.stat().st_size == 0:
                raise ValueError(f"Missing or empty face clip: {clip}")
            reference = transcript.read_text(encoding="utf-8-sig").strip()
            if not reference:
                raise ValueError(f"Empty transcript: {transcript}")
            sample_id = f"{video_id}/{transcript.stem}"
            rows.append(dict(id=sample_id, video_id=video_id, split=record["split"],
                             utterance_index=int(transcript.stem), face_video=str(clip),
                             reference=reference, previous_available_id=previous_id,
                             mouth_roi=None))
            previous_id = sample_id
    return rows


def write_jsonl(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                      encoding="utf-8")


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("JSONL input is empty")
    return rows
