"""Read KMSAV v0.2 ASD metadata without extracting untrusted ZIP paths."""

import bisect
import math
import re
import zipfile
from dataclasses import dataclass


@dataclass(frozen=True)
class Utterance:
    video_id: str
    index: str
    reference: str
    width: int
    height: int
    fps: int
    start_ms: int
    end_ms: int
    boxes: tuple  # (one-based frame index, x, y, width, height)

    @property
    def frame_range(self):
        # Half-open interval on the metadata's constant-rate timeline.
        return (self.start_ms * self.fps // 1000,
                math.ceil(self.end_ms * self.fps / 1000))

    def box_at(self, frame_index):
        """Only interpolate inside track; boundary extension is limited to 2 frames."""
        indices = [b[0] for b in self.boxes]
        offset = bisect.bisect_left(indices, frame_index)
        if offset < len(indices) and indices[offset] == frame_index:
            return self.boxes[offset][1:]
        if offset in (0, len(indices)):
            row = self.boxes[0 if offset == 0 else -1]
            if abs(row[0] - frame_index) > 2:
                raise ValueError(f"Face track missing at frame {frame_index}")
            return row[1:]
        left, right = self.boxes[offset - 1], self.boxes[offset]
        if right[0] - left[0] > 3:
            raise ValueError(f"Face track gap at frame {frame_index}")
        weight = (frame_index - left[0]) / (right[0] - left[0])
        return tuple(round(a + weight * (b - a)) for a, b in zip(left[1:], right[1:]))


def parse_asd(text, video_id, index):
    headers, boxes = {}, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if ":" in line:
                key, value = line[1:].split(":", 1)
                headers[key.strip()] = value.strip()
        else:
            row = tuple(map(int, line.split()))
            if len(row) != 5 or row[0] < 1 or min(row[3:]) < 1:
                raise ValueError("Invalid ASD bounding box")
            if boxes and row[0] <= boxes[-1][0]:
                raise ValueError("ASD frame indices must increase strictly")
            boxes.append(row)
    width, height = map(int, headers["ImageSize"].split())
    start, end = map(int, headers["Audio Duration(ms)"].split())
    fps = int(headers["FPS"])
    reference = headers["Script"]
    if min(width, height, fps) <= 0 or start < 0 or end <= start or not boxes or not reference:
        raise ValueError("Invalid ASD header or empty track")
    return Utterance(video_id, index, reference, width, height, fps, start, end, tuple(boxes))


def read_asd_archive(path, video_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Invalid video ID")
    pattern = re.compile(rf"(?:[^/]+/)?{re.escape(video_id)}/([0-9]{{6}})\.txt")
    utterances = []
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            match = pattern.fullmatch(entry.filename)
            if match:
                if entry.file_size > 10_000_000:
                    raise ValueError("ASD text unexpectedly large")
                utterances.append(parse_asd(archive.read(entry).decode("utf-8-sig"),
                                            video_id, match[1]))
    if not utterances:
        raise ValueError(f"No ASD utterances for {video_id}")
    if len({u.index for u in utterances}) != len(utterances):
        raise ValueError("Duplicate ASD utterance ID")
    return sorted(utterances, key=lambda u: (u.start_ms, int(u.index)))


def metadata_rows(utterances, split, max_gap_ms=10000):
    """Link a previous available turn only when it ends before this turn starts."""
    if max_gap_ms < 0:
        raise ValueError("max_gap_ms must be nonnegative")
    rows = []
    for utterance in utterances:
        previous = rows[-1] if rows else None
        gap = utterance.start_ms - previous["end_ms"] if previous else None
        usable = (previous is not None and previous["video_id"] == utterance.video_id
                  and 0 <= gap <= max_gap_ms)
        rows.append(dict(id=f"{utterance.video_id}/{utterance.index}",
                         video_id=utterance.video_id, split=split,
                         reference=utterance.reference, start_ms=utterance.start_ms,
                         end_ms=utterance.end_ms, fps=utterance.fps,
                         previous_available_id=previous["id"] if usable else None,
                         previous_gap_ms=gap, context_time_valid=usable))
    return rows
