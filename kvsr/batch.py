"""Frozen, failure-aware single-video development benchmark.

Run with python -m kvsr.batch --help. Never use this train subset as a test score.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .asd import read_asd_archive
from .data import read_catalog, read_jsonl, write_jsonl
from .evaluation import evaluate
from .preprocessing import sha256


PROFILES = {"size96_native": (96, None), "size96_30fps": (96, 30),
            "size224_native": (224, None), "size224_30fps": (224, 30)}


def summarize(references, predictions, failures):
    expected = {r["id"] for r in references}
    actual = {r["id"] for r in predictions}
    failed = {r["id"] for r in failures}
    if (not references or len(expected) != len(references) or len(actual) != len(predictions)
            or len(failed) != len(failures) or actual & failed or actual | failed != expected):
        raise ValueError("Each selected ID must have exactly one prediction or failure")
    # Empty predictions count all failed utterances as deletions in operational scoring.
    all_predictions = predictions + [dict(id=r["id"], hypothesis="") for r in failures]
    return dict(selected=len(references), succeeded=len(predictions), failed=len(failures),
                coverage=len(predictions) / len(references),
                all_selected_failure_as_empty=evaluate(references, all_predictions),
                successful_only=evaluate([r for r in references if r["id"] in actual], predictions)
                if predictions else None, failures=failures)


def run_command(arguments, log, timeout, env):
    with log.open("w", encoding="utf-8") as stream:
        try:
            result = subprocess.run([sys.executable, "-m", "kvsr", *map(str, arguments)],
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return f"Timeout after {timeout}s; see {log.name}"
    if result.returncode:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-1][:500] if lines else f"Process exit {result.returncode}"
    return None


def run(args):
    root = args.output.resolve()
    if root.exists():
        raise ValueError("Output exists; use a new run directory")
    if args.max_seconds <= 0:
        raise ValueError("max-seconds must be positive")
    record = next((r for r in read_catalog(args.catalog) if r["video_id"] == args.video_id), None)
    if record is None or record["split"] != "train" or record["flags"]:
        raise ValueError("Development benchmark requires an unflagged train video")
    for path in [args.video, args.archive, args.model, args.checkpoint]:
        if not path.is_file():
            raise ValueError(f"Missing input: {path}")
    items = read_asd_archive(args.archive, args.video_id)
    selected = [u for u in items if u.end_ms - u.start_ms <= args.max_seconds * 1000]
    if not selected:
        raise ValueError("No utterances satisfy the duration rule")
    root.mkdir(parents=True)
    plan = dict(scope="single-video train development; not held-out test performance",
                video_id=args.video_id, split="train", max_seconds=args.max_seconds,
                total_available=len(items), selection_rule="all utterances within duration limit",
                selected=[dict(id=f"{u.video_id}/{u.index}", utterance=u.index,
                               start_ms=u.start_ms, end_ms=u.end_ms) for u in selected],
                excluded=[dict(id=f"{u.video_id}/{u.index}", reason="duration") for u in items if u not in selected],
                profiles={k: dict(size=v[0], target_fps=v[1], decoder="ctc_greedy") for k, v in PROFILES.items()},
                hashes={name: sha256(path) for name, path in [("video", args.video), ("asd", args.archive),
                         ("catalog", args.catalog), ("landmarker", args.model), ("checkpoint", args.checkpoint)]})
    (root / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    references = [dict(id=f"{u.video_id}/{u.index}", reference=u.reference) for u in selected]
    write_jsonl(root / "references.jsonl", references)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLCONFIGDIR=str(root / "matplotlib"))
    predictions = {p: [] for p in PROFILES}
    failures = {p: [] for p in PROFILES}
    for number, item in enumerate(selected, 1):
        sample_id = f"{item.video_id}/{item.index}"
        output = root / "crops" / item.index
        log = root / f"preprocess_{item.index}.log"
        error = run_command(["preprocess", "--archive", args.archive, "--catalog", args.catalog,
                             "--video-id", args.video_id, "--utterance", item.index,
                             "--video", args.video, "--model", args.model, "--size", 224,
                             "--output", output], log, 180, env)
        print(f"[{number}/{len(selected)}] {sample_id}: preprocess {'FAILED' if error else 'OK'}", flush=True)
        for name, (size, fps) in PROFILES.items():
            failure = error
            destination = root / name / f"{item.index}.jsonl"
            destination.parent.mkdir(exist_ok=True)
            if not error:
                command = ["infer-olkavs", "--mouth", output / "mouth.npz", "--checkpoint", args.checkpoint,
                           "--repo", args.repo, "--id", sample_id, "--output", destination, "--size", size]
                if fps:
                    command += ["--target-fps", fps]
                failure = run_command(command, root / f"{name}_{item.index}.log", 300, env)
            if failure:
                failures[name].append(dict(id=sample_id, stage="preprocess" if error else "inference", error=failure))
            else:
                rows = read_jsonl(destination)
                if len(rows) != 1 or rows[0]["id"] != sample_id:
                    raise ValueError("Inference returned mismatched IDs")
                predictions[name].append(rows[0])
            print(f"  {name}: {'FAILED' if failure else 'OK'}", flush=True)
        progress = {p: dict(succeeded=len(predictions[p]), failed=len(failures[p])) for p in PROFILES}
        (root / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
    common = set.intersection(*[{r["id"] for r in rows} for rows in predictions.values()])
    summary = dict(plan_sha256=sha256(root / "plan.json"), plan=plan, paired_ids=sorted(common),
                   inference_provenance={p: {k: predictions[p][0][k] for k in
                       ["upstream_commit", "torch_version", "vocabulary_sha256", "preprocessing", "device"]}
                       if predictions[p] else None for p in PROFILES},
                   conditions={p: summarize(references, predictions[p], failures[p]) for p in PROFILES},
                   paired_metrics={p: evaluate([r for r in references if r["id"] in common],
                                              [r for r in predictions[p] if r["id"] in common])
                                   if common else None for p in PROFILES})
    for name, rows in predictions.items():
        write_jsonl(root / name / "predictions.jsonl", rows)
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Finished: {root / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["archive", "catalog", "video", "model", "checkpoint", "repo", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--max-seconds", type=float, default=10)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
