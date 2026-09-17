import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from .data import build_manifest, read_catalog, read_jsonl, select_subset, write_jsonl
from .evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description="Korean VSR research utilities (checkpoints downloaded separately)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report dependencies; does not install or run inference")
    catalog = commands.add_parser("catalog", help="Validate the official KMSAV list")
    catalog.add_argument("path", type=Path)
    subset = commands.add_parser("subset", help="Select whole videos within an official split")
    subset.add_argument("catalog", type=Path)
    subset.add_argument("--split", choices=["train", "valid", "test"], required=True)
    subset.add_argument("--count", type=int, default=1)
    subset.add_argument("--seed", type=int, default=2026)
    subset.add_argument("--output", type=Path, required=True)
    manifest = commands.add_parser("manifest", help="Index official cropped face clips and references")
    manifest.add_argument("--subset", type=Path, required=True)
    manifest.add_argument("--cropped-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("evaluate", help="Evaluate real or explicitly synthetic predictions")
    score.add_argument("--references", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    asd = commands.add_parser("asd", help="Index timed utterances from the official ASD ZIP")
    asd.add_argument("--archive", type=Path, required=True)
    asd.add_argument("--catalog", type=Path, required=True)
    asd.add_argument("--video-id", required=True)
    asd.add_argument("--output", type=Path, required=True)
    prep = commands.add_parser("preprocess", help="Extract a real face track and prototype mouth ROI")
    prep.add_argument("--archive", type=Path, required=True)
    prep.add_argument("--catalog", type=Path, required=True)
    prep.add_argument("--video-id", required=True)
    prep.add_argument("--utterance", required=True, help="Six-digit ASD utterance index")
    prep.add_argument("--video", type=Path, required=True)
    prep.add_argument("--model", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--size", type=int, choices=[96, 224], default=96)
    prep.add_argument("--crop-scale", type=float, default=1.7)
    prep.add_argument("--align", choices=["none", "eyes"], default="none")
    infer = commands.add_parser("infer-olkavs", help="Video-only CTC smoke inference; no transcript/audio input")
    infer.add_argument("--mouth", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--repo", type=Path, required=True)
    infer.add_argument("--id", required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--threads", type=int, default=2)
    infer.add_argument("--decoder", choices=["ctc_greedy", "hybrid"], default="ctc_greedy")
    infer.add_argument("--size", type=int, choices=[96, 224], default=96)
    infer.add_argument("--target-fps", type=int, choices=[30])
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            report = dict(python=platform.python_version(), platform=platform.system(),
                          packages={name: importlib.util.find_spec(name) is not None
                                    for name in ["torch", "cv2", "numpy", "fairseq"]},
                          executables={name: shutil.which(name) is not None
                                       for name in ["ffmpeg", "nvidia-smi"]},
                          baseline_status="Dependency inventory only; see experiment reports for verified runs")
        elif args.command == "catalog":
            rows = read_catalog(args.path)
            report = dict(videos=len(rows), splits=dict(Counter(r["split"] for r in rows)),
                          flagged=sum(bool(r["flags"]) for r in rows))
        elif args.command == "subset":
            rows = select_subset(read_catalog(args.catalog), args.split, args.count, args.seed)
            digest = hashlib.sha256(args.catalog.read_bytes()).hexdigest()
            rows = [dict(r, selection_seed=args.seed, catalog_sha256=digest) for r in rows]
            write_jsonl(args.output, rows)
            report = dict(selected=len(rows), output=str(args.output))
        elif args.command == "manifest":
            rows = build_manifest(read_jsonl(args.subset), args.cropped_root)
            write_jsonl(args.output, rows)
            report = dict(utterances=len(rows), output=str(args.output), status="face crops indexed; mouth ROI pending")
        elif args.command == "infer-olkavs":
            from .olkavs import infer_olkavs
            report = infer_olkavs(args.mouth, args.checkpoint, args.repo, args.id, args.output,
                                 args.threads, args.decoder, args.size, args.target_fps)
        elif args.command in {"asd", "preprocess"}:
            from .asd import metadata_rows, read_asd_archive
            record = next((r for r in read_catalog(args.catalog) if r["video_id"] == args.video_id), None)
            if record is None or record["flags"] or record["split"] == "pretrain":
                raise ValueError("Video must be an unflagged supervised entry in the official catalog")
            if args.command == "asd":
                rows = metadata_rows(read_asd_archive(args.archive, args.video_id), record["split"])
                write_jsonl(args.output, rows)
                report = dict(utterances=len(rows), context_links=sum(r["context_time_valid"] for r in rows),
                              output=str(args.output))
            else:
                from .preprocessing import preprocess_sample
                report = preprocess_sample(args.video, args.archive, args.video_id, args.utterance,
                                           args.model, args.output, record["split"], args.size,
                                           args.crop_scale, args.align)
        else:
            report = evaluate(read_jsonl(args.references), read_jsonl(args.predictions))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, TypeError, OSError, ImportError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
