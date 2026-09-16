import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
from collections import Counter
from pathlib import Path

from .data import build_manifest, read_catalog, read_jsonl, select_subset, write_jsonl
from .evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description="Korean VSR stage 1 utilities (no model bundled)")
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
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            report = dict(python=platform.python_version(), platform=platform.system(),
                          packages={name: importlib.util.find_spec(name) is not None
                                    for name in ["torch", "cv2", "numpy", "fairseq"]},
                          executables={name: shutil.which(name) is not None
                                       for name in ["ffmpeg", "nvidia-smi"]},
                          baseline_status="unverified: requires Korean checkpoint, tokenizer, mouth ROI and video-only decoding")
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
        else:
            report = evaluate(read_jsonl(args.references), read_jsonl(args.predictions))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()

